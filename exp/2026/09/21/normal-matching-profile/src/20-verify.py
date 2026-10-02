"""Independent endpoint and matched-step audit of the 2-D continuation study."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import normal_study as ns
import numpy as np
import pydantic_settings as ps
import scipy.sparse.linalg as spla
from experiment import Profile

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    input_dir: Path = Path("10-comparison")
    output: Path = Path("20-verification")
    force_residual_tolerance: float = 1e-10
    hessian_max_iterations: int = 2_000


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_normal_metrics(
    mesh: Any, full_u: np.ndarray, height: float
) -> dict[str, float]:
    """Recompute the oriented-top-segment normal metric without normal_loss()."""
    top = ns.gs.top_nodes(mesh)
    reference = mesh.p[top]
    points = reference + full_u[top]
    target = reference + np.column_stack(
        (np.zeros(len(top)), 4 * height * reference[:, 0] * (1 - reference[:, 0]))
    )
    edge = np.diff(points, axis=0)
    target_edge = np.diff(target, axis=0)
    length = np.linalg.norm(edge, axis=1)
    target_length = np.linalg.norm(target_edge, axis=1)
    assert np.all(length > 1e-12)
    tangent, target_tangent = (
        edge / length[:, None],
        target_edge / target_length[:, None],
    )
    weight = np.diff(reference[:, 0]) / (reference[-1, 0] - reference[0, 0])
    dot = np.sum(tangent * target_tangent, axis=1)
    angle = np.arctan2(
        tangent[:, 0] * target_tangent[:, 1] - tangent[:, 1] * target_tangent[:, 0], dot
    )
    loss = float(np.sum(weight * (1 - dot)))
    return {
        "normal_loss": loss,
        "normal_angle_rms_deg": float(np.degrees(np.sqrt(np.sum(weight * angle**2)))),
        "normal_chord_rms": float(np.sqrt(2 * loss)),
        "minimum_top_edge_length": float(length.min()),
    }


def independent_gradient_metrics(
    mesh: Any, full_u: np.ndarray, height: float
) -> dict[str, float]:
    top = ns.gs.top_nodes(mesh)
    x = mesh.p[top, 0]
    dx = np.diff(x)
    target = np.column_stack((np.zeros_like(x), 4 * height * x * (1 - x)))
    error = full_u[top] - target
    slope = np.diff(error, axis=0) / dx[:, None]
    gradient_loss = float(np.sum(dx[:, None] * slope**2) / (x[-1] - x[0]))
    center_dx = (dx[:-1] + dx[1:]) / 2
    curvature = np.diff(slope, axis=0) / center_dx[:, None]
    return {
        "gradient_loss": gradient_loss,
        "slope_rms": float(np.sqrt(gradient_loss)),
        "curvature_error": float(
            np.sqrt(np.sum(center_dx[:, None] * curvature**2) / np.sum(center_dx))
        ),
    }


def trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows, path
    assert [int(row["step"]) for row in rows] == list(range(len(rows))), path
    return rows


def feasibility(mode: str, q: np.ndarray, B: np.ndarray) -> dict[str, float]:
    offset = B - np.eye(2)
    eigen = np.linalg.eigvalsh(offset)
    result = {
        "minimum_offset_eigenvalue": float(eigen.min()),
        "rank_one_secondary_abs_max": float(np.max(np.abs(eigen[:, 0]))),
    }
    if mode == "contraction_only":
        assert eigen.min() >= -1e-12, result
    elif mode == "learned_direction":
        strength = q.reshape(-1, 2)[:, 0]
        assert strength.min() >= -1e-12, strength.min()
        assert result["rank_one_secondary_abs_max"] <= 1e-10, result
        result["minimum_strength"] = float(strength.min())
    elif mode == "x_contraction":
        assert q.min() >= -1e-12, q.min()
        assert np.max(np.abs(offset[:, 1, :])) <= 1e-12
        assert np.max(np.abs(offset[:, :, 1])) <= 1e-12
        result["minimum_strength"] = float(q.min())
    else:
        assert mode == "unconstrained"
    return result


def physical_hessian(hessian: Any, max_iterations: int) -> dict[str, Any]:
    symmetric = (hessian + hessian.T) * 0.5
    try:
        value, vector = spla.eigsh(
            symmetric, k=1, which="SA", tol=1e-8, maxiter=max_iterations, ncv=80
        )
    except spla.ArpackNoConvergence as exc:
        return {
            "physical_hessian_smallest_algebraic_eigenvalue": None,
            "physical_hessian_status": "ARPACK did not converge; no eigenvalue claim",
            "physical_hessian_converged_pairs": len(exc.eigenvalues),
        }
    eigenvalue = float(value[0])
    residual = float(
        np.linalg.norm(symmetric @ vector[:, 0] - eigenvalue * vector[:, 0])
        / max(np.linalg.norm(symmetric @ vector[:, 0]), 1.0)
    )
    assert residual <= 1e-7, residual
    return {
        "physical_hessian_smallest_algebraic_eigenvalue": eigenvalue,
        "physical_hessian_eigen_residual": residual,
        "physical_hessian_status": "ARPACK smallest algebraic eigenpair",
    }


def metric_at_step(
    mesh: Any, folder: Path, step: int, *, force_tolerance: float
) -> dict[str, Any]:
    history = np.load(folder / "history.npz", allow_pickle=False)
    index = np.flatnonzero(history["steps"] == step)
    assert len(index) == 1, (folder, step, history["steps"])
    i = int(index[0])
    q, full_u = history["controls"][i], history["u"][i]
    mode, height = str(history["mode"]), float(history["height"])
    assert np.array_equal(history["points"], mesh.p)
    B = ns.gs.study.matrices(mesh, q, mode)
    packed_u = ns.pack(mesh, full_u)
    _, residual, hessian, physical_j = ns.ph.assemble(mesh, packed_u, B)
    assert hessian is not None
    position_loss = ns.ph.loss(mesh, packed_u, height, "l2")[0]
    normal = independent_normal_metrics(mesh, full_u, height)
    gradient = independent_gradient_metrics(mesh, full_u, height)
    residual_norm = float(np.linalg.norm(residual, np.inf))
    assert residual_norm <= force_tolerance, (folder, step, residual_norm)
    fixed = mesh.lookup.reshape(-1, 2) < 0
    assert np.max(np.abs(full_u[fixed])) <= 1e-14
    assert physical_j.min() > 0
    row = next(row for row in trace(folder / "trace.csv") if int(row["step"]) == step)
    for name, actual in {
        "position_loss": position_loss,
        **normal,
        **gradient,
        "min_J": float(physical_j.min()),
    }.items():
        assert np.isclose(actual, float(row[name]), rtol=1e-9, atol=1e-11), (
            folder,
            step,
            name,
            actual,
            row[name],
        )
    top = mesh.top
    target = np.zeros_like(full_u[top])
    target[:, 1] = 4 * height * mesh.p[top, 0] * (1 - mesh.p[top, 0])
    prediction = full_u[top]
    top_all = ns.gs.top_nodes(mesh)
    result = {
        "case": folder.name,
        "step": int(step),
        "fit_rms": float(np.sqrt(position_loss)),
        "position_loss": position_loss,
        **normal,
        **gradient,
        "target_projection": float(np.sum(prediction * target) / np.sum(target**2)),
        "motion_rms": float(np.sqrt(np.mean(np.sum(prediction**2, axis=1)))),
        "peak_target_fraction": float(row["peak_target_fraction"]),
        "activation_roughness": float(row["activation_roughness"]),
        "min_J": float(physical_j.min()),
        "max_J": float(physical_j.max()),
        "inverted_cells": int(np.sum(physical_j <= 0)),
        "force_residual_inf": residual_norm,
        "fixed_displacement_inf": float(np.max(np.abs(full_u[fixed]))),
        "top_edge_backtracking_count": int(
            np.sum(np.diff((mesh.p[top_all] + full_u[top_all])[:, 0]) <= 0)
        ),
        "constraints": feasibility(mode, q, B[mesh.muscle]),
        "trace_objective": float(row["objective"]),
        "history_sha256": sha256(folder / "history.npz"),
    }
    for name in ("target_projection", "motion_rms", "top_edge_backtracking_count"):
        assert np.isclose(result[name], float(row[name]), rtol=1e-9, atol=1e-11), (
            folder,
            step,
            name,
            result[name],
            row[name],
        )
    return result, hessian


def all_trace_feasible(folder: Path) -> bool:
    rows = trace(folder / "trace.csv")
    assert all(
        int(row["inverted_cells"]) == 0 and float(row["min_J"]) > 0 for row in rows
    )
    return True


def no_failed_proposal_by(folder: Path, step: int) -> bool:
    failure = folder / "failure.json"
    if not failure.exists():
        return True
    failed_step = int(json.loads(failure.read_text())["step"])
    assert failed_step > step, (folder, failed_step, step)
    return True


def choose(kind: str, metrics: dict[str, dict[str, Any]]) -> str | None:
    eligible = []
    for name, metric in metrics.items():
        if name.startswith(kind) and metric["eligible"]:
            eligible.append(name)
    return min(eligible, key=lambda name: metrics[name]["normal_loss"], default=None)


def source_gate(source: Path) -> dict[str, Any]:
    checks = json.loads((source / "input-gates.json").read_text())
    assert checks["passed"]
    protocol = json.loads((source / "protocol.json").read_text())
    for source_path, digest in protocol["source_sha256"].items():
        snapshot = source / "source" / source_path
        assert snapshot.is_file(), snapshot
        assert sha256(snapshot) == digest, snapshot
        live = ns.ROOT / source_path
        assert live.is_file(), live
        assert sha256(live) == digest, live
    return {
        "input_gate_sha256": sha256(source / "input-gates.json"),
        "source_count": len(protocol["source_sha256"]),
        "source_snapshot_hashes_verified": True,
    }


def feasible_actual_directions(
    q: np.ndarray, mode: str
) -> list[tuple[str, np.ndarray]]:
    """Directions that stay in each control parameterization's feasible domain."""
    if mode == "contraction_only":
        return [("PSD-offset scaling", q / np.linalg.norm(q))]
    if mode == "x_contraction":
        return [("nonnegative-strength scaling", q / np.linalg.norm(q))]
    if mode == "learned_direction":
        values = q.reshape(-1, 2)
        strength = np.zeros_like(values)
        strength[:, 0] = values[:, 0]
        angle = np.zeros_like(values)
        angle[:, 1] = np.sin(np.arange(len(values)))
        return [
            (
                "nonnegative-strength scaling",
                strength.ravel() / np.linalg.norm(strength),
            ),
            ("free-angle perturbation", angle.ravel() / np.linalg.norm(angle)),
        ]
    assert mode == "unconstrained"
    direction = np.sin(np.arange(q.size) + 0.5)
    return [
        ("unconstrained deterministic direction", direction / np.linalg.norm(direction))
    ]


def historical_q200_derivatives() -> dict[str, Any]:
    """Recheck both mixed losses at each archived actual continuation start."""
    mesh = ns.ph.build_mesh(100, 10)
    output: dict[str, Any] = {}
    for folder in sorted(ns.BASELINE.glob("h*-w0")):
        history = np.load(folder / "history.npz", allow_pickle=False)
        index = np.flatnonzero(history["steps"] == 200)
        assert len(index) == 1
        i = int(index[0])
        q, full_u = history["controls"][i], history["u"][i]
        mode, height = str(history["mode"]), float(history["height"])
        seed = ns.pack(mesh, full_u)
        scales = ns.normalization(mesh, height)
        kinds: dict[str, Any] = {}
        for kind in ("gradient", "normal"):
            _, _, gradient, _ = ns.evaluate(
                mesh, q, mode, height, kind, 0.25, scales, seed
            )
            records = []
            for label, direction in feasible_actual_directions(q, mode):
                # The archived derivatives can be O(1e-5), below the forward
                # solve's repeatability at 1e-6.  Store two central differences
                # in the measured plateau rather than rely on that assertion.
                finite_differences = []
                for step in (1e-4, 3e-4):
                    numeric = (
                        ns.evaluate(
                            mesh,
                            q + step * direction,
                            mode,
                            height,
                            kind,
                            0.25,
                            scales,
                            seed,
                        )[3]["objective"]
                        - ns.evaluate(
                            mesh,
                            q - step * direction,
                            mode,
                            height,
                            kind,
                            0.25,
                            scales,
                            seed,
                        )[3]["objective"]
                    ) / (2 * step)
                    finite_differences.append(
                        {"epsilon": step, "value": float(numeric)}
                    )
                analytic = float(gradient @ direction)
                relative_errors = [
                    abs(analytic - item["value"])
                    / max(abs(analytic), abs(item["value"]), 1e-12)
                    for item in finite_differences
                ]
                plateau_relative_error = abs(
                    finite_differences[0]["value"] - finite_differences[1]["value"]
                ) / max(
                    abs(finite_differences[0]["value"]),
                    abs(finite_differences[1]["value"]),
                    1e-12,
                )
                assert max(relative_errors) < 5e-5, (
                    folder,
                    kind,
                    label,
                    analytic,
                    finite_differences,
                )
                assert plateau_relative_error < 5e-5, (
                    folder,
                    kind,
                    label,
                    finite_differences,
                )
                records.append(
                    {
                        "direction": label,
                        "analytic": analytic,
                        "finite_differences": finite_differences,
                        "maximum_relative_error": float(max(relative_errors)),
                        "plateau_relative_error": float(plateau_relative_error),
                    }
                )
            kinds[kind] = records
        output[folder.name] = kinds
    assert len(output) == 8
    return output


def main(cfg: Config) -> None:
    source = (
        cfg.input_dir if cfg.input_dir.is_dir() else ns.GROUP / "data" / cfg.input_dir
    )
    assert source.is_dir(), source
    mesh = ns.ph.build_mesh(100, 10)
    result: dict[str, Any] = {
        "source_gate": source_gate(source),
        "historical_q200_mixed_objective_derivatives": historical_q200_derivatives(),
        "selection_limits": {
            "fit_rms_ratio_maximum": 1.05,
            "target_projection_ratio_minimum": 0.95,
            "required_all_trace_states_inversion_free": True,
        },
        "cells": [],
    }
    hessian_requests: list[tuple[dict[str, Any], Any]] = []
    for cell in sorted(path for path in source.glob("h*-*") if path.is_dir()):
        variants = {
            path.name: path
            for path in cell.iterdir()
            if (path / "history.npz").is_file()
        }
        assert set(variants) == {item[0] for item in ns.VARIANTS}, variants
        shared = set.intersection(
            *(
                set(np.load(folder / "history.npz", allow_pickle=False)["steps"])
                for folder in variants.values()
            )
        )
        shared_step = max(shared)
        matched = {
            name: metric_at_step(
                mesh, folder, shared_step, force_tolerance=cfg.force_residual_tolerance
            )
            for name, folder in variants.items()
        }
        metrics = {name: value[0] for name, value in matched.items()}
        hessians = {name: value[1] for name, value in matched.items()}
        endpoints: dict[str, Any] = {}
        for name, folder in variants.items():
            history = np.load(folder / "history.npz", allow_pickle=False)
            endpoint_step = int(history["steps"][-1])
            checkpoint = np.load(folder / "checkpoint.npz", allow_pickle=False)
            assert int(checkpoint["step"]) == endpoint_step
            assert np.array_equal(checkpoint["controls"], history["controls"][-1])
            assert np.array_equal(checkpoint["u"], history["u"][-1])
            endpoint, _ = metric_at_step(
                mesh,
                folder,
                endpoint_step,
                force_tolerance=cfg.force_residual_tolerance,
            )
            endpoints[name] = endpoint
        for name, folder in variants.items():
            metrics[name]["all_trace_states_inversion_free"] = all_trace_feasible(
                folder
            )
            metrics[name]["no_failed_proposal_by_shared_step"] = no_failed_proposal_by(
                folder, shared_step
            )
        control = metrics["l2"]
        for metric in metrics.values():
            for name in (
                "fit_rms",
                "motion_rms",
                "target_projection",
                "gradient_loss",
                "normal_loss",
            ):
                metric[f"{name}_ratio_vs_l2"] = metric[name] / control[name]
            metric["eligible"] = (
                metric["fit_rms"] <= 1.05 * control["fit_rms"]
                and metric["target_projection"] >= 0.95 * control["target_projection"]
                and metric["all_trace_states_inversion_free"]
                and metric["no_failed_proposal_by_shared_step"]
            )
        selected = {kind: choose(kind, metrics) for kind in ("gradient", "normal")}
        hessian_requests.append((metrics["l2"], hessians["l2"]))
        hessian_requests.extend(
            (metrics[name], hessians[name])
            for name in selected.values()
            if name is not None
        )
        qualifying_backtracking = {
            name: metric["top_edge_backtracking_count"]
            for name, metric in metrics.items()
            if metric["eligible"] and metric["top_edge_backtracking_count"] > 0
        }
        result["cells"].append(
            {
                "name": cell.name,
                "mode": str(
                    np.load(variants["l2"] / "history.npz", allow_pickle=False)["mode"]
                ),
                "height": float(
                    np.load(variants["l2"] / "history.npz", allow_pickle=False)[
                        "height"
                    ]
                ),
                "shared_step": int(shared_step),
                "variants": metrics,
                "endpoints": endpoints,
                "selected": selected,
                "qualifying_candidate_backtracking_warning": qualifying_backtracking,
            }
        )
    for metric, hessian in hessian_requests:
        metric["physical_hessian"] = physical_hessian(
            hessian, cfg.hessian_max_iterations
        )
    result["physical_hessian_note"] = (
        "Forward-equilibrium displacement Hessian at selected matched-step states; it is not an inverse-objective Hessian."
    )
    result["passed"] = True
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "checks.json", result)
    write_json(
        output / "selection.json",
        {"cells": result["cells"], "limits": result["selection_limits"]},
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
