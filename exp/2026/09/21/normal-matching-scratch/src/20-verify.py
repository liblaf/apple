"""Independent audit of every neutral-start factorial endpoint and shared state."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import normal_smooth_study as ns
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
    hessian_max_iterations: int = 2000


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        result = list(csv.DictReader(stream))
    assert result
    assert [int(row["step"]) for row in result] == list(range(len(result)))
    return result


def normal_metrics(mesh: Any, u: np.ndarray, height: float) -> dict[str, float]:
    top = ns.gs.top_nodes(mesh)
    reference = mesh.p[top]
    target = reference + np.column_stack(
        (np.zeros(len(top)), 4 * height * reference[:, 0] * (1 - reference[:, 0]))
    )
    edge, target_edge = np.diff(reference + u[top], axis=0), np.diff(target, axis=0)
    length, target_length = (
        np.linalg.norm(edge, axis=1),
        np.linalg.norm(target_edge, axis=1),
    )
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


def gradient_metrics(mesh: Any, u: np.ndarray, height: float) -> dict[str, float]:
    top = ns.gs.top_nodes(mesh)
    x = mesh.p[top, 0]
    dx = np.diff(x)
    target = np.column_stack((np.zeros_like(x), 4 * height * x * (1 - x)))
    slope = np.diff(u[top] - target, axis=0) / dx[:, None]
    value = float(np.sum(dx[:, None] * slope**2) / (x[-1] - x[0]))
    center = (dx[:-1] + dx[1:]) / 2
    curvature = np.diff(slope, axis=0) / center[:, None]
    return {
        "gradient_loss": value,
        "slope_rms": float(np.sqrt(value)),
        "curvature_error": float(
            np.sqrt(np.sum(center[:, None] * curvature**2) / np.sum(center))
        ),
    }


def roughness(B: np.ndarray, edges: np.ndarray) -> float:
    delta = B[edges[:, 0]] - B[edges[:, 1]]
    return float(np.mean(np.sum(delta * delta, axis=(1, 2))))


def target_boundary_area_ratio(mesh: Any, height: float) -> float:
    """Area of the sampled prescribed boundary polygon relative to the box."""
    top = ns.gs.top_nodes(mesh)
    upper = mesh.p[top].copy()
    upper[:, 1] += 4 * height * upper[:, 0] * (1 - upper[:, 0])
    polygon = np.vstack(
        (mesh.p[0], mesh.p[mesh.nx], upper[-1], upper[-2::-1], mesh.p[0])
    )
    area = 0.5 * abs(
        np.sum(polygon[:-1, 0] * polygon[1:, 1] - polygon[1:, 0] * polygon[:-1, 1])
    )
    return float(area / mesh.area.sum())


def constraints(mode: str, q: np.ndarray, B: np.ndarray) -> dict[str, float]:
    offset = B - np.eye(2)
    eig = np.linalg.eigvalsh(offset)
    result = {"minimum_offset_eigenvalue": float(eig.min())}
    if mode == "contraction_only":
        assert eig.min() >= -1e-12
    elif mode == "learned_direction":
        strength = q.reshape(-1, 2)[:, 0]
        assert strength.min() >= -1e-12
        secondary = float(np.max(np.abs(eig[:, 0])))
        assert secondary <= 1e-10
        result.update(
            minimum_strength=float(strength.min()), rank_one_secondary_abs_max=secondary
        )
    elif mode == "x_contraction":
        assert q.min() >= -1e-12
        assert np.max(np.abs(offset[:, 1, :])) <= 1e-12
        assert np.max(np.abs(offset[:, :, 1])) <= 1e-12
        result["minimum_strength"] = float(q.min())
    else:
        assert mode == "unconstrained"
    return result


def hessian_metric(hessian: Any, max_iterations: int) -> dict[str, Any]:
    symmetric = (hessian + hessian.T) * 0.5
    try:
        value, vector = spla.eigsh(
            symmetric, k=1, which="SA", tol=1e-8, maxiter=max_iterations, ncv=80
        )
    except spla.ArpackNoConvergence as exc:
        return {
            "status": "ARPACK did not converge",
            "smallest_algebraic_eigenvalue": None,
            "converged_pairs": len(exc.eigenvalues),
        }
    residual = float(
        np.linalg.norm(symmetric @ vector[:, 0] - value[0] * vector[:, 0])
        / max(np.linalg.norm(symmetric @ vector[:, 0]), 1.0)
    )
    assert residual <= 1e-7
    return {
        "status": "ARPACK smallest algebraic eigenpair",
        "smallest_algebraic_eigenvalue": float(value[0]),
        "residual": residual,
    }


def metric(
    mesh: Any, folder: Path, step: int, scales: dict, tolerance: float
) -> tuple[dict[str, Any], Any]:
    history = np.load(folder / "history.npz", allow_pickle=False)
    index = np.flatnonzero(history["steps"] == step)
    assert len(index) == 1
    i = int(index[0])
    q, u = history["controls"][i], history["u"][i]
    mode, height, kind = (
        str(history["mode"]),
        float(history["height"]),
        str(history["kind"]),
    )
    beta, smooth_weight = float(history["beta"]), float(history["smooth_weight"])
    assert np.array_equal(history["points"], mesh.p)
    B = ns.gs.study.matrices(mesh, q, mode)
    packed = ns.ns.pack(mesh, u)
    _, residual, hessian, physical_j = ns.ph.assemble(mesh, packed, B)
    assert hessian is not None
    assert physical_j.min() > 0
    force = float(np.linalg.norm(residual, np.inf))
    assert force <= tolerance
    fixed = mesh.lookup.reshape(-1, 2) < 0
    assert np.max(np.abs(u[fixed])) <= 1e-14
    l2 = ns.ph.loss(mesh, packed, height, "l2")[0]
    normal, gradient = (
        normal_metrics(mesh, u, height),
        gradient_metrics(mesh, u, height),
    )
    R = roughness(B[mesh.muscle], np.asarray(mesh.edges))
    coefficient = (
        beta * scales["l2_0"] / scales["normal_0"] if kind == "normal" else 0.0
    )
    data_objective = l2 + coefficient * normal["normal_loss"]
    objective = data_objective + smooth_weight * height**2 * R
    trace_row = next(
        row for row in rows(folder / "trace.csv") if int(row["step"]) == step
    )
    area_ratio = float(np.sum(mesh.area * physical_j) / mesh.area.sum())
    for key, value in {
        "position_loss": l2,
        **normal,
        **gradient,
        "roughness": R,
        "data_objective": data_objective,
        "objective": objective,
        "min_J": float(physical_j.min()),
        "physical_area_ratio": area_ratio,
    }.items():
        assert np.isclose(value, float(trace_row[key]), rtol=1e-9, atol=1e-11), (
            folder,
            step,
            key,
            value,
            trace_row[key],
        )
    top = mesh.top
    target = np.zeros_like(u[top])
    target[:, 1] = 4 * height * mesh.p[top, 0] * (1 - mesh.p[top, 0])
    top_all = ns.gs.top_nodes(mesh)
    result = {
        "step": int(step),
        "fit_rms": float(np.sqrt(l2)),
        "position_loss": l2,
        **normal,
        **gradient,
        "roughness": R,
        "data_objective": data_objective,
        "total_objective": objective,
        "weighted_normal_loss": coefficient * normal["normal_loss"],
        "weighted_smoothness_loss": smooth_weight * height**2 * R,
        "motion_rms": float(np.sqrt(np.mean(np.sum(u[top] ** 2, axis=1)))),
        "target_projection": float(np.sum(u[top] * target) / np.sum(target**2)),
        "min_J": float(physical_j.min()),
        "max_J": float(physical_j.max()),
        "inverted_cells": int(np.sum(physical_j <= 0)),
        "physical_area_ratio": area_ratio,
        "force_residual_inf": force,
        "fixed_displacement_inf": float(np.max(np.abs(u[fixed]))),
        "top_edge_backtracking_count": int(
            np.sum(np.diff((mesh.p[top_all] + u[top_all])[:, 0]) <= 0)
        ),
        "constraints": constraints(mode, q, B[mesh.muscle]),
        "history_sha256": digest(folder / "history.npz"),
    }
    for key in ("motion_rms", "target_projection", "top_edge_backtracking_count"):
        assert np.isclose(result[key], float(trace_row[key]), rtol=1e-9, atol=1e-11)
    return result, hessian


def source_gate(source: Path) -> dict[str, Any]:
    checks = json.loads((source / "input-gates.json").read_text())
    assert checks["passed"]
    protocol = json.loads((source / "protocol.json").read_text())
    for path, expected in protocol["source_sha256"].items():
        snap = source / "source" / path
        live = ns.ROOT / path
        assert snap.is_file()
        assert live.is_file()
        assert digest(snap) == expected == digest(live)
    return {
        "source_snapshot_and_live_hashes_verified": True,
        "source_count": len(protocol["source_sha256"]),
        "gate_sha256": digest(source / "input-gates.json"),
    }


def neutral_receipt(mesh: Any, folder: Path, mode: str) -> dict[str, Any]:
    initial = np.load(folder / "initial-state.npz", allow_pickle=False)
    history = np.load(folder / "history.npz", allow_pickle=False)
    expected_q = ns.am.initialize(int(mesh.muscle.sum()), mode)
    for key, expected in {
        "controls": expected_q,
        "u": np.zeros_like(initial["u"]),
        "B": np.broadcast_to(np.eye(2), initial["B"].shape),
        "moment": np.zeros_like(initial["moment"]),
        "variance": np.zeros_like(initial["variance"]),
    }.items():
        assert np.array_equal(initial[key], expected), (folder, key)
    assert int(initial["step"]) == 0
    assert int(history["steps"][0]) == 0
    assert np.array_equal(history["controls"][0], initial["controls"])
    assert np.array_equal(history["u"][0], initial["u"])
    first = rows(folder / "trace.csv")[0]
    assert int(first["step"]) == 0
    return {"neutral_q_u_B_m_v_and_history_step0_verified": True}


def historical_l2_replays(source: Path) -> list[dict[str, Any]]:
    """Show fresh L2 controls reproduce, but do not reuse, archived trajectories."""
    legacy_root = ns.ROOT / "exp/2026/09/15/activation-direction-smoothness/data"
    records = []
    for mode in sorted(ns.MODES):
        for smooth_weight, variant, legacy_weight in (
            (0.0, "smooth-off-l2", "0"),
            (1.0, "smooth-on-l2", "1"),
        ):
            current = np.load(
                source / mode / variant / "checkpoint.npz", allow_pickle=False
            )
            legacy = np.load(
                legacy_root
                / f"tune-w{legacy_weight}"
                / f"h200-{mode}-w{legacy_weight}"
                / "checkpoint.npz",
                allow_pickle=False,
            )
            assert int(current["step"]) == int(legacy["step"])
            q_error = float(np.max(np.abs(current["controls"] - legacy["controls"])))
            u_error = float(np.max(np.abs(current["u"] - legacy["u"])))
            assert q_error <= 1e-8, (mode, smooth_weight, q_error)
            assert u_error <= 1e-10, (mode, smooth_weight, u_error)
            records.append(
                {
                    "mode": mode,
                    "smooth_weight": smooth_weight,
                    "variant": variant,
                    "historical_checkpoint": str(
                        (
                            legacy_root
                            / f"tune-w{legacy_weight}"
                            / f"h200-{mode}-w{legacy_weight}"
                            / "checkpoint.npz"
                        ).relative_to(ns.ROOT)
                    ),
                    "accepted_step": int(current["step"]),
                    "control_max_abs_error": q_error,
                    "displacement_max_abs_error": u_error,
                    "control_bitwise_equal": q_error == 0.0,
                    "displacement_bitwise_equal": u_error == 0.0,
                    "comparison_tolerance": {"controls": 1e-8, "displacements": 1e-10},
                    "initialization_note": "comparison only; current run initialized independently at q=u=m=v=0",
                }
            )
    assert len(records) == 8
    return records


def main(cfg: Config) -> None:
    source = (
        cfg.input_dir if cfg.input_dir.is_dir() else ns.GROUP / "data" / cfg.input_dir
    )
    assert source.is_dir()
    mesh, scales = (
        ns.ph.build_mesh(100, 10),
        ns.ns.normalization(ns.ph.build_mesh(100, 10), ns.HEIGHT),
    )
    result: dict[str, Any] = {
        "source_gate": source_gate(source),
        "historical_l2_replays": historical_l2_replays(source),
        "target_boundary_polygon_area_ratio": target_boundary_area_ratio(
            mesh, ns.HEIGHT
        ),
        "cells": [],
        "physical_hessian_note": "Forward equilibrium displacement Hessian, not the inverse objective Hessian or a mechanical-stability certificate.",
    }
    for mode in sorted(ns.MODES):
        cell = source / mode
        variants = {name: cell / name for name, *_ in ns.VARIANTS}
        assert all(folder.is_dir() for folder in variants.values())
        neutral = {
            name: neutral_receipt(mesh, folder, mode)
            for name, folder in variants.items()
        }
        shared_step = int(
            max(
                set.intersection(
                    *(
                        set(
                            np.load(folder / "history.npz", allow_pickle=False)["steps"]
                        )
                        for folder in variants.values()
                    )
                )
            )
        )
        matched = {
            name: metric(
                mesh, folder, shared_step, scales, cfg.force_residual_tolerance
            )
            for name, folder in variants.items()
        }
        endpoints = {}
        for name, folder in variants.items():
            history = np.load(folder / "history.npz", allow_pickle=False)
            checkpoint = np.load(folder / "checkpoint.npz", allow_pickle=False)
            summary = json.loads((folder / "summary.json").read_text())
            end_step = int(history["steps"][-1])
            assert int(checkpoint["step"]) == end_step
            assert np.array_equal(checkpoint["controls"], history["controls"][-1])
            assert np.array_equal(checkpoint["u"], history["u"][-1])
            assert int(summary["accepted_iterations"]) == end_step
            failure = summary["failure"]
            if failure is not None:
                assert int(failure["step"]) > shared_step
                assert (folder / "failure.json").is_file()
                assert (folder / "failed-proposal.npz").is_file()
            endpoint = metric(
                mesh, folder, end_step, scales, cfg.force_residual_tolerance
            )[0]
            endpoint.update(
                {
                    "accepted_iterations": int(summary["accepted_iterations"]),
                    "failure": failure,
                    "failure_receipt_saved": failure is None
                    or (
                        (folder / "failure.json").is_file()
                        and (folder / "failed-proposal.npz").is_file()
                    ),
                }
            )
            endpoints[name] = endpoint
        result["cells"].append(
            {
                "mode": mode,
                "height": ns.HEIGHT,
                "shared_step": shared_step,
                "neutral_receipts": neutral,
                "variants": {
                    name: {
                        **item[0],
                        "physical_hessian": hessian_metric(
                            item[1], cfg.hessian_max_iterations
                        ),
                    }
                    for name, item in matched.items()
                },
                "endpoints": endpoints,
            }
        )
    result["passed"] = True
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "comparison.json", result)
    write_json(output / "checks.json", result)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
