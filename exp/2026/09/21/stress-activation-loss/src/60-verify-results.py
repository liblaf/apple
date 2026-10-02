"""Independent CPU audit of staged active-stress inverse fits.

This deliberately reads saved arrays and receipts rather than importing the
GPU study or re-solving an equilibrium.  A completed finite budget is reported
separately from the consistency of whatever states were actually recorded.
"""
# ruff: noqa: C901, PT018

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
from experiment import Profile

from liblaf import cherries

GROUP = Path(__file__).parents[1]
L_REF_MM = 13.236093032531715
RTOL, ATOL = 1e-8, 1e-10
MODES = ("symmetric6", "psd6", "rankone_fixed", "rankone_learned")


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    source: Path
    output: Path = Path("60-verification")


def receipt(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require_close(
    actual: np.ndarray | float, expected: np.ndarray | float, label: str
) -> float:
    actual_a, expected_a = np.asarray(actual), np.asarray(expected)
    assert actual_a.shape == expected_a.shape, (label, actual_a.shape, expected_a.shape)
    error = float(np.max(np.abs(actual_a - expected_a), initial=0.0))
    scale = float(max(np.max(np.abs(expected_a), initial=0.0), 1.0))
    assert error <= ATOL + RTOL * scale, (label, error, scale)
    return error


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def state(path: Path) -> dict[str, np.ndarray | str | int | float | bool]:
    with np.load(path, allow_pickle=False) as data:
        value = {key: data[key].copy() for key in data.files}
    for key in ("mode",):
        raw = value[key]
        value[key] = str(
            raw.item() if isinstance(raw, np.ndarray) and raw.ndim == 0 else raw
        )
    for key in ("step",):
        value[key] = int(np.asarray(value[key]).item())
    for key in ("stress_reference_MPa",):
        value[key] = float(np.asarray(value[key]).item())
    for key in ("solver_valid",):
        value[key] = bool(np.asarray(value[key]).item())
    return value


def det_f(rest: np.ndarray, tets: np.ndarray, u: np.ndarray) -> np.ndarray:
    dm = np.transpose(rest[tets[:, 1:]] - rest[tets[:, :1]], (0, 2, 1))
    dm_inv = np.linalg.inv(dm)
    points = rest + u
    ds = np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
    return np.linalg.det(ds @ dm_inv)


def normals(points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = points[triangles]
    cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    assert np.isfinite(double_area).all() and np.all(double_area > 1e-14)
    return cross / double_area[:, None], double_area / 2


def losses(
    mesh: dict[str, np.ndarray],
    saved: dict[str, Any],
    normal_weight: float,
    smooth_weight: float,
) -> dict[str, float]:
    rest = mesh["rest_points"]
    skin_ids = mesh["skin_ids"].astype(int)
    triangles = mesh["triangles"].astype(int)
    u = np.asarray(saved["u"], dtype=float)
    target_u = mesh["target_displacement_skin"]
    weights = mesh["skin_vertex_weights"]
    error = u[skin_ids] - target_u
    position = float(
        (weights[:, None] * np.square(error)).sum() * (1e6 / 3) / L_REF_MM**2
    )
    reference = rest[skin_ids]
    current_normal, _ = normals(reference + u[skin_ids], triangles)
    target_normal, _ = normals(reference + target_u, triangles)
    _, reference_area = normals(reference, triangles)
    chord2 = np.square(current_normal - target_normal).sum(axis=1)
    normal = float((reference_area * chord2).sum() / reference_area.sum())
    dots = np.clip((current_normal * target_normal).sum(axis=1), -1, 1)
    angle = np.arccos(dots)
    angle_rms = float(
        np.degrees(
            np.sqrt((reference_area * np.square(angle)).sum() / reference_area.sum())
        )
    )
    qhat = np.asarray(saved["Qhat"], dtype=float)
    delta = qhat[mesh["edge_i"].astype(int)] - qhat[mesh["edge_j"].astype(int)]
    regularizer = float(
        mesh["regularizer_factor"].item()
        * (mesh["edge_weight"] * np.square(delta).sum(axis=(1, 2))).sum()
    )
    return {
        "position_contribution": position,
        "normal_loss": normal,
        "normal_contribution": normal_weight * normal,
        "activation_smoothness": regularizer,
        "regularizer_contribution": smooth_weight * regularizer,
        "objective": position + normal_weight * normal + smooth_weight * regularizer,
        "fit_rms_mm": float(
            np.sqrt((weights[:, None] * np.square(error)).sum()) * 1000
        ),
        "normal_angle_rms_deg": angle_rms,
    }


def matrix_from_controls(saved: dict[str, Any]) -> np.ndarray:
    q, mode = np.asarray(saved["q"], float), str(saved["mode"])
    if mode in {"symmetric6", "psd6"}:
        assert q.shape[-1] == 6
        out = np.zeros((*q.shape[:-1], 3, 3))
        out[..., 0, 0], out[..., 1, 1], out[..., 2, 2] = q[..., 0], q[..., 1], q[..., 2]
        out[..., 0, 1] = out[..., 1, 0] = q[..., 3] / math.sqrt(2)
        out[..., 1, 2] = out[..., 2, 1] = q[..., 4] / math.sqrt(2)
        out[..., 0, 2] = out[..., 2, 0] = q[..., 5] / math.sqrt(2)
        return out
    if mode == "rankone_fixed":
        axes = np.asarray(saved["fixed_axes"], float)
        assert q.shape[-1] == 1 and axes.shape == (len(q), 3)
        axes = canonical_axes(axes)
        return q[:, 0].clip(min=0)[:, None, None] * axes[:, :, None] * axes[:, None, :]
    assert mode == "rankone_learned" and q.shape[-1] == 4
    axes = canonical_axes(q[:, 1:])
    return q[:, 0].clip(min=0)[:, None, None] * axes[:, :, None] * axes[:, None, :]


def canonical_axes(axes: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(axes, axis=1, keepdims=True)
    assert np.all(length > 0)
    axes = axes / length
    maximum = np.abs(axes).argmax(axis=1)
    sign = np.sign(axes[np.arange(len(axes)), maximum])
    return axes * np.where(sign[:, None] == 0, 1, sign[:, None])


def validate_constraint(saved: dict[str, Any]) -> dict[str, float | int]:
    qhat = np.asarray(saved["Qhat"], float)
    assert qhat.ndim == 3 and qhat.shape[-2:] == (3, 3)
    assert np.isfinite(qhat).all()
    symmetry = float(np.abs(qhat - np.swapaxes(qhat, 1, 2)).max(initial=0.0))
    assert symmetry <= ATOL
    reconstructed_error = require_close(
        qhat, matrix_from_controls(saved), "controls_to_Qhat"
    )
    eig = np.linalg.eigvalsh(qhat)
    mode = str(saved["mode"])
    result: dict[str, float | int] = {
        "symmetry_max_abs": symmetry,
        "controls_reconstruction_max_abs": reconstructed_error,
        "eigenvalue_min": float(eig.min()),
        "eigenvalue_max": float(eig.max()),
        "Qhat_abs_peak": float(np.abs(qhat).max()),
        "Qhat_frobenius_peak": float(np.linalg.norm(qhat, axis=(1, 2)).max()),
    }
    if mode == "psd6":
        assert eig.min() >= -ATOL
    if mode.startswith("rankone"):
        assert eig.min() >= -ATOL
        rank_error = float(np.abs(eig[:, :2]).max(initial=0.0))
        assert rank_error <= ATOL + RTOL * max(float(np.abs(eig).max()), 1.0)
        result["rankone_small_eigenvalue_max_abs"] = rank_error
    return result


def rows(path: Path) -> list[dict[str, Any]]:
    def value(raw: str | None) -> float | str | None:
        if raw in {"", None}:
            return None
        try:
            return float(raw)
        except ValueError:
            return raw

    with path.open(newline="") as file:
        return [
            {key: value(raw) for key, raw in row.items()}
            for row in csv.DictReader(file)
        ]


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify_receipt(record: dict[str, str], *, require_live: bool = True) -> None:
    path = Path(record["path"])
    assert path.exists()
    assert receipt(path)["sha256"] == record["sha256"]
    snapshot = record.get("snapshot")
    if snapshot:
        assert receipt(Path(snapshot))["sha256"] == record["sha256"]
    elif not require_live:
        return


def verify_sources(protocol: dict[str, Any]) -> dict[str, int]:
    records = protocol["sources"]
    assert records
    for record in records.values():
        verify_receipt(record)
    return {
        "records": len(records),
        "unique_live_paths": len({value["path"] for value in records.values()}),
    }


def stage_files(stage_dir: Path) -> bool:
    required = (
        "initial-state.npz",
        "last.npz",
        "trace.csv",
        "solver-receipts.jsonl",
        "initialization.json",
    )
    return all((stage_dir / name).exists() for name in required)


def verify_stage(  # noqa: PLR0915
    source: Path,
    mesh: dict[str, np.ndarray],
    stage_id: str,
    summary: dict[str, Any],
    protocol: dict[str, Any],
    parent: dict[str, Any] | None,
) -> dict[str, Any]:
    folder = source / stage_id
    assert folder.is_dir()
    result: dict[str, Any] = {
        "status": summary["status"],
        "last_step": summary["last_step"],
        "budget": summary["budget"],
        "recorded": stage_files(folder),
    }
    if not result["recorded"]:
        assert summary["status"] in {"blocked_by_parent_failure", "failed"}
        return result
    initial, last = state(folder / "initial-state.npz"), state(folder / "last.npz")
    initialization = load_json(folder / "initialization.json")
    mode = str(initial["mode"])
    assert mode == stage_id.split("-", 1)[1] and mode in MODES
    assert str(last["mode"]) == mode
    assert int(initial["step"]) == 0
    assert int(last["step"]) == int(summary["last_step"])
    assert bool(initial["solver_valid"]) and bool(last["solver_valid"])
    assert math.isclose(
        float(initial["stress_reference_MPa"]),
        float(protocol["activation_reference_MPa"]),
        rel_tol=RTOL,
        abs_tol=ATOL,
    )
    assert math.isclose(
        float(last["stress_reference_MPa"]),
        float(protocol["activation_reference_MPa"]),
        rel_tol=RTOL,
        abs_tol=ATOL,
    )
    assert initialization["mode"] == mode and initialization["fresh_adam"] is True
    loss_name = stage_id.split("-", 1)[0]
    expected_beta = float(protocol["losses"][loss_name])
    assert float(initialization["normal_weight"]) == expected_beta
    assert math.isclose(
        float(initialization["smooth_weight"]),
        float(protocol["smooth_weight"]),
        rel_tol=RTOL,
        abs_tol=ATOL,
    )
    result["initial_constraint"] = validate_constraint(initial)
    result["last_constraint"] = validate_constraint(last)
    result["Q_snapshot"] = {
        "reference_stress_MPa": float(last["stress_reference_MPa"]),
        "component_peak_kPa": result["last_constraint"]["Qhat_abs_peak"]
        * float(last["stress_reference_MPa"])
        * 1000,
        "frobenius_peak_kPa": result["last_constraint"]["Qhat_frobenius_peak"]
        * float(last["stress_reference_MPa"])
        * 1000,
    }
    saved_states = sorted(folder.glob("step-*.npz"))
    saved_states = [folder / "initial-state.npz", *saved_states, folder / "last.npz"]
    state_j = {}
    for saved_path in dict.fromkeys(saved_states):
        saved = state(saved_path)
        j = det_f(
            mesh["rest_points"], mesh["tets"].astype(int), np.asarray(saved["u"], float)
        )
        assert np.isfinite(j).all() and j.min() > 0
        state_j[saved_path.name] = {
            "step": int(saved["step"]),
            "min": float(j.min()),
            "max": float(j.max()),
        }
    result["saved_state_detF"] = state_j
    initial_loss = losses(
        mesh, initial, expected_beta, float(protocol["smooth_weight"])
    )
    last_loss = losses(mesh, last, expected_beta, float(protocol["smooth_weight"]))
    trace = rows(folder / "trace.csv")
    assert (
        trace
        and int(trace[0]["step"]) == 0
        and int(trace[-1]["step"]) == int(last["step"])
    )
    for label, values, row in (
        ("initial", initial_loss, trace[0]),
        ("last", last_loss, trace[-1]),
    ):
        for key, value in values.items():
            if key in row and row[key] is not None:
                require_close(float(row[key]), value, f"{stage_id}:{label}:{key}")
    objectives = np.asarray([row["objective"] for row in trace], float)
    assert np.isfinite(objectives).all()
    assert np.all(
        np.diff(objectives) <= ATOL + RTOL * np.maximum(1, np.abs(objectives[:-1]))
    )
    proposals = (
        jsonl(folder / "proposals.jsonl")
        if (folder / "proposals.jsonl").exists()
        else []
    )
    for step in range(1, len(trace)):
        accepted = [
            row for row in proposals if int(row["step"]) == step and row.get("accepted")
        ]
        assert len(accepted) == 1, (stage_id, step, len(accepted))
        proposal = accepted[0]
        assert float(proposal["slope"]) < 0
        assert math.isclose(
            float(proposal["objective"]),
            float(trace[step]["objective"]),
            rel_tol=RTOL,
            abs_tol=ATOL,
        )
        assert (
            float(trace[step]["objective"])
            <= float(trace[step - 1]["objective"])
            + 1e-4 * float(proposal["slope"])
            + ATOL
        )
    solver = jsonl(folder / "solver-receipts.jsonl")
    assert len(solver) == len(trace)
    for expected_step, item in enumerate(solver):
        assert int(item["step"]) == expected_step
        forward, adjoint = item["forward"], item["adjoint"]
        assert forward["success"] is True and adjoint["success"] is True
        actual = item["actual_residual_receipt"]
        assert math.isclose(
            float(actual["adjoint_relative_residual"]),
            float(adjoint["relative_residual"]),
            rel_tol=RTOL,
            abs_tol=ATOL,
        )
        assert float(actual["adjoint_relative_residual"]) <= float(
            adjoint["relative_tolerance"]
        )
    result["metrics"] = last_loss
    result["trace"] = {
        "rows": len(trace),
        "objective_initial": float(objectives[0]),
        "objective_last": float(objectives[-1]),
        "accepted_updates": len(trace) - 1,
    }
    result["solver_receipts"] = {
        "count": len(solver),
        "maximum_adjoint_relative_residual": max(
            float(item["adjoint"]["relative_residual"]) for item in solver
        ),
    }
    if parent is None:
        require_close(
            np.asarray(initial["Qhat"]),
            np.zeros_like(np.asarray(initial["Qhat"])),
            f"{stage_id}:neutral_Q",
        )
    else:
        parent_state = state(source / parent["id"] / "last.npz")
        parent_receipt = load_json(source / f"{stage_id}-parent.json")
        assert parent_receipt["parent"] == parent["id"]
        verify_receipt(parent_receipt["state"])
        previous = np.asarray(parent_state["Qhat"], float)
        if mode == "psd6":
            values, vectors = np.linalg.eigh(
                (previous + np.swapaxes(previous, 1, 2)) / 2
            )
            expected = (vectors * np.maximum(values, 0)[:, None, :]) @ np.swapaxes(
                vectors, 1, 2
            )
        elif mode == "rankone_fixed":
            values, vectors = np.linalg.eigh(
                (previous + np.swapaxes(previous, 1, 2)) / 2
            )
            amplitude = np.maximum(values[:, -1], 0)
            axes = canonical_axes(vectors[:, :, -1])
            axes[np.linalg.norm(previous, axis=(1, 2)) == 0] = np.array([1.0, 0.0, 0.0])
            expected = amplitude[:, None, None] * axes[:, :, None] * axes[:, None, :]
        else:
            assert mode == "rankone_learned"
            expected = previous
        result["parent_initialization_max_abs"] = require_close(
            np.asarray(initial["Qhat"]), expected, f"{stage_id}:parent_initialization"
        )
    return result


def report(checks: dict[str, Any], path: Path) -> None:
    lines = [
        "# Active-stress chain verification",
        "",
        f"Source: `{checks['source']['path']}`",
        "",
        f"- Receipt-consistency audit: **{'passed' if checks['passed'] else 'failed'}**",
        f"- Every stage completed its configured budget: **{checks['all_completed']}**",
        "- Geometry determinants are independently checked only for saved state files (initial, every 50 updates, and final); unsaved accepted iterates have solver receipts and trace evidence, not stored displacements.",
        "",
    ]
    lines += [
        "| Stage | Status | Final update | Position RMS (mm) | Normal RMS (deg) | R |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, value in checks["stages"].items():
        metrics = value.get("metrics") or {}
        lines.append(
            f"| {name} | {value['status']} | {value.get('last_step')} | {metrics.get('fit_rms_mm', float('nan')):.6g} | {metrics.get('normal_angle_rms_deg', float('nan')):.6g} | {metrics.get('activation_smoothness', float('nan')):.6g} |"
        )
    figures = GROUP / "data/50-figures"
    if figures.exists():
        lines += [
            "",
            "Rendered stage assets were found under [`data/50-figures`](../data/50-figures/).",
        ]
    lines += [
        "",
        "Reproduction: `CHERRIES_NAME='Stress activation results verification' CHERRIES_TAGS='stress-activation,verification,cpu' uv run python src/60-verify-results.py --source "
        + checks["source"]["path"]
        + " --output 60-verification`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main(cfg: Config) -> None:
    source = cherries.input(cfg.source)
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = load_json(source / "protocol.json")
    assert tuple(protocol["modes"]) == MODES
    assert math.isclose(
        float(protocol["l_ref_mm"]), L_REF_MM, rel_tol=RTOL, abs_tol=ATOL
    )
    source_counts = verify_sources(protocol)
    verify_receipt(protocol["gate"])
    verify_receipt(protocol["calibration"])
    with np.load(source / "mesh.npz", allow_pickle=False) as raw:
        mesh = {key: raw[key].copy() for key in raw.files}
    assert mesh["rest_points"].ndim == 2 and mesh["rest_points"].shape[1] == 3
    assert mesh["tets"].ndim == 2 and mesh["tets"].shape[1] == 4
    assert len(mesh["active_ids"]) > 0
    assert mesh["edge_i"].shape == mesh["edge_j"].shape == mesh["edge_weight"].shape
    summaries = load_json(source / "summary.json")
    expected = [f"{loss}-{mode}" for loss in protocol["losses"] for mode in MODES]
    assert set(summaries) == set(expected)
    stages: dict[str, Any] = {}
    for loss in protocol["losses"]:
        parent = None
        for mode in MODES:
            stage_id = f"{loss}-{mode}"
            stages[stage_id] = verify_stage(
                source, mesh, stage_id, summaries[stage_id], protocol, parent
            )
            if stages[stage_id]["recorded"]:
                parent = {"id": stage_id, **summaries[stage_id]}
            else:
                parent = None
    completed = all(
        value["status"] == "completed_budget_not_convergence_certified"
        and value["last_step"] == value["budget"]
        for value in stages.values()
    )
    checks = {
        "passed": True,
        "all_completed": completed,
        "source": {
            "path": str(source.resolve()),
            "protocol": receipt(source / "protocol.json"),
            "counts": source_counts,
        },
        "stages": stages,
        "limitations": "CPU audit does not rerun GPU equilibrium; det(F) covers each saved state, while receipts and monotone trace cover every accepted update.",
    }
    write_json(output / "checks.json", checks)
    report(checks, GROUP / "docs/60-results.md")
    cherries.log_metric("all_completed", int(completed))
    cherries.log_metric(
        "stages_recorded", sum(value["recorded"] for value in stages.values())
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
