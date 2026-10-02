"""Analyze fixed-axis active-strain smoothness continuation endpoints."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
FROZEN_SRC = (
    GROUP / "data/00-frozen-source/apple/exp/2026/09/21/stress-activation-loss/src"
)
sys.path.insert(0, str(FROZEN_SRC))
from experiment import Profile  # noqa: E402

logger = logging.getLogger(__name__)

MULTIPLIERS = (1, 3, 10)
BASELINE_LIMIT = 1.05
DIRECTIONAL_LIMIT = 0.5


class Config(cherries.BaseConfig):
    source: Path = Path("10-sweep")
    output: Path = Path("20-analysis")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_receipt(row: dict, label: str) -> Path:
    path = Path(row["path"])
    actual = sha256(path)
    expected = row["sha256"]
    assert actual == expected, f"{label} sha256 mismatch: {path}"
    return path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def graph_roughness(q: np.ndarray, axes: np.ndarray, mesh: dict) -> dict:
    """Exactly split rank-one graph roughness into amplitude and axis terms."""
    mode = q.shape[1]
    assert mode in {1, 4}
    amplitude = q[:, 0]
    assert np.isfinite(amplitude).all()
    assert np.all(amplitude >= 0.0)
    if mode == 4:
        assert axes.shape == (0, 3)
        axes = q[:, 1:]
    assert axes.shape == (len(q), 3)
    axis_norms = np.linalg.norm(axes, axis=1)
    assert np.isfinite(axis_norms).all()
    assert np.all(axis_norms > 0.0)
    axes = axes / axis_norms[:, None]
    i, j, weight = mesh["edge_i"], mesh["edge_j"], mesh["edge_weight"]
    cosine2 = np.einsum("ij,ij->i", axes[i], axes[j]) ** 2
    cosine2 = np.clip(cosine2, 0.0, 1.0)
    directional_edge = 2.0 * amplitude[i] * amplitude[j] * (1.0 - cosine2)
    amplitude_edge = (amplitude[i] - amplitude[j]) ** 2
    factor = float(mesh["regularizer_factor"])
    amplitude_part = factor * float(np.dot(weight, amplitude_edge))
    direction_part = factor * float(np.dot(weight, directional_edge))
    S_i = amplitude[i, None, None] * axes[i, :, None] * axes[i, None, :]
    S_j = amplitude[j, None, None] * axes[j, :, None] * axes[j, None, :]
    direct = factor * float(np.dot(weight, np.sum((S_i - S_j) ** 2, axis=(1, 2))))
    split = amplitude_part + direction_part
    relative_error = abs(direct - split) / max(abs(direct), 1e-30)
    assert relative_error < 1e-10
    return {
        "amplitude": amplitude_part,
        "direction": direction_part,
        "total": direct,
        "direction_fraction": direction_part / direct if direct > 0 else 0.0,
        "identity_relative_error": relative_error,
    }


def amplitude_statistics(q: np.ndarray, mesh: dict) -> dict:
    """Summarize amplitudes using normalized active-volume weights."""
    amplitude = q[:, 0]
    volume = mesh["active_volume_weights"]
    volume = volume / volume.sum()
    order = np.argsort(amplitude)
    cumulative = np.cumsum(volume[order])
    p95_index = np.searchsorted(cumulative, 0.95, side="left")
    return {
        "volume_weighted_mean": float(np.dot(volume, amplitude)),
        "volume_weighted_rms": float(np.sqrt(np.dot(volume, amplitude**2))),
        "volume_weighted_p95": float(amplitude[order[p95_index]]),
        "maximum": float(amplitude.max()),
    }


def fixed_parent_direction_roughness(
    q: np.ndarray, axes: np.ndarray, parent_amplitude: np.ndarray, mesh: dict
) -> float:
    """Measure current direction roughness under the parent amplitude field."""
    if q.shape[1] == 4:
        axes = q[:, 1:]
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    i, j, weight = mesh["edge_i"], mesh["edge_j"], mesh["edge_weight"]
    cosine2 = np.einsum("ij,ij->i", axes[i], axes[j]) ** 2
    cosine2 = np.clip(cosine2, 0.0, 1.0)
    edge = 2.0 * parent_amplitude[i] * parent_amplitude[j] * (1.0 - cosine2)
    return float(mesh["regularizer_factor"]) * float(np.dot(weight, edge))


def checkpoint_metrics(state: dict, mesh: dict) -> dict:
    """Recompute fit, normal, and deformation diagnostics without a solver."""
    points = mesh["rest_points"]
    tets = mesh["tets"]
    u = state["u"]
    skin_ids = mesh["skin_ids"]
    triangles = mesh["triangles"]
    current = points + u

    displacement_error = u[skin_ids] - mesh["target_displacement_skin"]
    fit_rms_mm = 1000.0 * np.sqrt(
        np.sum(mesh["skin_vertex_weights"] * np.sum(displacement_error**2, axis=1))
    )

    reference = points[skin_ids]
    target = reference + mesh["target_displacement_skin"]
    actual = current[skin_ids]

    def unit_normals(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tri = x[triangles]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area2 = np.linalg.norm(cross, axis=1)
        assert np.all(area2 > 1e-14)
        return cross / area2[:, None], area2

    _, area2_ref = unit_normals(reference)
    n_target, _ = unit_normals(target)
    n_actual, _ = unit_normals(actual)
    dot = np.clip(np.einsum("ij,ij->i", n_actual, n_target), -1.0, 1.0)
    angle = np.arccos(dot)
    area = area2_ref / 2.0
    normal_rms_deg = float(np.rad2deg(np.sqrt(np.dot(area, angle**2) / area.sum())))

    ref_dm = np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
    current_dm = np.transpose(current[tets[:, 1:]] - current[tets[:, :1]], (0, 2, 1))
    det_f = np.linalg.det(current_dm) / np.linalg.det(ref_dm)
    return {
        "fit_rms_mm": float(fit_rms_mm),
        "normal_angle_rms_deg": normal_rms_deg,
        "detF_min": float(det_f.min()),
        "inverted_all_cells": int(np.count_nonzero(det_f <= 0.0)),
        "solver_valid": bool(state["solver_valid"]),
    }


def analyse_state(  # noqa: C901, PLR0915
    checkpoint: Path,
    mesh: dict,
    *,
    summary_path: Path | None = None,
    trace_path: Path | None = None,
    gradient_balance_path: Path | None = None,
    expected_mode: str,
    parent_amplitude: np.ndarray | None = None,
) -> dict:
    with np.load(checkpoint, allow_pickle=False) as saved:
        state = {key: saved[key].copy() for key in saved.files}
    assert str(state["mode"]) == expected_mode
    expected_q_width = 1 if expected_mode == "rankone_fixed" else 4
    assert state["q"].shape == (len(mesh["active_ids"]), expected_q_width)
    if expected_mode == "rankone_fixed":
        assert state["fixed_axes"].shape == (len(mesh["active_ids"]), 3)
    else:
        assert state["fixed_axes"].shape == (0, 3)
    assert int(state["step"]) > 0

    step = int(state["step"])
    if summary_path is not None:
        summary = load_json(summary_path)
        assert int(summary["last_step"]) == step
        assert summary["status"] == "completed_budget_not_convergence_certified"
        assert int(summary["budget"]) == 200
        assert step == 200
        assert int(summary["attempted_steps"]) == 200
        optimizer_updates = int(summary["optimizer_updates"])
        skipped_steps = int(summary["skipped_steps"])
        assert optimizer_updates + skipped_steps == 200
    final_row = None
    if trace_path is not None:
        with trace_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows, f"missing trace rows in {trace_path}"
        final_row = rows[-1]
        assert int(final_row["step"]) == step

    geometry_metrics = checkpoint_metrics(state, mesh)
    if final_row is not None:
        for key in ("fit_rms_mm", "normal_angle_rms_deg"):
            assert np.isclose(
                geometry_metrics[key], float(final_row[key]), rtol=2e-8, atol=1e-10
            ), f"{checkpoint}: recomputed {key} does not match final trace row"
    smoothness = graph_roughness(state["q"], state["fixed_axes"], mesh)
    result = {
        "step": step,
        "mode": expected_mode,
        **geometry_metrics,
        "amplitude": amplitude_statistics(state["q"], mesh),
        "graph_roughness": smoothness,
    }
    if summary_path is not None:
        result["status"] = summary["status"]
        result["budget"] = int(summary["budget"])
        result["optimizer_updates"] = optimizer_updates
        result["skipped_steps"] = skipped_steps
    if parent_amplitude is not None:
        result["direction_roughness_at_fixed_parent_amplitude"] = (
            fixed_parent_direction_roughness(
                state["q"], state["fixed_axes"], parent_amplitude, mesh
            )
        )
    if gradient_balance_path is not None:
        balance = load_json(gradient_balance_path)
        assert balance["status"] in {"available", "unavailable"}
        checkpoint_receipt = balance["checkpoint"]
        assert checkpoint_receipt["sha256"] == sha256(checkpoint)
        if balance["status"] == "available":
            result["smoothness_to_l2_gradient_ratio"] = balance[
                "smoothness_to_l2_gradient_ratio"
            ]
            result["gradient_diagnostic"] = {
                "checkpoint": checkpoint_receipt,
                "metric": balance["gradient_metric"],
                "solver_valid": balance["solver_valid"],
                "l2_gradient_dual_norm": balance["l2_gradient_dual_norm"],
                "weighted_smoothness_gradient_dual_norm": balance[
                    "weighted_smoothness_gradient_dual_norm"
                ],
                "forward_force_norm": balance["forward"]["accepted_force_norm"],
                "forward_force_threshold": balance["forward"][
                    "accepted_force_threshold"
                ],
                "forward_success": balance["forward"]["success"],
                "l2_adjoint": balance["l2_adjoint"],
            }
        else:
            result["smoothness_to_l2_gradient_ratio"] = None
            result["gradient_balance_unavailable"] = balance.get("failure")
    if final_row is not None:
        result["trace"] = {
            key: float(final_row[key])
            for key in (
                "objective",
                "regularizer_contribution",
                "neighbor_strain_rms_dimensionless",
            )
            if key in final_row
        }
    return result


def audit_initialization(source: Path, parent_state: dict) -> dict:
    """Require identical parent strain and cross-run initial displacements."""
    initial_states = {}
    strain_errors = {}
    for multiplier in MULTIPLIERS:
        path = source / f"multiplier-{multiplier}/stage/initial-state.npz"
        with np.load(path, allow_pickle=False) as saved:
            state = {key: saved[key].copy() for key in saved.files}
        assert str(state["mode"]) == "rankone_learned"
        error = float(np.max(np.abs(state["S"] - parent_state["S"])))
        assert error <= 1e-10, (multiplier, error)
        initial_states[multiplier] = state
        strain_errors[multiplier] = error
    reference_u = initial_states[MULTIPLIERS[0]]["u"]
    displacement_errors = {
        multiplier: float(np.max(np.abs(state["u"] - reference_u)))
        for multiplier, state in initial_states.items()
    }
    assert max(displacement_errors.values()) <= 1e-10, displacement_errors
    return {
        "parent_strain_max_abs_error_by_multiplier": strain_errors,
        "cross_run_displacement_max_abs_error_by_multiplier": displacement_errors,
        "tolerance": 1e-10,
    }


def relative_trajectories(source: Path, output: Path) -> dict:
    """Plot each branch trajectory as a ratio to the 1x row at each step."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    series = {}
    for multiplier in MULTIPLIERS:
        path = source / f"multiplier-{multiplier}/stage/trace.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        series[multiplier] = {
            "step": np.asarray([int(row["step"]) for row in rows]),
            "fit_rms_mm": np.asarray([float(row["fit_rms_mm"]) for row in rows]),
            "normal_angle_rms_deg": np.asarray(
                [float(row["normal_angle_rms_deg"]) for row in rows]
            ),
            "activation_smoothness": np.asarray(
                [float(row["activation_smoothness"]) for row in rows]
            ),
        }
    baseline = series[1]
    common_steps = baseline["step"]
    for multiplier in MULTIPLIERS[1:]:
        common_steps = np.intersect1d(common_steps, series[multiplier]["step"])
    assert len(common_steps) > 1

    fields = (
        ("fit_rms_mm", "Fit RMS / 1x", "Fit RMS"),
        ("normal_angle_rms_deg", "Normal RMS / 1x", "Normal RMS"),
        ("activation_smoothness", "Graph smoothness R / 1x", "Activation smoothness"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, (field, ylabel, title) in zip(axes, fields, strict=True):
        baseline_indices = np.searchsorted(baseline["step"], common_steps)
        denominator = baseline[field][baseline_indices]
        assert np.all(denominator > 0.0)
        for multiplier in MULTIPLIERS:
            indices = np.searchsorted(series[multiplier]["step"], common_steps)
            axis.plot(
                common_steps,
                series[multiplier][field][indices] / denominator,
                label=f"{multiplier}x",
            )
        axis.axhline(1.0, color="0.4", linewidth=0.8, linestyle="--")
        axis.set_title(title)
        axis.set_xlabel("Accepted update")
        axis.set_ylabel(ylabel)
        axis.grid(visible=True, alpha=0.25)
    axes[0].legend(frameon=False)
    path = output / "relative-trajectories.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return {"path": path, "matched_steps": common_steps.tolist()}


def markdown_report(result: dict, protocol: dict) -> str:
    rows = result["runs"]
    baseline = rows["multiplier-1"]
    selected = result["selected_multiplier"]
    decision = (
        "Neither 3x nor 10x passes all three predeclared criteria. "
        "Both reduce directional roughness by at least 50%, but exceed the "
        "allowed 5% increase in position RMS. No stronger coefficient is selected."
        if selected is None
        else f"Select {selected}x, the smallest multiplier passing all three predeclared criteria."
    )
    lines = [
        "# Released-axis smoothness continuation",
        "",
        decision,
        "",
        f"The limits are direction roughness at most {DIRECTIONAL_LIMIT:.0%} of new 1x, and both position and normal RMS at most {BASELINE_LIMIT:.0%} of new 1x. The position limit is {BASELINE_LIMIT * baseline['fit_rms_mm']:.4f} mm.",
        "",
        "This report compares released-axis rank-one endpoints initialized from the same fixed-axis parent state. Geometry metrics were recomputed from each saved displacement; no forward solve was run. Each branch used the full 200-update budget; that records budget completion, not convergence certification. The endpoints retain their numerical solver-validity and inversion diagnostics below.",
        "",
        "![Shapes and principal activation at a shared camera and scale](../data/30-comparison/smoothness-comparison-preview.png)",
        "",
        "[Full-resolution comparison (10240 x 5760)](../data/30-comparison/smoothness-comparison-16x9.png)",
        "",
        "## Exact roughness split",
        "",
        "For each graph edge, `S_i = a_i v_i v_i^T` gives `||S_i-S_j||_F^2 = (a_i-a_j)^2 + 2 a_i a_j [1-(v_i^T v_j)^2]`. The table sums the amplitude and unoriented-axis terms with the saved graph conductance and regularizer factor. Directional reduction is relative to the new 1x endpoint.",
        "",
        "Amplitude statistics use normalized active-volume weights; the RMS amplitude equals the active-strain tensor Frobenius RMS for unit axes. `Direction at parent amplitude` recomputes the direction term using the historical parent amplitudes and current axes, as a diagnostic only; selection still uses the actual field's direction term.",
        "",
        "## Selection metrics",
        "",
        "| Multiplier | Direction / 1x | Fit (mm) | Fit / 1x | Normal (deg) | Normal / 1x | Meets all criteria |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    amplitude_rows = []
    diagnostics_rows = []
    gradient_rows = []
    for name in ("multiplier-1", "multiplier-3", "multiplier-10"):
        row = rows[name]
        rough_ratio = (
            row["graph_roughness"]["direction"]
            / baseline["graph_roughness"]["direction"]
        )
        fit_ratio = row["fit_rms_mm"] / baseline["fit_rms_mm"]
        normal_ratio = row["normal_angle_rms_deg"] / baseline["normal_angle_rms_deg"]
        meets = (
            rough_ratio <= DIRECTIONAL_LIMIT
            and fit_ratio <= BASELINE_LIMIT
            and normal_ratio <= BASELINE_LIMIT
        )
        amplitude = row["amplitude"]
        lines.append(
            f"| {name.removeprefix('multiplier-')}x | {rough_ratio:.3f} | "
            f"{row['fit_rms_mm']:.4f} | {fit_ratio:.3f} | "
            f"{row['normal_angle_rms_deg']:.4f} | {normal_ratio:.3f} | {meets} |"
        )
        amplitude_rows.append(
            f"| {name.removeprefix('multiplier-')}x | {amplitude['volume_weighted_mean']:.4g} | "
            f"{amplitude['volume_weighted_rms']:.4g} | {amplitude['volume_weighted_p95']:.4g} | "
            f"{amplitude['maximum']:.4g} | {row['graph_roughness']['amplitude']:.6g} | "
            f"{row['graph_roughness']['direction']:.6g} | "
            f"{row['direction_roughness_at_fixed_parent_amplitude']:.6g} |"
        )
        gradient_ratio = row["smoothness_to_l2_gradient_ratio"]
        gradient_ratio_text = (
            "n/a" if gradient_ratio is None else f"{gradient_ratio:.4g}"
        )
        diagnostics_rows.append(
            f"| {name.removeprefix('multiplier-')}x | {row['optimizer_updates']} | "
            f"{row['skipped_steps']} | {gradient_ratio_text} | "
            f"{row['inverted_all_cells']} | {row['detF_min']:.5g} | {row['solver_valid']} |"
        )
        if "gradient_diagnostic" in row:
            diagnostic = row["gradient_diagnostic"]
            gradient_rows.append(
                f"| {name.removeprefix('multiplier-')}x | "
                f"{diagnostic['weighted_smoothness_gradient_dual_norm']:.6g} | "
                f"{diagnostic['l2_gradient_dual_norm']:.6g} | "
                f"{diagnostic['forward_force_norm']:.6g} | "
                f"{diagnostic['l2_adjoint']['relative_residual']:.6g} |"
            )
    lines.extend(
        [
            "",
            "## Amplitude and roughness",
            "",
            "| Multiplier | Mean amplitude | RMS amplitude | p95 amplitude | Max amplitude | Amplitude roughness | Direction roughness | Direction at parent amplitude |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *amplitude_rows,
            "",
            "## Numerical diagnostics",
            "",
            "| Multiplier | Adam updates | Skipped proposals | Gradient ratio | Inverted cells | min det(F) | solver_valid |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            *diagnostics_rows,
            "",
            "The gradient ratio is `||d(eta R)/dS||_* / ||d L2/dS||_*` in the full symmetric activation tensor S, with `||g||_* = sqrt(sum_i ||sym(g_i)||_F^2 / m_i)` and normalized effective active-cell volumes `m_i`. The numerator includes the smoothness coefficient; the denominator excludes the normal loss.",
            "",
            "The saved endpoint diagnostic performs a fresh approximate forward solve and a separate L2 adjoint at the final controls. Its displacement can differ from the final Adam checkpoint used for the geometry metrics above. These ratios therefore describe approximate gradient diagnostics, not certified equilibrium sensitivities.",
            "",
            "| Multiplier | Weighted smoothness gradient norm | L2 gradient norm | Diagnostic forward force norm | L2 adjoint relative residual |",
            "| ---: | ---: | ---: | ---: | ---: |",
            *gradient_rows,
            "",
            "All diagnostic forward and L2 adjoint solves miss their requested tolerances: force norm 1e-10 and adjoint relative residual 1e-7. The raw receipts are in each branch's `stage/gradient-balance.json`; all final checkpoints have `solver_valid=false` and inverted cells. Completing 200 Adam updates does not establish mechanical validity.",
        ]
    )
    parent = result["historical_parent"]
    lines.extend(
        [
            "",
            "## Historical parent",
            "",
            f"The fixed-axis parent has amplitude roughness {parent['graph_roughness']['amplitude']:.6g} and direction roughness {parent['graph_roughness']['direction']:.6g} on the same saved graph. Its fit and normal metrics are {parent['fit_rms_mm']:.4f} mm and {parent['normal_angle_rms_deg']:.4f} degrees.",
            "",
            f"Initialization audit: maximum tensor error from the parent is {max(result['initialization_audit']['parent_strain_max_abs_error_by_multiplier'].values()):.3g}; maximum cross-run initial displacement difference is {max(result['initialization_audit']['cross_run_displacement_max_abs_error_by_multiplier'].values()):.3g} (tolerance 1e-10).",
            "",
            "![Fit, normal, and graph smoothness trajectories relative to the 1x run](../data/20-analysis/relative-trajectories.png)",
            "",
            "## Interpretation limits",
            "",
            "`solver_valid`, minimum determinant, and inverted-cell count are reported as numerical diagnostics only. This analysis makes no physical-validity claim. The directional term is sign-invariant and amplitude-weighted; zero-amplitude cells contribute no directional roughness.",
            "",
            f"Parent checkpoint SHA-256: `{protocol['parent_checkpoint']['sha256']}`. Shared mesh SHA-256: `{protocol['mesh']['sha256']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main(cfg: Config) -> None:
    source = cherries.input(cfg.source)
    output = cherries.output(cfg.output)
    protocol = load_json(source / "protocol.json")
    parent_path = verify_receipt(protocol["parent_checkpoint"], "parent checkpoint")
    mesh_path = verify_receipt(protocol["mesh"], "shared mesh")
    with np.load(mesh_path, allow_pickle=False) as saved_mesh:
        mesh = {key: saved_mesh[key].copy() for key in saved_mesh.files}

    historical = analyse_state(parent_path, mesh, expected_mode="rankone_fixed")
    with np.load(parent_path, allow_pickle=False) as saved_parent:
        parent_state = {key: saved_parent[key].copy() for key in saved_parent.files}
    initialization_audit = audit_initialization(source, parent_state)
    assert list(protocol["multipliers"]) == list(MULTIPLIERS)
    assert (
        protocol["selection"]["directional_roughness_max_relative_to_1x"]
        == DIRECTIONAL_LIMIT
    )
    assert protocol["selection"]["fit_rms_max_relative_to_1x"] == BASELINE_LIMIT
    assert protocol["selection"]["normal_rms_max_relative_to_1x"] == BASELINE_LIMIT
    runs = {}
    for multiplier in MULTIPLIERS:
        branch_protocol = load_json(source / f"multiplier-{multiplier}/protocol.json")
        assert int(branch_protocol["multiplier"]) == multiplier
        assert (
            branch_protocol["parent_checkpoint"]["sha256"]
            == protocol["parent_checkpoint"]["sha256"]
        )
        assert (
            branch_protocol["smooth_weight"]
            == protocol["base_smooth_weight"] * multiplier
        )
        runs[f"multiplier-{multiplier}"] = analyse_state(
            source / f"multiplier-{multiplier}/stage/last.npz",
            mesh,
            summary_path=source / f"multiplier-{multiplier}/summary.json",
            trace_path=source / f"multiplier-{multiplier}/stage/trace.csv",
            gradient_balance_path=source
            / f"multiplier-{multiplier}/stage/gradient-balance.json",
            expected_mode="rankone_learned",
            parent_amplitude=parent_state["q"][:, 0],
        )
    output.mkdir(parents=True, exist_ok=True)
    trajectory_plot = relative_trajectories(source, output)
    baseline = runs["multiplier-1"]
    for row in runs.values():
        row["relative_to_multiplier_1"] = {
            "direction_roughness": row["graph_roughness"]["direction"]
            / baseline["graph_roughness"]["direction"],
            "fit_rms_mm": row["fit_rms_mm"] / baseline["fit_rms_mm"],
            "normal_angle_rms_deg": row["normal_angle_rms_deg"]
            / baseline["normal_angle_rms_deg"],
        }
        row["meets_target"] = (
            row["relative_to_multiplier_1"]["direction_roughness"] <= DIRECTIONAL_LIMIT
            and row["relative_to_multiplier_1"]["fit_rms_mm"] <= BASELINE_LIMIT
            and row["relative_to_multiplier_1"]["normal_angle_rms_deg"]
            <= BASELINE_LIMIT
        )
    result = {
        "schema": "activation-smoothness-continuation-analysis-v1",
        "roughness_definition": "edgewise ||a_i v_i v_i^T-a_j v_j v_j^T||_F^2=(a_i-a_j)^2+2*a_i*a_j*(1-(v_i dot v_j)^2), weighted by saved graph conductance and regularizer factor",
        "criteria": {
            "directional_roughness_at_most_fraction_of_multiplier_1": DIRECTIONAL_LIMIT,
            "fit_and_normal_rms_at_most_multiplier_1_fraction": BASELINE_LIMIT,
        },
        "selected_multiplier": next(
            (m for m in MULTIPLIERS[1:] if runs[f"multiplier-{m}"]["meets_target"]),
            None,
        ),
        "historical_parent": historical,
        "initialization_audit": initialization_audit,
        "relative_trajectories": {
            "path": str(trajectory_plot["path"]),
            "matched_updates": trajectory_plot["matched_steps"],
        },
        "runs": runs,
        "source": {
            "parent_checkpoint": protocol["parent_checkpoint"],
            "mesh": protocol["mesh"],
        },
    }
    report = markdown_report(result, protocol)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    report_path = Path(__file__).resolve().parents[1] / "docs/20-results.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    cherries.log_asset(output / "results.json")
    cherries.log_asset(trajectory_plot["path"])
    cherries.log_asset(report_path)
    logger.info("Wrote %s and %s", output / "results.json", report_path)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
