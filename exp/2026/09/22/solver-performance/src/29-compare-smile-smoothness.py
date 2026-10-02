# ruff: noqa: E402
"""Post-process the matched hybrid Smile fits with and without smoothness.

This script is intentionally read-only.  It requires two completed hybrid arms
and refuses to render unless their recorded protocol differs only in the
smoothness objective weight.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch

from liblaf import cherries

EXPERIMENT = Path(__file__).resolve().parent.parent
SOURCE_GROUP = EXPERIMENT.parent.parent / "21/joint-activation-material-mandible"
sys.path.insert(0, str(SOURCE_GROUP / "src"))

from joint_expression_inputs import EyeExpressionInputs


def load_helpers() -> Any:
    """Load the shared renderer without making its numbered filename an import API."""
    path = Path(__file__).with_name("21-compare-smile.py")
    spec = importlib.util.spec_from_file_location("smile_comparison_helpers", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helpers = load_helpers()


class Config(cherries.BaseConfig):
    baseline_dir: Path = EXPERIMENT / "data/smile-fit-adam03-unconditional-004"
    zero_dir: Path = EXPERIMENT / "data/smile-fit-adam03-no-smoothness-005"
    inputs_dir: Path = SOURCE_GROUP / "data/expression-inputs-002"
    output_dir: Path = EXPERIMENT / "data/smile-no-smoothness-visuals-001"


BASELINE_WEIGHT = 5.066584049455902


def protocol(directory: Path) -> dict[str, Any]:
    value = helpers.read_json(directory / "protocol.json")
    assert value["schema"] == "smile-solver-comparison-v1"
    assert value["target_expression"] == "Smile"
    return value


def validate_provenance(
    baseline: dict[str, Any], zero: dict[str, Any]
) -> dict[str, Any]:
    """Prove this is the planned single-variable diagnostic before rendering."""
    baseline_config = dict(baseline["comparison_config"])
    zero_config = dict(zero["comparison_config"])
    assert "smoothness_weight" not in baseline_config
    assert "effective_smoothness_weight" not in baseline_config
    assert float(zero_config.pop("smoothness_weight")) == 0.0
    assert float(zero_config.pop("effective_smoothness_weight")) == 0.0
    assert baseline_config == zero_config
    assert baseline["frozen_inputs"] == zero["frozen_inputs"]
    for key in (
        "scope",
        "target_expression",
        "source_expression_index",
        "common_initialization",
        "common_contact",
        "candidate_geometry_gate",
        "outer_step_policy",
        "trial_prescreen",
        "newton_switch_policy",
        "active_stress",
        "common_forward_atol",
    ):
        assert baseline[key] == zero[key], key
    baseline_objective = dict(baseline["objective"])
    zero_objective = dict(zero["objective"])
    baseline_weight = float(baseline_objective.pop("smoothness_weight"))
    zero_weight = float(zero_objective.pop("smoothness_weight"))
    assert "calibrated_smoothness_weight" not in baseline_objective
    assert "smoothness_weight_override" not in baseline_objective
    assert "smoothness_weight_mode" not in baseline_objective
    assert np.isclose(
        float(zero_objective.pop("calibrated_smoothness_weight")), baseline_weight
    )
    assert float(zero_objective.pop("smoothness_weight_override")) == 0.0
    assert zero_objective.pop("smoothness_weight_mode") == "explicit_override"
    assert baseline_objective == zero_objective
    assert np.isclose(baseline_weight, BASELINE_WEIGHT)
    assert zero_weight == 0.0
    return {
        "only_recorded_difference": "objective.smoothness_weight",
        "baseline_smoothness_weight": baseline_weight,
        "zero_smoothness_weight": zero_weight,
        "comparison_config": baseline_config,
        "zero_smoothness_metadata": {
            "smoothness_weight": 0.0,
            "effective_smoothness_weight": 0.0,
            "calibrated_smoothness_weight": baseline_weight,
            "smoothness_weight_override": 0.0,
            "smoothness_weight_mode": "explicit_override",
        },
        "frozen_inputs": baseline["frozen_inputs"],
        "implementation_receipts": {
            "baseline": baseline["implementation"],
            "zero_smoothness": zero["implementation"],
        },
    }


def contact_status(arm: Any) -> dict[str, Any]:
    last = {int(row["accepted_step"]): row for row in arm.timing}[
        int(arm.state["accepted_steps"])
    ]
    physical = last["physical"]
    contact = physical["contact"]
    assert contact["enabled"] is True
    assert contact["contact_numerically_valid"] is True
    assert int(physical["inverted_tetrahedra"]) == 0
    return {
        "contact_enabled": True,
        "contact_numerically_valid": True,
        "inverted_tetrahedra": 0,
        "minimum_active_distance_mm": 1000
        * float(contact["minimum_active_distance_m"]),
        "detF_min": float(physical["detF_min"]),
        "force_norm": float(physical["force_norm"]),
        "force_threshold": float(physical["force_threshold"]),
    }


def load_failed_last_valid_arm(
    directory: Path, label: str, color: str
) -> tuple[Any, dict[str, Any]]:
    """Load only the checkpoint that predates a visibly recorded failed proposal."""
    summary = helpers.read_json(directory / "arm-summary.json")
    assert summary["arm"] == "hybrid_diag"
    assert summary["success"] is False
    failure = helpers.read_json(directory / "failure.json")
    assert failure["status"] == "failed"
    assert failure["error_type"] == "ForwardConvergenceError"
    assert failure["failure"] == "candidate has inverted tetrahedra"
    checkpoint_path = helpers.checkpoint_path(summary, directory)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert isinstance(state, dict)
    accepted_steps = int(state["accepted_steps"])
    assert accepted_steps == int(failure["accepted_steps"])
    assert np.isclose(
        float(state["metrics"]["fit_rms_mm"]), float(failure["fit_rms_mm"])
    )
    trace = helpers.jsonl(directory / "expressions/Smile/trace.jsonl")
    timing = helpers.jsonl(directory / "iteration-timing.jsonl")
    assert trace
    assert timing
    assert max(int(row["accepted_steps"]) for row in trace) == accepted_steps
    assert max(int(row["accepted_step"]) for row in timing) == accepted_steps
    verification_path = directory.parent.parent / "local-copy-verification.json"
    verification = helpers.read_json(verification_path)
    assert verification["schema"] == "smile-no-smoothness-local-copy-verification-v1"
    outcome = verification["outcome"]
    assert outcome["status"] == "failed_on_twentieth_proposal"
    assert int(outcome["accepted_updates_preserved"]) == accepted_steps
    candidate = outcome["candidate_physical_metrics"]["candidate_metrics"]
    assert int(candidate["inverted_tetrahedra"]) == 1
    return (
        helpers.Arm(
            "smoothness_0",
            label,
            color,
            directory,
            {"path": str(checkpoint_path), "sha256": helpers.sha256(checkpoint_path)},
            state,
            trace,
            timing,
        ),
        {
            "failure_path": str((directory / "failure.json").resolve()),
            "failure_sha256": helpers.sha256(directory / "failure.json"),
            "error_type": failure["error_type"],
            "failure": failure["failure"],
            "last_valid_accepted_steps": accepted_steps,
            "verification_path": str(verification_path.resolve()),
            "verification_sha256": helpers.sha256(verification_path),
            "candidate_physical_metrics": candidate,
            "candidate_metrics_provenance": outcome["candidate_physical_metrics"][
                "provenance"
            ],
        },
    )


def initial_displacement(
    arm: Any, ids: np.ndarray
) -> tuple[np.ndarray, dict[str, str]]:
    path = arm.directory / "expressions/Smile/initial.pt"
    assert path.is_file(), path
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(state, dict)
    displacement = np.asarray(state["displacement_m"], dtype=np.float64)
    return displacement[ids], {
        "path": str(path.resolve()),
        "sha256": helpers.sha256(path),
    }


def endpoint_motion(
    arm: Any, initial_observed: np.ndarray, ids: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    final_observed = np.asarray(arm.state["displacement_m"], dtype=np.float64)[ids]
    motion_mm = 1000 * (final_observed - initial_observed)
    squared = np.sum(motion_mm**2, axis=1)
    return {
        "definition": "Endpoint displacement from the shared recorded initial equilibrium at observed skin nodes.",
        "weighted_rms_mm": float(np.sqrt(np.asarray(weights) @ squared)),
        "max_node_mm": float(np.sqrt(squared.max())),
    }


def endpoint_detail(arm: Any, run_label: str) -> str:
    metrics = arm.state["metrics"]
    status = contact_status(arm)
    return (
        f"{run_label}\nstep {arm.state['accepted_steps']} · objective {float(metrics['objective']):.4g} · "
        f"RMS {float(metrics['fit_rms_mm']):.3f} mm\n"
        f"contact valid · no inversions · min gap {status['minimum_active_distance_mm']:.4f} mm"
    )


def scalar_bar(title: str) -> dict[str, Any]:
    return {
        "title": title,
        "vertical": False,
        "position_x": 0.18,
        "position_y": 0.035,
        "width": 0.64,
        "height": 0.08,
        "label_font_size": 10,
        "title_font_size": 12,
    }


def render_clay(
    output: Path,
    target: pv.PolyData,
    baseline: pv.PolyData,
    zero: pv.PolyData,
    baseline_arm: Any,
    zero_arm: Any,
    cranium: pv.PolyData,
    mandibles: tuple[pv.PolyData, pv.PolyData, pv.PolyData],
    eyes: pv.PolyData,
    baseline_run_label: str,
    zero_run_label: str,
    view: str,
) -> str:
    plotter = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(2250, 760))
    plotter.set_background("#f7f7f5")
    plotter.enable_anti_aliasing("ssaa")
    setting = helpers.camera(target, view)
    items = (
        (
            "Smile target",
            target,
            mandibles[0],
            "target jaw pose unavailable; mandible at neutral reference",
        ),
        (
            "Hybrid · smoothness 5.066584",
            baseline,
            mandibles[1],
            endpoint_detail(baseline_arm, baseline_run_label),
        ),
        (
            "Hybrid · smoothness 0",
            zero,
            mandibles[2],
            endpoint_detail(zero_arm, zero_run_label),
        ),
    )
    for index, (title, mesh, mandible, detail) in enumerate(items):
        plotter.subplot(0, index)
        helpers.add_clay_context(plotter, cranium, mandible, eyes)
        plotter.add_mesh(mesh, color="#d5a18a", smooth_shading=True)
        plotter.add_text(
            f"{title}\n{view} clay geometry\n{detail}",
            position="upper_left",
            font_size=11,
            color="#202124",
        )
        helpers.set_camera(plotter, setting)
    name = f"smile-smoothness-clay-{view}.png"
    plotter.show(screenshot=output / name, auto_close=True)
    return name


def render_motion(
    output: Path,
    target: pv.PolyData,
    baseline: pv.PolyData,
    zero: pv.PolyData,
    cranium: pv.PolyData,
    mandibles: tuple[pv.PolyData, pv.PolyData, pv.PolyData],
    eyes: pv.PolyData,
    baseline_run_label: str,
    zero_run_label: str,
    view: str,
) -> str:
    maximum = max(float(mesh["motion_mm"].max()) for mesh in (target, baseline, zero))
    plotter = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(2250, 760))
    plotter.set_background("#f7f7f5")
    plotter.enable_anti_aliasing("ssaa")
    setting = helpers.camera(target, view)
    items = (
        (
            "Smile target",
            target,
            mandibles[0],
            "target jaw pose unavailable; mandible at neutral reference",
        ),
        ("Hybrid · smoothness 5.066584", baseline, mandibles[1], baseline_run_label),
        ("Hybrid · smoothness 0", zero, mandibles[2], zero_run_label),
    )
    for index, (title, mesh, mandible, detail) in enumerate(items):
        plotter.subplot(0, index)
        helpers.add_context(plotter, cranium, mandible, eyes)
        plotter.add_mesh(
            mesh,
            scalars="motion_mm",
            cmap="viridis",
            clim=(0, maximum),
            smooth_shading=True,
            scalar_bar_args=scalar_bar("Motion from frozen neutral (mm)"),
        )
        plotter.add_text(
            f"{title}\n{view} · common motion scale\n{detail}",
            position="upper_left",
            font_size=12,
            color="#202124",
        )
        helpers.set_camera(plotter, setting)
    name = f"smile-smoothness-motion-{view}.png"
    plotter.show(screenshot=output / name, auto_close=True)
    return name


def render_error(
    output: Path,
    baseline: pv.PolyData,
    zero: pv.PolyData,
    cranium: pv.PolyData,
    mandibles: tuple[pv.PolyData, pv.PolyData],
    eyes: pv.PolyData,
    baseline_run_label: str,
    zero_run_label: str,
    view: str,
) -> str:
    maximum = max(float(mesh["fit_error_mm"].max()) for mesh in (baseline, zero))
    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1600, 760))
    plotter.set_background("#f7f7f5")
    plotter.enable_anti_aliasing("ssaa")
    setting = helpers.camera(baseline, view)
    for index, (title, mesh, mandible, run_label) in enumerate(
        (
            (
                "Hybrid · smoothness 5.066584",
                baseline,
                mandibles[0],
                baseline_run_label,
            ),
            ("Hybrid · smoothness 0", zero, mandibles[1], zero_run_label),
        )
    ):
        plotter.subplot(0, index)
        helpers.add_context(plotter, cranium, mandible, eyes)
        plotter.add_mesh(
            mesh,
            scalars="fit_error_mm",
            cmap="magma",
            clim=(0, maximum),
            smooth_shading=True,
            scalar_bar_args=scalar_bar("Target residual (mm)"),
        )
        plotter.add_text(
            f"{title}\n{view} · common residual scale\n{run_label}",
            position="upper_left",
            font_size=12,
            color="#202124",
        )
        helpers.set_camera(plotter, setting)
    name = f"smile-smoothness-error-{view}.png"
    plotter.show(screenshot=output / name, auto_close=True)
    return name


def render_trends(
    output: Path, baseline: Any, zero: Any, zero_failure: dict[str, Any]
) -> str:
    arms = (baseline, zero)
    series = {arm.name: helpers.arm_series(arm) for arm in arms}
    common_last_step = int(zero.state["accepted_steps"])
    figure, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    charts = (
        (
            "steps",
            "objective",
            "Objective by accepted iteration\nnot comparable: smoothness terms differ",
            "Accepted iteration",
            "Objective",
        ),
        (
            "steps",
            "rms",
            "Positional fit RMS by accepted iteration",
            "Accepted iteration",
            "Fit RMS (mm)",
        ),
        (
            "steps",
            "smoothness",
            "Unweighted spatial roughness by iteration",
            "Accepted iteration",
            "Raw smoothness",
        ),
        (
            "seconds",
            "objective",
            "Objective by accumulated fit time\nnot comparable: smoothness terms differ",
            "Fit time (s)",
            "Objective",
        ),
        (
            "seconds",
            "rms",
            "Positional fit RMS by accumulated fit time",
            "Fit time (s)",
            "Fit RMS (mm)",
        ),
        (
            "steps",
            "neighbor_rms",
            "Neighbor active-stress roughness by iteration",
            "Accepted iteration",
            "Neighbor RMS",
        ),
    )
    for axis, (x_name, y_name, title, xlabel, ylabel) in zip(
        axes.flat, charts, strict=True
    ):
        for arm in arms:
            trace = [
                row
                for row in helpers.latest_rows(arm.trace)
                if int(row["accepted_steps"]) <= common_last_step
            ]
            values = series[arm.name]
            mask = values["steps"] <= common_last_step
            x = values[x_name][mask] if x_name in values else values["steps"][mask]
            y = (
                values[y_name][mask]
                if y_name in values
                else np.asarray([float(row[y_name]) for row in trace])
            )
            axis.plot(
                x,
                y,
                "o-",
                color=arm.color,
                markersize=3,
                linewidth=1.5,
                label=arm.label,
            )
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=9)
    figure.suptitle(
        f"Common valid trace through step {common_last_step}; smoothness 0 proposal "
        f"{common_last_step + 1} failed: {zero_failure['failure']}",
        fontsize=12,
    )
    name = "smile-smoothness-convergence-roughness.png"
    figure.savefig(output / name, dpi=200)
    plt.close(figure)
    return name


def main(cfg: Config) -> None:
    baseline_protocol = protocol(cfg.baseline_dir)
    zero_protocol = protocol(cfg.zero_dir)
    provenance = validate_provenance(baseline_protocol, zero_protocol)
    inputs = EyeExpressionInputs.load(cfg.inputs_dir)
    smile_index = int(baseline_protocol["source_expression_index"])
    assert inputs.expression_names[smile_index] == "Smile"
    baseline = replace(
        helpers.load_arm(
            cfg.baseline_dir / "arms/hybrid_diag",
            "hybrid_diag",
            "Hybrid · smoothness 5.066584",
            "#087d81",
        ),
        name="smoothness_5_066584",
    )
    zero, zero_failure = load_failed_last_valid_arm(
        cfg.zero_dir / "arms/hybrid_diag", "Hybrid · smoothness 0", "#b95c2e"
    )
    baseline_run_label = "completed budget endpoint · step 20"
    zero_run_label = (
        "LAST VALID endpoint · step 19\n"
        "proposal 20 failed: candidate had 1 inverted tetrahedron"
    )
    for arm in (baseline, zero):
        assert (
            np.asarray(arm.state["displacement_m"]).shape
            == inputs.arrays["neutral_displacement_m"].shape
        )
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    parent = Path(inputs.manifest["parent_frozen_neutral"]["directory"])
    frozen = helpers.read_json(parent / "manifest.json")
    prepared = helpers.read_json(Path(frozen["sources"]["prepared_manifest"]["path"]))
    skin = pv.read(Path(prepared["fixture"]["skin_path"])).triangulate()
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    assert np.array_equal(
        ids, np.asarray(inputs.arrays["observation_node_ids"], dtype=np.int64)
    )
    baseline_initial, baseline_initial_record = initial_displacement(baseline, ids)
    zero_initial, zero_initial_record = initial_displacement(zero, ids)
    assert np.array_equal(baseline_initial, zero_initial)
    neutral = skin.copy(deep=True)
    neutral.points = np.asarray(inputs.arrays["neutral_points_m"], dtype=np.float64)[
        ids
    ]
    target_increment = np.asarray(
        inputs.arrays["expression_displacement_m"], dtype=np.float64
    )[smile_index]
    origin = np.asarray(inputs.arrays["neutral_displacement_m"], dtype=np.float64)
    target = helpers.surface(neutral, target_increment, target_increment)
    baseline_surface = helpers.surface(
        neutral,
        target_increment,
        np.asarray(baseline.state["displacement_m"], dtype=np.float64)[ids]
        - origin[ids],
    )
    zero_surface = helpers.surface(
        neutral,
        target_increment,
        np.asarray(zero.state["displacement_m"], dtype=np.float64)[ids] - origin[ids],
    )
    cranium, neutral_mandible, eyes, pivot = helpers.load_context(inputs)
    axis = np.asarray(inputs.arrays["mandible_frame_world"], dtype=np.float64)[:, 0]
    baseline_mandible = helpers.transformed_mandible(
        neutral_mandible, pivot, axis, baseline.state["jaw_normalized"]
    )
    zero_mandible = helpers.transformed_mandible(
        neutral_mandible, pivot, axis, zero.state["jaw_normalized"]
    )
    figures = {}
    for view in ("front", "side"):
        figures[f"clay_{view}"] = render_clay(
            cfg.output_dir,
            target,
            baseline_surface,
            zero_surface,
            baseline,
            zero,
            cranium,
            (neutral_mandible, baseline_mandible, zero_mandible),
            eyes,
            baseline_run_label,
            zero_run_label,
            view,
        )
        figures[f"motion_{view}"] = render_motion(
            cfg.output_dir,
            target,
            baseline_surface,
            zero_surface,
            cranium,
            (neutral_mandible, baseline_mandible, zero_mandible),
            eyes,
            baseline_run_label,
            zero_run_label,
            view,
        )
        figures[f"error_{view}"] = render_error(
            cfg.output_dir,
            baseline_surface,
            zero_surface,
            cranium,
            (baseline_mandible, zero_mandible),
            eyes,
            baseline_run_label,
            zero_run_label,
            view,
        )
    figures["convergence_roughness"] = render_trends(
        cfg.output_dir, baseline, zero, zero_failure
    )
    weights = np.asarray(
        inputs.arrays["observation_weight_normalized"], dtype=np.float64
    )
    receipt = {
        "schema": "smile-smoothness-diagnostic-visuals-v1",
        "success": True,
        "scope": "CPU post-processing only: baseline completed step 20; zero-smoothness is the verified last valid step 19 before a failed proposal 20. No physics solve was run.",
        "provenance_validation": provenance,
        "shared_initial_equilibrium": {
            "exact_observed_displacements_equal": True,
            "baseline_checkpoint": baseline_initial_record,
            "zero_smoothness_checkpoint": zero_initial_record,
        },
        "comparison_steps": {
            "geometry": "baseline completed step 20 versus zero-smoothness last valid step 19; no baseline step-19 geometry checkpoint exists",
            "trends": "both arms are shown only through common valid step 19",
        },
        "baseline": {
            "label": baseline.label,
            "endpoint_status": "completed iteration budget",
            "checkpoint": baseline.checkpoint,
            "terminal_objective": float(baseline.state["metrics"]["objective"]),
            "terminal_fit_rms_mm": float(baseline.state["metrics"]["fit_rms_mm"]),
            "contact_status": contact_status(baseline),
            "endpoint_motion": endpoint_motion(
                baseline, baseline_initial, ids, weights
            ),
        },
        "zero_smoothness": {
            "label": zero.label,
            "endpoint_status": "last valid step 19; proposal 20 failed",
            "failure_receipt": zero_failure,
            "checkpoint": zero.checkpoint,
            "terminal_objective": float(zero.state["metrics"]["objective"]),
            "terminal_fit_rms_mm": float(zero.state["metrics"]["fit_rms_mm"]),
            "contact_status": contact_status(zero),
            "endpoint_motion": endpoint_motion(zero, zero_initial, ids, weights),
        },
        "target_expression": "Smile",
        "target_expression_index": smile_index,
        "context": {
            "cranium": "fixed source collision obstacle",
            "mandible": "neutral pose in target context; saved fitted hinge pose in each hybrid panel",
            "eyeballs": "fixed registered collision obstacles",
        },
        "figures": figures,
    }
    (cfg.output_dir / "summary.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main)
