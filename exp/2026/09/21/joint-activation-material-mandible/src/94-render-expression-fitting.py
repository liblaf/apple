"""Render only saved, accepted fixed-material expression-fit checkpoints."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs
from joint_expression_inputs import EyeExpressionInputs
from joint_fields import activation_stresses_mpa

from liblaf import cherries


class Config(cherries.BaseConfig):
    inputs_dir: Path = GROUP / "data/expression-inputs-002"
    fit_dir: Path = GROUP / "data/expression-fitting-008"
    output_dir: Path = GROUP / "data/expression-fitting-visuals-001"
    status_snapshot: Path | None = None
    max_activation_fields: int = 6


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return rows_from_bytes(stable_bytes(path))


def stable_bytes(path: Path) -> bytes:
    """Capture one immutable read of a concurrently refreshed artifact."""
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return payload


def rows_from_bytes(payload: bytes) -> list[dict]:
    result = []
    lines = payload.decode().splitlines()
    for number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines):
                break
            raise
        assert isinstance(value, dict)
        result.append(value)
    return result


def camera(surface: pv.PolyData, view: str) -> dict:
    xmin, xmax, ymin, ymax, zmin, zmax = surface.bounds
    center = np.array(((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2))
    span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    if view == "front":
        eye = center + np.array((0.0, 0.0, 2.7 * span))
    elif view == "side":
        eye = center + np.array((2.7 * span, 0.0, 0.0))
    else:
        raise ValueError(view)
    return {
        "position": [eye.tolist(), center.tolist(), [0.0, 1.0, 0.0]],
        "scale": span / 1.55,
    }


def set_camera(plot: pv.Plotter, setting: dict) -> None:
    plot.camera_position = setting["position"]
    plot.camera.parallel_projection = True
    plot.camera.parallel_scale = setting["scale"]
    plot.reset_camera_clipping_range()


def save_surface_triptych(
    output: Path,
    name: str,
    neutral: pv.PolyData,
    target_increment: np.ndarray,
    predicted_increment: np.ndarray,
    stage_label: str,
) -> dict:
    target = neutral.copy(deep=True)
    predicted = neutral.copy(deep=True)
    residual = neutral.copy(deep=True)
    target.points = neutral.points + target_increment
    predicted.points = neutral.points + predicted_increment
    residual.points = predicted.points.copy()
    target.point_data["motion_mm"] = 1000 * np.linalg.norm(target_increment, axis=1)
    predicted.point_data["motion_mm"] = 1000 * np.linalg.norm(
        predicted_increment, axis=1
    )
    residual.point_data["residual_mm"] = 1000 * np.linalg.norm(
        predicted_increment - target_increment, axis=1
    )
    maximum = max(
        float(target["motion_mm"].max()), float(predicted["motion_mm"].max()), 1e-8
    )
    residual_maximum = max(float(residual["residual_mm"].max()), 1e-8)
    plot = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1800, 750))
    plot.set_background("#f7f7f5")
    plot.enable_anti_aliasing("ssaa")
    framing = pv.PolyData(np.vstack((neutral.points, target.points, predicted.points)))
    setting = camera(framing, "front")
    setting["scale"] *= 1.55 / 1.45
    for column, title, mesh, scalar, limits, palette in (
        (0, "Target increment", target, "motion_mm", (0, maximum), "viridis"),
        (1, "Predicted increment", predicted, "motion_mm", (0, maximum), "viridis"),
        (2, "Residual norm", residual, "residual_mm", (0, residual_maximum), "magma"),
    ):
        plot.subplot(0, column)
        plot.add_text(
            f"{name}\n{stage_label}\n{title} (mm) · front · true scale",
            position="upper_left",
            font_size=13,
            color="#202124",
        )
        plot.add_mesh(
            mesh,
            scalars=scalar,
            cmap=palette,
            clim=limits,
            smooth_shading=True,
            scalar_bar_args={
                "title": f"{title} (mm)",
                "vertical": False,
                "position_x": 0.17,
                "position_y": 0.025,
                "width": 0.66,
                "height": 0.10,
                "label_font_size": 10,
                "title_font_size": 12,
                "color": "#202124",
            },
        )
        set_camera(plot, setting)
    filename = f"expression-{name}-surface.png"
    plot.show(screenshot=output / filename, auto_close=True)
    return {
        "filename": filename,
        "target_motion_max_mm": maximum,
        "residual_max_mm": residual_maximum,
    }


def save_trace(
    output: Path, name: str, trace: list[dict], *, pose_collision: bool
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    if not trace:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "No complete trace row was captured",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
        filename = f"expression-{name}-trend.png"
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)
        return filename
    for axis, key, label in (
        (axes[0], "objective", "Objective"),
        (axes[1], "fit_rms_mm", "Fit RMS (mm)"),
    ):
        pose_label = "Pose-only" if pose_collision else "Pose-only · collision off"
        stages = (
            ("pose_only", pose_label, "#bc622a"),
            ("joint", "Joint · collision on", "#087d81"),
        )
        present = set()
        for stage, stage_label, color in stages:
            rows_at_stage = [
                row for row in trace if row.get("fit_stage", "joint") == stage
            ]
            if rows_at_stage:
                present.add(stage)
                axis.plot(
                    [row["accepted_steps"] for row in rows_at_stage],
                    [row[key] for row in rows_at_stage],
                    "o-",
                    color=color,
                    markersize=3,
                    label=stage_label,
                )
        joint_steps = [
            row.get("pose_accepted_steps", row["accepted_steps"])
            for row in trace
            if row.get("fit_stage") == "joint"
        ]
        if "pose_only" in present and joint_steps:
            axis.axvline(
                min(joint_steps),
                color="#53565a",
                linestyle="--",
                linewidth=1,
                label="Joint unlocked",
            )
        axis.set(xlabel="Accepted outer step", ylabel=label, title=f"{name}: {label}")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=8)
    filename = f"expression-{name}-trend.png"
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)
    return filename


def save_activation_field(
    output: Path,
    name: str,
    volume: pv.UnstructuredGrid,
    reference_points: np.ndarray,
    active_cells: np.ndarray,
    activation: np.ndarray,
    displacement: np.ndarray,
    skin: pv.PolyData,
    stage_label: str,
) -> dict:
    """Render a declared Frobenius active-stress scalar on deformed muscle tets."""
    stress_kpa = (
        np.linalg.norm(
            activation_stresses_mpa(
                torch.as_tensor(activation), 0.012328767123287673
            ).numpy(),
            axis=(1, 2),
        )
        * 1000
    )
    values = np.full(volume.n_cells, np.nan)
    values[active_cells] = stress_kpa
    mesh = volume.copy(deep=True)
    mesh.points = reference_points + displacement
    mesh.cell_data["ActiveStressFrobenius_kPa"] = values
    active = mesh.extract_cells(active_cells).extract_surface(algorithm=None)
    deformed_skin = skin.copy(deep=True)
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    deformed_skin.points = reference_points[ids] + displacement[ids]
    plot = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    plot.set_background("#f7f7f5")
    plot.enable_anti_aliasing("ssaa")
    plot.add_text(
        f"{name}\n{stage_label}\nFrobenius norm of additive active stress (kPa) · deformed active muscle tets",
        position="upper_left",
        font_size=14,
        color="#202124",
    )
    plot.add_mesh(deformed_skin, color="#b96f59", opacity=0.12, smooth_shading=True)
    plot.add_mesh(
        active,
        scalars="ActiveStressFrobenius_kPa",
        cmap="magma",
        clim=(0, max(float(stress_kpa.max()), 1e-8)),
        smooth_shading=True,
        scalar_bar_args={
            "title": "kPa",
            "vertical": False,
            "position_x": 0.26,
            "position_y": 0.04,
            "width": 0.48,
            "height": 0.06,
        },
    )
    setting = camera(deformed_skin, "side")
    set_camera(plot, setting)
    filename = f"expression-{name}-active-stress.png"
    plot.show(screenshot=output / filename, auto_close=True)
    return {
        "filename": filename,
        "scalar": "Frobenius norm of additive active stress",
        "maximum_kpa": float(stress_kpa.max()),
        "rms_kpa": float(np.sqrt(np.mean(np.square(stress_kpa)))),
    }


def main(cfg: Config) -> None:  # noqa: PLR0915 - retain per-checkpoint provenance together
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    inputs = EyeExpressionInputs.load(cfg.inputs_dir)
    status_path = cfg.status_snapshot or cfg.fit_dir / "status.json"
    protocol_path = cfg.fit_dir / "protocol.json"
    status_bytes = stable_bytes(status_path)
    protocol_bytes = stable_bytes(protocol_path)
    status = json.loads(status_bytes)
    protocol = json.loads(protocol_bytes)
    assert status["schema"] == "joint-fixed-material-expression-fitting-v1"
    assert protocol["schema"] == "joint-fixed-material-expression-protocol-v1"
    assert protocol["inputs_manifest_sha256"] == sha256(
        cfg.inputs_dir / "manifest.json"
    )
    assert len(inputs.expression_names) == 36
    parent = Path(inputs.manifest["parent_frozen_neutral"]["directory"])
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    prepared = PreparedInputs.load(
        Path(parent_manifest["sources"]["prepared_npz"]["path"]),
        Path(parent_manifest["sources"]["prepared_manifest"]["path"]),
    )
    volume = pv.read(prepared.volume_path)
    reference_points = np.asarray(volume.points, dtype=np.float64).copy()
    skin = pv.read(prepared.skin_path).triangulate()
    obs = np.asarray(inputs.arrays["observation_node_ids"], dtype=np.int64)
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    assert np.array_equal(skin_ids, obs)
    neutral = skin.copy(deep=True)
    neutral.points = np.asarray(inputs.arrays["neutral_points_m"])[obs]
    target = np.asarray(inputs.arrays["expression_displacement_m"], dtype=np.float64)
    neutral_displacement = np.asarray(
        inputs.arrays["neutral_displacement_m"], dtype=np.float64
    )
    active_cells = np.asarray(inputs.arrays["active_cell_ids"], dtype=np.int64)
    pose_collision = bool(status.get("pose_collision", True))
    available: list[dict] = []
    queued: list[str] = []
    for index, name in enumerate(inputs.expression_names):
        directory = cfg.fit_dir / "expressions" / name
        latest = directory / "latest.pt"
        if not latest.is_file():
            queued.append(name)
            continue
        checkpoint_bytes = stable_bytes(latest)
        state = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False
        )
        if state.get("schema") is None:
            assert state["accepted_steps"] == 0
            checkpoint_kind = "initial accepted equilibrium"
        else:
            assert state["schema"] == "joint-fixed-material-expression-checkpoint-v1"
            assert state["expression"] == name
            assert state["expression_index"] == index
            checkpoint_kind = "accepted optimizer checkpoint"
        u = np.asarray(state["displacement_m"], dtype=np.float64)
        assert u.shape == neutral_displacement.shape
        assert np.isfinite(u).all()
        prediction = u[obs] - neutral_displacement[obs]
        trace = [
            row
            for row in rows(directory / "trace.jsonl")
            if row["accepted_steps"] <= state["accepted_steps"]
        ]
        metrics = state["metrics"]
        fit_stage = str(state.get("fit_stage", metrics.get("fit_stage", "joint")))
        stage_label = (
            "Pose-only · collision off (initialization)"
            if fit_stage == "pose_only" and not pose_collision
            else "Pose-only · collision on"
            if fit_stage == "pose_only"
            else "Joint · full skull + fixed-eye collision on"
        )
        surface = save_surface_triptych(
            cfg.output_dir, name, neutral, target[index], prediction, stage_label
        )
        trend = save_trace(cfg.output_dir, name, trace, pose_collision=pose_collision)
        field = None
        if len(available) < cfg.max_activation_fields:
            field = save_activation_field(
                cfg.output_dir,
                name,
                volume,
                reference_points,
                active_cells,
                np.asarray(state["activation"], dtype=np.float64),
                u,
                skin,
                stage_label,
            )
        pose = np.asarray(metrics["jaw_pose_rad_m"], dtype=np.float64)
        available.append(
            {
                "name": name,
                "status": status["expressions"][name]["status"],
                "fit_stage": fit_stage,
                "stage_label": stage_label,
                "pose_collision": pose_collision,
                "checkpoint_kind": checkpoint_kind,
                "checkpoint": {
                    "path": str(latest.resolve()),
                    "captured_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                },
                "accepted_steps": state["accepted_steps"],
                "pose_accepted_steps": state.get("pose_accepted_steps", 0),
                "joint_accepted_steps": state.get(
                    "joint_accepted_steps", state["accepted_steps"]
                ),
                "pose_converged": bool(state.get("pose_converged", False)),
                "fit_rms_mm": metrics["fit_rms_mm"],
                "initial_fit_rms_mm": metrics.get(
                    "initial_fit_rms_mm", metrics["target_motion_rms_mm"]
                ),
                "target_motion_rms_mm": metrics["target_motion_rms_mm"],
                "neighbor_rms": metrics["neighbor_rms"],
                "smoothness": metrics["smoothness"],
                "weighted_smoothness": metrics["weighted_smoothness"],
                "magnitude": metrics["magnitude"],
                "weighted_magnitude": metrics["weighted_magnitude"],
                "weighted_jaw_prior": metrics["weighted_jaw_prior"],
                "jaw_pose_rad_m": metrics["jaw_pose_rad_m"],
                "jaw_rotation_magnitude_deg": float(
                    np.linalg.norm(pose[:3]) * 180 / np.pi
                ),
                "jaw_translation_magnitude_mm": float(np.linalg.norm(pose[3:]) * 1000),
                "stationarity": metrics["stationarity"],
                "surface": surface,
                "trend": trend,
                "active_stress_field": field,
            }
        )
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)
    if available:
        names = [
            f"{row['name']}\n{row['fit_stage'].replace('_', ' ')} · "
            + (f"step {row['accepted_steps']}" if row["accepted_steps"] else "initial")
            for row in available
        ]
        values = [row["fit_rms_mm"] for row in available]
        axes[0].bar(range(len(names)), values, color="#087d81")
        axes[0].set(
            ylabel="Current saved state RMS (mm)",
            title="Saved expression states; initial or pose-only states are not joint fit results",
        )
        if status.get("mandible_dofs_per_expression") == 1:
            assert all(row["jaw_translation_magnitude_mm"] == 0 for row in available)
            axes[1].bar(
                range(len(names)),
                [row["jaw_rotation_magnitude_deg"] for row in available],
                color="#7da5ac",
            )
            axes[1].set(
                ylabel="Opening angle (degrees)",
                title="One mandible hinge angle; translation fixed at zero",
            )
        else:
            axes[1].bar(
                np.arange(len(names)) - 0.2,
                [row["jaw_rotation_magnitude_deg"] for row in available],
                0.4,
                label="rotation (deg)",
                color="#7da5ac",
            )
            axes[1].bar(
                np.arange(len(names)) + 0.2,
                [row["jaw_translation_magnitude_mm"] for row in available],
                0.4,
                label="translation (mm)",
                color="#bc622a",
            )
            axes[1].legend(fontsize=8)
            axes[1].set(ylabel="Jaw-pose magnitude", title="Saved mandible pose")
        for axis in axes:
            axis.set_xticks(
                range(len(names)), names, rotation=70, ha="right", fontsize=7
            )
    else:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "No expression checkpoint has been accepted yet",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    dashboard = "expression-fit-dashboard.png"
    fig.savefig(cfg.output_dir / dashboard, dpi=180)
    plt.close(fig)
    receipt = {
        "schema": "joint-fixed-material-expression-render-v1",
        "success": True,
        "scope": "Renders only existing accepted checkpoints. Queued or unavailable expressions are reported as no-data, never as failed or fitted. Pose-only collision policy is retained per saved state. Mutable status and trace inputs are captured as stable byte snapshots.",
        "inputs_manifest_sha256": sha256(cfg.inputs_dir / "manifest.json"),
        "fit_status": {
            "path": str(status_path.resolve()),
            "captured_sha256": hashlib.sha256(status_bytes).hexdigest(),
        },
        "fit_protocol": {
            "path": str(protocol_path.resolve()),
            "captured_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        },
        "pose_collision": pose_collision,
        "expression_count": 36,
        "available_count": len(available),
        "queued_or_unavailable": queued,
        "expressions": available,
        "dashboard": dashboard,
        "fit_running": status["running"],
        "fit_phase": status.get("phase"),
        "final_joint_is_deliverable": False,
    }
    write_json(cfg.output_dir / "summary.json", receipt)
    cherries.log_output(cfg.output_dir)
    cherries.log_metrics(
        {
            "available_expression_checkpoints": len(available),
            "queued_or_unavailable": len(queued),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
