# ruff: noqa: E402
"""Render a completed, matched Smile inverse-fit solver comparison.

This is deliberately a post-processing step.  It reads only the two accepted
endpoints and their accepted-step traces emitted by ``20-fit-smile.py``; it
does not rebuild physics or perform another forward solve.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
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
from joint_fields import activation_stresses_mpa

REFERENCE_MPA = 0.012328767123287673


class Config(cherries.BaseConfig):
    fit_dir: Path = EXPERIMENT / "data/smile-fit-001"
    inputs_dir: Path = SOURCE_GROUP / "data/expression-inputs-002"
    output_dir: Path = EXPERIMENT / "data/smile-fit-visuals-001"
    original_dir: Path | None = None
    hybrid_dir: Path | None = None


@dataclass(frozen=True)
class Arm:
    name: str
    label: str
    color: str
    directory: Path
    checkpoint: dict[str, Any]
    state: dict[str, Any]
    trace: list[dict[str, Any]]
    timing: list[dict[str, Any]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict), path
    return value


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        assert isinstance(value, dict), (path, number)
        values.append(value)
    return values


def faces(triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.int64)
    assert triangles.ndim == 2
    assert triangles.shape[1] == 3
    return np.column_stack(
        (np.full(len(triangles), 3, dtype=np.int64), triangles)
    ).ravel()


def camera(surface: pv.PolyData, view: str) -> dict[str, Any]:
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
        "scale": span / 1.5,
    }


def set_camera(plotter: pv.Plotter, setting: dict[str, Any]) -> None:
    plotter.camera_position = setting["position"]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = setting["scale"]
    plotter.reset_camera_clipping_range()


def checkpoint_path(arm_summary: dict[str, Any], arm_dir: Path) -> Path:
    record = arm_summary.get("latest_checkpoint")
    assert isinstance(record, dict), "arm summary must identify latest_checkpoint"
    # ``record['path']`` is runner provenance and may be a remote absolute
    # path.  The copied arm is self-contained at this stable relative path.
    path = arm_dir / "expressions" / "Smile" / "latest.pt"
    assert path.is_file(), path
    expected = record.get("sha256")
    if expected is not None:
        assert sha256(path) == expected, path
    return path.resolve()


def load_arm(directory: Path, name: str, label: str, color: str) -> Arm:
    summary_path = directory / "arm-summary.json"
    summary = read_json(summary_path)
    assert summary["arm"] == name
    assert summary["terminal"] != "failed", summary
    checkpoint = checkpoint_path(summary, directory)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert isinstance(state, dict)
    assert "displacement_m" in state
    assert "metrics" in state
    trace = jsonl(directory / "expressions/Smile/trace.jsonl")
    timing = jsonl(directory / "iteration-timing.jsonl")
    assert trace, f"no accepted trace rows: {name}"
    assert timing, f"no timing rows: {name}"
    return Arm(
        name,
        label,
        color,
        directory,
        {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        state,
        trace,
        timing,
    )


def latest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last log record at every accepted step, in monotone order."""
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = int(row["accepted_steps"])
        by_step[step] = row
    return [by_step[step] for step in sorted(by_step)]


def arm_series(arm: Arm) -> dict[str, np.ndarray]:
    trace = latest_rows(arm.trace)
    timing = {int(row["accepted_step"]): row for row in arm.timing}
    steps = np.asarray([int(row["accepted_steps"]) for row in trace], dtype=np.int64)
    missing = set(steps) - set(timing)
    assert missing <= {0}, (arm.name, missing)
    objective = np.asarray([float(row["objective"]) for row in trace], dtype=np.float64)
    rms = np.asarray([float(row["fit_rms_mm"]) for row in trace], dtype=np.float64)
    durations = np.asarray(
        [
            float(timing[int(step)]["total_seconds"]) if step in timing else 0.0
            for step in steps
        ]
    )
    assert np.isfinite(objective).all()
    assert np.isfinite(rms).all()
    assert np.isfinite(durations).all()
    assert np.all(durations >= 0)
    return {
        "steps": steps,
        "objective": objective,
        "rms": rms,
        "seconds": np.cumsum(durations),
    }


def load_context(
    inputs: EyeExpressionInputs,
) -> tuple[pv.PolyData, pv.PolyData, pv.PolyData, np.ndarray]:
    parent = Path(inputs.manifest["parent_frozen_neutral"]["directory"])
    frozen = read_json(parent / "manifest.json")
    geometry_path = Path(frozen["sources"]["geometry"]["path"])
    with np.load(geometry_path, allow_pickle=False) as geometry:
        cranium = pv.PolyData(
            geometry["cranium_points_m"], faces(geometry["cranium_faces"])
        )
        mandible = pv.PolyData(
            geometry["mandible_points_m"], faces(geometry["mandible_faces"])
        )
    eyes_manifest = Path(inputs.manifest["sources"]["eyes_manifest"]["path"])
    eyes = pv.read(eyes_manifest.parent / "eyes.vtp").triangulate()
    return (
        cranium,
        mandible,
        eyes,
        np.asarray(inputs.arrays["mandible_pivot_m"], dtype=np.float64),
    )


def add_context(
    plotter: pv.Plotter, cranium: pv.PolyData, mandible: pv.PolyData, eyes: pv.PolyData
) -> None:
    plotter.add_mesh(cranium, color="#d7c7ae", opacity=0.26, smooth_shading=True)
    plotter.add_mesh(mandible, color="#c5a87d", opacity=0.34, smooth_shading=True)
    plotter.add_mesh(eyes, color="#edf2f4", opacity=0.70, smooth_shading=True)


def transformed_mandible(
    source: pv.PolyData, pivot: np.ndarray, axis: np.ndarray, jaw_normalized: Any
) -> pv.PolyData:
    """Apply the fitter's one-axis hinge pose to the source mandible mesh."""
    amount = float(np.asarray(jaw_normalized, dtype=np.float64).reshape(()))
    rotation_vector = amount * np.pi / 18 * np.asarray(axis, dtype=np.float64)
    theta = float(np.linalg.norm(rotation_vector))
    if theta == 0.0:
        rotation = np.eye(3)
    else:
        unit = rotation_vector / theta
        x, y, z = unit
        skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
        rotation = (
            np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)
        )
    mesh = source.copy(deep=True)
    mesh.points = (np.asarray(source.points) - pivot) @ rotation.T + pivot
    return mesh


def render_motion(
    output: Path,
    target: pv.PolyData,
    original: pv.PolyData,
    hybrid: pv.PolyData,
    cranium: pv.PolyData,
    mandibles: tuple[pv.PolyData, pv.PolyData, pv.PolyData],
    eyes: pv.PolyData,
    view: str,
) -> str:
    maximum = max(float(mesh["motion_mm"].max()) for mesh in (target, original, hybrid))
    plotter = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(2100, 760))
    plotter.set_background("#f7f7f5")
    plotter.enable_anti_aliasing("ssaa")
    setting = camera(target, view)
    items = (
        (
            "Smile target",
            target,
            mandibles[0],
            "mandible shown at neutral reference; target jaw pose is unavailable",
        ),
        ("Original PNCG", original, mandibles[1], "fitted hinge pose"),
        ("Hybrid PNCG → Newton-CG", hybrid, mandibles[2], "fitted hinge pose"),
    )
    for index, (title, mesh, mandible, jaw_label) in enumerate(items):
        plotter.subplot(0, index)
        add_context(plotter, cranium, mandible, eyes)
        plotter.add_mesh(
            mesh,
            scalars="motion_mm",
            cmap="viridis",
            clim=(0, maximum),
            smooth_shading=True,
            scalar_bar_args={
                "title": "Motion (mm)",
                "vertical": False,
                "position_x": 0.18,
                "position_y": 0.035,
                "width": 0.64,
                "height": 0.08,
                "label_font_size": 10,
                "title_font_size": 12,
            },
        )
        plotter.add_text(
            f"{title}\n{view} · common motion scale\n{jaw_label}",
            position="upper_left",
            font_size=12,
            color="#202124",
        )
        set_camera(plotter, setting)
    name = f"smile-context-motion-{view}.png"
    plotter.show(screenshot=output / name, auto_close=True)
    return name


def render_neutral_target_preflight(
    output: Path,
    neutral: pv.PolyData,
    target: pv.PolyData,
    cranium: pv.PolyData,
    mandible: pv.PolyData,
    eyes: pv.PolyData,
    view: str,
) -> str:
    """Render real input geometry without labeling it as a fitted result."""
    maximum = max(float(neutral["motion_mm"].max()), float(target["motion_mm"].max()))
    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1500, 760))
    plotter.set_background("#f7f7f5")
    plotter.enable_anti_aliasing("ssaa")
    setting = camera(target, view)
    for index, (title, mesh) in enumerate(
        (("Frozen neutral input", neutral), ("Smile target input", target))
    ):
        plotter.subplot(0, index)
        add_context(plotter, cranium, mandible, eyes)
        plotter.add_mesh(
            mesh,
            scalars="motion_mm",
            cmap="viridis",
            clim=(0, maximum),
            smooth_shading=True,
            scalar_bar_args={
                "title": "Target motion (mm)",
                "vertical": False,
                "position_x": 0.18,
                "position_y": 0.035,
                "width": 0.64,
                "height": 0.08,
                "label_font_size": 10,
                "title_font_size": 12,
            },
        )
        plotter.add_text(
            f"RENDER PREFLIGHT — NOT A FIT RESULT\n{title}\n{view} · source collision context",
            position="upper_left",
            font_size=12,
            color="#202124",
        )
        set_camera(plotter, setting)
    name = f"smile-neutral-target-preflight-{view}.png"
    plotter.show(screenshot=output / name, auto_close=True)
    return name


def add_clay_context(
    plotter: pv.Plotter,
    cranium: pv.PolyData,
    mandible: pv.PolyData,
    eyes: pv.PolyData,
) -> None:
    """Use neutral materials so shape remains visible without a scalar field."""
    plotter.add_mesh(cranium, color="#eee8dc", opacity=0.38, smooth_shading=True)
    plotter.add_mesh(mandible, color="#e8dcc6", opacity=0.58, smooth_shading=True)
    plotter.add_mesh(eyes, color="#ffffff", opacity=0.85, smooth_shading=True)


def render_clay(
    output: Path,
    target: pv.PolyData,
    original: pv.PolyData,
    hybrid: pv.PolyData,
    original_arm: Arm,
    hybrid_arm: Arm,
    cranium: pv.PolyData,
    mandibles: tuple[pv.PolyData, pv.PolyData, pv.PolyData],
    eyes: pv.PolyData,
    view: str,
) -> str:
    """Render target and fit geometries without motion or error coloration."""
    plotter = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(2100, 760))
    plotter.set_background("#f7f7f5")
    plotter.enable_anti_aliasing("ssaa")
    setting = camera(target, view)
    items = (
        (
            "Smile target",
            target,
            mandibles[0],
            "target jaw pose unavailable; mandible at neutral reference",
        ),
        (
            "Original PNCG",
            original,
            mandibles[1],
            f"step {original_arm.state['accepted_steps']} · RMS {float(original_arm.state['metrics']['fit_rms_mm']):.3f} mm",
        ),
        (
            "Hybrid PNCG → Newton-CG",
            hybrid,
            mandibles[2],
            f"step {hybrid_arm.state['accepted_steps']} · RMS {float(hybrid_arm.state['metrics']['fit_rms_mm']):.3f} mm",
        ),
    )
    for index, (title, mesh, mandible, detail) in enumerate(items):
        plotter.subplot(0, index)
        add_clay_context(plotter, cranium, mandible, eyes)
        plotter.add_mesh(mesh, color="#d5a18a", smooth_shading=True)
        plotter.add_text(
            f"{title}\n{view} clay geometry\n{detail}",
            position="upper_left",
            font_size=12,
            color="#202124",
        )
        set_camera(plotter, setting)
    name = f"smile-clay-geometry-{view}.png"
    plotter.show(screenshot=output / name, auto_close=True)
    return name


def render_error(
    output: Path,
    original: pv.PolyData,
    hybrid: pv.PolyData,
    cranium: pv.PolyData,
    mandibles: tuple[pv.PolyData, pv.PolyData],
    eyes: pv.PolyData,
    view: str,
) -> str:
    maximum = max(float(mesh["fit_error_mm"].max()) for mesh in (original, hybrid))
    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1500, 760))
    plotter.set_background("#f7f7f5")
    plotter.enable_anti_aliasing("ssaa")
    setting = camera(original, view)
    for index, (title, mesh, mandible) in enumerate(
        (
            ("Original PNCG fit error", original, mandibles[0]),
            ("Hybrid PNCG → Newton-CG fit error", hybrid, mandibles[1]),
        )
    ):
        plotter.subplot(0, index)
        add_context(plotter, cranium, mandible, eyes)
        plotter.add_mesh(
            mesh,
            scalars="fit_error_mm",
            cmap="magma",
            clim=(0, maximum),
            smooth_shading=True,
            scalar_bar_args={
                "title": "Target residual (mm)",
                "vertical": False,
                "position_x": 0.18,
                "position_y": 0.035,
                "width": 0.64,
                "height": 0.08,
                "label_font_size": 10,
                "title_font_size": 12,
            },
        )
        plotter.add_text(
            f"{title}\n{view} · common residual scale\nfitted hinge pose",
            position="upper_left",
            font_size=12,
            color="#202124",
        )
        set_camera(plotter, setting)
    name = f"smile-fit-error-{view}.png"
    plotter.show(screenshot=output / name, auto_close=True)
    return name


def render_trends(output: Path, original: Arm, hybrid: Arm) -> str:
    arms = (original, hybrid)
    series = {arm.name: arm_series(arm) for arm in arms}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for arm in arms:
        values = series[arm.name]
        for axis, x, y, title, xlabel, ylabel in (
            (
                axes[0, 0],
                values["steps"],
                values["objective"],
                "Objective by accepted iteration",
                "Accepted iteration",
                "Objective",
            ),
            (
                axes[0, 1],
                values["steps"],
                values["rms"],
                "Fit RMS by accepted iteration",
                "Accepted iteration",
                "Fit RMS (mm)",
            ),
            (
                axes[1, 0],
                values["seconds"],
                values["objective"],
                "Objective by accumulated fit time",
                "Fit time (s)",
                "Objective",
            ),
            (
                axes[1, 1],
                values["seconds"],
                values["rms"],
                "Fit RMS by accumulated fit time",
                "Fit time (s)",
                "Fit RMS (mm)",
            ),
        ):
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
    for axis in axes.flat:
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=9)
    name = "smile-convergence.png"
    figure.savefig(output / name, dpi=200)
    plt.close(figure)
    return name


def render_active_stress(
    output: Path,
    original: Arm,
    hybrid: Arm,
    volume: pv.UnstructuredGrid,
    active_cells: np.ndarray,
) -> str:
    """Compare fitted six-coordinate symmetric active-stress fields."""
    fields: list[tuple[str, str, pv.PolyData]] = []
    for arm in (original, hybrid):
        coordinates = torch.as_tensor(arm.state["activation"], dtype=torch.float64)
        assert coordinates.shape == (len(active_cells), 6), (
            arm.name,
            coordinates.shape,
        )
        stress = activation_stresses_mpa(coordinates, REFERENCE_MPA)
        values = np.linalg.norm(stress.numpy(), axis=(1, 2)) * 1000
        assert np.isfinite(values).all()
        mesh = volume.copy(deep=True)
        mesh.points = np.asarray(volume.points) + np.asarray(
            arm.state["displacement_m"], dtype=np.float64
        )
        mesh.cell_data["active_stress_kpa"] = np.full(mesh.n_cells, np.nan)
        mesh.cell_data["active_stress_kpa"][active_cells] = values
        fields.append(
            (
                arm.label,
                arm.color,
                mesh.extract_cells(active_cells).extract_surface(algorithm=None),
            )
        )
    maximum = max(float(mesh["active_stress_kpa"].max()) for _, _, mesh in fields)
    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1600, 920))
    plotter.set_background("#f7f7f5")
    plotter.enable_anti_aliasing("ssaa")
    setting = camera(fields[0][2], "side")
    for index, (label, _color, mesh) in enumerate(fields):
        plotter.subplot(0, index)
        plotter.add_mesh(
            mesh,
            scalars="active_stress_kpa",
            cmap="magma",
            clim=(0, maximum),
            smooth_shading=True,
            scalar_bar_args={
                "title": "Frobenius norm (kPa)",
                "vertical": False,
                "position_x": 0.18,
                "position_y": 0.035,
                "width": 0.64,
                "height": 0.08,
                "label_font_size": 10,
                "title_font_size": 12,
            },
        )
        plotter.add_text(
            f"{label}\nFitted additive active stress · six symmetric coordinates/tet",
            position="upper_left",
            font_size=13,
            color="#202124",
        )
        set_camera(plotter, setting)
    name = "smile-active-stress.png"
    plotter.show(screenshot=output / name, auto_close=True)
    return name


def surface(
    neutral: pv.PolyData,
    target_increment: np.ndarray,
    prediction_increment: np.ndarray,
) -> pv.PolyData:
    result = neutral.copy(deep=True)
    result.points = neutral.points + prediction_increment
    result.point_data["motion_mm"] = 1000 * np.linalg.norm(prediction_increment, axis=1)
    result.point_data["fit_error_mm"] = 1000 * np.linalg.norm(
        prediction_increment - target_increment, axis=1
    )
    return result


def endpoint_difference(
    original: pv.PolyData, hybrid: pv.PolyData, weights: np.ndarray
) -> dict[str, Any]:
    weights = np.asarray(weights, dtype=np.float64)
    assert weights.shape == (original.n_points,)
    assert np.isclose(weights.sum(), 1.0)
    difference_mm = 1000 * (original.points - hybrid.points)
    squared_difference = np.sum(difference_mm**2, axis=1)
    return {
        "definition": "Direct original-versus-hybrid displacement difference at observed skin nodes; RMS uses the normalized fitting area weights.",
        "skin_weighted_rms_mm": float(np.sqrt(weights @ squared_difference)),
        "skin_max_node_mm": float(np.sqrt(squared_difference.max())),
    }


def main(cfg: Config) -> None:
    fit_summary = read_json(cfg.fit_dir / "summary.json")
    assert fit_summary["schema"] == "smile-solver-comparison-v1"
    assert fit_summary.get("success") is True, fit_summary.get("status")
    assert fit_summary["target_expression"] == "Smile"
    inputs = EyeExpressionInputs.load(cfg.inputs_dir)
    smile_index = int(fit_summary["source_expression_index"])
    assert inputs.expression_names[smile_index] == "Smile"
    assert set(fit_summary["arms"]) == {"original", "hybrid_diag"}
    original_dir = cfg.original_dir or cfg.fit_dir / "arms" / "original"
    hybrid_dir = cfg.hybrid_dir or cfg.fit_dir / "arms" / "hybrid_diag"
    original = load_arm(original_dir, "original", "Original PNCG", "#bc622a")
    hybrid = load_arm(hybrid_dir, "hybrid_diag", "Hybrid PNCG → Newton-CG", "#087d81")
    for arm in (original, hybrid):
        displacement = np.asarray(arm.state["displacement_m"], dtype=np.float64)
        assert displacement.shape == inputs.arrays["neutral_displacement_m"].shape
        assert np.isfinite(displacement).all(), arm.name
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    parent = Path(inputs.manifest["parent_frozen_neutral"]["directory"])
    frozen = read_json(parent / "manifest.json")
    prepared = read_json(Path(frozen["sources"]["prepared_manifest"]["path"]))
    skin = pv.read(Path(prepared["fixture"]["skin_path"])).triangulate()
    volume = pv.read(Path(prepared["fixture"]["volume_path"]))
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    observation_ids = np.asarray(inputs.arrays["observation_node_ids"], dtype=np.int64)
    assert np.array_equal(ids, observation_ids)
    neutral = skin.copy(deep=True)
    neutral.points = np.asarray(inputs.arrays["neutral_points_m"], dtype=np.float64)[
        ids
    ]
    target_increment = np.asarray(
        inputs.arrays["expression_displacement_m"], dtype=np.float64
    )[smile_index]
    origin = np.asarray(inputs.arrays["neutral_displacement_m"], dtype=np.float64)
    target = surface(neutral, target_increment, target_increment)
    original_surface = surface(
        neutral,
        target_increment,
        np.asarray(original.state["displacement_m"], dtype=np.float64)[ids]
        - origin[ids],
    )
    hybrid_surface = surface(
        neutral,
        target_increment,
        np.asarray(hybrid.state["displacement_m"], dtype=np.float64)[ids] - origin[ids],
    )
    cranium, neutral_mandible, eyes, pivot = load_context(inputs)
    hinge_axis = np.asarray(inputs.arrays["mandible_frame_world"], dtype=np.float64)[
        :, 0
    ]
    original_mandible = transformed_mandible(
        neutral_mandible, pivot, hinge_axis, original.state["jaw_normalized"]
    )
    hybrid_mandible = transformed_mandible(
        neutral_mandible, pivot, hinge_axis, hybrid.state["jaw_normalized"]
    )
    clay_front = render_clay(
        cfg.output_dir,
        target,
        original_surface,
        hybrid_surface,
        original,
        hybrid,
        cranium,
        (neutral_mandible, original_mandible, hybrid_mandible),
        eyes,
        "front",
    )
    clay_side = render_clay(
        cfg.output_dir,
        target,
        original_surface,
        hybrid_surface,
        original,
        hybrid,
        cranium,
        (neutral_mandible, original_mandible, hybrid_mandible),
        eyes,
        "side",
    )
    context_front = render_motion(
        cfg.output_dir,
        target,
        original_surface,
        hybrid_surface,
        cranium,
        (neutral_mandible, original_mandible, hybrid_mandible),
        eyes,
        "front",
    )
    context_side = render_motion(
        cfg.output_dir,
        target,
        original_surface,
        hybrid_surface,
        cranium,
        (neutral_mandible, original_mandible, hybrid_mandible),
        eyes,
        "side",
    )
    error_front = render_error(
        cfg.output_dir,
        original_surface,
        hybrid_surface,
        cranium,
        (original_mandible, hybrid_mandible),
        eyes,
        "front",
    )
    error_side = render_error(
        cfg.output_dir,
        original_surface,
        hybrid_surface,
        cranium,
        (original_mandible, hybrid_mandible),
        eyes,
        "side",
    )
    trend_image = render_trends(cfg.output_dir, original, hybrid)
    active_cells = np.asarray(inputs.arrays["active_cell_ids"], dtype=np.int64)
    active_stress_image = render_active_stress(
        cfg.output_dir, original, hybrid, volume, active_cells
    )
    receipt = {
        "schema": "smile-solver-comparison-visuals-v1",
        "success": True,
        "scope": "Post-processes completed accepted endpoints only; no physics solve was run.",
        "fit_summary": {
            "path": str((cfg.fit_dir / "summary.json").resolve()),
            "sha256": sha256(cfg.fit_dir / "summary.json"),
        },
        "inputs_manifest": {
            "path": str((cfg.inputs_dir / "manifest.json").resolve()),
            "sha256": sha256(cfg.inputs_dir / "manifest.json"),
        },
        "target_expression": "Smile",
        "target_expression_index": smile_index,
        "arms": {
            arm.name: {
                "label": arm.label,
                "checkpoint": arm.checkpoint,
                "accepted_trace_rows": len(arm.trace),
                "timing_rows": len(arm.timing),
                "terminal_fit_rms_mm": float(arm.state["metrics"]["fit_rms_mm"]),
                "terminal_objective": float(arm.state["metrics"]["objective"]),
            }
            for arm in (original, hybrid)
        },
        "endpoint_difference": endpoint_difference(
            original_surface,
            hybrid_surface,
            inputs.arrays["observation_weight_normalized"],
        ),
        "context": {
            "cranium": "fixed source collision obstacle",
            "mandible": "neutral pose in target context; saved fitted hinge pose in each solver panel",
            "eyeballs": "fixed registered collision obstacles",
        },
        "active_stress": {
            "definition": "Frobenius norm of fitted additive active stress in active tetrahedra",
            "parameterization": "six Frobenius-orthonormal symmetric stress coordinates per active tetrahedron",
            "reference_mpa": REFERENCE_MPA,
        },
        "figures": {
            "clay_geometry_front": clay_front,
            "clay_geometry_side": clay_side,
            "common_motion_front": context_front,
            "common_motion_side": context_side,
            "common_fit_error_front": error_front,
            "common_fit_error_side": error_side,
            "convergence": trend_image,
            "active_stress": active_stress_image,
        },
    }
    (cfg.output_dir / "summary.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main)
