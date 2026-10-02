"""Render the saved fixed-axis inverse refit and its optimization history."""

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
CAMERAS = ROOT / "exp/2026/09/08/physical-volume-closeups/data/20-regions/summary.json"
REFERENCE_RENDERER = (
    ROOT / "exp/2026/09/09/activation-space-smoothness/src/81-render-idea-shapes.py"
)
VIEWS = ("side-context", "region1-mouth-corner")
WINDOW_SIZE = (1800, 1800)
BACKGROUND = "#f4f2ed"
TARGET_COLOR = "#b8afa0"
SHAPE_COLOR = "#aeb7ba"
REFIT_COLOR = "#83aeb7"
EXPECTED_FROZEN_SHA256 = {
    REFERENCE_RENDERER: "c3b0543661c996a0f9409017d9c5de52ab833a753c072ca1e0ea18c807fd3138",
    CAMERAS: "f2628b1ff8fd82386fc02498739bb651f7b1e217479bc45350ec6b893ee7b58f",
    FIXTURE
    / "volume.vtu": "238962d0d27a2d35b6211a7a60204d362187b4190dfc7591756bc73ea26ff3b6",
    FIXTURE
    / "skin.vtp": "79eed2a5e2b5f23e84287fe729989ab356cb2780e3342b722b074a9231297833",
}
STATE_SPECS = (
    ("target", "Target smile\nFrozen fixture field", TARGET_COLOR),
    ("baseline-replay", "Full fitted activation\nReplayed equilibrium", SHAPE_COLOR),
    (
        "dominant-only",
        "Dominant-only start\nFrozen reference axes",
        SHAPE_COLOR,
    ),
    ("fixed-axis-refit", "Fixed-axis refit\nBest Adam iterate", REFIT_COLOR),
)
TRACE_COLUMNS = (
    "step",
    "objective_mm2",
    "uniform_fit_rms_mm",
    "area_weighted_fit_rms_mm",
    "area_weighted_motion_rms_mm",
    "projected_gradient_rms",
    "elapsed_seconds",
)


class Config(cherries.BaseConfig):
    """Saved refit inputs and Cherries-managed render destination."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    baseline_replay: Path = cherries.input("10-forward/baseline-replay.npz")
    dominant_only: Path = cherries.input("10-forward/dominant-only.npz")
    forward_summary: Path = cherries.input("10-forward/summary.json")
    initialization: Path = cherries.input("40-fixed-directions/initialization.npz")
    best: Path = cherries.input("40-fixed-directions/best.npz")
    last: Path = cherries.input("40-fixed-directions/last.npz")
    trace: Path = cherries.input("40-fixed-directions/trace.csv")
    refit_summary: Path = cherries.input("40-fixed-directions/summary.json")
    output_dir: Path = cherries.output("50-fixed-directions", mkdir=True)


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _digest(resolved),
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _snapshot_source(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o444)
    if _digest(source) != _digest(destination):
        raise AssertionError("source snapshot differs from executed source")
    return {"live_at_generation": _record(source), "snapshot": _record(destination)}


# Copied verbatim from the frozen reference renderer. Its source hash is checked
# before rendering so changes cannot silently alter the stated style provenance.
def _add_lights(plotter: pv.Plotter, camera: dict[str, Any]) -> None:
    focus = np.asarray(camera["focal_point"], dtype=np.float64)
    backward = np.asarray(camera["position"], dtype=np.float64) - focus
    backward /= np.linalg.norm(backward)
    right = np.cross(np.asarray(camera["view_up"], dtype=np.float64), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    key = focus + 0.3 * (0.72 * right + 0.35 * up + 0.60 * backward)
    fill = focus + 0.3 * backward
    for position, intensity in ((key, 0.85), (fill, 0.20)):
        plotter.add_light(
            pv.Light(
                position=position,
                focal_point=focus,
                intensity=intensity,
                light_type="scene light",
                positional=False,
            ),
            only_active=True,
        )


def _set_camera(plotter: pv.Plotter, camera: dict[str, Any]) -> None:
    plotter.enable_parallel_projection()
    plotter.camera.position = camera["position"]
    plotter.camera.focal_point = camera["focal_point"]
    plotter.camera.up = camera["view_up"]
    plotter.camera.parallel_scale = camera["parallel_scale"]
    plotter.set_background(BACKGROUND)
    plotter.reset_camera_clipping_range()


def _render_shape(
    mesh: pv.PolyData,
    camera: dict[str, Any],
    caption: str,
    color: str,
    path: Path,
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE, lighting="none")
    actor = plotter.add_mesh(
        mesh,
        color=color,
        smooth_shading=False,
        ambient=0.20,
        diffuse=0.80,
        specular=0.0,
    )
    if actor.GetProperty().GetInterpolation() != 0:
        raise AssertionError("shape rendering must use flat shading")
    _add_lights(plotter, camera)
    _set_camera(plotter, camera)
    annotation = plotter.add_text(
        caption, position="upper_left", color="black", font_size=13
    )
    annotation.GetTextProperty().SetBackgroundColor(244 / 255, 242 / 255, 237 / 255)
    annotation.GetTextProperty().SetBackgroundOpacity(0.90)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(path)
    plotter.close()


def _render_plate(images: list[Path], view_label: str, path: Path) -> None:
    if len(images) != 4:
        raise ValueError("comparison plate requires exactly four state panels")
    figure, axes = plt.subplots(1, 4, figsize=(21.6, 5.6), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    for axis, image in zip(axes, images, strict=True):
        axis.imshow(plt.imread(image))
        axis.set_axis_off()
    figure.suptitle(view_label, color="black", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _load_forward_state(path: Path) -> dict[str, np.ndarray]:
    required = {"u", "rest_points", "active_ids"}
    with np.load(path, allow_pickle=False) as saved:
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"{path} lacks required arrays: {sorted(missing)}")
        state = {name: np.asarray(saved[name]).copy() for name in required}
    if state["u"].shape != state["rest_points"].shape:
        raise ValueError(f"u/rest shape mismatch in {path}")
    if state["rest_points"].ndim != 2 or state["rest_points"].shape[1] != 3:
        raise ValueError(f"rest_points has invalid shape in {path}")
    if not np.isfinite(state["u"]).all() or not np.isfinite(state["rest_points"]).all():
        raise FloatingPointError(f"non-finite displacement or rest points in {path}")
    return state


def _load_initialization(path: Path) -> dict[str, np.ndarray]:
    required = {"axes", "initial_s", "active_ids", "rest_points"}
    with np.load(path, allow_pickle=False) as saved:
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"{path} lacks required arrays: {sorted(missing)}")
        state = {name: np.asarray(saved[name]).copy() for name in required}
    count = len(state["active_ids"])
    if state["axes"].shape != (count, 3):
        raise ValueError(f"axes has invalid shape in {path}: {state['axes'].shape}")
    if state["initial_s"].shape != (count,):
        raise ValueError(f"initial_s has invalid shape in {path}")
    if state["rest_points"].ndim != 2 or state["rest_points"].shape[1] != 3:
        raise ValueError(f"rest_points has invalid shape in {path}")
    if not all(
        np.isfinite(state[name]).all() for name in ("axes", "initial_s", "rest_points")
    ):
        raise FloatingPointError(f"non-finite initialization array in {path}")
    if np.any(state["initial_s"] < 0.0):
        raise ValueError("fixed-axis initial strengths must be nonnegative")
    axis_norm = np.linalg.norm(state["axes"], axis=1)
    if not np.allclose(axis_norm, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("fixed reference axes must be unit length")
    return state


def _load_refit_state(
    path: Path, point_count: int, active_count: int
) -> dict[str, np.ndarray]:
    required = {"s", "u", "step", "solver_valid", "active_ids"}
    with np.load(path, allow_pickle=False) as saved:
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"{path} lacks required arrays: {sorted(missing)}")
        state = {name: np.asarray(saved[name]).copy() for name in required}
    if state["s"].shape != (active_count,):
        raise ValueError(f"s has invalid shape in {path}: {state['s'].shape}")
    if state["u"].shape != (point_count, 3):
        raise ValueError(f"u has invalid shape in {path}: {state['u'].shape}")
    if state["active_ids"].shape != (active_count,):
        raise ValueError(f"active_ids has invalid shape in {path}")
    if state["step"].shape != () or state["solver_valid"].shape != ():
        raise ValueError(f"step and solver_valid must be scalar in {path}")
    if not np.isfinite(state["s"]).all() or not np.isfinite(state["u"]).all():
        raise FloatingPointError(f"non-finite refit state in {path}")
    if np.any(state["s"] < 0.0):
        raise ValueError(f"fixed-axis strengths must be nonnegative in {path}")
    if not bool(state["solver_valid"]):
        raise ValueError(f"saved refit state is solver-invalid: {path}")
    return state


def _read_trace(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        missing = set(TRACE_COLUMNS) - set(fieldnames)
        positions = [
            fieldnames.index(name) for name in TRACE_COLUMNS if name in fieldnames
        ]
        if missing or positions != sorted(positions):
            raise ValueError(
                "trace lacks required columns or changes their relative order: "
                f"missing={sorted(missing)}, columns={reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("optimization trace is empty")
    trace = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in TRACE_COLUMNS
    }
    if not all(np.isfinite(values).all() for values in trace.values()):
        raise FloatingPointError("optimization trace contains non-finite values")
    if not np.all(trace["step"] == np.floor(trace["step"])):
        raise ValueError("trace steps must be integers")
    if np.any(np.diff(trace["step"]) <= 0.0):
        raise ValueError("trace steps must be strictly increasing")
    nonnegative = set(TRACE_COLUMNS) - {"step"}
    if any(np.any(trace[name] < 0.0) for name in nonnegative):
        raise ValueError("trace metrics and elapsed time must be nonnegative")
    implied_objective = np.square(trace["uniform_fit_rms_mm"]) / 3.0
    if not np.allclose(
        trace["objective_mm2"], implied_objective, rtol=1e-6, atol=1e-12
    ):
        raise ValueError("objective_mm2 must equal uniform vector RMS squared / 3")
    return trace


def _baseline_levels(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    matches = [
        stage for stage in payload["stages"] if stage["name"] == "baseline-replay"
    ]
    if len(matches) != 1:
        raise ValueError("forward summary must contain one baseline-replay stage")
    metrics = matches[0]["metrics"]
    uniform = float(metrics["fit_rms_mm"])
    area_weighted = float(metrics["area_weighted_fit_rms_mm"])
    if (
        not np.isfinite([uniform, area_weighted]).all()
        or min(uniform, area_weighted) < 0.0
    ):
        raise ValueError("forward summary has invalid baseline fit levels")
    return {
        "uniform_fit_rms_mm": uniform,
        "area_weighted_fit_rms_mm": area_weighted,
        "objective_mm2": uniform**2 / 3.0,
    }


def _render_history(
    trace: dict[str, np.ndarray], baseline: dict[str, float], path: Path
) -> None:
    step = trace["step"]
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)
    figure.patch.set_facecolor("white")

    objective_axis, rms_axis = axes
    objective_axis.plot(
        step,
        trace["objective_mm2"],
        color="#35758a",
        linewidth=2.0,
        label="Fixed-axis Adam",
    )
    objective_axis.axhline(
        baseline["objective_mm2"],
        color="#3a3a3a",
        linestyle="--",
        linewidth=1.7,
        label="Full baseline: uniform MSE/component",
    )
    objective_axis.set_title("Optimization objective")
    objective_axis.set_xlabel("Adam step")
    objective_axis.set_ylabel("Uniform component MSE (mm²)")
    objective_axis.grid(alpha=0.22)
    objective_axis.legend(frameon=False)

    rms_axis.plot(
        step,
        trace["uniform_fit_rms_mm"],
        color="#35758a",
        linewidth=2.0,
        label="Refit: uniform RMS",
    )
    rms_axis.plot(
        step,
        trace["area_weighted_fit_rms_mm"],
        color="#b06b35",
        linewidth=2.0,
        label="Refit: area-weighted RMS",
    )
    rms_axis.axhline(
        baseline["uniform_fit_rms_mm"],
        color="#35758a",
        linestyle="--",
        linewidth=1.7,
        label="Full baseline: uniform RMS",
    )
    rms_axis.axhline(
        baseline["area_weighted_fit_rms_mm"],
        color="#b06b35",
        linestyle=":",
        linewidth=2.1,
        label="Full baseline: area-weighted RMS",
    )
    rms_axis.set_title("Surface fit")
    rms_axis.set_xlabel("Adam step")
    rms_axis.set_ylabel("Vector RMS (mm)")
    rms_axis.grid(alpha=0.22)
    rms_axis.legend(frameon=False)

    figure.suptitle(
        "Fixed reference direction per active tetrahedron; nonnegative scalar strength",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, facecolor=figure.get_facecolor())
    plt.close(figure)


def _array_records(state: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "sha256": _array_sha256(value),
        }
        for name, value in state.items()
    }


def _main(cfg: Config) -> dict[str, Any]:
    output = cfg.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    required = [
        *EXPECTED_FROZEN_SHA256,
        cfg.baseline_replay,
        cfg.dominant_only,
        cfg.forward_summary,
        cfg.initialization,
        cfg.best,
        cfg.last,
        cfg.trace,
        cfg.refit_summary,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path, expected in EXPECTED_FROZEN_SHA256.items():
        actual = _digest(path)
        if actual != expected:
            raise ValueError(
                f"frozen input/source changed: {path}; expected {expected}, got {actual}"
            )
    json.loads(cfg.refit_summary.read_text())

    volume = pv.read(FIXTURE / "volume.vtu")
    skin = pv.read(FIXTURE / "skin.vtp")
    fixture_rest = np.asarray(volume.points, dtype=np.float64)
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    fixture_active_ids = np.flatnonzero(
        np.asarray(volume.cell_data["ActivationMask"], dtype=bool)
    )
    if len(np.unique(skin_ids)) != len(skin_ids):
        raise ValueError("fixture skin GlobalPointId values are not unique")
    if not np.array_equal(
        np.asarray(skin.points, dtype=np.float64), fixture_rest[skin_ids]
    ):
        raise ValueError("fixture skin points do not match GlobalPointId indexing")

    replay = _load_forward_state(cfg.baseline_replay)
    dominant = _load_forward_state(cfg.dominant_only)
    initialization = _load_initialization(cfg.initialization)
    best = _load_refit_state(cfg.best, len(fixture_rest), len(fixture_active_ids))
    last = _load_refit_state(cfg.last, len(fixture_rest), len(fixture_active_ids))
    for identifier, state in (("baseline-replay", replay), ("dominant-only", dominant)):
        if not np.array_equal(state["rest_points"], fixture_rest):
            raise ValueError(f"{identifier} rest_points differ from the fixture")
        if not np.array_equal(state["active_ids"], fixture_active_ids):
            raise ValueError(f"{identifier} active_ids differ from the fixture")
    if not np.array_equal(initialization["rest_points"], fixture_rest):
        raise ValueError(
            "fixed-axis initialization rest_points differ from the fixture"
        )
    for identifier, state in (
        ("initialization", initialization),
        ("best", best),
        ("last", last),
    ):
        if not np.array_equal(state["active_ids"], fixture_active_ids):
            raise ValueError(f"{identifier} active_ids differ from the fixture")

    trace = _read_trace(cfg.trace)
    baseline = _baseline_levels(cfg.forward_summary)
    if int(best["step"]) not in trace["step"]:
        raise ValueError("best checkpoint step is absent from trace.csv")
    if int(last["step"]) != int(trace["step"][-1]):
        raise ValueError("last checkpoint step differs from the final trace step")

    target_displacement = np.asarray(volume.point_data["Smile"], dtype=np.float64)
    if not np.isfinite(target_displacement[skin_ids]).all():
        raise FloatingPointError("target Smile is non-finite on the skin")
    points_by_id = {
        "target": fixture_rest + target_displacement,
        "baseline-replay": fixture_rest + replay["u"],
        "dominant-only": fixture_rest + dominant["u"],
        "fixed-axis-refit": fixture_rest + best["u"],
    }
    cameras_doc = json.loads(CAMERAS.read_text())
    cameras = {view["id"]: view for view in cameras_doc["views"]}
    if set(VIEWS) - set(cameras):
        raise ValueError("frozen camera receipt lacks a requested view")

    assets: list[Path] = []
    meshes: dict[str, pv.PolyData] = {}
    state_records: dict[str, Any] = {}
    for identifier, caption, _color in STATE_SPECS:
        mesh = skin.copy(deep=True)
        mesh.points = points_by_id[identifier][skin_ids]
        mesh.field_data["StateId"] = np.asarray([identifier])
        mesh_path = output / "skins" / f"{identifier}.vtp"
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.save(mesh_path, binary=True)
        assets.append(mesh_path)
        meshes[identifier] = mesh
        state_records[identifier] = {
            "caption": caption.replace("\n", " | "),
            "skin": _record(mesh_path),
            "points_sha256": _array_sha256(np.asarray(mesh.points)),
            "views": {},
        }

    view_records: dict[str, Any] = {}
    for view_id in VIEWS:
        camera = cameras[view_id]["camera"]
        images: list[Path] = []
        for identifier, caption, color in STATE_SPECS:
            image_path = output / "geometry" / identifier / f"{view_id}.png"
            _render_shape(meshes[identifier], camera, caption, color, image_path)
            assets.append(image_path)
            images.append(image_path)
            state_records[identifier]["views"][view_id] = _record(image_path)
        plate_path = output / "composites" / f"{view_id}.png"
        _render_plate(images, cameras[view_id]["label"], plate_path)
        assets.append(plate_path)
        view_records[view_id] = {
            "camera": camera,
            "composite": _record(plate_path),
        }

    history_path = output / "history" / "loss-and-fit.png"
    _render_history(trace, baseline, history_path)
    assets.append(history_path)

    source_path = output / "sources/50-render-fixed-directions.py"
    source_receipt = _snapshot_source(Path(__file__), source_path)
    assets.append(source_path)
    summary = {
        "status": "completed_fixed_direction_refit_render",
        "scope": (
            "Pure rendering of saved target and equilibrium displacements plus a saved "
            "Adam trace; no fitting, equilibrium solve, smoothing, decimation, or "
            "deformation scaling"
        ),
        "render": {
            "views": list(VIEWS),
            "window_size": list(WINDOW_SIZE),
            "background": BACKGROUND,
            "shading": "flat",
            "deformation_scale": 1.0,
            "lighting": "exact copied _add_lights construction from the frozen reference renderer",
            "camera_policy": "unchanged frozen parallel cameras shared by every state",
            "comparison_order": [item[0] for item in STATE_SPECS],
        },
        "fixed_axis_refit": {
            "parameterization": (
                "one frozen unsigned unit reference axis and one optimized nonnegative "
                "scalar strength per active tetrahedron"
            ),
            "rendered_checkpoint": "best.npz",
            "best_step": int(best["step"]),
            "last_step": int(last["step"]),
            "active_tetrahedron_count": len(fixture_active_ids),
        },
        "history": {
            "trace_columns": list(TRACE_COLUMNS),
            "objective_definition": "uniform surface component MSE multiplied by 1e6 (mm^2)",
            "uniform_fit_definition": "uniform surface vector RMS in mm",
            "area_weighted_fit_definition": "surface-area-weighted vector RMS in mm",
            "full_baseline_levels": baseline,
            "steps": len(trace["step"]),
            "first_step": int(trace["step"][0]),
            "last_step": int(trace["step"][-1]),
            "figure": _record(history_path),
        },
        "states": state_records,
        "views": view_records,
        "inputs": {str(path.resolve()): _record(path) for path in required},
        "state_arrays": {
            "baseline-replay": _array_records(replay),
            "dominant-only": _array_records(dominant),
            "initialization": _array_records(initialization),
            "best": _array_records(best),
            "last": _array_records(last),
            "trace": _array_records(trace),
        },
        "reference_style_source": _record(REFERENCE_RENDERER),
        "executed_source": source_receipt,
        "outputs": {str(path.relative_to(output)): _record(path) for path in assets},
    }
    _write_json(output / "summary.json", summary)
    return summary


def main(cfg: Config) -> None:
    summary = _main(cfg)
    cherries.log_metrics(
        {
            "render/state_images": 4 * len(VIEWS),
            "render/composites": len(VIEWS),
            "render/history_images": 1,
            "refit/best_step": summary["fixed_axis_refit"]["best_step"],
            "refit/last_step": summary["fixed_axis_refit"]["last_step"],
        }
    )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
