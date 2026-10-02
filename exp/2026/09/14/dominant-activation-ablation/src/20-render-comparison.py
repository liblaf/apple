"""Render the frozen target, replayed fit, and dominant-mode ablation."""

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003

from __future__ import annotations

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
GROUP = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
CAMERAS = ROOT / "exp/2026/09/08/physical-volume-closeups/data/20-regions/summary.json"
REFERENCE_RENDERER = (
    ROOT / "exp/2026/09/09/activation-space-smoothness/src/81-render-idea-shapes.py"
)
ORIGINAL_BASELINE = (
    Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
    / "exp/2026/09/08/physical-volume-baseline/data/20-baseline/step-0200.npz"
)
VIEWS = ("side-context", "region1-mouth-corner")
WINDOW_SIZE = (1800, 1800)
BACKGROUND = "#f4f2ed"
TARGET_COLOR = "#b8afa0"
SHAPE_COLOR = "#aeb7ba"
DIFFERENCE_CMAP = "magma"
EXPECTED_FROZEN_SHA256 = {
    REFERENCE_RENDERER: "c3b0543661c996a0f9409017d9c5de52ab833a753c072ca1e0ea18c807fd3138",
    CAMERAS: "f2628b1ff8fd82386fc02498739bb651f7b1e217479bc45350ec6b893ee7b58f",
    FIXTURE
    / "volume.vtu": "238962d0d27a2d35b6211a7a60204d362187b4190dfc7591756bc73ea26ff3b6",
    FIXTURE
    / "skin.vtp": "79eed2a5e2b5f23e84287fe729989ab356cb2780e3342b722b074a9231297833",
    ORIGINAL_BASELINE: "21b7e546f04c5566629a1d3634659ec0c5738df215699ba23e24eb7ac856abd7",
}
STATE_SPECS = (
    ("target", "Target smile\nFrozen fixture field", TARGET_COLOR),
    ("baseline-replay", "Full fitted activation\nReplayed equilibrium", SHAPE_COLOR),
    (
        "dominant-only",
        "Strongest contraction mode only\nRe-solved equilibrium",
        SHAPE_COLOR,
    ),
)


class Config(cherries.BaseConfig):
    """Frozen renderer inputs and Cherries-managed destination."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    baseline_replay: Path = cherries.input("10-forward/baseline-replay.npz")
    dominant_only: Path = cherries.input("10-forward/dominant-only.npz")
    metrics_summary: Path = cherries.input("10-forward/summary.json")
    output_dir: Path = cherries.output("20-comparison", mkdir=True)


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
# before rendering so changes to that read-only source cannot silently alter the
# stated style provenance.
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


def _render_difference(
    mesh: pv.PolyData,
    camera: dict[str, Any],
    limit_mm: float,
    path: Path,
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE, lighting="none")
    actor = plotter.add_mesh(
        mesh,
        scalars="ReplayMinusDominantMm",
        cmap=DIFFERENCE_CMAP,
        clim=(0.0, limit_mm),
        smooth_shading=False,
        ambient=0.20,
        diffuse=0.80,
        specular=0.0,
        scalar_bar_args={
            "title": "|full - strongest mode| (mm)",
            "vertical": True,
            "color": "black",
            "position_x": 0.80,
            "position_y": 0.10,
            "width": 0.08,
            "height": 0.45,
            "title_font_size": 17,
            "label_font_size": 14,
            "fmt": "%.3g",
            "background_color": BACKGROUND,
            "fill": True,
        },
    )
    if actor.GetProperty().GetInterpolation() != 0:
        raise AssertionError("difference rendering must use flat shading")
    _add_lights(plotter, camera)
    _set_camera(plotter, camera)
    annotation = plotter.add_text(
        "Surface displacement difference\nShown on the full replay geometry",
        position="upper_left",
        color="black",
        font_size=13,
    )
    annotation.GetTextProperty().SetBackgroundColor(244 / 255, 242 / 255, 237 / 255)
    annotation.GetTextProperty().SetBackgroundOpacity(0.90)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(path)
    plotter.close()


def _render_plate(images: list[Path], view_label: str, path: Path) -> None:
    if len(images) != 3:
        raise ValueError("comparison plate requires exactly three state panels")
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 5.6), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    for axis, image in zip(axes, images, strict=True):
        axis.imshow(plt.imread(image))
        axis.set_axis_off()
    figure.suptitle(view_label, color="black", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _load_state(path: Path) -> dict[str, np.ndarray]:
    required = {"u", "rest_points", "active_ids", "B", "Z"}
    with np.load(path, allow_pickle=False) as saved:
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"{path} lacks required arrays: {sorted(missing)}")
        state = {name: np.asarray(saved[name]).copy() for name in required}
    if state["u"].shape != state["rest_points"].shape:
        raise ValueError(f"u/rest shape mismatch in {path}")
    if state["rest_points"].ndim != 2 or state["rest_points"].shape[1] != 3:
        raise ValueError(f"rest_points has invalid shape in {path}")
    active_count = len(state["active_ids"])
    for name in ("B", "Z"):
        if state[name].shape != (active_count, 3, 3):
            raise ValueError(f"{name} has invalid shape in {path}: {state[name].shape}")
    if not all(
        np.isfinite(state[name]).all() for name in ("u", "rest_points", "B", "Z")
    ):
        raise FloatingPointError(f"non-finite saved state array in {path}")
    if not np.allclose(state["Z"], np.swapaxes(state["Z"], 1, 2), rtol=0.0, atol=1e-10):
        raise ValueError(f"Z is not symmetric in {path}")
    effective = state["B"] @ np.swapaxes(state["B"], 1, 2) - np.eye(3)
    if not np.allclose(effective, state["Z"], rtol=1e-10, atol=1e-10):
        raise ValueError(f"Z != B @ B.T - I in {path}")
    return state


def _validate_dominant_mode(
    replay_z: np.ndarray, dominant_z: np.ndarray
) -> dict[str, Any]:
    values, vectors = np.linalg.eigh(replay_z)
    strongest = np.maximum(values[:, -1], 0.0)
    direction = vectors[:, :, -1]
    expected = strongest[:, None, None] * (
        direction[:, :, None] * direction[:, None, :]
    )
    error = np.linalg.norm(dominant_z - expected, axis=(1, 2))
    scale = np.maximum(np.linalg.norm(expected, axis=(1, 2)), 1.0)
    if np.max(error / scale) > 1e-9:
        raise ValueError(
            "dominant-only Z is not the positive strongest eigenmode of replay Z"
        )
    return {
        "definition": "positive strongest eigenmode of replay Z, cell by cell",
        "positive_cell_count": int(np.count_nonzero(strongest > 0.0)),
        "max_reconstruction_error": float(error.max()),
        "strongest_eigenvalue_min": float(strongest.min()),
        "strongest_eigenvalue_max": float(strongest.max()),
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
        cfg.metrics_summary,
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
    json.loads(cfg.metrics_summary.read_text())

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

    replay = _load_state(cfg.baseline_replay)
    dominant = _load_state(cfg.dominant_only)
    with np.load(ORIGINAL_BASELINE, allow_pickle=False) as saved:
        original_rest = np.asarray(saved["rest_points"], dtype=np.float64)
        original_active_ids = np.asarray(saved["active_ids"], dtype=np.int64)
        original_b = np.asarray(saved["Ainv"], dtype=np.float64)
    for identifier, state in (("baseline-replay", replay), ("dominant-only", dominant)):
        if not np.array_equal(state["rest_points"], fixture_rest):
            raise ValueError(f"{identifier} rest_points differ from the fixture")
        if not np.array_equal(state["active_ids"], fixture_active_ids):
            raise ValueError(f"{identifier} active_ids differ from the fixture")
    if not np.array_equal(replay["rest_points"], original_rest):
        raise ValueError("replay rest_points differ from the original fitted baseline")
    if not np.array_equal(replay["active_ids"], original_active_ids):
        raise ValueError("replay active_ids differ from the original fitted baseline")
    if not np.array_equal(replay["B"], original_b):
        raise ValueError("replay B differs from the original fitted baseline Ainv")
    if dominant["B"].shape != replay["B"].shape:
        raise ValueError("dominant-only B shape differs from replay B")
    dominant_receipt = _validate_dominant_mode(replay["Z"], dominant["Z"])

    target_displacement = np.asarray(volume.point_data["Smile"], dtype=np.float64)
    if not np.isfinite(target_displacement[skin_ids]).all():
        raise FloatingPointError("target Smile is non-finite on the skin")
    points_by_id = {
        "target": fixture_rest + target_displacement,
        "baseline-replay": fixture_rest + replay["u"],
        "dominant-only": fixture_rest + dominant["u"],
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

    difference_mm = 1000.0 * np.linalg.norm(
        replay["u"][skin_ids] - dominant["u"][skin_ids], axis=1
    )
    difference_max_mm = float(difference_mm.max())
    if not np.isfinite(difference_mm).all() or difference_max_mm <= 0.0:
        raise ValueError(
            "replay/dominant surface difference must be finite and nonzero"
        )
    difference_mesh = meshes["baseline-replay"].copy(deep=True)
    difference_mesh.point_data["ReplayMinusDominantMm"] = difference_mm
    difference_mesh.field_data["GeometryState"] = np.asarray(["baseline-replay"])
    difference_path = output / "skins/replay-minus-dominant.vtp"
    difference_mesh.save(difference_path, binary=True)
    assets.append(difference_path)
    difference_views: dict[str, Any] = {}
    for view_id in VIEWS:
        image_path = output / "difference" / f"{view_id}.png"
        _render_difference(
            difference_mesh,
            cameras[view_id]["camera"],
            difference_max_mm,
            image_path,
        )
        assets.append(image_path)
        difference_views[view_id] = _record(image_path)

    source_receipt = _snapshot_source(
        Path(__file__), output / "sources/20-render-comparison.py"
    )
    assets.append(output / "sources/20-render-comparison.py")
    summary = {
        "status": "completed_dominant_activation_shape_comparison",
        "scope": (
            "Pure rendering of saved equilibrium displacements and the frozen target; "
            "no fitting, equilibrium solve, smoothing, decimation, or deformation scaling"
        ),
        "render": {
            "views": list(VIEWS),
            "window_size": list(WINDOW_SIZE),
            "background": BACKGROUND,
            "target_color": TARGET_COLOR,
            "shape_color": SHAPE_COLOR,
            "difference_cmap": DIFFERENCE_CMAP,
            "difference_range_mm": [0.0, difference_max_mm],
            "shading": "flat",
            "deformation_scale": 1.0,
            "lighting": "exact copied _add_lights construction from the frozen reference renderer",
            "camera_policy": "unchanged frozen parallel cameras shared by every state",
        },
        "states": state_records,
        "views": view_records,
        "difference": {
            "definition": "Euclidean norm of replay u minus dominant-only u at each skin vertex",
            "units": "mm",
            "geometry": "baseline-replay deformed skin",
            "minimum_mm": float(difference_mm.min()),
            "mean_mm": float(difference_mm.mean()),
            "rms_mm": float(np.sqrt(np.mean(np.square(difference_mm)))),
            "maximum_mm": difference_max_mm,
            "values_sha256": _array_sha256(difference_mm),
            "skin": _record(difference_path),
            "views": difference_views,
        },
        "dominant_mode_validation": dominant_receipt,
        "inputs": {str(path.resolve()): _record(path) for path in required},
        "state_arrays": {
            identifier: {
                name: {
                    "shape": list(value.shape),
                    "dtype": value.dtype.str,
                    "sha256": _array_sha256(value),
                }
                for name, value in state.items()
            }
            for identifier, state in (
                ("baseline-replay", replay),
                ("dominant-only", dominant),
            )
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
            "render/state_images": 3 * len(VIEWS),
            "render/composites": len(VIEWS),
            "render/difference_images": len(VIEWS),
            "difference/surface_rms_mm": summary["difference"]["rms_mm"],
            "difference/surface_max_mm": summary["difference"]["maximum_mm"],
        }
    )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
