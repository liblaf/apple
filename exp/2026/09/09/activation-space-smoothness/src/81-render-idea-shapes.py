"""Render separate common-camera skin shapes for the selected idea comparison."""

# ruff: noqa: C901, CPY001, EM101, EM102, PLR0912, PLR0915, TRY003

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]
GROUP = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
CAMERAS = ROOT / "exp/2026/09/08/physical-volume-closeups/data/20-regions/summary.json"
SELECTION = GROUP / "data/80-idea-result-selection/selection.json"
OUTPUT = GROUP / "data/81-idea-shapes"
VIEWS = ("side-context", "region1-mouth-corner")
BACKGROUND = "#f4f2ed"
SHAPE_COLOR = "#aeb7ba"
TARGET_COLOR = "#b8afa0"
STATE_METADATA = {
    "raw6-refit-off-200": {
        "caption": "Corrected Raw6 · smoothness off · update 200",
        "model": "corrected Raw6",
        "smoothing": "off",
    },
    "raw6-refit-on-200": {
        "caption": "Corrected Raw6 · smoothness on · update 200",
        "model": "corrected Raw6",
        "smoothing": "on",
    },
    "axis-on-128": {
        "caption": "Learned axis · smoothness on · update 128",
        "model": "learned axis",
        "smoothing": "on",
    },
    "axis-off-best-15": {
        "caption": "Learned axis · smoothness off · best-fit update 15",
        "model": "learned axis",
        "smoothing": "off",
    },
    "raw6-corrected-rest-start-200": {
        "caption": "Corrected Raw6 rest-start baseline · smoothness off · update 200",
        "model": "corrected Raw6",
        "smoothing": "off",
    },
    "psd-off-1024": {
        "caption": "PSD active stress · smoothness off · update 1024",
        "model": "PSD active stress",
        "smoothing": "off",
    },
}


class Config(cherries.BaseConfig):
    """Destination and optional one-state preview selection."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = OUTPUT
    preview_id: str | None = None


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


def _render(
    mesh: pv.PolyData, camera: dict[str, Any], caption: str, color: str, path: Path
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1800), lighting="none")
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


def _load_state(
    selected: dict[str, Any],
    volume: pv.UnstructuredGrid,
    active_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(selected["path"])
    receipt = _record(path)
    if receipt["sha256"] != selected["sha256"] or receipt["bytes"] != selected["bytes"]:
        raise ValueError(f"selected checkpoint differs from frozen selection: {path}")
    with np.load(path, allow_pickle=False) as saved:
        if int(saved["step"]) != int(selected["step"]):
            raise ValueError(f"checkpoint step differs from selection: {path}")
        if not bool(saved["solver_valid"]):
            raise ValueError(f"selected checkpoint is solver-invalid: {path}")
        if not np.array_equal(saved["active_ids"], active_ids):
            raise ValueError(f"checkpoint active IDs differ from fixture: {path}")
        if "rest_points" in saved.files:
            rest = np.asarray(saved["rest_points"], dtype=np.float64)
            if not np.array_equal(rest, np.asarray(volume.points, dtype=np.float64)):
                raise ValueError(f"checkpoint rest points differ from fixture: {path}")
        else:
            rest = np.asarray(volume.points, dtype=np.float64)
        displacement = np.asarray(saved["u"], dtype=np.float64)
    if displacement.shape != rest.shape or not np.isfinite(displacement).all():
        raise ValueError(f"checkpoint displacement is invalid: {path}")
    return rest + displacement, receipt


def _main(cfg: Config) -> dict[str, Any]:
    output = cfg.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_receipt = _snapshot_source(
        Path(__file__), output / "sources/81-render-idea-shapes.py"
    )
    required = [SELECTION, FIXTURE / "volume.vtu", FIXTURE / "skin.vtp", CAMERAS]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    selection = json.loads(SELECTION.read_text())
    if selection["scope"] != "Curated existing results; no new fitting or rendering":
        raise ValueError("unexpected frozen selection scope")
    selected_by_id = {
        state["id"]: state for state in selection["states"] if state["kind"] == "full"
    }
    if set(selected_by_id) != set(STATE_METADATA):
        raise ValueError("full selected state IDs differ from the renderer contract")
    state_ids = [
        state["id"] for state in selection["states"] if state["id"] in STATE_METADATA
    ]
    if cfg.preview_id is not None:
        if cfg.preview_id not in selected_by_id:
            raise ValueError(f"unknown preview state: {cfg.preview_id}")
        state_ids = [cfg.preview_id]

    volume = pv.read(FIXTURE / "volume.vtu")
    skin = pv.read(FIXTURE / "skin.vtp")
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    if len(np.unique(skin_ids)) != len(skin_ids):
        raise ValueError("fixture skin GlobalPointId values are not unique")
    if not np.array_equal(
        np.asarray(skin.points, dtype=np.float64),
        np.asarray(volume.points, dtype=np.float64)[skin_ids],
    ):
        raise ValueError(
            "fixture skin points do not match volume GlobalPointId indexing"
        )
    active_ids = np.flatnonzero(
        np.asarray(volume.cell_data["MuscleFraction"], dtype=np.float64) > 0.0
    )
    cameras_doc = json.loads(CAMERAS.read_text())
    cameras = {view["id"]: view for view in cameras_doc["views"]}
    if set(VIEWS) - set(cameras):
        raise ValueError("frozen camera receipt lacks a requested view")

    assets: list[Path] = []
    state_records: list[dict[str, Any]] = []
    for state_id in state_ids:
        full_points, checkpoint_receipt = _load_state(
            selected_by_id[state_id], volume, active_ids
        )
        deformed = skin.copy(deep=True)
        deformed.points = full_points[skin_ids]
        deformed.field_data["StateId"] = np.asarray([state_id])
        skin_path = output / "skins" / f"{state_id}.vtp"
        skin_path.parent.mkdir(parents=True, exist_ok=True)
        deformed.save(skin_path, binary=True)
        assets.append(skin_path)
        views: dict[str, Any] = {}
        for view_id in VIEWS:
            image_path = output / "geometry" / state_id / f"{view_id}.png"
            _render(
                deformed,
                cameras[view_id]["camera"],
                STATE_METADATA[state_id]["caption"],
                SHAPE_COLOR,
                image_path,
            )
            assets.append(image_path)
            views[view_id] = _record(image_path)
        state_records.append(
            {
                "id": state_id,
                **STATE_METADATA[state_id],
                "step": int(selected_by_id[state_id]["step"]),
                "checkpoint": checkpoint_receipt,
                "skin": _record(skin_path),
                "views": views,
            }
        )

    target = skin.copy(deep=True)
    target.points = (
        np.asarray(skin.points, dtype=np.float64)
        + np.asarray(volume.point_data["Smile"], dtype=np.float64)[skin_ids]
    )
    target.field_data["StateId"] = np.asarray(["target"])
    target_skin = output / "skins/target.vtp"
    target_skin.parent.mkdir(parents=True, exist_ok=True)
    target.save(target_skin, binary=True)
    assets.append(target_skin)
    target_views: dict[str, Any] = {}
    for view_id in VIEWS:
        image_path = output / "geometry/target" / f"{view_id}.png"
        _render(
            target,
            cameras[view_id]["camera"],
            "Target smile · frozen fixture field",
            TARGET_COLOR,
            image_path,
        )
        assets.append(image_path)
        target_views[view_id] = _record(image_path)

    summary = {
        "status": "completed_idea_shape_preview"
        if cfg.preview_id
        else "completed_idea_shape_render",
        "scope": "Pure rendering of selected saved deformed coordinates and the frozen target; no fitting, smoothing, decimation, geometry processing, or metric recomputation",
        "render": {
            "views": list(VIEWS),
            "window_size": [1800, 1800],
            "background": BACKGROUND,
            "shape_color": SHAPE_COLOR,
            "target_color": TARGET_COLOR,
            "shading": "flat",
            "deformation_scale": 1.0,
            "lighting": "exact _add_lights construction from src/40-compare.py",
        },
        "states": state_records,
        "target": {"skin": _record(target_skin), "views": target_views},
        "inputs": {str(path.resolve()): _record(path) for path in required},
        "outputs": {str(path.relative_to(output)): _record(path) for path in assets},
        "source": source_receipt,
    }
    _write_json(output / "summary.json", summary)
    return summary


def main(cfg: Config) -> None:
    summary = _main(cfg)
    cherries.log_metric("render/states", len(summary["states"]))
    cherries.log_metric(
        "render/images",
        sum(len(state["views"]) for state in summary["states"])
        + len(summary["target"]["views"]),
    )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
