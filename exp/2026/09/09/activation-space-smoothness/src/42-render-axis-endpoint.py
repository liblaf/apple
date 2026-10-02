"""Render bounded, derived plates for the saved learned-axis endpoint."""

# ruff: noqa: C901, EM101, EM102, PLR0915, TRY003

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[6]
GROUP = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
CAMERAS = ROOT / "exp/2026/09/08/physical-volume-closeups/data/20-regions/summary.json"
COMPARISON = GROUP / "data/40-comparison/summary.json"
AXIS_ON = GROUP / "data/25-learned-axis-smooth"
AXIS_OFF = GROUP / "data/24-learned-axis"
RAW6_ON = GROUP / "data/28-raw6-smooth"
RAW6_OFF = ROOT / "exp/2026/09/08/local-skin-prestrain/data/30-refit-no-skin"
VIEWS = ("side-context", "region1-mouth-corner", "nasolabial-region")
AXIS_MATCH_STEP = 11
RAW6_MATCH_STEP = 200
BACKGROUND = "#242c36"
COLORS = {
    "target": "#202020",
    "axis-on": "#8aa53c",
    "axis-off": "#43845b",
    "raw6-on": "#256f9c",
    "raw6-off": "#687787",
}


class Config(cherries.BaseConfig):
    """Inputs and destination for the bounded derived render."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    axis_on_dir: Path = AXIS_ON
    output_dir: Path = GROUP / "data/42-axis-on-endpoint"
    step: int = 128


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _digest(path)}


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _snapshot_source(source: Path, destination: Path) -> dict[str, Any]:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o444)
    live = _record(source)
    snapshot = _record(destination)
    if live["sha256"] != snapshot["sha256"]:
        raise ValueError("renderer source snapshot differs from live source")
    return {"snapshot": snapshot, "live_at_generation": live}


def _fail_if_nonempty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {path}")


def _load_surface(
    directory: Path, step: int, point_ids: np.ndarray
) -> tuple[np.ndarray, Path]:
    path = directory / f"surface-{step:04d}.npz"
    with np.load(path, allow_pickle=False) as saved:
        if int(saved["step"]) != step:
            raise ValueError(f"surface step receipt mismatch: {path}")
        if not np.array_equal(saved["point_ids"], point_ids):
            raise ValueError(f"surface point IDs differ from frozen fixture: {path}")
        displacement = np.asarray(saved["u"], dtype=np.float64)
    if displacement.shape != (len(point_ids), 3) or not np.isfinite(displacement).all():
        raise ValueError(f"invalid surface displacement: {path}")
    return displacement, path


def _add_lights(plotter: pv.Plotter, camera: dict[str, Any]) -> None:
    focus = np.asarray(camera["focal_point"], dtype=np.float64)
    backward = np.asarray(camera["position"], dtype=np.float64) - focus
    backward /= np.linalg.norm(backward)
    right = np.cross(np.asarray(camera["view_up"], dtype=np.float64), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    for position, intensity in (
        (focus + 0.3 * (0.72 * right + 0.35 * up + 0.60 * backward), 0.85),
        (focus + 0.3 * backward, 0.20),
    ):
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


def _render_surface(
    base: pv.PolyData, displacement: np.ndarray, camera: dict[str, Any], path: Path
) -> None:
    mesh = base.copy(deep=True)
    mesh.points = np.asarray(base.points, dtype=np.float64) + displacement
    plotter = pv.Plotter(off_screen=True, window_size=(900, 900), lighting="none")
    actor = plotter.add_mesh(
        mesh,
        color="#eeeeea",
        smooth_shading=False,
        ambient=0.20,
        diffuse=0.80,
        specular=0.0,
    )
    if actor.GetProperty().GetInterpolation() != 0:
        raise AssertionError("derived geometry must use flat shading")
    _add_lights(plotter, camera)
    plotter.enable_parallel_projection()
    plotter.camera.position = camera["position"]
    plotter.camera.focal_point = camera["focal_point"]
    plotter.camera.up = camera["view_up"]
    plotter.camera.parallel_scale = camera["parallel_scale"]
    plotter.set_background(BACKGROUND)
    plotter.reset_camera_clipping_range()
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(path)
    plotter.close()


def _render_plate(
    images: list[Path], titles: list[str], title: str, path: Path
) -> None:
    if len(images) != 2 or len(titles) != 2:
        raise ValueError("endpoint plate requires target and Axis-on panels")
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.8), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    for axis, image, panel_title in zip(axes, images, titles, strict=True):
        axis.imshow(plt.imread(image))
        axis.set_title(panel_title, color="white", fontsize=10)
        axis.set_axis_off()
    figure.suptitle(title, color="white", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _triangle_segments(
    points: np.ndarray, triangles: np.ndarray, y: float
) -> np.ndarray:
    segments: list[np.ndarray] = []
    for triangle in points[triangles]:
        signed = triangle[:, 1] - y
        hits: list[np.ndarray] = []
        for left, right in ((0, 1), (1, 2), (2, 0)):
            a, b = triangle[left], triangle[right]
            sa, sb = signed[left], signed[right]
            if sa == 0.0 and sb == 0.0:
                continue
            if sa == 0.0:
                hits.append(a)
            elif sb == 0.0:
                hits.append(b)
            elif (sa < 0.0) != (sb < 0.0):
                hits.append(a + (-sa / (sb - sa)) * (b - a))
        unique = [
            hit
            for index, hit in enumerate(hits)
            if not any(
                np.allclose(hit, prior, rtol=0.0, atol=1e-13) for prior in hits[:index]
            )
        ]
        if len(unique) == 2:
            segment = np.stack(unique)
            if (
                segment[:, 0].max() >= 1.414
                and segment[:, 0].min() <= 1.460
                and segment[:, 2].max() >= 0.040
            ):
                segments.append(segment)
    return np.asarray(segments, dtype=np.float64)


def _render_sections(
    base: pv.PolyData, states: list[tuple[str, np.ndarray, str]], path: Path
) -> None:
    triangles = np.asarray(base.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    for axis, y in zip(axes, (2.170, 2.180, 2.190), strict=True):
        for label, displacement, color in states:
            segments = _triangle_segments(
                np.asarray(base.points, dtype=np.float64) + displacement, triangles, y
            )
            for index, segment in enumerate(segments):
                axis.plot(
                    1000.0 * segment[:, 0],
                    1000.0 * segment[:, 2],
                    color=color,
                    linewidth=0.8,
                    label=label if index == 0 else None,
                )
        axis.set(
            title=f"y = {1000.0 * y:.0f} mm",
            xlim=(1414, 1460),
            ylim=(40, 115),
            xlabel="x (mm)",
            ylabel="z (mm)",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper left", frameon=False, fontsize=7)
    figure.suptitle("Exact skin-triangle sections", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _main(cfg: Config) -> dict[str, Any]:
    _fail_if_nonempty(cfg.output_dir)
    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)
    source_receipt = _snapshot_source(
        Path(__file__), output / "sources/42-render-axis-endpoint.py"
    )
    if cfg.step != 128:
        raise ValueError("this bounded renderer is fixed to Axis-on update 128")
    required_paths = (
        FIXTURE / "volume.vtu",
        FIXTURE / "skin.vtp",
        CAMERAS,
        COMPARISON,
        cfg.axis_on_dir / "summary.json",
        cfg.axis_on_dir / "surface-0128.npz",
        AXIS_OFF / f"surface-{AXIS_MATCH_STEP:04d}.npz",
        cfg.axis_on_dir / f"surface-{AXIS_MATCH_STEP:04d}.npz",
        RAW6_OFF / f"surface-{RAW6_MATCH_STEP:04d}.npz",
        RAW6_ON / f"surface-{RAW6_MATCH_STEP:04d}.npz",
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    skin = pv.read(FIXTURE / "skin.vtp")
    volume = pv.read(FIXTURE / "volume.vtu")
    point_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    target = np.asarray(volume.point_data["Smile"], dtype=np.float64)[point_ids]
    if target.shape != (len(point_ids), 3) or not np.isfinite(target).all():
        raise ValueError("frozen target field is invalid")
    axis_on, axis_source = _load_surface(cfg.axis_on_dir, cfg.step, point_ids)
    cameras = json.loads(CAMERAS.read_text())
    views = {view["id"]: view for view in cameras["views"]}
    if set(VIEWS) - set(views):
        raise ValueError("frozen camera receipt lacks a requested view")
    axis_summary = json.loads((cfg.axis_on_dir / "summary.json").read_text())
    if not str(axis_summary["status"]).startswith("completed_"):
        raise ValueError("Axis-on source run did not complete its fixed budget")
    if int(axis_summary["last_evaluated_step"]) != cfg.step:
        raise ValueError("Axis-on endpoint does not match the requested update")
    if int(axis_summary["best_step"]) != cfg.step:
        raise ValueError("Axis-on update 128 is not the recorded best-fit state")
    metrics = axis_summary["best_metrics"]
    fit = float(metrics["fit_rms_mm"])
    motion = float(metrics["motion_rms_mm"])
    assets: dict[str, Any] = {"endpoint_plates": [], "sections": []}
    for view_id in VIEWS:
        view = views[view_id]
        target_png = output / "geometry" / view_id / "target.png"
        axis_png = output / "geometry" / view_id / "axis-on-0128.png"
        _render_surface(skin, target, view["camera"], target_png)
        _render_surface(skin, axis_on, view["camera"], axis_png)
        plate = output / "geometry" / view_id / "target-axis-on-0128.png"
        _render_plate(
            [target_png, axis_png],
            [
                "Target smile",
                f"Axis-on update 128\nunpaired; fit {fit:.4f} mm; motion {motion:.4f} mm",
            ],
            f"{view['label']} · endpoint",
            plate,
        )
        assets["endpoint_plates"].append(
            {
                "view": view_id,
                "file": _record(plate),
                "panels": [_record(target_png), _record(axis_png)],
            }
        )

    axis_off_match, axis_off_match_source = _load_surface(
        AXIS_OFF, AXIS_MATCH_STEP, point_ids
    )
    axis_on_match, axis_on_match_source = _load_surface(
        cfg.axis_on_dir, AXIS_MATCH_STEP, point_ids
    )
    matched = {
        "learned-axis-matched": [
            ("Target", target, "target"),
            (f"Axis-off matched update {AXIS_MATCH_STEP}", axis_off_match, "axis-off"),
            (f"Axis-on matched update {AXIS_MATCH_STEP}", axis_on_match, "axis-on"),
        ]
    }
    section_sources = {
        "learned_axis_off_matched": _record(axis_off_match_source),
        "learned_axis_on_matched": _record(axis_on_match_source),
    }
    raw6_off_match, raw6_off_match_source = _load_surface(
        RAW6_OFF, RAW6_MATCH_STEP, point_ids
    )
    raw6_on_match, raw6_on_match_source = _load_surface(
        RAW6_ON, RAW6_MATCH_STEP, point_ids
    )
    matched["raw6-matched"] = [
        ("Target", target, "target"),
        (f"Raw6-off matched update {RAW6_MATCH_STEP}", raw6_off_match, "raw6-off"),
        (f"Raw6-on matched update {RAW6_MATCH_STEP}", raw6_on_match, "raw6-on"),
    ]
    section_sources.update(
        raw6_off_matched=_record(raw6_off_match_source),
        raw6_on_matched=_record(raw6_on_match_source),
    )
    endpoint_sections = output / "sections" / "axis-on-endpoint-0128.png"
    _render_sections(
        skin,
        [
            ("Target", target, COLORS["target"]),
            ("Axis-on update 128 (unpaired)", axis_on, COLORS["axis-on"]),
        ],
        endpoint_sections,
    )
    assets["sections"].append(
        {
            "name": "axis-on-endpoint-0128",
            "file": _record(endpoint_sections),
            "pairing": "unpaired",
        }
    )
    for name, state_rows in matched.items():
        section = output / "sections" / f"{name}.png"
        _render_sections(
            skin,
            [
                (label, displacement, COLORS[color])
                for label, displacement, color in state_rows
            ],
            section,
        )
        assets["sections"].append(
            {
                "name": name,
                "file": _record(section),
                "pairing": "matched",
                "updates": [
                    AXIS_MATCH_STEP if name.startswith("learned") else RAW6_MATCH_STEP
                ],
            }
        )
    inputs = {
        "fixture_volume": _record(FIXTURE / "volume.vtu"),
        "fixture_skin": _record(FIXTURE / "skin.vtp"),
        "frozen_cameras": _record(CAMERAS),
        "axis_on_surface_0128": _record(axis_source),
        "axis_on_summary": _record(cfg.axis_on_dir / "summary.json"),
        "comparison_summary": _record(COMPARISON),
        "section_surfaces": section_sources,
        "source": source_receipt,
    }
    summary = {
        "status": "completed_bounded_derived_render",
        "scope": "Target versus exact saved Axis-on surface at update 128; no solve, interpolation, deformation exaggeration, or geometry smoothing.",
        "axis_on": {
            "step": 128,
            "pairing": "unpaired",
            "fit_rms_mm": fit,
            "motion_rms_mm": motion,
        },
        "views": list(VIEWS),
        "inputs": inputs,
        "outputs": assets,
    }
    _write_json(output / "summary.json", summary)
    provenance = {
        "renderer": source_receipt,
        "inputs": inputs,
        "outputs": assets,
        "dependency": {
            "40_compare": "not imported; comparison summary is receipt-only"
        },
    }
    _write_json(output / "provenance.json", provenance)
    return summary


def main(cfg: Config) -> None:
    summary = _main(cfg)
    cherries.log_metric(
        "render/endpoint_plates", len(summary["outputs"]["endpoint_plates"])
    )
    cherries.log_metric("render/section_overlays", len(summary["outputs"]["sections"]))
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
