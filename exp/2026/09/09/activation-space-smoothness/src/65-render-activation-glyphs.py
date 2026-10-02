"""Render sparse rest-frame learned-axis activation glyphs from saved states."""

# ruff: noqa: C901, CPY001, EM101, EM102, PLR0912, PLR0915, RUF001, TRY003

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import zipfile
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
OUTPUT = GROUP / "data/65-activation-glyphs"
VIEWS = ("side-context", "region1-mouth-corner")
SAMPLE_SPACING_M = 0.006
DISPLAY_LENGTH_M = 0.012
DISPLAY_MIN_LENGTH_M = 0.0015
CONTEXT_OPACITY = 0.12
BACK_DEPTH_M = 0.030
BACKGROUND = "#202832"
STATES = (
    {
        "id": "original-off",
        "label": "Original rate · smoothing off · last full checkpoint",
        "directory": "24-learned-axis",
        "step": 16,
        "rate": 7.857985794554741,
        "smoothing": False,
        "case": "learned-axis",
    },
    {
        "id": "original-on",
        "label": "Original rate · smoothing on",
        "directory": "25-learned-axis-smooth",
        "step": 128,
        "rate": 7.857985794554741,
        "smoothing": True,
        "case": "learned-axis-smooth",
    },
    {
        "id": "quarter-off",
        "label": "Quarter rate · smoothing off",
        "directory": "48-axis-off-lr-quarter-64",
        "step": 64,
        "rate": 1.9644964486386853,
        "smoothing": False,
        "case": "learned-axis",
    },
    {
        "id": "quarter-on",
        "label": "Quarter rate · smoothing on",
        "directory": "49-axis-on-lr-quarter-64",
        "step": 64,
        "rate": 1.9644964486386853,
        "smoothing": True,
        "case": "learned-axis-smooth",
    },
    {
        "id": "rate03-on-128",
        "label": "Rate 0.3 · smoothing on · phase boundary",
        "directory": "55-axis-on-lr03-128",
        "step": 128,
        "rate": 0.3,
        "smoothing": True,
        "case": "learned-axis-smooth",
    },
    {
        "id": "rate03-on",
        "label": "Rate 0.3 · smoothing on",
        "directory": "56-axis-on-lr03-256",
        "step": 256,
        "rate": 0.3,
        "smoothing": True,
        "case": "learned-axis-smooth",
    },
)


class Config(cherries.BaseConfig):
    """Destination and optional single-state preview mode."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = OUTPUT
    preview_only: bool = False


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


def _trace_row(path: Path, step: int) -> dict[str, Any]:
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["step"]) == step]
    if len(rows) != 1:
        raise ValueError(f"expected one trace row for step {step}: {path}")
    row = rows[0]
    if row["forward_success"] != "True" or row["adjoint_success"] != "True":
        raise ValueError(
            f"selected trace row is not a valid forward/adjoint state: {path}"
        )
    return {
        "fit_rms_mm": float(row["fit_rms_mm"]),
        "motion_rms_mm": float(row["motion_rms_mm"]),
        "inverted_all_cells": int(row["inverted_all_cells"]),
        "detF_min": float(row["detF_min"]),
    }


def _sample_indices(centroids: np.ndarray, global_ids: np.ndarray) -> np.ndarray:
    origin = np.floor(np.min(centroids, axis=0) / 0.001) * 0.001
    voxels = np.floor((centroids - origin) / SAMPLE_SPACING_M).astype(np.int64)
    order = np.argsort(global_ids, kind="stable")
    _, first = np.unique(voxels[order], axis=0, return_index=True)
    chosen = order[first]
    return chosen[np.argsort(global_ids[chosen], kind="stable")]


def _view_mask(points: np.ndarray, camera: dict[str, Any]) -> np.ndarray:
    focus = np.asarray(camera["focal_point"], dtype=np.float64)
    backward = np.asarray(camera["position"], dtype=np.float64) - focus
    backward /= np.linalg.norm(backward)
    right = np.cross(np.asarray(camera["view_up"], dtype=np.float64), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    relative = points - focus
    scale = float(camera["parallel_scale"])
    return (
        (np.abs(relative @ right) <= scale)
        & (np.abs(relative @ up) <= scale)
        & (relative @ backward >= -BACK_DEPTH_M)
    )


def _point_mesh(
    centroids: np.ndarray,
    global_ids: np.ndarray,
    muscle_ids: np.ndarray,
    muscle_labels: np.ndarray,
    control_ids: np.ndarray,
    muscle_fraction: np.ndarray,
    axes: np.ndarray,
    strength: np.ndarray,
    shortening: np.ndarray,
) -> pv.PolyData:
    mesh = pv.PolyData(centroids)
    mesh.point_data["GlobalCellId"] = global_ids
    mesh.point_data["MuscleId"] = muscle_ids
    mesh.point_data["MuscleLabel"] = muscle_labels
    mesh.point_data["ActivationControlId"] = control_ids
    mesh.point_data["MuscleFraction"] = muscle_fraction
    mesh.point_data["LearnedAxisRest"] = axes
    mesh.point_data["LearnedAxisDyadRest"] = np.einsum(
        "ni,nj->nij", axes, axes
    ).reshape(-1, 9)
    mesh.point_data["qNormSquared"] = strength
    mesh.point_data["CommandedShorteningFraction"] = shortening
    mesh.point_data["CommandedShorteningPercent"] = 100.0 * shortening
    mesh.field_data["CoordinateFrame"] = np.asarray(["reference/rest coordinates"])
    mesh.field_data["AxisMeaning"] = np.asarray(
        [
            "arbitrary signed representative of unoriented learned parameter axis; not anatomical fiber"
        ]
    )
    mesh.field_data["LengthMeaning"] = np.asarray(
        ["display only: 1.5 mm visibility floor to 12 mm maximum"]
    )
    return mesh


def _line_mesh(
    centroids: np.ndarray,
    global_ids: np.ndarray,
    muscle_ids: np.ndarray,
    muscle_labels: np.ndarray,
    control_ids: np.ndarray,
    muscle_fraction: np.ndarray,
    axes: np.ndarray,
    strength: np.ndarray,
    shortening: np.ndarray,
) -> pv.PolyData:
    display_length = (
        DISPLAY_MIN_LENGTH_M + (DISPLAY_LENGTH_M - DISPLAY_MIN_LENGTH_M) * shortening
    )
    half = 0.5 * display_length[:, None] * axes
    points = np.empty((2 * len(centroids), 3), dtype=np.float64)
    points[0::2] = centroids - half
    points[1::2] = centroids + half
    lines = np.column_stack(
        (
            np.full(len(centroids), 2, dtype=np.int64),
            2 * np.arange(len(centroids)),
            2 * np.arange(len(centroids)) + 1,
        )
    ).ravel()
    mesh = pv.PolyData(points, lines=lines)
    mesh.cell_data["GlobalCellId"] = global_ids
    mesh.cell_data["MuscleId"] = muscle_ids
    mesh.cell_data["MuscleLabel"] = muscle_labels
    mesh.cell_data["ActivationControlId"] = control_ids
    mesh.cell_data["MuscleFraction"] = muscle_fraction
    mesh.cell_data["LearnedAxisRest"] = axes
    mesh.cell_data["LearnedAxisDyadRest"] = np.einsum("ni,nj->nij", axes, axes).reshape(
        -1, 9
    )
    mesh.cell_data["qNormSquared"] = strength
    mesh.cell_data["CommandedShorteningFraction"] = shortening
    mesh.cell_data["CommandedShorteningPercent"] = 100.0 * shortening
    mesh.field_data["CoordinateFrame"] = np.asarray(["reference/rest coordinates"])
    mesh.field_data["AxisMeaning"] = np.asarray(
        [
            "arbitrary signed representative of unoriented learned parameter axis; not anatomical fiber"
        ]
    )
    mesh.field_data["LengthMeaning"] = np.asarray(
        ["display only: 1.5 mm visibility floor to 12 mm maximum"]
    )
    return mesh


def _render(
    skin: pv.PolyData,
    glyphs: pv.PolyData,
    mask: np.ndarray,
    camera: dict[str, Any],
    state: dict[str, Any],
    metrics: dict[str, Any],
    path: Path,
) -> None:
    shown = glyphs.extract_cells(mask)
    plotter = pv.Plotter(
        off_screen=True, window_size=(1000, 1000), lighting="three lights"
    )
    plotter.set_background(BACKGROUND)
    plotter.add_mesh(
        skin, color="#d8d6cf", opacity=CONTEXT_OPACITY, smooth_shading=False
    )
    plotter.add_mesh(
        shown,
        color="#dce4e8",
        opacity=0.38,
        line_width=5.0,
        render_lines_as_tubes=True,
    )
    plotter.add_mesh(
        shown,
        scalars="CommandedShorteningPercent",
        cmap="viridis",
        clim=(0.0, 100.0),
        line_width=3.0,
        render_lines_as_tubes=True,
        scalar_bar_args={
            "title": "Shortening (%)",
            "color": "white",
            "title_font_size": 17,
            "label_font_size": 14,
            "n_labels": 5,
            "vertical": True,
        },
    )
    plotter.add_text(
        f"{state['label']} · update {state['step']}\n"
        f"fit {metrics['fit_rms_mm']:.4f} mm · motion {metrics['motion_rms_mm']:.4f} mm · "
        f"inversions {metrics['inverted_all_cells']}\n"
        "Reference-frame learned axes · commanded shortening\n"
        "Display length 1.5–12 mm (visibility floor to maximum)",
        position="upper_left",
        color="white",
        font_size=12,
    )
    plotter.enable_parallel_projection()
    plotter.camera.position = camera["position"]
    plotter.camera.focal_point = camera["focal_point"]
    plotter.camera.up = camera["view_up"]
    plotter.camera.parallel_scale = camera["parallel_scale"]
    plotter.reset_camera_clipping_range()
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(path)
    plotter.close()


def _readme(states: list[dict[str, Any]]) -> str:
    rows = "\n".join(
        f"- `{state['id']}`: update {state['step']}, learning rate {state['rate']:.15g}, "
        f"smoothing {'on' if state['smoothing'] else 'off'}"
        for state in states
    )
    return f"""# Learned-axis glyph data

Coordinates and lengths use metres. `LearnedAxisRest = q / ||q||` is an arbitrary
signed representative of an unoriented axis in the reference/rest coordinate frame;
`LearnedAxisDyadRest = n n^T` is its sign-invariant 9-component dyad. These axes are
learned parameter directions, not anatomical muscle fibers.

`qNormSquared = ||q||^2` and `CommandedShorteningFraction = qNormSquared /
(1 + qNormSquared)`. The percentage field is fixed to 0–100% in every state; there
is no per-state normalization. Glyph line length is `0.0015 m + 0.0105 m *
shortening`: a 1.5 mm visibility floor and 12 mm maximum. Both are display scales,
not muscle displacement or physical fiber length.

Sampling uses a fixed 6 mm rest-space voxel grid and the smallest GlobalCellId in
each occupied voxel. The same sample is used in every state, independently of
activation. This deterministic sample covers all 103 activation-region IDs, but has
no per-muscle coverage guarantee: 1,222 occupied voxels contain multiple MuscleId
values and choosing one cell omits 1,558 additional muscle/voxel combinations.

States:
{rows}

In ParaView, open `context/rest-skin.vtp` and one `glyphs/<state>.vtp`, then use
"Apply" for both. Color the glyph object by `CommandedShorteningPercent` with a
fixed 0–100 range. The glyph VTP contains prebuilt centered line cells; optionally
apply the Tube filter for cylindrical rods. `samples/<state>.vtp` contains one point
per sampled cell with the same scalar fields and `LearnedAxisRest` for custom Glyph
filters. Use `GlobalCellId`, `MuscleId`, and `MuscleLabel` to inspect provenance.
Use centered lines or cylinders for custom glyphs; do not use a single arrowhead,
because the learned axis is unoriented. `ActivationControlId` and `MuscleFraction`
are also included for filtering.
"""


def _main(cfg: Config) -> dict[str, Any]:
    output = cfg.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_receipt = _snapshot_source(
        Path(__file__), output / "sources/65-render-activation-glyphs.py"
    )
    required = [FIXTURE / "volume.vtu", FIXTURE / "skin.vtp", CAMERAS]
    states = [
        dict(state)
        for state in STATES
        if not cfg.preview_only or state["id"] == "rate03-on"
    ]
    for state in states:
        directory = GROUP / "data" / state["directory"]
        state["checkpoint"] = directory / f"step-{state['step']:04d}.npz"
        state["trace"] = directory / "trace.csv"
        state["summary_source"] = directory / "summary.json"
        state["config_source"] = directory / "config.json"
        state["provenance_source"] = directory / "provenance.json"
        required.extend(
            [
                state["checkpoint"],
                state["trace"],
                state["summary_source"],
                state["config_source"],
                state["provenance_source"],
            ]
        )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    volume = pv.read(FIXTURE / "volume.vtu")
    skin = pv.read(FIXTURE / "skin.vtp")
    active_fixture_ids = np.flatnonzero(
        np.asarray(volume.cell_data["ActivationMask"], dtype=bool)
    )
    muscle_active_ids = np.flatnonzero(
        np.asarray(volume.cell_data["MuscleFraction"], dtype=np.float64) > 0.0
    )
    if not np.array_equal(active_fixture_ids, muscle_active_ids):
        raise ValueError("fixture ActivationMask differs from MuscleFraction > 0")
    all_centroids = np.asarray(volume.cell_centers().points, dtype=np.float64)
    active_centroids = all_centroids[active_fixture_ids]
    chosen = _sample_indices(active_centroids, active_fixture_ids)
    sampled_ids = active_fixture_ids[chosen]
    centroids = active_centroids[chosen]
    muscle_ids = np.asarray(volume.cell_data["MuscleId"], dtype=np.int32)[sampled_ids]
    control_ids = np.asarray(volume.cell_data["ActivationControlId"], dtype=np.int64)[
        sampled_ids
    ]
    muscle_fraction = np.asarray(volume.cell_data["MuscleFraction"], dtype=np.float64)[
        sampled_ids
    ]
    grid_origin = np.floor(np.min(active_centroids, axis=0) / 0.001) * 0.001
    active_voxels = np.floor(
        (active_centroids - grid_origin) / SAMPLE_SPACING_M
    ).astype(np.int64)
    active_muscle_ids = np.asarray(volume.cell_data["MuscleId"], dtype=np.int32)[
        active_fixture_ids
    ]
    voxel_muscle_pairs = np.unique(
        np.column_stack((active_voxels, active_muscle_ids)), axis=0
    )
    _, muscle_counts_per_voxel = np.unique(
        voxel_muscle_pairs[:, :3], axis=0, return_counts=True
    )
    multi_muscle_voxels = int(np.count_nonzero(muscle_counts_per_voxel > 1))
    omitted_muscle_voxel_pairs = int(np.sum(muscle_counts_per_voxel - 1))
    if (multi_muscle_voxels, omitted_muscle_voxel_pairs) != (1222, 1558):
        raise ValueError(
            "observed voxel/muscle suppression differs from the audited values"
        )
    muscle_names = np.asarray(volume.field_data["MuscleName"]).astype(str)
    if np.any((muscle_ids < 0) | (muscle_ids >= len(muscle_names))):
        raise ValueError("sampled active cells have invalid MuscleId values")
    muscle_labels = muscle_names[muscle_ids]
    cameras_doc = json.loads(CAMERAS.read_text())
    cameras = {view["id"]: view for view in cameras_doc["views"]}
    if set(VIEWS) - set(cameras):
        raise ValueError("frozen camera file lacks a requested view")
    view_masks = {
        view_id: _view_mask(centroids, cameras[view_id]["camera"]) for view_id in VIEWS
    }

    context = output / "context/rest-skin.vtp"
    context.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURE / "skin.vtp", context)
    state_records: list[dict[str, Any]] = []
    generated: list[Path] = [context]
    for state in states:
        checkpoint = state.pop("checkpoint")
        trace = state.pop("trace")
        summary_source = state.pop("summary_source")
        config_source = state.pop("config_source")
        provenance_source = state.pop("provenance_source")
        config_doc = json.loads(config_source.read_text())
        provenance_doc = json.loads(provenance_source.read_text())
        run_summary_doc = json.loads(summary_source.read_text())
        for document, label in (
            (config_doc, "config"),
            (provenance_doc, "provenance"),
            (run_summary_doc, "summary"),
        ):
            if document["case"] != state["case"]:
                raise ValueError(
                    f"{label} case disagrees with selected state: {state['id']}"
                )
        settings_path = Path(config_doc["settings"])
        if not settings_path.is_absolute():
            settings_path = GROUP / settings_path
        settings_doc = json.loads(settings_path.read_text())
        recorded_rate = float(settings_doc["learning_rates"]["learned-axis"])
        if recorded_rate != state["rate"]:
            raise ValueError(
                f"settings rate disagrees with selected state: {state['id']}"
            )
        if bool(provenance_doc["smoothness_weight"] > 0.0) != state["smoothing"]:
            raise ValueError(
                f"provenance smoothing weight disagrees with selected state: {state['id']}"
            )
        with np.load(checkpoint, allow_pickle=False) as saved:
            step = int(saved["step"])
            q = np.asarray(saved["q"], dtype=np.float64)
            C = np.asarray(saved["C"], dtype=np.float64)
            active_ids = np.asarray(saved["active_ids"], dtype=np.int64)
            rest_points = np.asarray(saved["rest_points"], dtype=np.float64)
            model = str(saved["model"])
            solver_valid = bool(saved["solver_valid"])
        if step != state["step"]:
            raise ValueError(f"checkpoint step mismatch: {checkpoint}")
        if not np.array_equal(active_ids, active_fixture_ids):
            raise ValueError(
                f"active IDs differ from fixture ActivationMask: {checkpoint}"
            )
        if model != "learned-axis" or not solver_valid:
            raise ValueError(f"checkpoint model/solver validity mismatch: {checkpoint}")
        if not np.array_equal(rest_points, np.asarray(volume.points, dtype=np.float64)):
            raise ValueError(
                f"checkpoint rest points differ from fixture: {checkpoint}"
            )
        if q.shape != (len(active_ids), 3) or not np.isfinite(q).all():
            raise ValueError(f"invalid learned axes: {checkpoint}")
        reconstructed_C = np.einsum("ni,nj->nij", q, q)
        c_error = float(np.max(np.abs(C - reconstructed_C)))
        if c_error != 0.0:
            raise ValueError(f"saved C is not exactly q q^T: {checkpoint}")
        sample_q = q[chosen]
        norm = np.linalg.norm(sample_q, axis=1)
        if np.any(norm == 0.0):
            raise ValueError(
                f"zero sampled learned axis cannot define a direction: {checkpoint}"
            )
        axes = sample_q / norm[:, None]
        strength = norm * norm
        shortening = strength / (1.0 + strength)
        if not (np.all(shortening >= 0.0) and np.all(shortening < 1.0)):
            raise AssertionError("shortening must lie in [0, 1)")
        points = _point_mesh(
            centroids,
            sampled_ids,
            muscle_ids,
            muscle_labels,
            control_ids,
            muscle_fraction,
            axes,
            strength,
            shortening,
        )
        glyphs = _line_mesh(
            centroids,
            sampled_ids,
            muscle_ids,
            muscle_labels,
            control_ids,
            muscle_fraction,
            axes,
            strength,
            shortening,
        )
        samples_path = output / "samples" / f"{state['id']}.vtp"
        glyphs_path = output / "glyphs" / f"{state['id']}.vtp"
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        glyphs_path.parent.mkdir(parents=True, exist_ok=True)
        points.save(samples_path, binary=True)
        glyphs.save(glyphs_path, binary=True)
        generated.extend([samples_path, glyphs_path])
        metrics = _trace_row(trace, state["step"])
        views: dict[str, Any] = {}
        for view_id in VIEWS:
            png = output / "geometry" / state["id"] / f"{view_id}.png"
            _render(
                skin,
                glyphs,
                view_masks[view_id],
                cameras[view_id]["camera"],
                state,
                metrics,
                png,
            )
            generated.append(png)
            views[view_id] = {
                "sample_count": int(np.count_nonzero(view_masks[view_id])),
                "file": _record(png),
            }
        state_records.append(
            {
                **state,
                "checkpoint": _record(checkpoint),
                "trace": _record(trace),
                "run_summary": _record(summary_source),
                "config": _record(config_source),
                "provenance": _record(provenance_source),
                "settings": _record(settings_path),
                "metrics": metrics,
                "sampled_shortening_percentiles": {
                    str(p): float(np.percentile(100.0 * shortening, p))
                    for p in (0, 50, 90, 95, 99, 100)
                },
                "C_equals_q_qT_max_abs_error": c_error,
                "samples": _record(samples_path),
                "glyphs": _record(glyphs_path),
                "views": views,
            }
        )

    readme = output / "README.md"
    readme.write_text(_readme(states))
    generated.append(readme)
    archive = output / "glyph-data.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as bundle:
        bundle.write(readme, "README.md")
        for folder in ("samples", "glyphs", "context"):
            for path in sorted((output / folder).glob("*")):
                bundle.write(path, path.relative_to(output))
    generated.append(archive)
    summary = {
        "status": "completed_activation_glyph_preview"
        if cfg.preview_only
        else "completed_activation_glyph_render",
        "semantics": {
            "coordinate_frame": "reference/rest coordinates at active tetrahedron centroids",
            "axis": "arbitrary signed representative q / ||q|| of an unoriented learned parameter axis; not an anatomical muscle fiber",
            "axis_dyad": "sign-invariant n n^T stored as 9 components in row-major order",
            "strength": "s = ||q||^2 and saved C = q q^T",
            "commanded_shortening": "s / (1 + s), shown on one fixed 0–100% scale without per-state normalization",
            "glyph_length": "1.5 mm + 10.5 mm * commanded shortening; display-only visibility floor and scale, not muscle displacement or physical fiber length",
        },
        "sampling": {
            "method": "6 mm rest-space voxels; smallest GlobalCellId in each occupied voxel",
            "spacing_m": SAMPLE_SPACING_M,
            "activation_independent": True,
            "per_muscle_coverage_guarantee": False,
            "activation_region_ids_present": len(np.unique(control_ids)),
            "activation_region_ids_total": len(
                np.unique(volume.cell_data["ActivationControlId"][active_fixture_ids])
            ),
            "occupied_voxels_with_multiple_muscle_ids": multi_muscle_voxels,
            "additional_muscle_voxel_combinations_omitted": omitted_muscle_voxel_pairs,
            "active_cell_count": len(active_fixture_ids),
            "sample_count": len(chosen),
            "sampled_global_cell_ids_sha256": hashlib.sha256(
                sampled_ids.tobytes()
            ).hexdigest(),
        },
        "render": {
            "views": list(VIEWS),
            "context_opacity": CONTEXT_OPACITY,
            "line_width_pixels": 3.0,
            "visibility_understroke": {
                "color": "#dce4e8",
                "opacity": 0.38,
                "line_width_pixels": 5.0,
            },
            "back_depth_clip_m": BACK_DEPTH_M,
            "clip_rule": "centroid inside the frozen square parallel viewport and no farther than 30 mm behind its focal plane",
            "color_map": "viridis",
            "color_range_percent": [0.0, 100.0],
        },
        "states": state_records,
        "inputs": {str(path.resolve()): _record(path) for path in required},
        "outputs": {str(path.relative_to(output)): _record(path) for path in generated},
        "archive": _record(archive),
        "source": source_receipt,
    }
    _write_json(output / "summary.json", summary)
    return summary


def main(cfg: Config) -> None:
    summary = _main(cfg)
    cherries.log_metric("render/states", len(summary["states"]))
    cherries.log_metric("render/samples", summary["sampling"]["sample_count"])
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
