"""Render full activation axes on each saved deformed shape."""

# ruff: noqa: C901, CPY001, EM101, EM102, PLR0912, PLR0915, RUF001, TRY003

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from muscle_glyph_context import (
    RegionVisibility,
    build_muscle_region_context,
    save_muscle_region_context,
    save_region_visibility,
    visible_region_mask,
)

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]
GROUP = Path(__file__).resolve().parents[1]
INPUT_GROUP = ROOT / "exp/2026/09/09/activation-space-smoothness"
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
CAMERAS = ROOT / "exp/2026/09/08/physical-volume-closeups/data/20-regions/summary.json"
OUTPUT = GROUP / "data/76-deformed-activation-glyphs"
VIEWS = ("side-context", "region1-mouth-corner")
CONTEXT_OPACITY = 0.06
BACKGROUND = "#f4f2ed"
DISPLAY_MAX_LENGTH_M = 0.0045
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
    mesh.point_data["LearnedAxisSpatial"] = axes
    mesh.point_data["LearnedAxisDyadSpatial"] = np.einsum(
        "ni,nj->nij", axes, axes
    ).reshape(-1, 9)
    mesh.point_data["qNormSquared"] = strength
    mesh.point_data["CommandedShorteningFraction"] = shortening
    mesh.point_data["CommandedShorteningPercent"] = 100.0 * shortening
    mesh.field_data["CoordinateFrame"] = np.asarray(["saved deformed coordinates"])
    mesh.field_data["AxisMeaning"] = np.asarray(
        [
            "arbitrary signed representative of unoriented learned parameter axis; not anatomical fiber"
        ]
    )
    mesh.field_data["LengthMeaning"] = np.asarray(
        ["global display rule: 4.5 mm multiplied linearly by commanded shortening"]
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
    half = (0.5 * DISPLAY_MAX_LENGTH_M * shortening)[:, None] * axes
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
    mesh.cell_data["LearnedAxisSpatial"] = axes
    mesh.cell_data["LearnedAxisDyadSpatial"] = np.einsum(
        "ni,nj->nij", axes, axes
    ).reshape(-1, 9)
    mesh.cell_data["qNormSquared"] = strength
    mesh.cell_data["CommandedShorteningFraction"] = shortening
    mesh.cell_data["CommandedShorteningPercent"] = 100.0 * shortening
    mesh.field_data["CoordinateFrame"] = np.asarray(["saved deformed coordinates"])
    mesh.field_data["AxisMeaning"] = np.asarray(
        [
            "arbitrary signed representative of unoriented learned parameter axis; not anatomical fiber"
        ]
    )
    mesh.field_data["LengthMeaning"] = np.asarray(
        ["global display rule: 4.5 mm multiplied linearly by commanded shortening"]
    )
    return mesh


def _render(
    skin: pv.PolyData,
    glyphs: pv.PolyData,
    visibility: RegionVisibility,
    camera: dict[str, Any],
    state: dict[str, Any],
    metrics: dict[str, Any],
    path: Path,
) -> None:
    plotter = pv.Plotter(
        off_screen=True, window_size=(1800, 1800), lighting="three lights"
    )
    plotter.set_background(BACKGROUND)
    plotter.add_mesh(
        skin, color="#8d969b", opacity=CONTEXT_OPACITY, smooth_shading=False
    )
    shown = glyphs.extract_cells(visibility.mask)
    assert np.array_equal(
        shown.cell_data["GlobalCellId"],
        glyphs.cell_data["GlobalCellId"][visibility.mask],
    )
    assert shown.n_cells == visibility.retained_count
    plotter.add_mesh(
        shown,
        scalars="CommandedShorteningPercent",
        cmap="viridis",
        clim=(0.0, 100.0),
        line_width=1.0,
        render_lines_as_tubes=False,
        scalar_bar_args={
            "title": "Shortening (%)",
            "color": "black",
            "title_font_size": 17,
            "label_font_size": 14,
            "n_labels": 5,
            "vertical": True,
            "background_color": BACKGROUND,
            "fill": True,
        },
    )
    annotation = plotter.add_text(
        f"{state['label']} · update {state['step']}\n"
        f"fit {metrics['fit_rms_mm']:.4f} mm · motion {metrics['motion_rms_mm']:.4f} mm · "
        f"inversions {metrics['inverted_all_cells']}\n"
        "Saved deformed shape · transported learned axes\n"
        "Visible muscles only · 100% shortening = 4.5 mm · no length floor",
        position="upper_left",
        color="black",
        font_size=12,
    )
    annotation.GetTextProperty().SetBackgroundColor(244 / 255, 242 / 255, 237 / 255)
    annotation.GetTextProperty().SetBackgroundOpacity(0.90)
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
    return f"""# Learned-axis glyphs on saved deformed shapes

Coordinates are metres. Each complete field contains one centered line for each of
288,235 active tetrahedra across 103 activation regions, without spatial sampling.
Centers are mean((rest_points + u)[tetrahedron vertices]). F maps reference edges to
saved deformed edges. LearnedAxisRest = q / ||q|| and LearnedAxisSpatial =
normalize(F @ LearnedAxisRest). Both are arbitrary signed representatives of an
unoriented learned control axis; their dyads are sign-invariant. They are not
anatomical fibers or observed tissue strain. DeformationGradient stores F in
row-major order; AxisTransportStretch = ||F n_rest|| does not scale line length.

The saved control obeys C = q q^T, B = I+C = A^-1. qNormSquared = ||q||^2.
CommandedShorteningFraction = qNormSquared / (1 + qNormSquared).
DisplayLineLengthM = 0.0045 m * commanded shortening fraction. This same 4.5 mm
scale applies to every cell and state, without a floor or cell-size normalization.
Color uses one linear 0–100% range. A line is a display glyph, not displacement
or physical fiber length. MuscleFraction is exported but does not multiply a.

States (unmatched descriptive checkpoints):
{rows}

In ParaView, open context/<state>-skin.vtp and glyphs/<state>.vtp. Color glyphs by
CommandedShorteningPercent with fixed 0–100 range. The glyph VTP has prebuilt lines;
a Tube filter is optional. samples/<state>.vtp contains all cell centers and the
same attributes for custom centered Glyph filters using LearnedAxisSpatial.
RestCentroid, LearnedAxisRest, DeformationGradient, and GlobalCellId retain the
reference-to-spatial mapping. GlobalCellId maps through checkpoint active_ids,
not ActivationControlId. Use centered lines: single arrowheads imply a false sign.

The skin uses saved deformed volume points indexed by its verified GlobalPointId.
No interpolation, deformation amplification, surface smoothing, or decimation is
applied. Saved inverted cells are retained and panel annotations report inversions.

context/<state>-muscle-regions.vtp contains 103 regions extracted separately from
that state's deformed volume, retaining shared interfaces and source tetrahedron
IDs. These surfaces are used for visibility only and are not drawn in the panels.
No outlines are drawn. Faint deformed skin (opacity 0.06) provides face context.
visibility/<state>--<view>.npz stores the full IDs, pixel projection, front-label
image, and mask. A cell is retained when the frontmost region at its projected
centroid matches its own ActivationControlId. Internal tetrahedra of that region
remain present; other muscles behind it are hidden. Each state and camera has its
own recomputed mask. Full exported fields remain unfiltered. This centroid rule
can leave line ends crossing visible boundaries; it does not clip every line pixel.
"""


def _main(cfg: Config) -> dict[str, Any]:
    output = cfg.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_receipt = _snapshot_source(
        Path(__file__), output / "sources/76-render-deformed-activation-glyphs.py"
    )
    helper_source = Path(__file__).with_name("muscle_glyph_context.py")
    helper_receipt = _snapshot_source(
        helper_source, output / "sources/muscle_glyph_context.py"
    )
    profile_receipt = _snapshot_source(
        Path(__file__).with_name("experiment_profile.py"),
        output / "sources/experiment_profile.py",
    )
    required = [FIXTURE / "volume.vtu", FIXTURE / "skin.vtp", CAMERAS, helper_source]
    states = [
        dict(state)
        for state in STATES
        if not cfg.preview_only or state["id"] == "rate03-on"
    ]
    states.sort(key=lambda state: state["id"] != "rate03-on")
    for state in states:
        directory = INPUT_GROUP / "data" / state["directory"]
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
    if len(active_fixture_ids) != 288235:
        raise ValueError("unexpected active tetrahedron count")
    cells = np.asarray(volume.cells, dtype=np.int64).reshape(-1, 5)
    if not np.all(cells[:, 0] == 4):
        raise ValueError("fixture contains a non-tetrahedral cell")
    active_tetrahedra = np.asarray(volume.points, dtype=np.float64)[
        cells[active_fixture_ids, 1:]
    ]
    direct_centroids = np.mean(active_tetrahedra, axis=1)
    if not np.allclose(active_centroids, direct_centroids, rtol=0.0, atol=1.0e-12):
        raise ValueError("PyVista cell centers differ from tetrahedron centroids")
    full_ids = active_fixture_ids
    centroids = direct_centroids
    muscle_ids = np.asarray(volume.cell_data["MuscleId"], dtype=np.int32)[full_ids]
    control_ids = np.asarray(volume.cell_data["ActivationControlId"], dtype=np.int64)[
        full_ids
    ]
    muscle_fraction = np.asarray(volume.cell_data["MuscleFraction"], dtype=np.float64)[
        full_ids
    ]
    muscle_names = np.asarray(volume.field_data["MuscleName"]).astype(str)
    if np.any((muscle_ids < 0) | (muscle_ids >= len(muscle_names))):
        raise ValueError("active cells have invalid MuscleId values")
    muscle_labels = muscle_names[muscle_ids]
    cameras_doc = json.loads(CAMERAS.read_text())
    cameras = {view["id"]: view for view in cameras_doc["views"]}
    if set(VIEWS) - set(cameras):
        raise ValueError("frozen camera file lacks a requested view")

    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    np.testing.assert_array_equal(skin.points, volume.points[skin_ids])
    rest_edges = np.swapaxes(active_tetrahedra[:, 1:] - active_tetrahedra[:, :1], 1, 2)
    inverse_rest_edges = np.linalg.inv(rest_edges)
    state_records: list[dict[str, Any]] = []
    generated: list[Path] = []
    visibility_records: dict[str, Any] = {}
    for state in states:
        logging.getLogger(__name__).info("Rendering %s", state["id"])
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
            settings_path = INPUT_GROUP / settings_path
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
            displacement = np.asarray(saved["u"], dtype=np.float64)
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
        norm = np.linalg.norm(q, axis=1)
        if np.any(norm == 0.0):
            raise ValueError(
                f"zero learned axis cannot define a direction: {checkpoint}"
            )
        rest_axes = q / norm[:, None]
        assert displacement.shape == rest_points.shape
        assert np.isfinite(displacement).all()
        deformed_points = rest_points + displacement
        deformed_tetrahedra = deformed_points[cells[full_ids, 1:]]
        centroids = deformed_tetrahedra.mean(axis=1)
        deformed_edges = np.swapaxes(
            deformed_tetrahedra[:, 1:] - deformed_tetrahedra[:, :1], 1, 2
        )
        deformation_gradient = deformed_edges @ inverse_rest_edges
        transported = np.einsum("nij,nj->ni", deformation_gradient, rest_axes)
        axis_stretch = np.linalg.norm(transported, axis=1)
        assert np.isfinite(axis_stretch).all()
        assert np.all(axis_stretch > 0)
        axes = transported / axis_stretch[:, None]
        deformed_skin = skin.copy(deep=True)
        deformed_skin.points = deformed_points[skin_ids]
        deformed_skin.point_data["RestPosition"] = rest_points[skin_ids]
        deformed_skin.point_data["Displacement"] = displacement[skin_ids]
        deformed_skin.field_data["CoordinateFrame"] = np.asarray(
            ["saved deformed coordinates"]
        )
        skin_path = output / "context" / f"{state['id']}-skin.vtp"
        skin_path.parent.mkdir(parents=True, exist_ok=True)
        deformed_skin.save(skin_path, binary=True)
        deformed_volume = volume.copy(deep=True)
        deformed_volume.points = deformed_points
        muscles = build_muscle_region_context(deformed_volume)
        muscle_surface = save_muscle_region_context(
            muscles, output / "context" / f"{state['id']}-muscle-regions.vtp"
        )
        generated.extend([skin_path, muscle_surface])
        visibility_by_view: dict[str, RegionVisibility] = {}
        visibility_records[state["id"]] = {}
        for view_id in VIEWS:
            visibility = visible_region_mask(
                muscles,
                centroids,
                full_ids,
                control_ids,
                cameras[view_id]["camera"],
                window_size=(1800, 1800),
            )
            visibility_path = save_region_visibility(
                visibility, output / "visibility" / f"{state['id']}--{view_id}.npz"
            )
            visibility_by_view[view_id] = visibility
            generated.append(visibility_path)
            visibility_records[state["id"]][view_id] = {
                "file": _record(visibility_path),
                **{
                    key: getattr(visibility, key)
                    for key in (
                        "retained_count",
                        "retained_interior_count",
                        "retained_boundary_source_count",
                        "projected_inside_count",
                        "projected_background_count",
                        "occluded_by_other_region_count",
                        "front_surface_control_ids",
                        "retained_control_ids",
                    )
                },
            }
        strength = norm * norm
        shortening = strength / (1.0 + strength)
        if not (np.all(shortening >= 0.0) and np.all(shortening < 1.0)):
            raise AssertionError("shortening must lie in [0, 1)")
        points = _point_mesh(
            centroids,
            full_ids,
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
            full_ids,
            muscle_ids,
            muscle_labels,
            control_ids,
            muscle_fraction,
            axes,
            strength,
            shortening,
        )
        for mesh, arrays in ((points, points.point_data), (glyphs, glyphs.cell_data)):
            arrays["LearnedAxisRest"] = rest_axes
            arrays["LearnedAxisDyadRest"] = np.einsum(
                "ni,nj->nij", rest_axes, rest_axes
            ).reshape(-1, 9)
            arrays["DeformationGradient"] = deformation_gradient.reshape(-1, 9)
            arrays["DetF"] = np.linalg.det(deformation_gradient)
            arrays["AxisTransportStretch"] = axis_stretch
            arrays["RestCentroid"] = direct_centroids
            arrays["DisplayLineLengthM"] = DISPLAY_MAX_LENGTH_M * shortening
            mesh.field_data["DirectionTransport"] = np.asarray(
                [
                    "normalize(F @ LearnedAxisRest); length is 0.0045 m * commanded shortening, independent of ||F n||"
                ]
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
            visibility = visibility_by_view[view_id]
            _render(
                deformed_skin,
                glyphs,
                visibility,
                cameras[view_id]["camera"],
                state,
                metrics,
                png,
            )
            generated.append(png)
            views[view_id] = {
                "line_count_submitted": visibility.retained_count,
                "clipping": "frontmost muscle-region label at cell center; interior cells retained",
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
                "full_field_shortening_percentiles": {
                    str(p): float(np.percentile(100.0 * shortening, p))
                    for p in (0, 50, 90, 95, 99, 100)
                },
                "C_equals_q_qT_max_abs_error": c_error,
                "display_line_length_mm": {
                    "min": float(1000.0 * DISPLAY_MAX_LENGTH_M * np.min(shortening)),
                    "median": float(
                        1000.0 * DISPLAY_MAX_LENGTH_M * np.median(shortening)
                    ),
                    "max": float(1000.0 * DISPLAY_MAX_LENGTH_M * np.max(shortening)),
                },
                "deformed_skin": _record(skin_path),
                "deformed_muscle_regions": _record(muscle_surface),
                "deformation": {
                    "max_point_displacement_mm": float(
                        1000 * np.linalg.norm(displacement, axis=1).max()
                    ),
                    "max_centroid_displacement_mm": float(
                        1000
                        * np.linalg.norm(centroids - direct_centroids, axis=1).max()
                    ),
                    "active_inverted_count": int(
                        np.count_nonzero(np.linalg.det(deformation_gradient) < 0)
                    ),
                    "axis_transport_stretch_min": float(axis_stretch.min()),
                    "axis_transport_stretch_max": float(axis_stretch.max()),
                },
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
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as bundle:
        bundle.write(readme, "README.md")
        for folder in ("samples", "glyphs", "context", "sources", "visibility"):
            for path in sorted((output / folder).glob("*")):
                bundle.write(path, path.relative_to(output))
    generated.append(archive)
    summary = {
        "status": "completed_activation_glyph_preview"
        if cfg.preview_only
        else "completed_activation_glyph_render",
        "semantics": {
            "coordinate_frame": "saved deformed coordinates at mean((rest_points + u)[tetrahedron vertices])",
            "axis": "spatial unoriented learned axis normalize(F @ (q / ||q||)); not anatomical fiber or measured strain",
            "deformation_gradient": "F = [x1-x0, x2-x0, x3-x0] @ inverse([X1-X0, X2-X0, X3-X0]); columns are edges",
            "skin": "deformed volume points indexed by skin GlobalPointId, verified against reference coordinates",
            "deformation_scale": 1.0,
            "axis_dyad": "sign-invariant n n^T stored as 9 components in row-major order",
            "strength": "s = ||q||^2 and saved C = q q^T",
            "commanded_shortening": "s / (1 + s), shown on one fixed 0–100% scale without per-state normalization",
            "glyph_length": "global length = 0.0045 m * commanded shortening fraction; no floor or cell-size scaling",
        },
        "coverage": {
            "method": "one line for every active tetrahedron; no spatial sampling",
            "activation_region_ids_present": len(np.unique(control_ids)),
            "activation_region_ids_total": len(
                np.unique(volume.cell_data["ActivationControlId"][active_fixture_ids])
            ),
            "active_cell_count": len(active_fixture_ids),
            "line_count": len(full_ids),
            "every_active_id_present": np.array_equal(full_ids, active_fixture_ids),
            "global_cell_ids_sha256": hashlib.sha256(full_ids.tobytes()).hexdigest(),
        },
        "render": {
            "views": list(VIEWS),
            "context_opacity": CONTEXT_OPACITY,
            "line_width_pixels": 1.0,
            "visibility_understroke": None,
            "clip_rule": "frontmost muscle-region label at each projected centroid must match cell label; retain all depths of matching muscle",
            "visibility": visibility_records,
            "visibility_geometry": "independently rebuilt deformed muscle surfaces and projected centers for each state and camera",
            "maximum_display_length_m": DISPLAY_MAX_LENGTH_M,
            "color_map": "viridis",
            "color_range_percent": [0.0, 100.0],
            "muscle_context": "muscle shapes shown by activation glyphs only; no muscle surfaces or outlines drawn; faint saved deformed skin",
        },
        "states": state_records,
        "inputs": {str(path.resolve()): _record(path) for path in required},
        "outputs": {str(path.relative_to(output)): _record(path) for path in generated},
        "archive": _record(archive),
        "source": source_receipt,
        "muscle_context_source": helper_receipt,
        "profile_source": profile_receipt,
        "input_root": str(INPUT_GROUP),
        "output_root": str(output),
    }
    _write_json(output / "summary.json", summary)
    return summary


def main(cfg: Config) -> None:
    summary = _main(cfg)
    cherries.log_metric("render/states", len(summary["states"]))
    cherries.log_metric("render/lines_per_state", summary["coverage"]["line_count"])
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
