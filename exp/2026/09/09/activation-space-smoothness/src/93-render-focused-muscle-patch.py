# ruff: noqa: PLR0915
"""Render one fixed mouth-region muscle patch at four saved states."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from PIL import Image, ImageDraw, ImageFont

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]
INPUT_ROOT = ROOT / "exp/2026/09/07"
FIXTURE = INPUT_ROOT / "face-actuation-diagnosis/data/12-historical-fixture/volume.vtu"
OLD = INPUT_ROOT / "face-actuation-diagnosis/data/11-historical-no-skin/final.npz"
CORRECTED = (
    Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
    / "exp/2026/09/08/physical-volume-baseline/data/20-baseline/final.npz"
)
PSD = INPUT_ROOT / "tensor-active-stress/data/102-fit1024/final.npz"
SOURCE_RENDERER = (
    Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
    / "exp/2026/09/08/analytical-active-mechanics/src/41-render-tetra-comparison.py"
)
SOURCE_RECEIPT = SOURCE_RENDERER.parents[1] / "data/41-tetra-comparison/receipt.json"
BG = "#303946"
GOLD = "#ffce57"
CAMERA_POSITION = np.array([1.6970235649558214, 2.162, 0.281])
CAMERA_FOCAL = np.array([1.4070235649558214, 2.162, 0.071])
CAMERA_UP = np.array([0.0, 1.0, 0.0])


class Config(cherries.BaseConfig):
    cell_id: int = 27306
    output_dir: Path = cherries.output("93-focused-muscle-patch", mkdir=True)


def record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
    }


def tetra(points: np.ndarray) -> pv.UnstructuredGrid:
    return pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3]), np.array([10], dtype=np.uint8), points
    )


def set_camera(plotter: pv.Plotter, focal: np.ndarray, scale: float) -> None:
    plotter.enable_parallel_projection()
    plotter.camera.position = focal + CAMERA_POSITION - CAMERA_FOCAL
    plotter.camera.focal_point = focal
    plotter.camera.up = CAMERA_UP
    plotter.camera.parallel_scale = scale
    plotter.set_background(BG)


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    ref = pv.read(FIXTURE)
    cells = np.asarray(ref.cells).reshape(-1, 5)
    assert np.all(cells[:, 0] == 4)
    tets = cells[:, 1:]
    selected_vertices = tets[cfg.cell_id]
    assert ref.cell_data["MuscleFraction"][cfg.cell_id] == 1.0
    active_ids = np.flatnonzero(ref.cell_data["ActivationMask"])

    positions = [ref.points.copy()]
    state_specs = (
        (OLD, 194, None),
        (CORRECTED, 200, True),
        (PSD, 1024, None),
    )
    for path, expected_step, physical_volume in state_specs:
        with np.load(path) as state:
            if "rest_points" in state:
                assert np.array_equal(state["rest_points"], ref.points)
            assert np.array_equal(state["active_ids"], active_ids)
            assert np.isfinite(state["u"]).all()
            if "step" in state:
                assert int(state["step"]) == expected_step
            if "solver_valid" in state:
                assert bool(state["solver_valid"])
            if physical_volume is not None:
                assert bool(state["physical_volume_energy"]) is physical_volume
            positions.append(ref.points + state["u"])

    labels = (
        "Rest",
        "Old baseline | step 194",
        "Corrected baseline | step 200",
        "PSD active stress | step 1024",
    )
    selected_points = [points[selected_vertices] for points in positions]

    # Preserve the original baseline-only selection and fixed rest-space neighborhood.
    centers = ref.points[tets].mean(axis=1)
    muscle = np.asarray(ref.cell_data["MuscleFraction"]) > 0
    patch_ids = np.flatnonzero(
        muscle
        & (np.linalg.norm(centers - selected_points[0].mean(axis=0), axis=1) <= 0.006)
    )
    assert len(patch_ids) == 694
    np.save(out / "patch-source-cell-ids.npy", patch_ids)
    ref.point_data["VolumePointIndex"] = np.arange(ref.n_points)
    patch_ref = ref.extract_cells(patch_ids)
    original_ids = np.asarray(patch_ref.point_data["VolumePointIndex"])

    # Freeze the focal point and scale to the original REST/OLD/PSD renderer.
    original_state_indices = (0, 1, 3)
    patch_focal = np.mean(
        [selected_points[i].mean(axis=0) for i in original_state_indices], axis=0
    )
    patch_scale = (
        max(
            np.linalg.norm(positions[i][original_ids] - patch_focal, axis=1).max()
            for i in original_state_indices
        )
        * 1.08
    )

    width = height = 1000
    font = ImageFont.truetype("DejaVuSans.ttf", 26)
    small = ImageFont.truetype("DejaVuSans.ttf", 22)
    panels = []
    panel_records = []
    for index, (label, points) in enumerate(zip(labels, positions, strict=True)):
        plotter = pv.Plotter(
            off_screen=True,
            window_size=(width, height),
            lighting="three lights",
            border=False,
        )
        patch = patch_ref.copy(deep=True)
        patch.points = points[original_ids]
        plotter.add_mesh(
            patch,
            color="#e4a9a6",
            opacity=0.3,
            show_edges=True,
            edge_color="#73717a",
            smooth_shading=False,
            line_width=0.5,
        )
        plotter.add_mesh(
            tetra(selected_points[index]),
            color=GOLD,
            show_edges=True,
            edge_color="#aa7200",
            line_width=3,
        )
        set_camera(plotter, patch_focal, patch_scale)
        raster = Image.fromarray(plotter.screenshot(return_img=True)).convert("RGB")
        plotter.close()
        draw = ImageDraw.Draw(raster)
        draw.rectangle((0, 0, width, 64), fill=BG)
        draw.text((24, 18), label, fill="white", font=font)
        path = out / f"panel-{index}-{('rest', 'old', 'corrected', 'psd')[index]}.png"
        raster.save(path)
        panels.append(raster)
        panel_records.append(record(path))

    composite = Image.new("RGB", (4 * width, height + 120), BG)
    for index, panel in enumerate(panels):
        composite.paste(panel, (index * width, 0))
    draw = ImageDraw.Draw(composite)
    draw.text(
        (24, height + 18),
        "Fixed 694-cell mouth-region muscle neighborhood within 6 mm of cell 27306",
        fill="white",
        font=font,
    )
    draw.text(
        (24, height + 64),
        "Same rest-selected cells, world coordinates, camera, scale, translucency and gold selected tetrahedron; deformation scale 1.",
        fill="white",
        font=small,
    )
    composite_path = out / "cell-patch.png"
    composite.save(composite_path)

    patch_tets = tets[patch_ids]
    patch_rest = positions[0][patch_tets]
    dm = np.swapaxes(patch_rest[:, 1:] - patch_rest[:, :1], 1, 2)
    dm_inv = np.linalg.inv(dm)
    weights = (
        np.abs(np.linalg.det(dm))
        / 6
        * np.asarray(ref.cell_data["MuscleFraction"])[patch_ids]
    )
    metrics = []
    selected_dm = (selected_points[0][1:] - selected_points[0][:1]).T
    for label, points, cell_points in zip(
        labels, positions, selected_points, strict=True
    ):
        deformed = points[patch_tets]
        F = np.swapaxes(deformed[:, 1:] - deformed[:, :1], 1, 2) @ dm_inv
        J = np.linalg.det(F)
        stretches = np.linalg.svd(F, compute_uv=False)
        deviation = np.linalg.norm(stretches - 1, axis=1)
        selected_F = (cell_points[1:] - cell_points[:1]).T @ np.linalg.inv(selected_dm)
        metrics.append(
            {
                "label": label,
                "selected_cell_detF": float(np.linalg.det(selected_F)),
                "selected_cell_principal_stretches_descending": np.linalg.svd(
                    selected_F, compute_uv=False
                ).tolist(),
                "patch_inverted_cells": int((J <= 0).sum()),
                "patch_muscle_volume_weighted_RMS_detF_minus_1": float(
                    np.sqrt(np.average((J - 1) ** 2, weights=weights))
                ),
                "patch_muscle_volume_weighted_RMS_stretch_deviation": float(
                    np.sqrt(np.average(deviation**2, weights=weights))
                ),
                "patch_unweighted_stretch_deviation_quantiles_50_95_100": np.quantile(
                    deviation, [0.5, 0.95, 1.0]
                ).tolist(),
            }
        )

    summary = {
        "status": "completed_saved_state_render",
        "scope": "Read-only rendering of existing saved geometries; no solve, fitting, interpolation, displacement amplification, or geometry smoothing",
        "global_cell_id": cfg.cell_id,
        "vertex_ids": selected_vertices.tolist(),
        "patch_cell_count": len(patch_ids),
        "patch_rule": "MuscleFraction > 0 and rest centroid within 6 mm of selected cell 27306 rest centroid",
        "selection": "Cell and neighborhood frozen before examining corrected or PSD deformation",
        "state_order": list(labels),
        "metrics": metrics,
        "camera": {
            "projection": "parallel",
            "orientation_source_position": CAMERA_POSITION.tolist(),
            "orientation_source_focal_point": CAMERA_FOCAL.tolist(),
            "view_up": CAMERA_UP.tolist(),
            "actual_patch_focal_point": patch_focal.tolist(),
            "parallel_scale": float(patch_scale),
            "freeze_rule": "Exact original REST/OLD/PSD focal and radius calculation; corrected state does not alter framing",
        },
        "rendering": {
            "geometry_smoothing": False,
            "deformation_scale": 1.0,
            "coordinates": "Exact saved world coordinates x = X + u",
            "centroid_normalization": False,
            "fixed_connectivity": True,
            "patch_opacity": 0.3,
            "selected_tetrahedron_color": GOLD,
            "panel_dimensions": [width, height],
            "composite_dimensions": [4 * width, height + 120],
        },
        "outputs": {
            "composite": record(composite_path),
            "panels": panel_records,
            "patch_source_cell_ids": record(out / "patch-source-cell-ids.npy"),
        },
        "sources": [
            record(path)
            for path in (
                FIXTURE,
                OLD,
                CORRECTED,
                PSD,
                SOURCE_RENDERER,
                SOURCE_RECEIPT,
                Path(__file__),
            )
        ],
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    cherries.log_metrics(
        {
            "report/patch_cells": len(patch_ids),
            "report/states": len(labels),
            "report/cell_id": cfg.cell_id,
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
