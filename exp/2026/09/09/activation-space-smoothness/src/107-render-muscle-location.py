"""Locate the fixed section-3 muscle neighborhood on an oblique rest overview."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from vtkmodules.vtkRenderingCore import vtkActor2D, vtkPolyDataMapper2D

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
FIXTURE = GROUP.parents[1] / "07/face-actuation-diagnosis/data/12-historical-fixture"
PATCH = GROUP / "data/93-focused-muscle-patch"
RENDERER = GROUP / "src/81-render-idea-shapes.py"
SIZE = 1800
PADDING = 18


class Config(cherries.BaseConfig):
    output_dir: Path = cherries.output("107-muscle-location", mkdir=True)


def record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def main(cfg: Config) -> None:
    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)
    assert not any(output.iterdir()), output
    patch_summary = json.loads((PATCH / "summary.json").read_text())
    patch_path = PATCH / "patch-source-cell-ids.npy"
    assert (
        record(patch_path)["sha256"]
        == patch_summary["outputs"]["patch_source_cell_ids"]["sha256"]
    )
    volume = pv.read(FIXTURE / "volume.vtu")
    skin = pv.read(FIXTURE / "skin.vtp")
    patch_ids = np.load(patch_path)
    assert len(patch_ids) == 694
    cells = np.asarray(volume.cells).reshape(-1, 5)
    assert np.all(cells[:, 0] == 4)
    tets = cells[:, 1:]
    centers = volume.points[tets].mean(axis=1)
    selected = np.flatnonzero(
        (volume.cell_data["MuscleFraction"] > 0)
        & (np.linalg.norm(centers - centers[27306], axis=1) <= 0.006)
    )
    np.testing.assert_array_equal(patch_ids, selected)
    point_ids = np.unique(tets[patch_ids])
    points = volume.points[point_ids]
    np.testing.assert_array_equal(
        skin.points, volume.points[skin.point_data["GlobalPointId"]]
    )
    baseline_path = GROUP / "data/97-baseline-oblique/summary.json"
    baseline = json.loads(baseline_path.read_text())
    camera = baseline["camera"]
    spec = importlib.util.spec_from_file_location("study81_shape_renderer", RENDERER)
    assert spec is not None and spec.loader is not None
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    plotter = pv.Plotter(off_screen=True, window_size=(SIZE, SIZE), lighting="none")
    plotter.add_mesh(
        skin,
        color=renderer.SHAPE_COLOR,
        smooth_shading=False,
        ambient=0.20,
        diffuse=0.80,
        specular=0.0,
    )
    renderer._add_lights(plotter, camera)
    renderer._set_camera(plotter, camera)
    plotter.show(auto_close=False, interactive=False)
    pixels = []
    for point in points:
        plotter.renderer.SetWorldPoint(*point, 1.0)
        plotter.renderer.WorldToDisplay()
        pixels.append(plotter.renderer.GetDisplayPoint()[:2])
    pixels = np.asarray(pixels)
    minimum = pixels.min(axis=0) - PADDING
    maximum = pixels.max(axis=0) + PADDING
    assert np.all(minimum > 0) and np.all(maximum < SIZE)
    xmin, ymin = minimum
    xmax, ymax = maximum
    box = pv.PolyData(
        np.array([[xmin, ymin, 0], [xmax, ymin, 0], [xmax, ymax, 0], [xmin, ymax, 0]])
    )
    box.lines = np.array([5, 0, 1, 2, 3, 0])
    for color, width in (((1.0, 1.0, 1.0), 11), ((0.82, 0.48, 0.02), 6)):
        mapper = vtkPolyDataMapper2D()
        mapper.SetInputData(box)
        actor = vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(width)
        plotter.renderer.AddActor2D(actor)
    title = plotter.add_text(
        "Muscle patch location · rest configuration",
        position="upper_left",
        font_size=13,
        color="black",
    )
    title.GetTextProperty().SetBackgroundColor(244 / 255, 242 / 255, 237 / 255)
    title.GetTextProperty().SetBackgroundOpacity(0.9)
    label = plotter.add_text(
        "694 tetrahedra",
        position=(float(xmin), float(ymax + 20)),
        font_size=16,
        color="black",
    )
    label.GetTextProperty().SetBackgroundColor(244 / 255, 242 / 255, 237 / 255)
    label.GetTextProperty().SetBackgroundOpacity(0.95)
    image_path = output / "overview.png"
    plotter.screenshot(image_path)
    plotter.close()
    np.savez(
        output / "projection.npz",
        cell_ids=patch_ids,
        point_ids=point_ids,
        rest_points=points,
        display_pixels=pixels,
        padded_minimum=minimum,
        padded_maximum=maximum,
    )
    receipt = {
        "status": "completed_reference_location_render",
        "scope": "Reference geometry rendering with a screen-space locator; no fitting or geometry alteration",
        "reference_configuration": "rest",
        "patch_cell_count": len(patch_ids),
        "patch_vertex_count": len(point_ids),
        "patch_rule": patch_summary["patch_rule"],
        "camera": camera,
        "style": baseline["style"],
        "box": {
            "meaning": "Projected bounds of all vertices of the exact 694-cell neighborhood",
            "padding_pixels": PADDING,
            "display_origin": "bottom left",
            "minimum": minimum.tolist(),
            "maximum": maximum.tolist(),
            "image_bounds_left_top_right_bottom": [
                float(xmin),
                float(SIZE - ymax),
                float(xmax),
                float(SIZE - ymin),
            ],
        },
        "outputs": [record(image_path), record(output / "projection.npz")],
        "sources": [
            record(path)
            for path in (
                FIXTURE / "volume.vtu",
                FIXTURE / "skin.vtp",
                patch_path,
                PATCH / "summary.json",
                baseline_path,
                RENDERER,
                Path(__file__),
            )
        ],
    }
    (output / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    cherries.log_metrics(
        {"report/patch_cells": len(patch_ids), "report/patch_vertices": len(point_ids)}
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
