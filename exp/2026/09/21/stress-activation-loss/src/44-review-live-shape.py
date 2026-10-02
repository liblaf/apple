"""Render a frozen live checkpoint beside the earlier corrected active-strain fit."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from experiment import Profile

from liblaf import cherries

REPO = Path(__file__).resolve().parents[6]
PREVIOUS = (
    REPO / "exp/2026/09/14/dominant-activation-ablation/data/five-deformed-meshes"
)
CAMERAS = (
    REPO
    / "exp/2026/09/07/tensor-active-stress/data/107-learning-rate-render/viewer-manifest.json"
)


class Config(cherries.BaseConfig):
    source: Path = Path("41-l2-unrestricted-inverse-v2-lr0p05-nosmooth-001")
    output: Path = Path("44-live-shape-review-001")


def main(cfg: Config) -> None:
    source = cherries.input(cfg.source)
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    blob = (source / "l2-symmetric6/last.npz").read_bytes()
    with (
        np.load(io.BytesIO(blob), allow_pickle=False) as state,
        np.load(source / "mesh.npz", allow_pickle=False) as mesh,
    ):
        step = int(state["step"])
        ids, triangles = mesh["skin_ids"], mesh["triangles"]
        rest = mesh["rest_points"][ids]
        target = rest + mesh["target_displacement_skin"]
        fit = rest + state["u"][ids]
        weights = mesh["skin_vertex_weights"]
    previous = pv.read(PREVIOUS / "01-free-activation.vtp")
    index = {int(value): i for i, value in enumerate(previous["GlobalPointId"])}
    order = np.array([index[int(value)] for value in ids])
    old = previous.points[order]
    np.testing.assert_allclose(
        previous["TargetPosition"][order], target, atol=1e-12, rtol=0
    )
    rows = list(csv.DictReader((source / "l2-symmetric6/trace.csv").open()))
    row = next(row for row in rows if int(row["step"]) == step)
    faces = np.column_stack([np.full(len(triangles), 3), triangles]).ravel()
    cameras = json.loads(CAMERAS.read_text())["cameras"]
    panels = [
        ("Smile target", target),
        ("Earlier corrected active strain | step 200", old),
        (f"Current unrestricted stress | step {step}", fit),
    ]
    plotter = pv.Plotter(
        shape=(2, 3), off_screen=True, window_size=(2100, 1500), lighting="three lights"
    )
    for r, view in enumerate(["three_quarter", "mouth"]):
        camera = cameras[view]
        for c, (label, points) in enumerate(panels):
            plotter.subplot(r, c)
            plotter.set_background("#f4f2ed")
            plotter.add_mesh(
                pv.PolyData(points, faces), color="#b8c2c5", smooth_shading=False
            )
            plotter.add_text(label, font_size=12, color="#16222b")
            plotter.enable_parallel_projection()
            plotter.camera.position = camera["position"]
            plotter.camera.focal_point = camera["focal_point"]
            plotter.camera.up = camera["view_up"]
            plotter.camera.parallel_scale = camera["parallel_scale"]
            plotter.reset_camera_clipping_range()
    plotter.screenshot(out / "comparison.png")
    plotter.close()
    np.savez_compressed(
        out / "surfaces.npz",
        rest=rest,
        target=target,
        previous=old,
        current=fit,
        triangles=triangles,
        global_ids=ids,
    )
    metrics = {}
    for label, points in panels:
        delta = points - target
        neighbors = triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
        neighbors = np.unique(np.sort(neighbors, axis=1), axis=0)
        accumulated = np.zeros_like(delta)
        counts = np.zeros(len(delta))
        for a, b in ((0, 1), (1, 0)):
            np.add.at(accumulated, neighbors[:, a], delta[neighbors[:, b]])
            np.add.at(counts, neighbors[:, a], 1)
        local_residual = delta - accumulated / counts[:, None]
        metrics[label] = {
            "position_rms_mm": float(
                1000 * np.sqrt(np.sum(weights * np.sum(delta**2, axis=1)))
            ),
            "neighbor_residual_rms_mm": float(
                1000 * np.sqrt(np.mean(np.sum(local_residual**2, axis=1)))
            ),
            "neighbor_residual_max_mm": float(
                1000 * np.max(np.linalg.norm(local_residual, axis=1))
            ),
        }
    report = {
        "checkpoint_step": step,
        "checkpoint_sha256": hashlib.sha256(blob).hexdigest(),
        "source": str(source.resolve()),
        "previous_surface": str((PREVIOUS / "01-free-activation.vtp").resolve()),
        "target_agreement_atol_m": 1e-12,
        "current_trace_row": row,
        "metrics": metrics,
        "render": "same cameras, flat shading, no geometry smoothing; software OpenGL",
        "comparison_limit": "different iteration budgets and optimizer/solver settings; descriptive comparison",
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    cherries.log_metrics({"checkpoint_step": step, "shape_metrics": metrics})


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
