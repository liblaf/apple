"""Render the MouthOpen target, chin jaw estimate, and prescribed-cell conflict."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
OLD_SRC = ROOT / "exp/2026/09/21/stress-activation-loss/src"
sys.path.insert(0, str(OLD_SRC))
from experiment import Profile  # noqa: E402
from shape_scene import CRANIUM_PATH, EYES_PATH, MANDIBLE_PATH  # noqa: E402

WINDOW = (1560, 1620)
BACKGROUND = "#f4f2ed"


class Config(cherries.BaseConfig):
    source: Path = Path("10-mandible")
    output: Path = Path("20-preflight")


def record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def rigid(points: np.ndarray, pivot: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (
        (points - pivot) @ Rotation.from_rotvec(pose[:3]).as_matrix().T
        + pivot
        + pose[3:]
    )


def camera_for(point_sets: list[np.ndarray]) -> dict:
    toward = np.array([0.65, 0.03, 1.0])
    toward /= np.linalg.norm(toward)
    right = np.cross([0.0, 1.0, 0.0], toward)
    right /= np.linalg.norm(right)
    up = np.cross(toward, right)
    basis = np.column_stack((right, up, toward))
    projected = np.vstack(point_sets) @ basis
    low, high = projected.min(axis=0), projected.max(axis=0)
    center = basis @ ((low + high) / 2)
    half = (high - low) / 2
    return {
        "position": (center + 0.6 * toward).tolist(),
        "focal_point": center.tolist(),
        "up": up.tolist(),
        "parallel_scale": float(1.05 * max(half[1], half[0] / (WINDOW[0] / WINDOW[1]))),
    }


def main(cfg: Config) -> None:  # noqa: PLR0915
    source = cherries.input(cfg.source)
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(source / "prepared.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    points = arrays["X"]
    ids = arrays["skin_ids"]
    triangles = arrays["triangles"]
    target = arrays["target_skin"]
    assert target.shape == (len(ids), 3)
    pivot, pose = arrays["pivot"], arrays["pose"]
    jaw = pv.read(MANDIBLE_PATH)
    cranium = pv.read(CRANIUM_PATH)
    eyes = pv.read(EYES_PATH)
    posed_jaw = jaw.copy(deep=True)
    posed_jaw.points = rigid(np.asarray(jaw.points), pivot, pose)
    faces = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    skin = pv.PolyData(points[ids], faces)
    target_skin = pv.PolyData(target, faces)
    bad_ids = arrays["inverted_allfixed_cell_ids"]
    cells = np.column_stack((np.full(len(bad_ids), 4), arrays["tets"][bad_ids]))
    bad = pv.UnstructuredGrid(
        cells.ravel(),
        np.full(len(bad_ids), pv.CellType.TETRA, dtype=np.uint8),
        points + arrays["boundary_u"],
    ).extract_surface(algorithm=None)
    camera = camera_for(
        [skin.points, target, cranium.points, jaw.points, posed_jaw.points]
    )
    panel_paths = []
    for index in range(3):
        plot = pv.Plotter(off_screen=True, window_size=WINDOW)
        plot.set_background(BACKGROUND)
        if index < 2:
            plot.add_mesh(cranium, color="#eee8d6", smooth_shading=True)
            plot.add_mesh(
                jaw if index == 0 else posed_jaw, color="#eee8d6", smooth_shading=True
            )
            plot.add_mesh(eyes, color="#f9f7ef", smooth_shading=True)
            plot.add_mesh(
                skin if index == 0 else target_skin,
                color="#8d969b",
                smooth_shading=True,
            )
            patch = arrays["patch_ids"]
            patch_points = points[ids][patch] if index == 0 else target[patch]
            plot.add_mesh(
                pv.PolyData(patch_points),
                color="#20b7b2",
                point_size=8,
                render_points_as_spheres=True,
            )
        else:
            plot.add_mesh(cranium, color="#c2bcb2", opacity=0.18)
            plot.add_mesh(posed_jaw, color="#7b8f9b", opacity=0.30)
            plot.add_mesh(bad, color="#ca3935", show_edges=True, edge_color="#7c2427")
        plot.camera.position = camera["position"]
        plot.camera.focal_point = camera["focal_point"]
        plot.camera.up = camera["up"]
        plot.camera.parallel_projection = True
        plot.camera.parallel_scale = camera["parallel_scale"]
        plot.reset_camera_clipping_range()
        plot.enable_anti_aliasing("ssaa")
        path = output / f"panel-{index}.png"
        plot.screenshot(path)
        plot.close()
        panel_paths.append(path)
    page = Image.new("RGB", (4800, 2160), "#101315")
    draw = ImageDraw.Draw(page)
    fonts = Path("/usr/share/fonts/TTF")
    title = ImageFont.truetype(str(fonts / "DejaVuSans-Bold.ttf"), 94)
    heading = ImageFont.truetype(str(fonts / "DejaVuSans-Bold.ttf"), 57)
    small = ImageFont.truetype(str(fonts / "DejaVuSans.ttf"), 39)
    draw.text(
        (2400, 30),
        "MouthOpen: mandible pose preflight",
        font=title,
        fill="white",
        anchor="mt",
    )
    headings = [
        "Historical neutral",
        "MouthOpen target + jaw estimate",
        "Conflicting prescribed motion",
    ]
    subtitles = [
        "Chin patch in cyan",
        "Kinematic target; no activation fit",
        f"{len(bad_ids)} inverted, fully fixed tetrahedra",
    ]
    for i, path in enumerate(panel_paths):
        left = 20 + 1600 * i
        draw.text(
            (left + 780, 175), headings[i], font=heading, fill="white", anchor="mt"
        )
        draw.text(
            (left + 780, 259), subtitles[i], font=small, fill="#bdc7cb", anchor="mt"
        )
        with Image.open(path) as panel:
            assert panel.size == WINDOW
            page.paste(panel, (left, 345))
    draw.text(
        (40, 2010),
        "Chin alignment estimates jaw motion from skin. It is not a measured bone pose or an equilibrium solution.",
        font=small,
        fill="#d7dfe2",
    )
    draw.text(
        (40, 2074),
        "IsFixed is preserved. Jaw motion applies only to IsFixed intersect Mandible; free-node activation cannot repair red cells.",
        font=small,
        fill="#d7dfe2",
    )
    full = output / "mouthopen-preflight.png"
    preview = output / "mouthopen-preflight-preview.png"
    page.save(full)
    page.resize((1920, 864), Image.Resampling.LANCZOS).save(preview)
    manifest = {
        "prepared": record(source / "prepared.npz"),
        "sources": [
            record(p) for p in [Path(__file__), CRANIUM_PATH, MANDIBLE_PATH, EYES_PATH]
        ],
        "panels": [record(p) for p in panel_paths],
        "camera": camera,
        "figure": record(full),
        "preview": record(preview),
        "target_is_observed_kinematic_surface_not_simulated": True,
        "jaw_pose_is_chin_derived_initialization_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
