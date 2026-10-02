# Copyright (c) 2026 liblaf
# ruff: noqa: E402, EM101, EM102, PLR0915, TRY003
"""Render fixed-only boundary geometry at neutral and prescribed MouthOpen."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
PARENT = ROOT / "exp/2026/09/29/mouthopen-activation"
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))

from experiment import Profile

PAGE_SIZE = (1600, 900)
PANEL_SIZE = (780, 610)
FONT_DIR = Path("/usr/share/fonts/TTF")
BACKGROUND = "#101315"
PANEL_BACKGROUND = "#f6f5f1"
COLORS = {
    "cranium": "#5c8fa8",
    "mandible": "#d46a4b",
    "mixed_or_other": "#d4aa42",
}
POSE_INTERSECTION_FRACTION = 0.05160293579101563


class Config(cherries.BaseConfig):
    output: Path = Path("30-prescribed-boundary")
    fixture: Path = PARENT / "data/30-pruned-fixture/volume.vtu"
    pose: Path = PARENT / "data/10-mandible/prepared.npz"
    audit: Path = Path("18-prescribed-boundary/summary.json")


def digest(path: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return {
        "path": str(path.resolve()),
        "sha256": hasher.hexdigest(),
        "bytes": path.stat().st_size,
    }


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "-Bold" if bold else ""
    return ImageFont.truetype(str(FONT_DIR / f"DejaVuSans{suffix}.ttf"), size)


def posed_points(
    points: np.ndarray, jaw: np.ndarray, pivot: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    output = points.copy()
    output[jaw] = (
        (points[jaw] - pivot) @ Rotation.from_rotvec(pose[:3]).as_matrix().T
        + pivot
        + pose[3:]
    )
    return output


def set_camera(plotter: pv.Plotter, camera: dict[str, Any]) -> None:
    plotter.camera_position = (
        camera["position"],
        camera["focal_point"],
        camera["view_up"],
    )
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = camera["parallel_scale"]


def render_panel(
    points: np.ndarray,
    triangles: np.ndarray,
    groups: np.ndarray,
    camera: dict[str, Any],
    path: Path,
) -> np.ndarray:
    plotter = pv.Plotter(
        off_screen=True, window_size=PANEL_SIZE, lighting="three lights"
    )
    plotter.set_background(PANEL_BACKGROUND)
    plotter.ren_win.SetMultiSamples(8)
    for group, color in COLORS.items():
        selected = groups == group
        if not selected.any():
            continue
        faces = triangles[selected]
        packed = np.column_stack(
            (np.full(len(faces), 3, dtype=np.int64), faces)
        ).ravel()
        surface = pv.PolyData(points, packed)
        plotter.add_mesh(
            surface,
            color=color,
            smooth_shading=False,
            opacity=1.0,
            ambient=0.24,
            diffuse=0.76,
            specular=0.06,
            show_edges=False,
        )
    set_camera(plotter, camera)
    screenshot = plotter.screenshot(str(path), return_img=True)
    plotter.close()
    return screenshot


def camera_for(points: np.ndarray, *, aspect: float) -> dict[str, Any]:
    direction = np.array([0.65, 0.015, 1.0], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    right = np.cross([0.0, 1.0, 0.0], direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    basis = np.column_stack((right, up, direction))
    projected = points @ basis
    low, high = projected.min(axis=0), projected.max(axis=0)
    center = basis @ ((low + high) / 2)
    half = (high - low) / 2
    scale = 1.07 * max(float(half[1]), float(half[0]) / aspect)
    return {
        "position": (center + 0.8 * direction).tolist(),
        "focal_point": center.tolist(),
        "view_up": up.tolist(),
        "parallel_scale": scale,
        "projected_bounds_m": [low.tolist(), high.tolist()],
        "projection": "shared orthographic camera fitted to the neutral and full-pose prescribed boundary",
    }


def main(cfg: Config) -> None:
    fixture_path = cherries.input(cfg.fixture)
    pose_path = cherries.input(cfg.pose)
    audit_path = cherries.input(cfg.audit)
    output_path = cherries.output(cfg.output / "prescribed-boundary.png", mkdir=True)
    manifest_path = cherries.output(cfg.output / "manifest.json", mkdir=True)
    if manifest_path.exists():
        raise FileExistsError(manifest_path)

    audit = json.loads(audit_path.read_text())
    if audit["schema"] != "prescribed-fem-boundary-collision-audit-v1":
        raise ValueError("unexpected prescribed-boundary audit schema")
    if not audit["endpoints"][-1]["has_intersections"]:
        raise ValueError("saved audit no longer reports full-pose intersections")
    if audit["endpoints"][0]["has_intersections"]:
        raise ValueError("saved audit no longer reports a clear neutral endpoint")
    np.testing.assert_allclose(
        audit["first_endpoint_intersection_bracket"]["first_detected_fraction"],
        POSE_INTERSECTION_FRACTION,
        rtol=0,
        atol=1e-12,
    )
    for source, expected in audit["sources_sha256"].items():
        if digest(Path(source))["sha256"] != expected:
            raise ValueError(f"prescribed-boundary audit receipt mismatch: {source}")

    volume = pv.read(fixture_path)
    rest = np.asarray(volume.points, dtype=np.float64).copy()
    fixed = np.asarray(volume.point_data["IsFixed"], dtype=bool)
    group_id = np.asarray(volume.point_data["GroupId"], dtype=np.int64)
    names = [str(value) for value in np.asarray(volume.field_data["GroupName"]).ravel()]
    cranium_id, mandible_id = names.index("Cranium"), names.index("Mandible")
    jaw = fixed & (group_id == mandible_id)
    np.testing.assert_array_equal(
        volume.point_data["GlobalPointId"], np.arange(volume.n_points)
    )

    extracted = volume.extract_surface(algorithm=None, pass_pointid=True)
    point_ids = np.asarray(extracted.point_data["vtkOriginalPointIds"], dtype=np.int64)
    packed = np.asarray(extracted.faces, dtype=np.int64).reshape(-1, 4)
    if not np.all(packed[:, 0] == 3):
        raise ValueError("the extracted FEM boundary contains non-triangle faces")
    global_faces = point_ids[packed[:, 1:]]
    chosen = global_faces[fixed[global_faces].all(axis=1)]
    vertex_ids, inverse = np.unique(chosen, return_inverse=True)
    triangles = inverse.reshape(-1, 3)
    local_rest = rest[vertex_ids]
    local_jaw = jaw[vertex_ids]
    face_group_ids = group_id[chosen]
    groups = np.full(len(chosen), "mixed_or_other", dtype=object)
    groups[np.all(face_group_ids == cranium_id, axis=1)] = "cranium"
    groups[np.all(face_group_ids == mandible_id, axis=1)] = "mandible"
    np.testing.assert_equal(len(chosen), audit["boundary"]["selected_faces"])
    np.testing.assert_equal(len(vertex_ids), audit["boundary"]["selected_vertices"])
    np.testing.assert_equal(
        int(local_jaw.sum()), audit["boundary"]["selected_jaw_vertices"]
    )
    face_counts = {key: int(np.count_nonzero(groups == key)) for key in COLORS}
    audit_group_keys = {
        "cranium": "cranium_pure",
        "mandible": "mandible_pure",
        "mixed_or_other": "mixed_or_other",
    }
    for key, count in face_counts.items():
        np.testing.assert_equal(
            count, audit["boundary"]["face_groups"][audit_group_keys[key]]
        )

    with np.load(pose_path, allow_pickle=False) as archive:
        pose, pivot = archive["pose"].copy(), archive["pivot"].copy()
    full_points = posed_points(local_rest, local_jaw, pivot, pose)
    camera = camera_for(
        np.concatenate((local_rest, full_points)), aspect=PANEL_SIZE[0] / PANEL_SIZE[1]
    )

    temporary = output_path.parent
    neutral_png = temporary / "temporary-neutral.png"
    posed_png = temporary / "temporary-mouthopen.png"
    neutral = render_panel(local_rest, triangles, groups, camera, neutral_png)
    mouthopen = render_panel(full_points, triangles, groups, camera, posed_png)
    neutral_png.unlink()
    posed_png.unlink()

    page = Image.new("RGB", PAGE_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(page)
    draw.text(
        (800, 22),
        "Prescribed-only FEM boundary",
        font=font(34, bold=True),
        fill="white",
        anchor="mt",
    )
    draw.text(
        (400, 84), "Neutral pose", font=font(24, bold=True), fill="#e5e5e5", anchor="mt"
    )
    draw.text(
        (1200, 84),
        "Full prescribed MouthOpen pose",
        font=font(24, bold=True),
        fill="#e5e5e5",
        anchor="mt",
    )
    page.paste(Image.fromarray(neutral), (20, 122))
    page.paste(Image.fromarray(mouthopen), (800, 122))

    y = 755
    x = 35
    for group, color in COLORS.items():
        draw.rounded_rectangle((x, y, x + 22, y + 22), radius=3, fill=color)
        draw.text(
            (x + 32, y - 2),
            f"{group.replace('_', ' ')} · {face_counts[group]:,} faces",
            font=font(18),
            fill="#f0f0f0",
        )
        x += 325
    draw.text(
        (35, 798),
        "Only boundary triangles with all three vertices IsFixed are shown. IsFixed ∩ Mandible receives the saved rigid pose; other fixed vertices stay still.",
        font=font(17),
        fill="#d4d4d4",
    )
    draw.text(
        (35, 830),
        f"First detected endpoint intersection: {POSE_INTERSECTION_FRACTION * 100:.4f}% pose. Full pose intersects. No free-tissue equilibrium is shown or claimed.",
        font=font(18, bold=True),
        fill="#f0b6a4",
    )
    page.save(output_path)
    page.close()

    manifest = {
        "schema": "prescribed-boundary-comparison-render-v1",
        "status": "complete",
        "purpose": "visualize fixed-only FEM boundary geometry at neutral and full prescribed MouthOpen pose; no free-tissue solve",
        "figure": digest(output_path),
        "inputs": [digest(path) for path in (fixture_path, pose_path, audit_path)],
        "source": digest(Path(__file__)),
        "boundary": {
            "policy": "all extracted surface triangles with three IsFixed vertices",
            "triangles": len(chosen),
            "vertices": len(vertex_ids),
            "face_groups": face_counts,
            "prescribed_vertices": int(local_jaw.sum()),
            "fixed_other_vertices_stationary": True,
        },
        "pose": {"rotvec_translation": pose.tolist(), "pivot_m": pivot.tolist()},
        "intersection": {
            "first_detected_pose_fraction": POSE_INTERSECTION_FRACTION,
            "full_pose_intersects": True,
            "source": str(audit_path.resolve()),
            "free_tissue_equilibrium_claim": False,
        },
        "camera": camera,
        "image_size": list(PAGE_SIZE),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    cherries.log_metrics(
        {
            "boundary/faces": len(chosen),
            "boundary/vertices": len(vertex_ids),
            "pose/first_intersection_fraction": POSE_INTERSECTION_FRACTION,
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
