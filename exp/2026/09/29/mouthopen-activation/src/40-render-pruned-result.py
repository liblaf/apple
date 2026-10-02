# ruff: noqa: C901, EM101, EM102, PLR0915, TRY003
"""Render neutral, target, and accepted pruned-mouthopen states."""

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
TERMINAL_STATUSES = {
    "completed",
    "wall_budget_exhausted",
    "blocked_at_minimum_pose_step",
    "failed",
    "numerical_failure",
    "terminal_physical_or_force_gate_failed",
    "blocked_by_neutral_surface_intersections",
}


class Config(cherries.BaseConfig):
    source: Path = Path("10-mandible")
    fixture: Path = Path("30-pruned-fixture")
    forward: Path = Path("35-forward-pruned-002")
    output: Path = Path("40-pruned-comparison-002")


def record(path: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def rigid(points: np.ndarray, pivot: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (
        (points - pivot) @ Rotation.from_rotvec(pose[:3]).as_matrix().T
        + pivot
        + pose[3:]
    )


def camera_for(point_sets: list[np.ndarray]) -> dict[str, object]:
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


def render_panel(
    output: Path,
    camera: dict[str, object],
    cranium: pv.PolyData,
    eyes: pv.PolyData,
    jaw: pv.PolyData,
    skin: pv.PolyData,
) -> None:
    plot = pv.Plotter(off_screen=True, window_size=WINDOW)
    plot.set_background(BACKGROUND)
    plot.add_mesh(cranium, color="#eee8d6", smooth_shading=True)
    plot.add_mesh(jaw, color="#eee8d6", smooth_shading=True)
    plot.add_mesh(eyes, color="#f9f7ef", smooth_shading=True)
    plot.add_mesh(skin, color="#8d969b", smooth_shading=True)
    plot.camera.position = camera["position"]
    plot.camera.focal_point = camera["focal_point"]
    plot.camera.up = camera["up"]
    plot.camera.parallel_projection = True
    plot.camera.parallel_scale = camera["parallel_scale"]
    plot.reset_camera_clipping_range()
    plot.enable_anti_aliasing("ssaa")
    plot.screenshot(output)
    plot.close()


def main(cfg: Config) -> None:
    source = cherries.input(GROUP / "data" / cfg.source)
    fixture = cherries.input(GROUP / "data" / cfg.fixture)
    forward = cherries.input(GROUP / "data" / cfg.forward)
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)

    summary_path = forward / "summary.json"
    checkpoint_path = forward / "final.npz"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            "rendering requires a terminal forward summary and final.npz checkpoint"
        )
    summary = json.loads(summary_path.read_text())
    status = str(summary.get("status", "unknown"))
    if status not in TERMINAL_STATUSES:
        raise RuntimeError(f"forward run is not terminal (status={status!r})")

    with np.load(source / "prepared.npz", allow_pickle=False) as z:
        prepared = {name: z[name].copy() for name in z.files}
    with np.load(checkpoint_path, allow_pickle=False) as z:
        required = {"displacement", "pose", "fraction"}
        if not required.issubset(z.files):
            raise ValueError(
                f"final checkpoint is missing {sorted(required - set(z.files))}"
            )
        displacement = z["displacement"].copy()
        pose = z["pose"].copy()
        fraction = float(z["fraction"])
    if not np.isfinite(displacement).all() or not np.isfinite(pose).all():
        raise ValueError("final checkpoint contains nonfinite state")
    if pose.shape != (6,) or not 0.0 <= fraction <= 1.0:
        raise ValueError("final checkpoint pose or fraction is malformed")

    volume = pv.read(fixture / "volume.vtu")
    skin_mesh = pv.read(fixture / "skin.vtp")
    skin_ids = np.asarray(skin_mesh.point_data["GlobalPointId"], dtype=np.int64)
    if displacement.shape != (volume.n_points, 3):
        raise ValueError(
            f"checkpoint displacement shape {displacement.shape} does not match "
            f"fixture point count {volume.n_points}"
        )
    np.testing.assert_array_equal(skin_mesh.points, np.asarray(volume.points)[skin_ids])
    target_skin = prepared["target_skin"]
    triangles = prepared["triangles"]
    np.testing.assert_array_equal(
        triangles, np.asarray(skin_mesh.faces).reshape(-1, 4)[:, 1:]
    )
    if target_skin.shape != skin_mesh.points.shape:
        raise ValueError("prepared target skin does not match the fixture skin")

    cranium = pv.read(CRANIUM_PATH)
    eyes = pv.read(EYES_PATH)
    mandible = pv.read(MANDIBLE_PATH)
    pivot = prepared["pivot"]
    jaw_neutral = mandible.copy(deep=True)
    jaw_target = mandible.copy(deep=True)
    jaw_target.points = rigid(np.asarray(mandible.points), pivot, pose)
    jaw_accepted = mandible.copy(deep=True)
    jaw_accepted.points = rigid(np.asarray(mandible.points), pivot, fraction * pose)

    skin_neutral = skin_mesh.copy(deep=True)
    skin_target = skin_mesh.copy(deep=True)
    skin_target.points = target_skin
    skin_accepted = skin_mesh.copy(deep=True)
    skin_accepted.points = np.asarray(volume.points)[skin_ids] + displacement[skin_ids]
    camera = camera_for(
        [
            np.asarray(cranium.points),
            np.asarray(eyes.points),
            np.asarray(jaw_neutral.points),
            np.asarray(jaw_target.points),
            np.asarray(jaw_accepted.points),
            np.asarray(skin_neutral.points),
            np.asarray(skin_target.points),
            np.asarray(skin_accepted.points),
        ]
    )

    panel_paths = []
    for index, (jaw, surface) in enumerate(
        (
            (jaw_neutral, skin_neutral),
            (jaw_target, skin_target),
            (jaw_accepted, skin_accepted),
        )
    ):
        panel_path = output / f"panel-{index}.png"
        render_panel(panel_path, camera, cranium, eyes, jaw, surface)
        panel_paths.append(panel_path)

    page = Image.new("RGB", (4800, 2160), "#101315")
    draw = ImageDraw.Draw(page)
    fonts = Path("/usr/share/fonts/TTF")
    title_font = ImageFont.truetype(str(fonts / "DejaVuSans-Bold.ttf"), 94)
    heading_font = ImageFont.truetype(str(fonts / "DejaVuSans-Bold.ttf"), 57)
    small_font = ImageFont.truetype(str(fonts / "DejaVuSans.ttf"), 39)
    draw.text(
        (2400, 30),
        "MouthOpen: zero-activation forward continuation",
        font=title_font,
        fill="white",
        anchor="mt",
    )
    headings = [
        "Historical neutral",
        "Full MouthOpen target",
        f"Forward state: {fraction:.1%} pose",
    ]
    subtitles = [
        "Source skin and jaw; zero activation",
        "Transferred blendshape; chin-derived jaw pose",
        "Zero activation; continuation stopped early",
    ]
    for index, panel_path in enumerate(panel_paths):
        left = 20 + 1600 * index
        draw.text(
            (left + 780, 175),
            headings[index],
            font=heading_font,
            fill="white",
            anchor="mt",
        )
        draw.text(
            (left + 780, 259),
            subtitles[index],
            font=small_font,
            fill="#bdc7cb",
            anchor="mt",
        )
        with Image.open(panel_path) as panel:
            if panel.size != WINDOW:
                raise ValueError(f"unexpected rendered panel size {panel.size}")
            page.paste(panel, (left, 345))
    draw.text(
        (40, 2010),
        "Jaw pose is chin-derived. The forward panel is S=0 and has no activation fit or contact forces.",
        font=small_font,
        fill="#d7dfe2",
    )
    draw.text(
        (40, 2074),
        "A complete outer FEM surface was checked for self-intersection; separate bone contact was not modeled.",
        font=small_font,
        fill="#d7dfe2",
    )
    figure = output / "mouthopen-pruned-comparison.png"
    preview = output / "mouthopen-pruned-comparison-preview.png"
    page.save(figure)
    page.resize((1920, 864), Image.Resampling.LANCZOS).save(preview)
    (output / "source.py").write_text(Path(__file__).read_text())
    manifest = {
        "schema": "mouthopen-pruned-comparison-v1",
        "forward_status": status,
        "saved_pose_fraction": fraction,
        "valid_forward": bool(summary.get("valid_forward", False)),
        "inputs": {
            "prepared": record(source / "prepared.npz"),
            "forward_summary": record(summary_path),
            "forward_checkpoint": record(checkpoint_path),
            "volume": record(fixture / "volume.vtu"),
            "skin": record(fixture / "skin.vtp"),
        },
        "sources": [
            record(Path(__file__)),
            record(CRANIUM_PATH),
            record(MANDIBLE_PATH),
            record(EYES_PATH),
        ],
        "panels": [record(path) for path in panel_paths],
        "camera": camera,
        "figure": record(figure),
        "preview": record(preview),
        "panel_semantics": [
            "historical neutral source surface and neutral jaw",
            "MouthOpen kinematic target and full fitted jaw pose",
            "last accepted pruned-volume forward state and jaw at checkpoint pose",
        ],
        "activation": "all-zero native active-strain tensor S=0",
        "no_contact_forces": True,
        "surface_gate_scope": "complete extracted FEM boundary self-intersection; no separate bone obstacles or containment test",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
