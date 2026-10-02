"""Render all transferred expressions and the new neutral on an 8K slide."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402

LOGGER = logging.getLogger(__name__)
BACKGROUND = "#f1f3f5"
INK = "#1e2935"
MUTED = "#64707d"
FONT_DIR = Path("/usr/share/fonts/TTF")
WIDTH, HEIGHT = 7680, 4320
TILE_WIDTH, TILE_HEIGHT = 656, 696


class Config(cherries.BaseConfig):
    bundle_dir: Path = GROUP / "data/blendshapes-005"
    output: Path = cherries.output("93-blendshape-slide-no-highlights", mkdir=False)


def font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / f"MonaSans-{weight}.ttf"), size)


def display_name(name: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name.replace("_", " "))


def wrapped_label(draw: ImageDraw.ImageDraw, label: str) -> list[str]:
    lines = [""]
    for word in label.split():
        candidate = f"{lines[-1]} {word}".strip()
        if draw.textlength(candidate, font=font(45, "Medium")) > TILE_WIDTH - 20:
            lines.append(word)
        else:
            lines[-1] = candidate
    assert len(lines) <= 2, lines
    return lines


def render_faces(
    points: np.ndarray, triangles: np.ndarray, output: Path
) -> tuple[list[Image.Image], dict]:
    low, high = points.min(axis=(0, 1)), points.max(axis=(0, 1))
    center = (low + high) / 2
    span = high - low
    aspect = TILE_WIDTH / TILE_HEIGHT
    scale = float(max(span[1] / 2, span[0] / (2 * aspect)) * 1.045)
    eye = center + np.array([0, 0, 3 * float(span.max())])
    camera_position = [eye.tolist(), center.tolist(), [0, 1, 0]]
    camera = {"position": camera_position, "parallel_scale_m": scale}
    faces = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    geometry_dir = output.parent / "geometry"
    geometry_dir.mkdir()
    surfaces = []
    for index, vertices in enumerate(points):
        surface = pv.PolyData(vertices, faces).compute_normals(
            cell_normals=False, point_normals=True, split_vertices=False
        )
        surface_path = geometry_dir / f"{index:02d}.vtp"
        surface.save(surface_path)
        surfaces.append(str(surface_path))
    spec_path = output.parent / "paraview-input.json"
    write_json(
        spec_path,
        {
            "camera": camera,
            "resolution": [TILE_WIDTH * 2, TILE_HEIGHT * 2],
            "background": [int(BACKGROUND[i : i + 2], 16) / 255 for i in (1, 3, 5)],
            "surfaces": surfaces,
            "output": str(output),
        },
    )
    subprocess.run(
        [
            "/usr/bin/pvpython",
            str(GROUP / "src/94-paraview-blendshape-faces.py"),
            str(spec_path),
        ],
        check=True,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    images = []
    for index in range(len(points)):
        image = (
            Image.open(output / f"{index:02d}.png")
            .convert("RGB")
            .resize((TILE_WIDTH, TILE_HEIGHT), Image.Resampling.LANCZOS)
        )
        images.append(image)
    return images, camera


def compose(images: list[Image.Image], ranking: list[dict]) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (164, 120), "Transferred blendshapes", font=font(154, "SemiBold"), fill=INK
    )
    draw.text(
        (174, 337),
        "New neutral  ·  36 expressions  ·  Largest deformation first",
        font=font(63),
        fill=MUTED,
    )
    draw.line((170, 495, 7510, 495), fill="#cfd5dc", width=3)

    # The neutral uses exactly the same viewport, camera and mesh scale as targets.
    neutral_center = 647
    draw.text(
        (neutral_center, 1190),
        "REFERENCE",
        anchor="mt",
        font=font(44, "SemiBold"),
        fill=MUTED,
    )
    draw.text(
        (neutral_center, 1290),
        "New neutral",
        anchor="mt",
        font=font(79, "SemiBold"),
        fill=INK,
    )
    canvas.paste(images[0], (neutral_center - TILE_WIDTH // 2, 1470))
    draw.line((310, 2400, 980, 2400), fill="#cfd5dc", width=3)
    for line, text in enumerate(
        ("Front view", "Identical camera and scale", "Expression weight = 1.0")
    ):
        draw.text(
            (neutral_center, 2490 + 75 * line),
            text,
            anchor="mt",
            font=font(43),
            fill=MUTED,
        )
    draw.line((1230, 645, 1230, 4040), fill="#cfd5dc", width=3)

    for index, entry in enumerate(ranking):
        row, col = divmod(index, 9)
        x, y = 1350 + col * 684, 590 + row * 885
        center_x = x + TILE_WIDTH // 2
        canvas.paste(images[index + 1], (x, y))
        draw.text((x + 16, y + 12), f"{index + 1:02d}", font=font(36), fill="#7c8792")
        lines = wrapped_label(draw, display_name(entry["expression"]))
        label_y = y + 714
        for line_index, line in enumerate(lines):
            draw.text(
                (center_x, label_y + 51 * line_index),
                line,
                anchor="mt",
                font=font(45, "Medium"),
                fill=INK,
            )
    return canvas


def main(cfg: Config) -> None:
    output = cfg.output.resolve()
    assert not output.exists(), output
    manifest_path = cfg.bundle_dir / "manifest.json"
    bundle_path = cfg.bundle_dir / "blendshapes.npz"
    manifest = json.loads(manifest_path.read_text())
    assert sha256(bundle_path) == manifest["artifacts"]["blendshapes.npz"]["sha256"]
    with np.load(bundle_path, allow_pickle=False) as archive:
        names = archive["expression_names"].tolist()
        neutral = archive["new_neutral_points_m"]
        targets = archive["target_points_m"]
        displacement = archive["expression_displacement_m"]
        triangles = archive["skin_triangles"]
    assert names == manifest["expression_names"]
    assert len(names) == 36
    assert np.isfinite(targets).all()
    assert np.isfinite(neutral).all()
    np.testing.assert_array_equal(targets, neutral[None] + displacement)
    rms = 1000 * np.sqrt(np.mean(np.sum(displacement**2, axis=2), axis=1))
    maximum = 1000 * np.linalg.norm(displacement, axis=2).max(axis=1)
    for name, value in zip(names, rms, strict=True):
        np.testing.assert_allclose(
            value, manifest["expression_statistics"][name]["skin_rms_mm"]
        )
    order = np.argsort(-rms, kind="stable")
    assert np.all(np.diff(rms[order]) <= 0)
    ranking = [
        {
            "rank": rank,
            "expression": names[index],
            "source_index": int(index),
            "rms_displacement_mm": float(rms[index]),
            "max_displacement_mm": float(maximum[index]),
        }
        for rank, index in enumerate(order, 1)
    ]
    output.mkdir(parents=True)
    renders = output / "faces"
    renders.mkdir()
    points = np.concatenate((neutral[None], targets[order]))
    images, camera = render_faces(points, triangles, renders)
    slide = compose(images, ranking)
    slide_path = output / "transferred-blendshapes-new-neutral-8k.png"
    slide.save(slide_path, dpi=(576, 576))
    preview_path = output / "transferred-blendshapes-new-neutral-preview.png"
    slide.resize((1920, 1080), Image.Resampling.LANCZOS).save(preview_path)
    assert Image.open(slide_path).size == (7680, 4320)
    ranking_path = output / "deformation-ranking.json"
    write_json(ranking_path, ranking)
    shutil.copy2(__file__, output / Path(__file__).name)
    shutil.copy2(GROUP / "src/94-paraview-blendshape-faces.py", output)
    write_json(
        output / "render-receipt.json",
        {
            "schema": "transferred-blendshape-slide-v1",
            "bundle": {"path": str(bundle_path), "sha256": sha256(bundle_path)},
            "manifest_sha256": sha256(manifest_path),
            "dimensions_px": [WIDTH, HEIGHT],
            "aspect_ratio": "16:9",
            "panels": 37,
            "grid": [4, 9],
            "neutral": "new_neutral_points_m; separate reference panel",
            "expression_weight": 1.0,
            "order": "descending RMS; left to right then top to bottom",
            "metric": "1000 * sqrt(mean_vertices(sum_xyz(expression_displacement_m ** 2)))",
            "mesh_color": "white",
            "renderer": json.loads((renders / "paraview.json").read_text()),
            "background": BACKGROUND,
            "camera": camera,
            "neutral_status": manifest["neutral_status"],
            "scope": manifest["scope"],
            "versions": {"pyvista": pv.__version__, "numpy": np.__version__},
            "outputs": {
                file.name: {"sha256": sha256(file), "bytes": file.stat().st_size}
                for file in (slide_path, preview_path, ranking_path)
            },
        },
    )
    cherries.log_input(manifest_path)
    cherries.log_output(output)
    cherries.log_metrics(
        {"expressions": 36, "panels": 37, "width_px": WIDTH, "height_px": HEIGHT}
    )
    LOGGER.info("Wrote %s", slide_path)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
