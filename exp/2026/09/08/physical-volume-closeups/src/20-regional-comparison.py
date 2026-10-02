"""Compare corresponding rest, corrected, and target face regions."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "closeups", GROUP / "src/10-render-closeups.py"
)
assert spec is not None and spec.loader is not None
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
DONE = False


class Config(cherries.BaseConfig):
    output_dir: Path = cherries.output("20-regions", mkdir=True)
    resolution: int = 1000


def render(
    surfaces: list[pv.PolyData], view: dict, output: Path, resolution: int
) -> None:
    labels = ["Rest geometry", "Corrected baseline | step 200", "Target geometry"]
    cam = view["camera"]
    focus = np.array(cam["focal_point"])
    backward = np.array(cam["position"]) - focus
    backward /= np.linalg.norm(backward)
    right = np.cross([0, 1, 0], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    key = focus + 0.3 * (0.72 * right + 0.35 * up + 0.60 * backward)
    fill = focus + 0.3 * backward
    p = pv.Plotter(
        shape=(1, 3),
        off_screen=True,
        window_size=(3 * resolution, resolution),
        lighting="none",
        border=False,
    )
    for index, surface in enumerate(surfaces):
        p.subplot(0, index)
        actor = p.add_mesh(
            surface,
            color="#eeeeea",
            smooth_shading=False,
            ambient=0.20,
            diffuse=0.80,
            specular=0.0,
        )
        assert actor.GetProperty().GetInterpolation() == 0
        p.add_light(
            pv.Light(
                position=key,
                focal_point=focus,
                intensity=0.85,
                light_type="scene light",
                positional=False,
            ),
            only_active=True,
        )
        p.add_light(
            pv.Light(
                position=fill,
                focal_point=focus,
                intensity=0.20,
                light_type="scene light",
                positional=False,
            ),
            only_active=True,
        )
        assert len(p.renderer.lights) == 2
        p.enable_parallel_projection()
        p.camera.position = cam["position"]
        p.camera.focal_point = cam["focal_point"]
        p.camera.up = cam["view_up"]
        p.camera.parallel_scale = cam["parallel_scale"]
        p.reset_camera_clipping_range()
        p.set_background(c.BACKGROUND)
    raster = p.screenshot(return_img=True)
    p.close()
    image = Image.new("RGB", (3 * resolution, resolution + 130), c.BACKGROUND)
    image.paste(Image.fromarray(raster).convert("RGB"), (0, 90))
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    font = ImageFont.truetype("DejaVuSans.ttf", 21)
    for i, label in enumerate(labels):
        draw.text((i * resolution + 20, 12), label, fill="white", font=title)
        draw.text((i * resolution + 20, 51), view["label"], fill="#c7d0da", font=font)
        draw.text(
            (i * resolution + 20, resolution + 99),
            "Same camera and lights | actual geometry",
            fill="#c7d0da",
            font=font,
        )
    image.save(output)


def main(cfg: Config) -> None:
    global DONE
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), out
    current, evidence = c.load_surface()
    ids = np.array(current["VolumePointIndex"])
    ref = pv.read(c.REFERENCE)
    target_u = np.array(ref["Smile"])
    rest = current.copy(deep=True)
    rest.points = np.array(ref.points[ids])
    target = current.copy(deep=True)
    assert np.isfinite(target_u[ids]).all()
    target.points = ref.points[ids] + target_u[ids]
    surfaces = [rest, current, target]
    views = [
        {
            "id": "side-context",
            "label": "Oblique face context",
            "camera": c.camera([1.425, 2.202, 0.047], [1, 0.05, 0.72], 0.105),
        },
        {
            "id": "region1-mouth-corner",
            "label": "Region 1 | mouth corner",
            "camera": c.camera([1.440, 2.162, 0.073], [1, 0.02, 0.72], 0.024),
        },
        {
            "id": "region2-lateral-cheek",
            "label": "Region 2 | lateral cheek",
            "camera": c.camera([1.463, 2.166, 0.033], [1, 0.02, 0.72], 0.032),
        },
        {
            "id": "region3-lower-cheek",
            "label": "Region 3 | lower cheek and jaw",
            "camera": c.camera([1.455, 2.146, 0.043], [1, 0.02, 0.72], 0.023),
        },
        {
            "id": "nasolabial-region",
            "label": "Nose-to-mouth region",
            "camera": c.camera([1.430, 2.181, 0.081], [0.75, 0.02, 1], 0.034),
        },
    ]
    for view in views:
        render(surfaces, view, out / (view["id"] + ".png"), cfg.resolution)
        logging.info("Rendered %s", view["id"])
    for name, surface in zip(["rest", "corrected", "target"], surfaces, strict=True):
        surface.save(out / (name + "-skin.vtp"))
    (out / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed_saved_state_comparison",
                **evidence,
                "source": c.record(Path(__file__)),
                "helper": c.record(GROUP / "src/10-render-closeups.py"),
                "views": views,
                "target_field": "Smile",
                "same_topology": True,
                "geometry_smoothing": False,
                "deformation_scale": 1,
                "selection_note": "Camera centers approximate the three anatomical regions marked in the user screenshot; no exact screen-to-world registration is claimed.",
                "outputs": [
                    c.record(p) for p in out.iterdir() if p.name != "summary.json"
                ],
            },
            indent=2,
        )
        + "\n"
    )
    for path in out.iterdir():
        cherries.log_output(path)
    DONE = True


if __name__ == "__main__":
    cherries.main(
        main, profile="debug" if os.environ.get("DEBUG") else c.ProfileCometNoCommit
    )
    if not DONE:
        raise SystemExit(1)
