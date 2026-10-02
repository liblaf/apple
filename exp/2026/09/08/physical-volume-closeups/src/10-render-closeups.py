"""Render exact step-200 face close-ups with flat shading and raking light."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pyvista as pv
from liblaf.cherries import core, plugins, profiles
from PIL import Image, ImageDraw, ImageFont

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]
BASELINE = (
    Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
    / "exp/2026/09/08/physical-volume-baseline"
)
REFERENCE = (
    ROOT
    / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture/volume.vtu"
)
SAVED = BASELINE / "data/20-baseline"
RECEIPT = BASELINE / "data/30-comparison/summary.json"
BACKGROUND = "#242c36"
DONE = False


class Config(cherries.BaseConfig):
    output_dir: Path = cherries.output("10-closeups", mkdir=True)
    resolution: int = 1400


class ProfileCometNoCommit(profiles.Profile):
    def init(self) -> core.Run:
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


def record(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": h.hexdigest(),
    }


def array_hash(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(array.dtype.str.encode())
    h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    h.update(array.tobytes())
    return h.hexdigest()


def load_surface() -> tuple[pv.PolyData, dict]:
    receipt = json.loads(RECEIPT.read_text())
    inputs = {}
    for path in (
        REFERENCE,
        SAVED / "final.npz",
        SAVED / "final.vtu",
        SAVED / "summary.json",
    ):
        actual = record(path)
        assert actual == receipt["inputs"][str(path)], path
        inputs[str(path)] = actual
    summary = json.loads((SAVED / "summary.json").read_text())
    assert summary["best_step"] == summary["last_evaluated_step"] == 200
    assert summary["failure"] is None
    reference = pv.read(REFERENCE)
    with np.load(SAVED / "final.npz", allow_pickle=False) as saved:
        assert int(saved["step"]) == 200 and bool(saved["solver_valid"])
        assert np.array_equal(saved["rest_points"], reference.points)
        points = saved["rest_points"] + saved["u"]
    final = pv.read(SAVED / "final.vtu")
    assert np.array_equal(final.cells, reference.cells)
    assert np.allclose(final.points, points, rtol=0, atol=3e-15)
    reference.point_data["VolumePointIndex"] = np.arange(
        reference.n_points, dtype=np.int64
    )
    exterior = reference.extract_surface(algorithm="dataset_surface").triangulate()
    triangles = exterior.faces.reshape(-1, 4)
    assert np.all(triangles[:, 0] == 3)
    mask = np.asarray(exterior["IsFace"], dtype=bool)
    selected = np.flatnonzero(np.all(mask[triangles[:, 1:]], axis=1))
    surface = (
        exterior.extract_cells(selected)
        .extract_surface(algorithm="dataset_surface")
        .triangulate()
    )
    ids = np.asarray(surface["VolumePointIndex"], dtype=np.int64).copy()
    assert surface.n_points == 15299 and surface.n_cells == 29899
    assert array_hash(ids) == receipt["skin"]["source_point_ids_sha256"]
    surface.points = points[ids].copy()
    # Remove stored field normals so only the current triangles determine shading.
    surface.clear_data()
    surface.point_data["VolumePointIndex"] = ids
    assert np.array_equal(surface.points, points[ids])
    return surface, {
        "inputs": inputs,
        "comparison_receipt": record(RECEIPT),
        "point_ids_sha256": array_hash(ids),
        "positions_sha256": array_hash(surface.points),
        "faces_sha256": array_hash(surface.faces),
    }


def camera(focus, direction, scale) -> dict:
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    return {
        "focal_point": list(focus),
        "position": (np.asarray(focus) + 0.25 * direction).tolist(),
        "view_up": [0, 1, 0],
        "parallel_scale": scale,
    }


def render(
    surface: pv.PolyData, view: dict, path: Path, resolution: int, *, edges=False
) -> dict:
    cam = view["camera"]
    focus = np.asarray(cam["focal_point"])
    backward = np.asarray(cam["position"]) - focus
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array(cam["view_up"]), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    key = focus + 0.3 * (
        view.get("light_side", -1) * 0.86 * right + 0.3 * up + 0.40 * backward
    )
    fill = focus + 0.3 * backward
    plotter = pv.Plotter(
        off_screen=True, window_size=(resolution, resolution), lighting="none"
    )
    plotter.set_background(BACKGROUND)
    actor = plotter.add_mesh(
        surface,
        color="#eeeeea",
        smooth_shading=False,
        show_edges=edges,
        edge_color="#424b56",
        line_width=0.55,
        ambient=0.20,
        diffuse=0.80,
        specular=0.0,
    )
    assert actor.GetProperty().GetInterpolation() == 0
    plotter.add_light(
        pv.Light(
            position=key,
            focal_point=focus,
            color="white",
            intensity=0.85,
            light_type="scene light",
            positional=False,
        )
    )
    plotter.add_light(
        pv.Light(
            position=fill,
            focal_point=focus,
            color="white",
            intensity=0.20,
            light_type="scene light",
            positional=False,
        )
    )
    plotter.enable_parallel_projection()
    plotter.camera.position = cam["position"]
    plotter.camera.focal_point = cam["focal_point"]
    plotter.camera.up = cam["view_up"]
    plotter.camera.parallel_scale = cam["parallel_scale"]
    plotter.reset_camera_clipping_range()
    raster = plotter.screenshot(return_img=True)
    plotter.close()
    image = Image.new("RGB", (resolution, resolution + 142), BACKGROUND)
    image.paste(Image.fromarray(raster).convert("RGB"), (0, 84))
    draw = ImageDraw.Draw(image)
    regular = ImageFont.truetype("DejaVuSans.ttf", 23)
    heading = ImageFont.truetype("DejaVuSans-Bold.ttf", 31)
    draw.text(
        (26, 12),
        view["label"] + (" | mesh edges" if edges else ""),
        fill="white",
        font=heading,
    )
    draw.text(
        (26, 51),
        "Corrected physical-volume baseline | step 200",
        fill="#c7d0da",
        font=regular,
    )
    draw.text(
        (26, resolution + 101),
        "Flat shading | actual deformation",
        fill="#c7d0da",
        font=regular,
    )
    # Orthographic projection: the image height represents twice parallel_scale.
    length_mm = 10 if cam["parallel_scale"] >= 0.03 else 5
    width = resolution * (length_mm / 1000) / (2 * cam["parallel_scale"])
    x1, y = resolution - 30, resolution + 119
    x0 = x1 - width
    draw.line((x0, y, x1, y), fill="white", width=3)
    draw.line((x0, y - 6, x0, y + 6), fill="white", width=3)
    draw.line((x1, y - 6, x1, y + 6), fill="white", width=3)
    draw.text(
        ((x0 + x1) / 2, y - 29),
        f"{length_mm} mm",
        anchor="mm",
        fill="white",
        font=regular,
    )
    image.save(path)
    assert array_hash(surface.points) == view["positions_sha256"]
    return {
        "file": record(path),
        "camera": cam,
        "key_light_position": key.tolist(),
        "fill_light_position": fill.tolist(),
        "key_intensity": 0.85,
        "fill_intensity": 0.20,
        "ambient": 0.20,
        "diffuse": 0.80,
        "specular": 0.0,
        "interpolation": "VTK_FLAT",
        "edges": edges,
        "scale_bar_mm": length_mm,
        "size": list(image.size),
    }


def main(cfg: Config) -> None:
    global DONE
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), out
    surface, evidence = load_surface()
    logging.info(
        "Verified step-200 state, source hashes, and all 29,899 exterior triangles."
    )
    views = [
        {
            "id": "mouth-overview",
            "label": "Mouth and chin",
            "camera": camera([1.407, 2.145, 0.087], [0, -0.10, 1], 0.045),
        },
        {
            "id": "mouth-corner-left",
            "label": "Mouth corner | image-left side",
            "camera": camera([1.374, 2.152, 0.078], [-0.48, -0.08, 1], 0.024),
        },
        {
            "id": "mouth-corner-right",
            "label": "Mouth corner | image-right side",
            "light_side": 1,
            "camera": camera([1.440, 2.152, 0.078], [0.48, -0.08, 1], 0.024),
        },
        {
            "id": "chin-oblique",
            "label": "Lower lip and chin | oblique view",
            "camera": camera([1.407, 2.133, 0.080], [0.35, -0.20, 1], 0.027),
        },
    ]
    surface.save(out / "corrected-step200-skin.vtp")
    figures = {}
    for view in views:
        view["positions_sha256"] = evidence["positions_sha256"]
        figures[view["id"]] = render(
            surface, view, out / (view["id"] + ".png"), cfg.resolution
        )
        logging.info("Rendered %s", view["id"])
    view = views[1]
    figures["mouth-corner-left-edges"] = render(
        surface, view, out / "mouth-corner-left-edges.png", cfg.resolution, edges=True
    )
    summary = {
        "status": "completed_saved_state_postprocessing",
        "step": 200,
        "geometry": {
            "points": surface.n_points,
            "triangles": surface.n_cells,
            "deformation_scale": 1,
            "geometry_smoothing": False,
            "displacement_exaggeration": False,
            "camera_only_crops": True,
        },
        **evidence,
        "source": record(Path(__file__)),
        "surface": record(out / "corrected-step200-skin.vtp"),
        "figures": figures,
        "runtime": {"pyvista": pv.__version__, "vtk": pv.vtk_version_info._asdict()},
        "interpretation": "Raking light reveals local triangle-normal variation. Flat facets also reflect mesh discretization; these images do not quantify bumpiness.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for path in out.iterdir():
        cherries.log_output(path)
    DONE = True


if __name__ == "__main__":
    cherries.main(
        main, profile="debug" if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
    if not DONE:
        raise SystemExit(1)
