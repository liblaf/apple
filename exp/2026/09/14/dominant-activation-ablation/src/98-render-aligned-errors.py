"""Render corresponding-vertex target errors on five saved deformed skins."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from matplotlib.colors import to_hex
from muscle_glyph_context import set_parallel_camera
from PIL import Image

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]

LOG = logging.getLogger(__name__)
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
WINDOW = (1800, 1800)
LIMITS = (0.0, 10.0)
CMAP = mpl.colormaps["viridis"]


class Config(cherries.BaseConfig):
    comparison: Path = cherries.input("96-aligned-five-way/summary.json")
    fixed_initialization: Path = cherries.input(
        "42-fixed-directions-400/initialization.npz"
    )
    output_dir: Path = cherries.output("98-aligned-errors", mkdir=True)


def record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def verify(item: dict) -> Path:
    path = Path(item["path"])
    assert record(path) == item, path
    return path


def render(skin: pv.PolyData, camera: dict, path: Path) -> dict:
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW)
    plotter.set_background("#f4f2ed")
    plotter.add_mesh(
        skin,
        scalars="PointToPointErrorMm",
        preference="point",
        cmap=CMAP,
        clim=LIMITS,
        n_colors=256,
        lighting=False,
        smooth_shading=False,
        interpolate_before_map=True,
        show_scalar_bar=False,
    )
    set_parallel_camera(plotter, camera)
    projection = np.asarray(
        plotter.camera.GetCompositeProjectionTransformMatrix(1.0, -1.0, 1.0).GetData()
    ).reshape(4, 4)
    alignment = {
        "position": list(plotter.camera.position),
        "focal_point": list(plotter.camera.focal_point),
        "view_up": list(plotter.camera.up),
        "parallel_scale": plotter.camera.parallel_scale,
        "screen_projection_rows": projection[[0, 1, 3]].tolist(),
    }
    plotter.screenshot(path)
    plotter.close()
    return alignment


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), out
    (out / "sources").mkdir()
    sources = []
    for source in (
        Path(__file__),
        Path(__file__).with_name("experiment_profile.py"),
        Path(__file__).with_name("muscle_glyph_context.py"),
    ):
        dest = out / "sources" / source.name
        shutil.copyfile(source, dest)
        sources.append({"source": record(source), "snapshot": record(dest)})
    previous = json.loads(cfg.comparison.read_text())
    assert previous["status"] == "completed"
    volume = pv.read(FIXTURE / "volume.vtu")
    template = pv.read(FIXTURE / "skin.vtp")
    rest = np.asarray(volume.points)
    target = np.asarray(volume.point_data["Smile"])
    face = np.asarray(volume.point_data["IsFace"], dtype=bool)
    face_ids = np.flatnonzero(face & np.isfinite(target).all(axis=1))
    ids = np.asarray(template.point_data["GlobalPointId"], dtype=np.int64)
    assert len(np.unique(ids)) == len(ids)
    assert np.array_equal(template.points, rest[ids])
    assert face[ids].all() and np.isfinite(target[ids]).all()
    states = []
    inputs = [
        record(p)
        for p in (
            cfg.comparison,
            cfg.fixed_initialization,
            FIXTURE / "volume.vtu",
            FIXTURE / "skin.vtp",
        )
    ]
    for index, old in enumerate(previous["states"]):
        cherries.set_step(index)
        source = verify(old["source"])
        inputs.append(old["source"])
        with np.load(source, allow_pickle=False) as saved:
            assert bool(saved["solver_valid"])
            u = saved["u"]
            if "rest_points" in saved:
                assert np.array_equal(saved["rest_points"], rest)
            else:
                assert old["id"] == "fixed"
                with np.load(cfg.fixed_initialization, allow_pickle=False) as initial:
                    assert np.array_equal(initial["rest_points"], rest)
        assert u.shape == rest.shape and np.isfinite(u).all()
        errors = 1000 * np.linalg.norm(u[ids] - target[ids], axis=1)
        fit = float(
            1000
            * np.sqrt(np.mean(np.sum((u[face_ids] - target[face_ids]) ** 2, axis=1)))
        )
        assert np.isclose(
            fit, old["metrics"]["uniform_fit_rms_mm"], rtol=1e-13, atol=1e-13
        )
        assert np.isfinite(errors).all() and errors.min() >= LIMITS[0]
        assert errors.max() < LIMITS[1], (old["id"], errors.max())
        metrics = {
            "skin_min_mm": float(errors.min()),
            "skin_max_mm": float(errors.max()),
            "skin_p95_mm": float(np.percentile(errors, 95)),
            "skin_p99_mm": float(np.percentile(errors, 99)),
            "skin_rms_mm": float(np.sqrt(np.mean(errors**2))),
            "uniform_fit_rms_mm": fit,
        }
        skin = template.copy(deep=True)
        skin.points = rest[ids] + u[ids]
        skin.point_data["PointToPointErrorMm"] = errors
        skin.point_data["TargetPosition"] = rest[ids] + target[ids]
        skin_path = out / f"{old['id']}-error-skin.vtp"
        skin.save(skin_path, binary=True)
        error_path = out / f"{old['id']}-errors.npz"
        np.savez_compressed(error_path, GlobalPointId=ids, error_mm=errors)
        views = {}
        for name, camera in previous["cameras"].items():
            shape = old["views"][name]["shape"]
            verify(shape)
            inputs.append(shape)
            path = out / f"{name}-{old['id']}-error.png"
            LOG.info("Rendering %s: %s", old["id"], name)
            alignment = render(skin, camera, path)
            assert alignment == previous["verified_alignment"][name]
            with Image.open(path) as image:
                assert image.size == WINDOW
                image.verify()
            views[name] = {
                "shape": shape,
                "error": record(path),
                "camera_alignment": alignment,
            }
        states.append(
            {
                **{k: old[k] for k in ("id", "title", "subtitle", "source", "metrics")},
                "error_metrics": metrics,
                "error_skin": record(skin_path),
                "error_values": record(error_path),
                "views": views,
            }
        )
        cherries.log_metrics(
            {f"{old['id']}/{key}": value for key, value in metrics.items()}
        )
        LOG.info("%s: %s", old["id"], metrics)
    summary = {
        "status": "completed",
        "inputs": inputs,
        "sources": sources,
        "states": states,
        "cameras": previous["cameras"],
        "verified_alignment": previous["verified_alignment"],
        "domain": {
            "skin_vertices": len(ids),
            "finite_face_objective_vertices": len(face_ids),
            "all_skin_vertices_have_finite_face_targets": True,
            "fit_vertices_absent_from_skin": np.setdiff1d(face_ids, ids).tolist(),
        },
        "encoding": {
            "definition": "1000 * norm(u[GlobalPointId] - Smile[GlobalPointId])",
            "units": "mm",
            "correspondence": "GlobalPointId; no registration or nearest-neighbor remapping",
            "rendered_geometry": "Saved deformed skin, X+u",
            "colormap": "viridis",
            "color_limits_mm": list(LIMITS),
            "color_stops": [to_hex(CMAP(x)) for x in np.linspace(0, 1, 33)],
            "clipping": "None; all vertex errors are inside the common limits",
            "surface_interpolation": "Point scalars interpolated across triangles before color mapping",
            "lighting": False,
            "panel_pixels": list(WINDOW),
        },
        "limitations": previous["limitations"][:1]
        + [
            "Displayed error is an unsigned corresponding-vertex distance, not closest-surface distance.",
            "Fit RMS uses 15302 face vertices; error maps contain the 15299 skin vertices.",
            "Top images and inversion counts are preserved from the verified prior comparison.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for image in out.glob("*.png"):
        cherries.log_output(image)
    cherries.log_output(out / "summary.json")
    LOG.info("Completed aligned point-to-point error renders in %s", out)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
