"""Render saved transferred blendshapes without evaluating the forward model."""

from __future__ import annotations

import json
import re
import sys
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402


class Config(cherries.BaseConfig):
    bundle_dir: Path = GROUP / "data/blendshapes-005"
    review_dir: Path = GROUP / "data/review-repaired-reference-005"


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verified(item: dict[str, Any]) -> Path:
    path = Path(item["path"])
    actual = record(path)
    assert actual["path"] == item["path"]
    assert actual["sha256"] == item["sha256"]
    return path


def artifact(manifest: dict[str, Any], filename: str) -> Path:
    item = manifest["artifacts"][filename]
    return verified(item)


def mesh(points: np.ndarray, triangles: np.ndarray) -> pv.PolyData:
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert triangles.ndim == 2
    assert triangles.shape[1] == 3
    assert np.isfinite(points).all()
    assert np.issubdtype(triangles.dtype, np.integer)
    assert triangles.min() >= 0
    assert triangles.max() < len(points)
    faces = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    return pv.PolyData(points, faces)


def camera(meshes: list[pv.PolyData], view: str) -> tuple[list[list[float]], float]:
    points = np.concatenate([item.points for item in meshes])
    low, high = points.min(0), points.max(0)
    center, span = (low + high) / 2, float(max(high - low))
    if view == "front":
        eye = center + np.array([0.0, 0.0, 2.7 * span])
    elif view == "side":
        eye = center + np.array([2.7 * span, 0.0, 0.0])
    else:
        raise ValueError(view)
    return [eye.tolist(), center.tolist(), [0.0, 1.0, 0.0]], 0.60 * span


def apply_camera(plot: pv.Plotter, position: list[list[float]], scale: float) -> None:
    plot.camera_position = position
    plot.camera.parallel_projection = True
    plot.camera.parallel_scale = scale
    plot.reset_camera_clipping_range()


def label(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def detail(
    output: Path,
    name: str,
    target: pv.PolyData,
    position: list[list[float]],
    scale: float,
    status: str,
) -> str:
    plot = pv.Plotter(off_screen=True, window_size=(900, 900))
    plot.set_background("#f7f7f5")
    plot.add_text(
        f"{name}\nTransferred expression · {status}",
        position="upper_left",
        font_size=15,
        color="#202124",
    )
    plot.add_mesh(target, color="#c75f42", smooth_shading=True)
    apply_camera(plot, position, scale)
    filename = f"{label(name)}.png"
    plot.show(screenshot=output / filename, auto_close=True)
    return filename


def gallery(
    output: Path,
    names: list[str],
    targets: np.ndarray,
    triangles: np.ndarray,
    position: list[list[float]],
    scale: float,
    status: str,
) -> str:
    columns = 6
    rows = int(np.ceil(len(names) / columns))
    plot = pv.Plotter(
        off_screen=True, shape=(rows, columns), window_size=(2400, 400 * rows)
    )
    plot.set_background("#f7f7f5")
    for index, name in enumerate(names):
        plot.subplot(index // columns, index % columns)
        plot.add_text(
            f"{name}\n{status}", position="upper_left", font_size=10, color="#202124"
        )
        plot.add_mesh(
            mesh(targets[index], triangles), color="#c75f42", smooth_shading=True
        )
        apply_camera(plot, position, scale)
    filename = "gallery-front.png"
    plot.show(screenshot=output / filename, auto_close=True)
    return filename


def neutral_comparison(
    output: Path,
    source: pv.PolyData,
    target: pv.PolyData,
    status: str,
) -> str:
    position, scale = camera([source, target], "front")
    plot = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1800, 900))
    plot.set_background("#f7f7f5")
    for column, (title, surface, color) in enumerate(
        (
            ("Historical source neutral", source, "#737b86"),
            (f"Transferred new neutral · {status}", target, "#c75f42"),
        )
    ):
        plot.subplot(0, column)
        plot.add_text(
            f"{title}\nfront · identical true scale",
            position="upper_left",
            font_size=15,
            color="#202124",
        )
        plot.add_mesh(surface, color=color, smooth_shading=True)
        apply_camera(plot, position, scale)
    filename = "neutral-comparison-front.png"
    plot.show(screenshot=output / filename, auto_close=True)
    return filename


def main(cfg: Config) -> None:  # noqa: PLR0915
    bundle_dir, review = cfg.bundle_dir.resolve(), cfg.review_dir.resolve()
    output = review / "blendshapes"
    assert not output.exists(), output
    manifest_path = bundle_dir / "manifest.json"
    assert manifest_path.is_file(), manifest_path
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "new-neutral-blendshape-transfer-v1"
    assert (
        manifest["topology_convention"]
        == "skin_triangles contains local zero-based indices into skin_global_ids"
    )
    bundle_path = artifact(manifest, "blendshapes.npz")
    skin_path = artifact(manifest, "neutral-with-blendshapes.vtp")
    volume_path = artifact(manifest, "neutral-with-blendshapes.vtu")
    targets_zip_path = artifact(manifest, "targets.zip")
    review_receipt = json.loads((review / "receipt.json").read_text())
    neutral_status = manifest["neutral_status"]
    assert neutral_status["solver_converged"] is True
    assert neutral_status["valid_forward"] is False
    assert neutral_status["inverted_tetrahedra"] == 2
    assert review_receipt["state_label"] == "invalid geometry"
    assert int(review_receipt["geometry"]["inverted_tetrahedra"]) == 2
    with np.load(bundle_path, allow_pickle=False) as archive:
        required = {
            "expression_names",
            "skin_global_ids",
            "skin_triangles",
            "source_neutral_points_m",
            "new_neutral_points_m",
            "expression_displacement_m",
            "target_points_m",
        }
        assert set(archive.files) == required, archive.files
        names = [str(value) for value in archive["expression_names"]]
        ids = archive["skin_global_ids"]
        triangles = archive["skin_triangles"]
        source_points = archive["source_neutral_points_m"]
        new_points = archive["new_neutral_points_m"]
        displacement = archive["expression_displacement_m"]
        targets = archive["target_points_m"]
    assert names == manifest["expression_names"]
    assert len(names) == 36
    assert len(set(names)) == len(names)
    assert ids.shape == (len(new_points),)
    assert len(np.unique(ids)) == len(ids)
    assert source_points.shape == new_points.shape == (len(ids), 3)
    assert displacement.shape == targets.shape == (len(names), len(ids), 3)
    assert np.isfinite(source_points).all()
    assert np.isfinite(new_points).all()
    assert np.isfinite(displacement).all()
    assert np.isfinite(targets).all()
    np.testing.assert_allclose(
        targets, new_points[None] + displacement, rtol=0, atol=5e-16
    )
    served_skin = pv.read(skin_path)
    assert np.array_equal(np.asarray(served_skin.point_data["GlobalPointId"]), ids)
    np.testing.assert_allclose(served_skin.points, new_points, rtol=0, atol=5e-16)
    target_meshes = manifest["target_meshes"]
    assert set(target_meshes) == set(names)
    for index, name in enumerate(names):
        target_path = verified(target_meshes[name])
        target_skin = pv.read(target_path)
        assert np.array_equal(target_skin.point_data["GlobalPointId"], ids)
        np.testing.assert_allclose(
            target_skin.points, targets[index], rtol=0, atol=5e-16
        )
    source = mesh(source_points, triangles)
    neutral = mesh(new_points, triangles)
    position, scale = camera(
        [neutral, *[mesh(points, triangles) for points in targets]], "front"
    )
    neutral_status_label = "invalid geometry (2 inverted tetrahedra)"
    base_status_label = "Base neutral: 2 inverted tetrahedra"
    output.mkdir()
    details = output / "expressions"
    details.mkdir()
    neutral_image = neutral_comparison(output, source, neutral, neutral_status_label)
    gallery_image = gallery(
        output, names, targets, triangles, position, scale, base_status_label
    )
    images = {
        name: detail(
            details,
            name,
            mesh(targets[index], triangles),
            position,
            scale,
            base_status_label,
        )
        for index, name in enumerate(names)
    }
    assets = output / "assets"
    assets.mkdir()
    bundle_assets = {
        "blendshapes.npz": bundle_path,
        "neutral-with-blendshapes.vtp": skin_path,
        "neutral-with-blendshapes.vtu": volume_path,
        "targets.zip": targets_zip_path,
    }
    for filename, source_path in bundle_assets.items():
        link = assets / filename
        link.symlink_to(source_path)
        assert link.resolve() == source_path
    transfer_validation = manifest["transfer_validation"]
    receipt = {
        "schema": "saved-blendshape-review-v1",
        "bundle_manifest": record(manifest_path),
        "inputs": {
            "bundle": record(bundle_path),
            "skin": record(skin_path),
            "volume": record(volume_path),
            "targets_zip": record(targets_zip_path),
            "review": record(review / "receipt.json"),
        },
        "neutral_status": neutral_status,
        "expression_names": names,
        "transfer_validation": transfer_validation,
        "skin_vertices": len(ids),
        "skin_triangles": len(triangles),
        "target_equals_new_neutral_plus_displacement": True,
        "images": {
            "neutral_comparison": neutral_image,
            "gallery_front": gallery_image,
            "details": images,
        },
        "download_assets": {name: record(path) for name, path in bundle_assets.items()},
        "scope": "Saved transfer-bundle review only. It runs no solver and does not adopt or validate a neutral endpoint. Expression geometry inherits the saved neutral diagnostic status.",
    }
    write_json(output / "receipt.json", receipt)
    options = "".join(
        f'<option value="expressions/{escape(images[name])}">{escape(name)}</option>'
        for name in names
    )
    downloads = " · ".join(
        f'<a href="assets/{escape(filename)}">{escape(filename)}</a>'
        for filename in bundle_assets
    )
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Transferred blendshapes</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#202124;background:#f7f7f5}}img{{max-width:100%;height:auto}}.status{{background:#ffe2dc;padding:1rem}}select{{font:inherit;padding:.4rem;max-width:100%}}a{{color:#236595}}</style>
<p><a href="../">Back to neutral review</a></p><h1>Transferred blendshapes</h1>
<p class="status">36 geometric expression targets use original offsets. Base neutral has two inverted tetrahedra; these expressions are not forward-solved equilibria.</p>
<p>All 36 expressions use the saved additive displacement transfer: target = new neutral + expression displacement. Bones and eyes are not rendered here; this gallery changes skin geometry only.</p>
<h2>0. Neutral comparison</h2><a href="{neutral_image}"><img src="{neutral_image}" alt="Historical and transferred new neutral comparison"></a>
<h2>36-expression front gallery</h2><a href="{gallery_image}"><img src="{gallery_image}" alt="All transferred blendshapes at identical front camera scale"></a>
<h2>Expression detail</h2><label for="expression">Choose an expression: <select id="expression">{options}</select></label><p><img id="detail" src="expressions/{escape(images[names[0]])}" alt="Selected transferred blendshape"></p>
<script>document.getElementById('expression').addEventListener('change', e => document.getElementById('detail').src = e.target.value);</script>
<p>Downloads: {downloads}. <a href="receipt.json">Review receipt</a>.</p></html>"""
    (output / "index.html").write_text(html)
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "blendshapes/expressions": len(names),
            "blendshapes/skin_vertices": len(ids),
            "blendshapes/skin_triangles": len(triangles),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
