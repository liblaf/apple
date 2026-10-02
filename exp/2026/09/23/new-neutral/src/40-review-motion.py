# ruff: noqa: PLR0915
"""Measure and visualize the saved neutral motion without running a solver."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint  # noqa: E402


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/forward-repaired-reference-001"
    review_dir: Path = GROUP / "data/review-repaired-reference-001"


def record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


def stats(displacement: np.ndarray) -> dict:
    mm = 1000 * np.linalg.norm(displacement, axis=1)
    return {
        "vertex_rms_mm": float(np.sqrt(np.mean(mm**2))),
        "median_mm": float(np.median(mm)),
        "p95_mm": float(np.quantile(mm, 0.95)),
        "max_mm": float(mm.max()),
        "vertices_under_1mm_fraction": float(np.mean(mm < 1)),
    }


def render(
    output: Path,
    skin: pv.PolyData,
    u: np.ndarray,
    view: str,
    reference_label: str,
    rigid_meshes: dict[str, pv.PolyData] | None = None,
) -> str:
    reference = np.asarray(skin.points).copy()
    magnitude = 1000 * np.linalg.norm(u, axis=1)
    surfaces = []
    for scale in (0, 1, 10):
        surface = skin.copy(deep=True)
        surface.points = reference + scale * u
        surface["Actual displacement (mm)"] = magnitude
        surfaces.append(surface)
    rigid_meshes = {} if rigid_meshes is None else rigid_meshes
    points = np.concatenate(
        [
            *(surface.points for surface in surfaces),
            *(mesh.points for mesh in rigid_meshes.values()),
        ]
    )
    low, high = points.min(0), points.max(0)
    center, span = (low + high) / 2, float(max(high - low))
    eye = (
        center
        + (np.array([0, 0, 2.7]) if view == "front" else np.array([2.7, 0, 0])) * span
    )
    plot = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1800, 820))
    plot.set_background("#f7f7f5")
    labels = (
        reference_label,
        "Solved motion at 1x (actual)",
        "Solved motion at 10x display scale",
    )
    for col, surface in enumerate(surfaces):
        plot.subplot(0, col)
        caption = labels[col]
        if rigid_meshes:
            caption += "\nBones/eyes fixed; 10x is display only"
        plot.add_text(caption, position="upper_left", font_size=15, color="#202124")
        if col == 0:
            plot.add_mesh(surface, color="#a2a6ac", smooth_shading=True)
        else:
            plot.add_mesh(
                surface,
                scalars="Actual displacement (mm)",
                cmap="viridis",
                clim=(0, float(magnitude.max())),
                smooth_shading=True,
                show_scalar_bar=col == 1,
                scalar_bar_args={
                    "title": "Actual motion (mm)",
                    "vertical": False,
                    "position_y": 0.03,
                    "height": 0.10,
                    "width": 0.78,
                    "position_x": 0.11,
                    "color": "#202124",
                },
            )
        for name, mesh in rigid_meshes.items():
            color = "#9fc5e8" if name == "eyes" else "#e8dfc8"
            plot.add_mesh(mesh, color=color, smooth_shading=True, opacity=0.82)
        plot.camera_position = [eye.tolist(), center.tolist(), [0, 1, 0]]
        plot.camera.parallel_projection = True
        plot.camera.parallel_scale = 0.60 * span
        plot.reset_camera_clipping_range()
    name = f"motion-{view}.png"
    plot.show(screenshot=output / name, auto_close=True)
    return name


def main(cfg: Config) -> None:
    run, review = cfg.run_dir.resolve(), cfg.review_dir.resolve()
    output = review / "motion"
    assert not output.exists(), output
    receipt = json.loads((review / "receipt.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    skin_path = Path(receipt["inputs"]["constitutive_skin"]["path"])
    assert record(skin_path) == receipt["inputs"]["constitutive_skin"]
    skin = pv.read(skin_path).triangulate()
    ids = np.asarray(skin["GlobalPointId"], dtype=np.int64)
    endpoint_path = run / "endpoint.npz"
    assert record(endpoint_path) == receipt["run"]["endpoint"]
    with np.load(endpoint_path, allow_pickle=False) as archive:
        u = archive["displacement_m"][ids]
    seed_path = Path(protocol["fixture"]["seed"]["path"])
    assert record(seed_path) == protocol["fixture"]["seed"]
    with np.load(seed_path, allow_pickle=False) as archive:
        seed = archive["displacement_m"][ids]
    solved = pv.read(review / "neutral-skin.vtp")
    assert np.max(np.abs(solved.points - (skin.points + u))) <= 5e-16
    mm = 1000 * np.linalg.norm(u, axis=1)
    np.testing.assert_allclose(
        solved["DisplacementMagnitudeMm"], mm, rtol=1e-12, atol=1e-12
    )
    output.mkdir()
    reference_configuration = receipt.get("reference_configuration")
    reference_label = (
        "Clearance-repaired constitutive reference"
        if reference_configuration is not None
        else "Original constitutive reference"
    )
    reference_description = receipt["reference_coordinate_contract"]["definition"]
    valid = bool(receipt["result"]["valid_forward"])
    inverted = int(receipt["geometry"]["inverted_tetrahedra"])
    summary = {
        "schema": "saved-neutral-motion-review-v1",
        "reference": reference_label + "; meters converted to millimeters.",
        "reference_coordinate_contract": reference_description,
        "endpoint": record(endpoint_path),
        "skin": record(skin_path),
        "seed": record(seed_path),
        "vertex_count": len(ids),
        "face_span_mm": (1000 * np.ptp(skin.points, axis=0)).tolist(),
        "reference_to_result": stats(u),
        "reference_to_seed": stats(seed),
        "seed_to_result": stats(u - seed),
        "fitting_area_weighted_rms_mm": receipt["result"]["geometry"][
            "surface_motion_rms_mm"
        ],
        "served_mesh_equals_reference_plus_displacement": True,
        "valid_forward": receipt["result"]["valid_forward"],
        "inverted_tetrahedra": receipt["geometry"]["inverted_tetrahedra"],
        "display_factors": [0, 1, 10],
        "display_note": "All colors encode actual displacement. The 10x geometry is a visualization only, not a solved state.",
    }
    images = [
        render(output, skin, u, view, reference_label) for view in ("front", "side")
    ]
    summary["images"] = [record(output / name) for name in images]
    (output / "motion-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    s = summary["reference_to_result"]
    figures = "".join(
        f'<figure><a href="{name}"><img src="{name}" alt="Constitutive reference, actual solved motion, and 10 times display motion"></a></figure>'
        for name in images
    )
    if valid:
        validity_label = "Valid forward endpoint"
        validity_detail = (
            "Solver convergence and physical geometry/contact gates passed."
        )
        status_color = "#d9f2e6"
    elif inverted:
        validity_label = "Diagnostic forward endpoint; geometry gate failed"
        validity_detail = f"{inverted} tetrahedra are inverted."
        status_color = "#ffe2dc"
    else:
        failure = str(
            receipt["result"].get("failure", "forward solver did not converge")
        )
        validity_label = "Diagnostic forward endpoint; force convergence failed"
        validity_detail = f"{failure}. Geometry has no inverted tetrahedra and the saved contact gate passed."
        status_color = "#fff1c7"
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Neutral motion detail</title>
<style>body{{font:16px system-ui,sans-serif;margin:2rem auto;max-width:1500px;padding:0 1rem;background:#f7f7f5;color:#202124}}img{{width:100%}}figure{{margin:1rem 0}}.status{{background:{status_color};padding:1rem}}table{{border-collapse:collapse}}th,td{{border:1px solid #aaa;padding:.5rem;text-align:left}}</style>
<p><a href="../">Back to full neutral review</a></p><h1>How far did the solved face move?</h1>
<p class="status">{validity_label}. {validity_detail} The 10&times; panels enlarge only the displayed displacement; they are not solved states.</p>
<p>The skin moved a median {s["median_mm"]:.3f} mm on a face about {summary["face_span_mm"][1]:.0f} mm tall. The camera and scale match across panels. Colors always show the actual displacement in millimeters.</p>
<table><tr><th>Skin vertex statistic</th><th>Actual motion</th></tr><tr><td>Median</td><td>{s["median_mm"]:.3f} mm</td></tr><tr><td>95th percentile</td><td>{s["p95_mm"]:.3f} mm</td></tr><tr><td>Maximum</td><td>{s["max_mm"]:.3f} mm</td></tr><tr><td>Vertices moving less than 1 mm</td><td>{100 * s["vertices_under_1mm_fraction"]:.1f}%</td></tr></table>
<p>{reference_description}</p>
{figures}<p>The skin vertices start at the displayed constitutive reference. The middle panels show the saved solved displacement at 1&times;; the right panels show the same displacement multiplied by 10 only for inspection. The exported solved mesh was checked against reference + displacement. <a href="motion-summary.json">Download measurement receipt</a>.</p></html>"""
    (output / "index.html").write_text(html)
    index = review / "index.html"
    review_html = index.read_text()
    assert 'id="motion-detail"' not in review_html
    review_html = review_html.replace(
        "</h1>",
        '</h1><p id="motion-detail"><a href="motion/">Displacement map and 10x motion view</a> '
        "(visual magnification only).</p>",
        1,
    )
    index.write_text(review_html)
    cherries.log_output(output)
    cherries.log_metrics({f"motion/{key}": value for key, value in s.items()})
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
