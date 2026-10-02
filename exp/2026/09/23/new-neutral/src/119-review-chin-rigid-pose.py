"""Visualize a saved chin rigid estimate before any smoothed-carry solve."""

# ruff: noqa: E402, PLR0915
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint, sha256, write_json


class Config(cherries.BaseConfig):
    review_dir: Path = GROUP / "data/review-repaired-reference-005"
    estimate_path: Path = GROUP / "data/chin-rigid-pose-001/estimate.json"
    plan_path: Path = GROUP / "data/chin-rigid-pose-001/proposed-continuation.json"


def record(path: Path):
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verified(item: dict):
    path = Path(item["path"])
    assert record(path)["sha256"] == item["sha256"]
    return path


def poly(points: np.ndarray, triangles: np.ndarray):
    return pv.PolyData(
        points, np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    )


def camera(plot: pv.Plotter, points: np.ndarray, view: str):
    lo, hi = points.min(0), points.max(0)
    center, span = (lo + hi) / 2, float(max(hi - lo))
    direction = {"front": [0, 0, 2.7], "side": [2.7, 0, 0], "oblique": [2, 0, 2]}[view]
    plot.camera_position = [
        (center + np.array(direction) * span).tolist(),
        center.tolist(),
        [0, 1, 0],
    ]
    plot.camera.parallel_projection = True
    plot.camera.parallel_scale = 0.69 * span
    plot.reset_camera_clipping_range()


def main(cfg: Config):
    out = cfg.review_dir / "mandible-estimate"
    out.mkdir(exist_ok=True)
    estimate = json.loads(cfg.estimate_path.read_text())
    protocol = json.loads(verified(estimate["sources"]["source_protocol"]).read_text())
    plan = json.loads(cfg.plan_path.read_text())
    assert verified(plan["estimate"]) == cfg.estimate_path.resolve()
    topology = verified(protocol["rendering"]["archive"])
    with np.load(topology) as data:
        a = {k: data[k] for k in data.files}
    with np.load(verified(estimate["sources"]["blendshapes"])) as data:
        index = list(data["expression_names"]).index("MouthOpen")
        target_points = data["target_points_m"][index]
        neutral_points = data["new_neutral_points_m"]
        np.testing.assert_array_equal(data["skin_global_ids"], a["skin_global_ids"])
        np.testing.assert_array_equal(data["skin_triangles"], a["skin_triangles"])
    x = a["full_reference_points_m"]
    target = poly(target_points, a["skin_triangles"])
    jaw0 = poly(x[a["mandible_global_ids"]], a["mandible_triangles"])
    pose = np.asarray(estimate["pose_rad_m"])
    pivot = np.asarray(estimate["mandible_pivot_m"])
    rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
    jaw1 = jaw0.copy()
    jaw1.points = (jaw0.points - pivot) @ rotation.T + pivot + pose[3:]
    cranium = poly(x[a["cranium_global_ids"]], a["cranium_triangles"])
    eyes = poly(x[a["eye_global_ids"]], a["eye_triangles"])
    patch = np.asarray(estimate["patch_local_ids"])
    fitted = (neutral_points[patch] - pivot) @ rotation.T + pivot + pose[3:]
    target_patch = target_points[patch]
    all_points = np.concatenate(
        [mesh.points for mesh in (target, jaw0, jaw1, cranium, eyes)]
    )
    assets = []
    for view in ("front", "side", "oblique"):
        plot = pv.Plotter(off_screen=True, window_size=(1200, 1000))
        plot.set_background("#f7f7f5")
        plot.add_mesh(
            target,
            color="#737b86",
            opacity=0.20,
            label="MouthOpen target skin",
            smooth_shading=True,
        )
        plot.add_mesh(
            cranium,
            color="#d3c6ad",
            opacity=0.80,
            label="Fixed cranium",
            smooth_shading=True,
        )
        plot.add_mesh(eyes, color="#9fc5e8", label="Fixed eyes", smooth_shading=True)
        plot.add_mesh(
            jaw0,
            color="#dd812f",
            style="wireframe",
            opacity=0.65,
            label="Neutral mandible",
        )
        plot.add_mesh(
            jaw1,
            color="#087d81",
            opacity=0.95,
            label="Estimated mandible",
            smooth_shading=True,
        )
        plot.add_points(
            target_patch,
            color="#62b840",
            point_size=7,
            render_points_as_spheres=True,
            label="Target chin patch",
        )
        plot.add_text(
            f"Chin-fitted 6-DoF mandible | {view}\nGeometry estimate; no FEM solve\nOrange: neutral mandible | Teal: estimated mandible\nGray: target skin | Green: target chin patch",
            position="upper_left",
            font_size=15,
            color="#222222",
        )
        camera(plot, all_points, view)
        name = f"pose-{view}.png"
        plot.show(screenshot=out / name, auto_close=True)
        assets.append(name)
    plot = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1900, 950))
    plot.set_background("#f7f7f5")
    for col, view in enumerate(("front", "side")):
        plot.subplot(0, col)
        plot.add_mesh(cranium, color="#d3c6ad", smooth_shading=True)
        plot.add_mesh(eyes, color="#9fc5e8", smooth_shading=True)
        plot.add_mesh(jaw1, color="#087d81", smooth_shading=True)
        plot.add_mesh(target, color="#bc806d", smooth_shading=True)
        plot.add_text(
            f"MouthOpen target + estimated mandible\n{view} | target skin, not a simulated result",
            position="upper_left",
            font_size=14,
            color="#222222",
        )
        camera(plot, all_points, view)
    plot.show(screenshot=out / "target-context.png", auto_close=True)
    assets.append("target-context.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    center = target_patch.mean(0)
    pred, truth = (fitted - center) * 1000, (target_patch - center) * 1000
    for ax, axis, label in zip(
        axes, (0, 2), ("Front (x, y)", "Side (z, y)"), strict=True
    ):
        ax.scatter(
            truth[:, axis],
            truth[:, 1],
            facecolors="none",
            edgecolors="#237c3d",
            s=45,
            label="Target chin vertices",
        )
        ax.scatter(
            pred[:, axis],
            pred[:, 1],
            color="#cc772e",
            marker="x",
            s=22,
            label="Rigidly transformed chin",
        )
        for before, after in zip(truth, pred, strict=True):
            ax.plot(
                [before[axis], after[axis]],
                [before[1], after[1]],
                color="#999999",
                lw=0.7,
            )
        ax.set(
            title=label,
            xlabel=("x" if axis == 0 else "z") + " offset (mm)",
            ylabel="y offset (mm)",
            aspect="equal",
        )
        ax.grid(alpha=0.2)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(
        f"27-vertex chin fit | weighted RMS {estimate['weighted_rms_after_m'] * 1000:.3f} mm"
    )
    fig.savefig(out / "chin-fit.png", dpi=170)
    plt.close(fig)
    assets.append("chin-fit.png")
    jaw1.save(out / "estimated-mandible.vtp")
    translation = pose[3:] * 1000
    receipt = {
        "schema": "chin-rigid-pose-review-v1",
        "estimate": record(cfg.estimate_path),
        "plan": record(cfg.plan_path),
        "topology": record(topology),
        "source_protocol": record(verified(estimate["sources"]["source_protocol"])),
        "pose_rad_m": pose.tolist(),
        "physics_solved": False,
        "awaiting_user_pose_review": True,
        "assets": {name: record(out / name) for name in assets},
    }
    write_json(out / "receipt.json", receipt)
    (out / "estimate.json").write_text(cfg.estimate_path.read_text())
    text = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Estimated mandible pose</title>
<style>body{{font:16px/1.55 system-ui;max-width:1250px;margin:30px auto;padding:0 22px;color:#242424;background:#f7f7f5}}img{{width:100%;height:auto}}.status{{padding:16px;background:#fff0ce;border-radius:8px}}table{{border-collapse:collapse}}td,th{{padding:8px 16px;border-bottom:1px solid #ccc;text-align:left}}a{{color:#176e93}}h1{{font-size:28px}}figure{{margin:26px 0}}</style>
<p><a href="/">Neutral review</a> · <a href="/pose-jump/">Previous pose-jump tests</a></p>
<h1>Estimated 6-DoF mandible pose</h1>
<p class="status"><b>Awaiting your pose review.</b> This is the rigid pose estimated from the chin. No smoothed carry or forward solve has run for this pose.</p>
<p>Orange wireframe: neutral mandible. Teal: estimated mandible. Gray: MouthOpen target skin. Green points: the 27 target chin vertices used for the fit. Cranium and eyes stay in their neutral poses.</p>
<table><tr><th>Total rotation</th><td>{estimate["fit_rotation_degrees"]:.3f}°</td></tr><tr><th>Translation (world x, y, z)</th><td>({translation[0]:+.3f}, {translation[1]:+.3f}, {translation[2]:+.3f}) mm</td></tr><tr><th>Translation length</th><td>{estimate["translation_norm_m"] * 1000:.3f} mm</td></tr><tr><th>Chin fitting error</th><td>{estimate["weighted_rms_after_m"] * 1000:.3f} mm RMS; {estimate["residual_max_m"] * 1000:.3f} mm maximum</td></tr></table>
<p>Pose convention: rotation about the existing jaw pivot, then world translation. The chin is soft tissue, so this fit is an initialization estimate. Force and contact have not been evaluated for it.</p>
<p>Planned updates after review: at most <b>1° relative rotation and 1 mm pivot translation</b> each. From the saved 4.13° state, the geometric schedule contains {plan["paths"]["current_4deg_checkpoint"]["steps_count"]} increments. These limits apply to each update, not the total pose.</p>
<p><a href="receipt.json">Visualization evidence</a> · <a href="estimate.json">Pose parameters</a> · <a href="estimated-mandible.vtp">Estimated mandible mesh</a></p>
"""
    for name in assets:
        text += f'<figure><img src="{name}" loading="lazy"></figure>'
    (out / "index.html").write_text(text + "</html>")
    for index in (
        cfg.review_dir / "index.html",
        cfg.review_dir / "pose-jump/index.html",
    ):
        content = index.read_text()
        if 'id="mandible-estimate-link"' not in content:
            assert "</h1>" in content
            content = content.replace(
                "</h1>",
                '</h1><p id="mandible-estimate-link"><a href="/mandible-estimate/">Review the new chin-fitted 6-DoF mandible pose</a></p>',
                1,
            )
            index.write_text(content)
    cherries.log_output(out)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
