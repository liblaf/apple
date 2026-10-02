"""Render saved pose-jump stages without evaluating physics."""

# ruff: noqa: E402, PLR0915
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint, sha256, write_json


class Config(cherries.BaseConfig):
    review_dir: Path = GROUP / "data/review-repaired-reference-005"
    direct_run: Path = GROUP / "data/pose-jump-004"


def record(path: Path):
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def mesh(x: np.ndarray, ids: np.ndarray, triangles: np.ndarray, u: np.ndarray):
    return pv.PolyData(
        x[ids] + u[ids],
        np.column_stack((np.full(len(triangles), 3), triangles)).ravel(),
    )


def main(cfg: Config):
    out = cfg.review_dir / "pose-jump"
    out.mkdir(exist_ok=True)
    topology = GROUP / "data/inverse-mouthopen-003/rendering.npz"
    with np.load(topology) as data:
        arrays = {key: data[key] for key in data.files}
    x = arrays["full_reference_points_m"]
    direct = cfg.direct_run / "direct-relaxed-partial.pt"
    rows = [
        json.loads(line)
        for line in (cfg.direct_run / "direct-relaxed-force.jsonl")
        .read_text()
        .splitlines()
    ]
    saved = [row for row in rows if "checkpoint" in row][-1]
    assert sha256(direct) == saved["checkpoint"]["sha256"]
    harmonic = GROUP / "data/pose-jump-002"
    assert json.loads((harmonic / "summary.json").read_text())["success"]
    stages = [
        {
            "title": "Starting pose",
            "label": "4.13 degrees | contact on",
            "path": harmonic / "source.pt",
        },
        {
            "title": "Direct jump + no collision",
            "label": f"11.56 degrees | {saved['seconds']:.0f} s | {saved['force_norm'] * 1e6:.3g} N\nUnconverged; before push-out",
            "path": direct,
        },
        {
            "title": "Smoothed carry + no collision",
            "label": "11.56 degrees | 177 s | 0.00969 N\nForce converged; before push-out",
            "path": harmonic / "relaxed.pt",
        },
    ]
    for stage in stages:
        state = torch.load(stage["path"], map_location="cpu", weights_only=False)
        stage["u"] = state["displacement_m"].numpy()
        assert stage["u"].shape == x.shape
        assert np.isfinite(stage["u"]).all()
    ids, tri = arrays["skin_global_ids"], arrays["skin_triangles"]
    rigid = [("cranium", "cranium"), ("mandible", "mandible"), ("eyes", "eye")]
    visible_ids = np.unique(
        np.concatenate([ids, *[arrays[f"{key}_global_ids"] for _, key in rigid]])
    )
    points = np.concatenate(
        [x[visible_ids] + stage["u"][visible_ids] for stage in stages]
    )
    low, high = points.min(0), points.max(0)
    center, span = (low + high) / 2, float(max(high - low))
    assets = []
    for view, translucent in [("front", False), ("side", False), ("side", True)]:
        name = f"comparison-{view}{'-anatomy' if translucent else ''}.png"
        plot = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(2100, 850))
        plot.set_background("#f7f7f5")
        eye = (
            center
            + (np.array([0, 0, 2.7]) if view == "front" else np.array([2.7, 0, 0]))
            * span
        )
        for col, stage in enumerate(stages):
            plot.subplot(0, col)
            plot.add_text(
                stage["title"] + "\n" + stage["label"],
                font_size=13,
                position="upper_left",
                color="#222222",
            )
            u = stage["u"]
            for label, key in rigid:
                plot.add_mesh(
                    mesh(x, arrays[f"{key}_global_ids"], arrays[f"{key}_triangles"], u),
                    color="#9dc9ee" if label == "eyes" else "#e4d8bf",
                    smooth_shading=True,
                )
            plot.add_mesh(
                mesh(x, ids, tri, u),
                color="#c77960",
                smooth_shading=True,
                opacity=0.25 if translucent else 1.0,
            )
            plot.camera_position = [eye.tolist(), center.tolist(), [0, 1, 0]]
            plot.camera.parallel_projection = True
            horizontal_span = high[0] - low[0] if view == "front" else high[2] - low[2]
            plot.camera.parallel_scale = max(
                0.65 * (high[1] - low[1]), 0.58 * horizontal_span * 850 / 700
            )
            plot.reset_camera_clipping_range()
        plot.show(screenshot=out / name, auto_close=True)
        assets.append(name)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.semilogy(
        [r["seconds"] for r in rows],
        [r["force_norm"] * 1e6 for r in rows],
        color="#b65c41",
        label="Direct jump: contact disabled",
    )
    ax.axhline(0.01, color="#555555", linestyle=":", label="Force threshold: 0.01 N")
    ax.set(xlabel="Relaxation wall time (seconds)", ylabel="Free force norm (N)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(out / "direct-force.png", dpi=180)
    plt.close(fig)
    receipt = {
        "schema": "saved-pose-jump-review-v1",
        "topology": record(topology),
        "stages": [
            {k: v for k, v in s.items() if k not in {"u", "path"}}
            | {"checkpoint": record(s["path"])}
            for s in stages
        ],
        "direct_last_snapshot": saved,
        "harmonic_summary": record(harmonic / "summary.json"),
    }
    write_json(out / "receipt.json", receipt)
    force = saved["force_norm"] * 1e6
    inverted = saved["geometry"]["inverted_tetrahedra"]
    body = f"""<!doctype html><html><meta charset="utf-8"><title>Mandible jump: no-collision relaxation</title>
<style>body{{font:16px/1.55 system-ui;max-width:1500px;margin:30px auto;padding:0 22px;background:#f7f7f5;color:#252525}}img{{width:100%;height:auto}}.note{{padding:16px;background:#fff0ce;border-radius:8px}}a{{color:#176e93}}figure{{margin:24px 0}}h1{{font-size:28px}}.links{{display:flex;gap:24px}}</style>
<p><a href="/">← Neutral review</a></p><h1>Mandible jump: no-collision relaxation</h1>
<p>Same saved starting state at 4.13°, with the jaw moved directly to 11.56°. The direct carry moves 173 free vertices within d_hat = 0.1 mm, together with the prescribed jaw attachments. The smoothed carry extends those increments through the volume.</p>
<p class="note"><b>Direct saved iterate at {saved["seconds"]:.1f} s: {force:.4g} N</b>, versus the 0.01 N threshold. It is unconverged and has {inverted:,} inverted tetrahedra. No push-out has been applied. The earlier direct trial exceeded its 600 s budget at about 0.106 N; its terminal geometry was not saved.</p>
<p>The smoothed contact-off state converged in 177 s but has 299 inverted tetrahedra. After push-out and a contact-on solve, force is 0.00992 N and surface contact passes; 282 tetrahedra remain inverted. These are numerical diagnostics, not physically valid inverse fits. Muscle activation is zero in all displayed stages.</p>
<div class="links"><a href="comparison-front.png">Front</a><a href="comparison-side.png">Side</a><a href="comparison-side-anatomy.png">Translucent anatomy</a><a href="receipt.json">Saved-state evidence</a></div>
"""
    for asset in assets:
        body += f'<figure><img src="{html.escape(asset)}"><figcaption>Identical camera and true scale for all three stages.</figcaption></figure>'
    body += '<figure><img src="direct-force.png"><figcaption>Accepted states from the direct repeat; contact remains disabled.</figcaption></figure></html>'
    (out / "index.html").write_text(body)
    root_index = cfg.review_dir / "index.html"
    root = root_index.read_text()
    if 'id="pose-jump-link"' not in root:
        assert "</h1>" in root
        root = root.replace(
            "</h1>",
            '</h1><p id="pose-jump-link"><a href="/pose-jump/">Mandible jump: no-collision preview</a></p>',
            1,
        )
        root_index.write_text(root)
    cherries.log_output(out)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
