"""Render saved contact-equilibrated rigid-pose checkpoints and live progress."""

# ruff: noqa: E402, PLR0915
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint, sha256, write_json


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/pose-rigid-diagnostic-001"
    live_run_dir: Path | None = None
    review_dir: Path = GROUP / "data/review-repaired-reference-005"


def record(path: Path):
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def poly(points: np.ndarray, triangles: np.ndarray):
    return pv.PolyData(
        points, np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    )


def main(cfg: Config):
    run = cfg.run_dir.resolve()
    live_run = cfg.live_run_dir.resolve() if cfg.live_run_dir is not None else run
    protocol = json.loads((run / "protocol.json").read_text())
    progress = run / "continuation/summary.json"
    summary = json.loads(progress.read_text())
    completed = [r for r in summary["steps"] if "proposal" in r]
    row = completed[-1] if completed else None
    source_receipt_path = run / "source-corrector.json"
    source_receipt = (
        json.loads(source_receipt_path.read_text())
        if source_receipt_path.exists()
        else None
    )
    snapshot_result = row["proposal"] if row else source_receipt
    checkpoint = (
        Path(row["proposal"]["checkpoint"]["path"]) if row else run / "source.pt"
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    u = state["displacement_m"].numpy()
    pose = state["pose_rad_m"].numpy()
    rotation_degrees = float(np.linalg.norm(pose[:3]) * 180 / np.pi)
    translation_mm = float(np.linalg.norm(pose[3:]) * 1000)
    topology = Path(protocol["rendering"]["archive"]["path"])
    assert sha256(topology) == protocol["rendering"]["archive"]["sha256"]
    with np.load(topology) as data:
        a = {k: data[k] for k in data.files}
    target_path = Path(protocol["blendshapes"]["path"])
    assert sha256(target_path) == protocol["blendshapes"]["sha256"]
    with np.load(target_path) as data:
        triangles = data["skin_triangles"]
        xyz = data["new_neutral_points_m"][triangles]
        area = 0.5 * np.linalg.norm(
            np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
        )
        weights = np.zeros(len(data["skin_global_ids"]))
        np.add.at(weights, triangles.ravel(), np.repeat(area / 3.0, 3))
        weights /= weights.sum()
        target = poly(
            data["target_points_m"][list(data["expression_names"]).index("MouthOpen")],
            data["skin_triangles"],
        )
    full = a["full_reference_points_m"] + u
    fit = poly(full[a["skin_global_ids"]], a["skin_triangles"])
    fit_rms_mm = float(
        np.sqrt((weights[:, None] * (fit.points - target.points) ** 2).sum()) * 1000
    )
    anatomy = {
        key: poly(full[a[f"{key}_global_ids"]], a[f"{key}_triangles"])
        for key in ("cranium", "mandible", "eye")
    }
    points = np.concatenate(
        [target.points, fit.points, *[x.points for x in anatomy.values()]]
    )
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2
    span = float(max(hi - lo))
    out = cfg.review_dir / "rigid-forward"
    out.mkdir(exist_ok=True)
    label = f"Jaw {rotation_degrees:.3f}° · {translation_mm:.3f} mm"
    for view in ("front", "side"):
        p = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1800, 900))
        p.set_background("#f7f7f5")
        for col, skin in enumerate((target, fit)):
            p.subplot(0, col)
            p.add_text(
                ("MouthOpen target" if col == 0 else label) + "\n" + view,
                position="upper_left",
                font_size=16,
                color="#222222",
            )
            if col == 1:
                for key, mesh in anatomy.items():
                    p.add_mesh(
                        mesh,
                        color="#9fc5e8" if key == "eye" else "#e1d5bb",
                        smooth_shading=True,
                    )
            p.add_mesh(
                skin, color="#7e858c" if col == 0 else "#bd7c63", smooth_shading=True
            )
            direction = (
                np.array([0, 0, 2.7]) if view == "front" else np.array([2.7, 0, 0])
            )
            p.camera_position = [
                (center + span * direction).tolist(),
                center.tolist(),
                [0, 1, 0],
            ]
            p.camera.parallel_projection = True
            p.camera.parallel_scale = 0.65 * span
            p.reset_camera_clipping_range()
        p.show(screenshot=out / f"current-{view}.png", auto_close=True)
    live_paths = {
        "continuation-progress.json": live_run / "continuation/summary.json",
        "source-relaxation.json": live_run / "source-relaxation/latest.json",
    }
    for name, target_path in live_paths.items():
        link = out / name
        assert not link.exists() or link.is_symlink(), link
        if link.is_symlink():
            link.unlink()
        link.symlink_to(target_path)
    snapshot = {
        "schema": "rigid-forward-progress-review-v1",
        "fit_rms_mm": fit_rms_mm,
        "pose_rad_m": pose.tolist(),
        "rotation_degrees": rotation_degrees,
        "translation_mm": translation_mm,
        "checkpoint": record(checkpoint),
        "progress_snapshot": summary,
        "snapshot_run_dir": str(run),
        "live_run_dir": str(live_run),
        "live_paths": {name: str(path) for name, path in live_paths.items()},
        "geometry": snapshot_result["geometry"] if snapshot_result else None,
        "force_norm_n": (
            snapshot_result["forward"]["grad_norm"] * 1e6 if snapshot_result else None
        ),
        "physics_scope": "Pose initialization at zero muscle activation, not an inverse fit",
    }
    write_json(out / "receipt.json", snapshot)
    force_plot = out / "rigid-forward-force.png"
    force_receipt = out / "rigid-forward-force-receipt.json"
    force_section = (
        '<p><a href="rigid-forward-force.png">Saved no-contact force snapshots</a> '
        '· <a href="rigid-forward-force-receipt.json">snapshot receipt</a></p>'
        if force_plot.is_file() and force_receipt.is_file()
        else ""
    )
    (out / "index.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>Rigid jaw forward progress</title><style>body{{font:16px/1.5 system-ui;max-width:1200px;margin:30px auto;padding:0 20px;background:#f7f7f5;color:#242424}}img{{width:100%}}.status{{background:#fff0ce;padding:16px}}</style><p><a href="/mandible-estimate/">Accepted pose estimate</a> · <a href="/">Neutral review</a></p><h1>Bounded 6-DoF jaw forward solve</h1><p>Each update uses at most 1° rotation and 1 mm pivot translation, followed by smoothed carry, no-collision relaxation, push-out and a contact equilibrium. Muscle activation is still zero.</p><p class="status">Force/contact convergence and tetrahedron validity are reported separately. Inverted cells remain a validity limitation.</p><pre id="status">Loading status…</pre><p>Images show the saved contact-equilibrated checkpoint from <code>{run.name}</code>, identified in <a href="receipt.json">the image receipt</a>. Live status follows <code>{live_run.name}</code> independently; missing future files mean that phase has not started, not failure.</p>{force_section}<img src="current-front.png"><img src="current-side.png"><script>const runName={json.dumps(live_run.name)};async function load(name){{let r=await fetch(name+'?'+Date.now());if(!r.ok)throw new Error('not ready');return r.json()}}async function tick(){{try{{let s=await load('continuation-progress.json');let r=s.steps.at(-1);let done=s.steps.filter(x=>x.proposal);document.querySelector('#status').textContent=JSON.stringify({{live_run:runName,phase:'continuation',status:s.status,completed_steps:done.length,total_steps:s.schedule.steps.length,current_phase:r?.status,last_force_N:done.at(-1)?.proposal.forward.grad_norm*1e6,last_inverted_tetrahedra:done.at(-1)?.proposal.geometry.inverted_tetrahedra,failure:s.failure?.message}},null,2)}}catch(_continuationMissing){{try{{let s=await load('source-relaxation.json');document.querySelector('#status').textContent=JSON.stringify({{live_run:runName,phase:'source no-contact relaxation',status:s.kind,accepted:s.accepted,no_contact_force_N:s.force_norm*1e6,elapsed_seconds:s.seconds,checkpoint:s.checkpoint?.path,failure:s.failure}},null,2)}}catch(_sourceMissing){{document.querySelector('#status').textContent=JSON.stringify({{live_run:runName,phase:'waiting for source no-contact or continuation progress'}},null,2)}}}}}}tick();setInterval(tick,10000)</script>"""
    )
    for index in (
        cfg.review_dir / "index.html",
        cfg.review_dir / "mandible-estimate/index.html",
    ):
        text = index.read_text()
        text = text.replace(
            "<b>Awaiting your pose review.</b> This is the rigid pose estimated from the chin. No smoothed carry or forward solve has run for this pose.",
            "<b>Pose accepted.</b> The bounded smoothed-carry forward solve is now running; see its progress link below.",
        )
        if 'id="rigid-forward-link"' not in text:
            text = text.replace(
                "</h1>",
                '</h1><p id="rigid-forward-link"><a href="/rigid-forward/">View bounded rigid-pose forward progress</a></p>',
                1,
            )
        index.write_text(text)
    cherries.log_output(out)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
