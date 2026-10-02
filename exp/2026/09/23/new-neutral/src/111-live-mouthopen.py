"""Publish a polling-only MouthOpen initialization status page."""
# ruff: noqa: ANN001, PT018

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint, sha256, write_json  # noqa:E402


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/inverse-mouthopen-003"
    review_dir: Path = GROUP / "data/review-repaired-reference-005"
    overwrite: bool = False


def rec(p):
    return {"path": str(p.resolve()), "sha256": sha256(p)}


def main(c: Config):
    run, review = c.run_dir.resolve(), c.review_dir.resolve()
    out = review / "mouthopen"
    if out.exists():
        assert c.overwrite
        shutil.rmtree(out)
    est = run / "chin-estimate.json"
    preview = run / "chin-patch-preview.png"
    progress = run / "initialization-progress.json"
    assert est.is_file() and preview.is_file() and progress.is_file()
    chin = json.loads(est.read_text())
    assert abs(chin["angle_deg"] - 11.558760337864761) < 1e-9
    pose = GROUP / "data/pose-jump-002/summary.json"
    assert pose.is_file()
    out.mkdir()
    shutil.copy2(est, out / "chin-estimate.json")
    shutil.copy2(preview, out / "chin-patch-preview.png")
    (out / "initialization-progress.json").symlink_to(progress)
    (out / "pose-jump-summary.json").symlink_to(pose)
    assert (out / "initialization-progress.json").resolve() == progress
    html = """<!doctype html><meta charset="utf-8"><title>MouthOpen initialization</title><style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}.status{padding:1rem;background:#fff1c7}img{max-width:100%}</style><p><a href="../">Back to neutral review</a></p><h1>MouthOpen: chin initialization, then joint optimization</h1><p class="status">Live initialization status only. No inverse endpoint, force convergence, contact, or geometry validity is claimed until a saved endpoint review replaces this page.</p><p>Geometry-only chin seed: <b>11.55876°</b>. Main run paused at 4.1294°. The smoothed jump reached 11.5588° and passed the final force/contact gates. It retains 282 inverted tetrahedra, so this remains a diagnostic. Joint optimization has not started. <a href="../pose-jump/">View the direct and smoothed no-collision shapes.</a></p><p><a href="pose-jump-summary.json">Pose-jump test summary</a></p><p> The current proposed angle is progress_proposed x this seed.</p><img src="chin-patch-preview.png" alt="Selected connected anterior chin patch"><pre id="state">Loading…</pre><script>async function tick(){try{let x=await fetch('initialization-progress.json?'+Date.now()).then(r=>r.json());let q=x[x.length-1]||x;let a=q.progress_proposed*11.558760337864761;document.querySelector('#state').textContent=JSON.stringify({iteration:q.iteration,progress_proposed:q.progress_proposed,current_angle_deg:a,internal_corrector_grad_norm:q.internal_corrector?.grad_norm},null,2)}catch(e){document.querySelector('#state').textContent='Waiting for initialization progress: '+e} }tick();setInterval(tick,10000)</script><p><a href="chin-estimate.json">Chin estimate receipt</a> · <a href="initialization-progress.json">Initialization progress</a></p>"""
    (out / "index.html").write_text(html)
    index = review / "index.html"
    text = index.read_text()
    if 'id="mouthopen"' not in text:
        index.write_text(
            text.replace(
                "</h1>",
                r'<\/h1><p id="mouthopen"><a href="mouthopen\/">Live MouthOpen initialization<\/a><\/p>',
                1,
            )
        )
    write_json(
        out / "live-receipt.json",
        {
            "schema": "mouthopen-live-init-v1",
            "estimate": rec(est),
            "preview": rec(preview),
            "progress": rec(progress),
            "scope": "Polling initialization only; no endpoint validity.",
        },
    )
    cherries.log_output(out)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
