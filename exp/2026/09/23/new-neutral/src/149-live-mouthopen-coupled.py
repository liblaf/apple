"""Publish a small polling view of an in-progress coupled MouthOpen fit."""

from __future__ import annotations

import html
import sys
from pathlib import Path

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible"
sys.path.insert(0, str(JOINT / "src"))
from joint_common import ProfileJoint, write_json  # noqa: E402


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/inverse-mouthopen-coupled-003"
    output_dir: Path = GROUP / "data/review-isfixed-001/mouthopen-continuation"


def _link(path: Path, target: Path) -> None:
    if path.exists() or path.is_symlink():
        assert path.is_symlink(), path
        path.unlink()
    path.symlink_to(target.resolve())


def _page(run_name: str) -> str:
    title = f"MouthOpen coupled continuation: {html.escape(run_name)}"
    return f"""<!doctype html>
<html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>{title}</title>
<style>
body{{font:16px system-ui,sans-serif;max-width:850px;margin:2rem auto;padding:0 1rem;color:#1d2433}} h1{{margin-bottom:.2rem}}
.notice{{background:#fff3cd;border-left:5px solid #b7791f;padding:1rem;font-weight:650}} .live{{color:#0b6b35;font-weight:650}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:.75rem;margin:1rem 0}} .card{{border:1px solid #cbd5e1;border-radius:8px;padding:.8rem;background:#f8fafc}}
.label{{color:#52606d;font-size:.85rem}} .value{{font-size:1.25rem;font-variant-numeric:tabular-nums;margin-top:.2rem}} code{{font-size:.9em}}
</style>
<h1>{title}</h1>
<p class=\"live\" id=\"live\">Loading live receipts…</p>
<p class=\"notice\">Inverse status: <strong>NOT converged and not independently audited.</strong> Values below are the latest accepted progress row only; they are not a published fit result.</p>
<div class=\"grid\">
<div class=\"card\"><div class=\"label\">Accepted updates in this continuation</div><div class=\"value\" id=\"accepted\">—</div></div>
<div class=\"card\"><div class=\"label\">Positional fit RMS</div><div class=\"value\" id=\"rms\">—</div></div>
<div class=\"card\"><div class=\"label\">Objective components</div><div class=\"value\" id=\"objective\">—</div></div>
<div class=\"card\"><div class=\"label\">Forward force</div><div class=\"value\" id=\"force\">—</div></div>
<div class=\"card\"><div class=\"label\">Jaw rotation</div><div class=\"value\" id=\"rotation\">—</div></div>
<div class=\"card\"><div class=\"label\">Jaw translation</div><div class=\"value\" id=\"translation\">—</div></div>
<div class=\"card\"><div class=\"label\">Inverted tetrahedra</div><div class=\"value\" id=\"inversions\">—</div></div>
</div>
<p id=\"details\">Waiting for <code>progress.jsonl</code>.</p>
<p><a href=\"../mouthopen-coupled/\">Prior independently audited full-surface MouthOpen page</a></p>
<p>Receipts: <a href=\"progress.jsonl\">progress.jsonl</a> · <a href=\"summary.json\">summary.json</a></p>
<script>
const put=(id,value)=>document.getElementById(id).textContent=value;
const number=(v,d=3)=>Number.isFinite(v)?v.toFixed(d):"—";
const components=(v)=>Object.entries(v||{{}}).filter(([,x])=>Number.isFinite(x)).map(([k,x])=>`${{k}}=${{Number(x).toExponential(3)}}`).join(" · ")||"position-only objective";
async function refresh(){{
  try {{
    const response=await fetch(`progress.jsonl?now=${{Date.now()}}`,{{cache:"no-store"}});
    if(!response.ok) throw new Error(`progress HTTP ${{response.status}}`);
    const rows=(await response.text()).trim().split("\\n").filter(Boolean).map(JSON.parse);
    if(!rows.length) throw new Error("no completed progress row yet");
    const row=rows.at(-1), geometry=row.geometry||{{}};
    const protocolResponse=await fetch(`protocol.json?now=${{Date.now()}}`,{{cache:"no-store"}});
    if(!protocolResponse.ok) throw new Error(`protocol HTTP ${{protocolResponse.status}}`);
    const protocol=await protocolResponse.json();
    const hasMixedObjective=protocol.objective_terms!==undefined;
    const positional=hasMixedObjective ? row.positional_fit_rms_mm : row.fit_rms_mm;
    if(!Number.isFinite(positional)) throw new Error("missing explicit positional_fit_rms_mm for regularized objective");
    if(hasMixedObjective && (row.loss_components===null || typeof row.loss_components!=="object")) throw new Error("missing loss_components for regularized objective");
    put("accepted", `${{row.local_iteration ?? row.iteration ?? "—"}} (Adam step ${{row.optimizer_step ?? "—"}})`);
    put("rms", `${{number(positional)}} mm`);
    put("objective", components(row.loss_components));
    put("force", `${{number(row.force_norm_n,6)}} N / ${{number(row.force_threshold_n,6)}} N`);
    put("rotation", `${{number(row.pose_rotation_degrees)}}°`);
    put("translation", `${{number(row.pose_translation_mm)}} mm`);
    put("inversions", `${{geometry.inverted_tetrahedra ?? "—"}}; min detF ${{number(geometry.detF_min,6)}}`);
    put("details", `Progress rows: ${{rows.length}} · forward converged: ${{String(row.forward_converged)}} · contact valid: ${{String(row.contact_valid)}} · live refresh every 3 s.`);
    const summaryResponse=await fetch(`summary.json?now=${{Date.now()}}`,{{cache:"no-store"}});
    if(!summaryResponse.ok) throw new Error(`summary HTTP ${{summaryResponse.status}}`);
    const summary=await summaryResponse.json();
    put("live", `Run status: ${{summary.status}} · receipt loaded at ${{new Date().toLocaleTimeString()}}`);
  }} catch(error) {{
    put("live", `Waiting for readable progress receipt: ${{error.message}}`);
  }}
}}
refresh(); setInterval(refresh,3000);
</script></html>
"""


def main(cfg: Config) -> None:
    run_dir = cfg.run_dir.resolve()
    assert run_dir.is_dir(), run_dir
    output = cfg.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _link(output / "progress.jsonl", run_dir / "progress.jsonl")
    _link(output / "summary.json", run_dir / "summary.json")
    _link(output / "protocol.json", run_dir / "protocol.json")
    (output / "index.html").write_text(_page(run_dir.name))
    receipt = {
        "schema": "mouthopen-coupled-live-progress-page-v2",
        "run_dir": str(run_dir),
        "exposed_files": ["progress.jsonl", "summary.json", "protocol.json"],
        "checkpoint_exposed": False,
        "inverse_status": "not_converged_not_independently_audited",
        "prior_audited_page": "../mouthopen-coupled/",
    }
    write_json(output / "receipt.json", receipt)
    cherries.log_output(output / "index.html")
    cherries.log_output(output / "receipt.json")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
