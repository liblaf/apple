"""Package selected saved-state shape and activation figures for the report."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
OUTPUT = GROUP / "data/84-idea-comparison"
VIEWS = ("side-context", "region1-mouth-corner")
STATES = {
    "raw6-refit-off-200": {
        "label": "Corrected Raw6 · smoothness off",
        "step": 200,
        "fit": 1.397767,
        "motion": 4.818188,
        "inversions": 4,
    },
    "raw6-refit-on-200": {
        "label": "Corrected Raw6 · smoothness on",
        "step": 200,
        "fit": 1.404671,
        "motion": 4.807193,
        "inversions": 4,
    },
    "axis-on-128": {
        "label": "Learned axis · smoothness on",
        "step": 128,
        "fit": 0.681094,
        "motion": 5.179578,
        "inversions": 94,
    },
    "axis-off-best-15": {
        "label": "Learned axis · smoothness off · best fit",
        "step": 15,
        "fit": 3.539459,
        "motion": 4.121976,
        "inversions": 541,
    },
    "raw6-corrected-rest-start-200": {
        "label": "Corrected Raw6 · smoothness off",
        "step": 200,
        "fit": 1.835697,
        "motion": 4.150593,
        "inversions": 1,
    },
    "psd-off-1024": {
        "label": "PSD active stress · smoothness off",
        "step": 1024,
        "fit": 1.610246,
        "motion": 4.125968,
        "inversions": 0,
    },
}
PAIRS = [
    {
        "id": "learned-axis",
        "title": "Learned axis",
        "states": ["raw6-refit-on-200", "axis-on-128"],
        "question": "How does a single contraction axis change the achieved shape and field?",
        "observation": "The selected learned-axis result fits more closely, while retaining more inverted tetrahedra. Its field has one contractile mode per cell; Raw6 can use several signed modes.",
        "limit": "Achieved results with different starts, rates, regularization coefficients, and update budgets. This pair does not isolate the causal effect of the axis restriction.",
        "weighting": "Fit and motion use uniform fitted-face vector RMS.",
    },
    {
        "id": "smoothness",
        "title": "Spatial smoothness",
        "states": ["raw6-refit-off-200", "raw6-refit-on-200"],
        "question": "Does smoother activation produce a smoother reconstructed shape?",
        "observation": "With the same start and update budget, S(C) decreases 11.0%, but the surface residual decreases only 0.20%. Compare the field texture with the nearly unchanged mouth geometry.",
        "limit": "This pair meets the frozen fit and motion matching tolerances. The prior acts on C; the glyphs display the strongest mode of the common effective field Z.",
        "weighting": "Fit and motion use uniform fitted-face vector RMS.",
    },
    {
        "id": "active-stress",
        "title": "Active stress",
        "states": ["raw6-corrected-rest-start-200", "psd-off-1024"],
        "question": "How does PSD active stress compare with the corrected Raw6 formulation?",
        "observation": "Both selected unsmoothed states have relatively regular geometry at nearly equal motion. Raw6 can have negative effective activation modes; PSD constrains them to be nonnegative.",
        "limit": "This is the corrected physical-volume, rest-start baseline, a different Raw6 run from the other pairs. Fit and optimization histories are not matched. The earlier incorrect-volume baseline is excluded.",
        "weighting": "This pair uses area-weighted fit and motion RMS. Do not rank these values against the other pairs.",
    },
]

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shape and activation comparisons</title>
<style>
:root{color-scheme:light;--bg:#f4f2ed;--panel:#fffef9;--ink:#172322;--muted:#55625d;--line:#c9d1cc;--accent:#176a70}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:24px clamp(12px,3vw,48px) 60px}a{color:var(--accent);text-underline-offset:3px}h1{font-size:clamp(2rem,4.4vw,4rem);line-height:1.05;letter-spacing:-.04em;margin:24px 0 14px}h2{line-height:1.2}.intro{max-width:850px;color:var(--muted)}.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:20px 0}.tabs{display:flex;gap:8px;flex-wrap:wrap}button,select{font:inherit;border:1px solid var(--line);border-radius:7px;background:var(--panel);color:var(--ink);padding:9px 14px}button{cursor:pointer}button[aria-pressed=true]{background:var(--accent);color:white;border-color:var(--accent)}:focus-visible{outline:3px solid #b97d24;outline-offset:3px}.muted{color:var(--muted)}.lead{font-size:1.12rem;max-width:950px}.note{border-left:4px solid var(--accent);padding:10px 16px;background:var(--panel);margin:18px 0}.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}.card header{padding:14px 18px;border-bottom:1px solid var(--line)}.card h3{margin:0 0 5px;font-size:1.14rem}.metrics{font-variant-numeric:tabular-nums;font-size:.9rem;color:var(--muted)}figure{margin:0;border-top:1px solid var(--line)}figure a{display:block}img{display:block;width:100%;height:auto}figcaption{padding:8px 14px;font-size:.9rem;color:var(--muted)}details{margin-top:20px;padding:12px 16px;border:1px solid var(--line);border-radius:8px;background:var(--panel)}summary{cursor:pointer;font-weight:650}.target{max-width:550px;margin-top:12px}.downloads{display:flex;gap:16px;flex-wrap:wrap;margin:25px 0}.method{max-width:1000px}.method li{margin:8px 0}.footer{margin-top:32px;border-top:1px solid var(--line);padding-top:16px;font-size:.88rem}.support{display:flex;gap:12px;flex-wrap:wrap}@media(max-width:720px){.pair{grid-template-columns:1fr}.controls{align-items:flex-start}.card header{padding:12px}main{padding-top:15px}}@media print{.controls{display:none}.pair{grid-template-columns:1fr 1fr}.card{break-inside:avoid}}
</style></head><body><main>
<a href="index.html">← Full study report</a><h1>Shape and activation</h1>
<p class="intro">Three comparisons from existing saved states. Within every pair, camera, deformation scale, glyph length, and color scales are fixed. Overview is shown first. Click any image to open the original 1,800 × 1,800 figure and zoom in.</p>
<nav class="tabs" id="pair-tabs" aria-label="Comparison"></nav>
<section aria-live="polite"><h2 id="question"></h2><p class="lead" id="observation"></p></section>
<div class="controls"><label>View <select id="view"><option value="side-context">Overview</option><option value="region1-mouth-corner">Mouth corner</option></select></label><label>Show <select id="mode"><option value="both">Shape + activation</option><option value="shape">Shape only</option><option value="activation">Activation only</option><option value="omitted">Omitted tensor modes</option></select></label></div>
<p class="muted" id="weighting"></p><div id="pair" class="pair"></div>
<p class="note" id="limit"></p>
<details><summary>Target shape in the same camera</summary><figure class="target"><a id="target-link"><img id="target" alt="Target shape in the selected comparison camera"></a><figcaption>Supplied target deformation, scale 1. No target activation field is assumed.</figcaption></figure></details>
<details class="method"><summary>How to read the activation comparison</summary>
<ul><li><strong>One line per active tetrahedron.</strong> Its center follows the saved deformation and its direction is the transported strongest contractile mode. Muscles behind the front muscle are hidden; interior tetrahedra of the visible muscle remain.</li>
<li><strong>More contraction means a longer line.</strong> Length = 4.5 mm × a, with a = 1 − 1/√(1 + max(λₘₐₓ(Z), 0)). The same linear 0–100% range colors every field. For learned axis, this is exactly its commanded shortening. For Raw6 and PSD, it is a display-equivalent contraction of the dominant effective mode.</li>
<li><strong>The line does not show every tensor mode.</strong> “Omitted tensor modes” maps the fraction of squared Frobenius norm outside the displayed positive mode: 1 − max(λₘₐₓ, 0)² / ‖Z‖². A zero tensor has zero omitted fraction. The map uses one fixed 0–100% range. Negative and secondary positive modes both contribute.</li>
<li>The common field is Z = BBᵀ − I for Raw6 and learned axis, and Z = Q/μ for PSD. The line is unsigned and does not identify an anatomical fiber or observed tissue strain. Full tensors remain in the exported data.</li>
<li>Glyph panels draw no muscle outlines or muscle surfaces. The separate omitted-mode map colors actual muscle surface cells, without edge or silhouette overlays. No shape smoothing, deformation amplification, or inverse fitting was performed.</li></ul>
</details>
<details><summary>Supporting result: learned axis without smoothness</summary><p>The best-fit checkpoint at step 15 has fit RMS 3.539 mm and 541 inverted tetrahedra. It is supporting evidence, not a matched comparison.</p><div class="support" id="support"></div></details>
<div class="downloads"><a href="downloads/ideas/figures.zip">Download separate figures</a><a href="downloads/ideas/report.md">Methods and observations</a><a href="downloads/ideas/selection.json">Exact checkpoint selection</a><a href="downloads/ideas/activation-summary.json">Activation receipt</a><a href="downloads/ideas/shape-summary.json">Shape receipt</a><a href="downloads/ideas/audit-summary.json">Independent tensor audit</a><a href="downloads/ideas/verification.json">Geometry and field verification</a></div>
<p class="footer">Existing saved states · reproducible standalone figure assets · <a href="manifest.json">Site file hashes</a></p>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);let selected=data.pairs[0];const byId=id=>document.getElementById(id);
function picture(state,kind,view){const f=document.createElement('figure'),a=document.createElement('a'),img=document.createElement('img'),caption=document.createElement('figcaption');const path=`assets/ideas/${kind}/${state}/${view}.png`;a.href=path;img.src=path;img.width=1800;img.height=1800;img.alt=`${data.states[state].label}, update ${data.states[state].step}: ${kind}, ${view}`;img.loading='lazy';a.append(img);f.append(a);caption.textContent={shape:'Saved deformed shape · scale 1 · flat shading',activation:'Dominant contractile mode · 4.5 mm at 100% · same color scale',omitted:'Squared tensor magnitude outside the line · fixed 0–100%'}[kind];f.append(caption);return f;}
function render(){const v=byId('view').value,m=byId('mode').value;byId('question').textContent=selected.question;byId('observation').textContent=selected.observation;byId('limit').textContent=selected.limit;byId('weighting').textContent=selected.weighting;byId('pair').replaceChildren();for(const id of selected.states){const s=data.states[id],card=document.createElement('article'),header=document.createElement('header'),h=document.createElement('h3'),metrics=document.createElement('div');card.className='card';h.textContent=`${s.label} · update ${s.step}`;metrics.className='metrics';metrics.textContent=`Fit ${s.fit.toFixed(3)} mm · motion ${s.motion.toFixed(3)} mm · inversions ${s.inversions}`;header.append(h,metrics);card.append(header);for(const kind of m==='both'?['shape','activation']:[m])card.append(picture(id,kind,v));byId('pair').append(card);}const target=`assets/ideas/shape/target/${v}.png`;byId('target').src=target;byId('target-link').href=target;for(const b of byId('pair-tabs').children)b.setAttribute('aria-pressed',String(b.dataset.id===selected.id));}
for(const p of data.pairs){const b=document.createElement('button');b.textContent=p.title;b.dataset.id=p.id;b.onclick=()=>{selected=p;location.hash=p.id;render();};byId('pair-tabs').append(b);}for(const id of ['view','mode'])byId(id).addEventListener('change',render);for(const v of ['side-context','region1-mouth-corner'])for(const k of ['shape','activation','omitted']){const a=document.createElement('a');a.href=`assets/ideas/${k}/axis-off-best-15/${v}.png`;a.textContent=`${v==='side-context'?'Overview':'Mouth'} · ${k}`;byId('support').append(a);}const initial=data.pairs.find(p=>p.id===location.hash.slice(1));if(initial)selected=initial;render();
</script></main></body></html>"""


def record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


class Config(cherries.BaseConfig):
    output_dir: Path = OUTPUT


def main(cfg: Config) -> None:
    output = cfg.output_dir.resolve()
    assert not output.exists(), output
    web = output / "web"
    web.mkdir(parents=True)
    copied = []

    def copy(source: Path, target: str) -> None:
        assert source.is_file(), source
        destination = web / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_record, output_record = record(source), record(destination)
        assert source_record["sha256"] == output_record["sha256"]
        copied.append(
            {"source": source_record, "output": output_record, "site_path": target}
        )

    for state in (*STATES, "target"):
        for view in VIEWS:
            copy(
                GROUP / f"data/81-idea-shapes/geometry/{state}/{view}.png",
                f"assets/ideas/shape/{state}/{view}.png",
            )
            if state != "target":
                for folder, name in (
                    ("geometry", "activation"),
                    ("companions", "omitted"),
                ):
                    copy(
                        GROUP / f"data/83-idea-activation/{folder}/{state}/{view}.png",
                        f"assets/ideas/{name}/{state}/{view}.png",
                    )
    for source, name in (
        ("docs/84-shape-activation-comparison.md", "report.md"),
        ("data/80-idea-result-selection/selection.json", "selection.json"),
        ("data/81-idea-shapes/summary.json", "shape-summary.json"),
        ("data/82-idea-activation-audit-v2/summary.json", "audit-summary.json"),
        ("data/83-idea-activation/summary.json", "activation-summary.json"),
        ("data/85-idea-verification/summary.json", "verification.json"),
    ):
        copy(GROUP / source, f"downloads/ideas/{name}")
    for number, name in (
        (81, "render-idea-shapes"),
        (82, "audit-idea-activation"),
        (83, "render-idea-activation"),
        (84, "build-idea-comparison"),
        (85, "verify-idea-comparison"),
    ):
        copy(
            GROUP / f"src/{number}-{name}.py",
            f"downloads/ideas/sources/{number}-{name}.py",
        )
    for name in ("muscle_glyph_context.py", "experiment_profile.py"):
        copy(GROUP / "src" / name, f"downloads/ideas/sources/{name}")
    (web / "ideas.html").write_text(
        HTML.replace(
            "__DATA__",
            json.dumps({"pairs": PAIRS, "states": STATES}, ensure_ascii=False),
        )
    )
    archive = web / "downloads/ideas/figures.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as bundle:
        for path in sorted(web.rglob("*.png")):
            bundle.write(path, path.relative_to(web / "assets/ideas"))
        bundle.writestr(
            "README.md",
            """# Shape and activation comparison figures

This bundle contains 38 separate 1800 x 1800 PNGs for later slide layout.
Open the interactive comparison and full methods at:
the locally generated ideas.html

- shape/: six saved deformed states plus target, each in two frozen cameras.
- activation/: one dominant contractile-mode line per visible active tetrahedron.
- omitted/: fraction of squared tensor magnitude omitted by that line.

Use matching state and view filenames together. All images use deformation scale
1. Glyph length is 4.5 mm times display-equivalent contraction, with no length
floor or per-state scaling. Glyph color and omitted-mode maps each use fixed
0-100% scales. The line is exact for the rank-one learned-axis model; it shows
only the strongest positive mode for Raw6 and PSD. It is not an anatomical fiber
or observed strain. Muscle occlusion retains internal tetrahedra of the front
muscle, while hiding other muscles behind it.

The three selected pairs are:
1. Learned axis: raw6-refit-on-200 vs axis-on-128.
2. Spatial smoothness: raw6-refit-off-200 vs raw6-refit-on-200.
3. Active stress: raw6-corrected-rest-start-200 vs psd-off-1024.

axis-off-best-15 is supporting evidence. Model pairs have different optimization
histories; only the Raw6 smoothness pair is a controlled same-start comparison.
selection.json records exact source checkpoint paths and hashes. No new fitting
was performed. File names identify the saved state, not a common update budget.
""",
        )
        bundle.write(web / "downloads/ideas/selection.json", "selection.json")
    receipt = {
        "status": "completed",
        "scope": "Static packaging only; no fitting or new metric computation",
        "sources_and_copies": copied,
        "page": record(web / "ideas.html"),
        "figure_archive": record(archive),
        "pairs": PAIRS,
        "states": STATES,
    }
    (output / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    cherries.log_metrics({"report/figures": 38, "report/comparisons": 3})
    cherries.log_output(output)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
