"""Build the self-contained static activation-space smoothness report."""

# ruff: noqa: C901, EM102, RUF001, TRY003

from __future__ import annotations

import hashlib
import json
import os
import shutil
from html.parser import HTMLParser
from pathlib import Path

GROUP = Path(__file__).resolve().parents[1]
SITE = GROUP / "site"
STAGING = GROUP / ".site-staging"

COPIES = {
    "downloads/activation-glyphs.md": "docs/76-deformed-activation-glyphs.md",
    "downloads/activation-glyphs.json": "data/76-deformed-activation-glyphs/summary.json",
    "downloads/activation-glyphs-verification.json": "data/77-deformed-glyph-verification/activation-glyphs.json",
    "downloads/activation-glyph-data.zip": "data/76-deformed-activation-glyphs/glyph-data.zip",
    "downloads/rate03-full-activation-glyphs.vtp": "data/76-deformed-activation-glyphs/glyphs/rate03-on.vtp",
    "downloads/sources/76-render-deformed-activation-glyphs.py": "data/76-deformed-activation-glyphs/sources/76-render-deformed-activation-glyphs.py",
    "downloads/sources/77-verify-deformed-activation-glyphs.py": "data/77-deformed-glyph-verification/77-verify-deformed-activation-glyphs.py",
    "downloads/sources/muscle_glyph_context.py": "data/76-deformed-activation-glyphs/sources/muscle_glyph_context.py",
    # Report-facing exact figures.
    "assets/endpoint-side.png": "data/42-axis-on-endpoint-v2/geometry/side-context/target-axis-on-0128.png",
    "assets/endpoint-mouth.png": "data/42-axis-on-endpoint-v2/geometry/region1-mouth-corner/target-axis-on-0128.png",
    "assets/endpoint-sections.png": "data/44-report-figures-v2/sections/axis-on-endpoint-0128.png",
    "assets/trajectories.png": "data/44-report-figures-v2/new-pairs-trajectories.png",
    "assets/fit-motion-tradeoff.png": "data/44-report-figures-v2/new-pairs-fit-motion-surface-tradeoff.png",
    "assets/activation-variation.png": "data/44-report-figures-v2/new-pairs-activation-variation.png",
    "assets/learned-matched-mouth.png": "data/40-comparison/geometry/learned-axis-matched/region1-mouth-corner-target-off-on.png",
    "assets/learned-matched-sections.png": "data/44-report-figures-v2/sections/learned-axis-matched.png",
    "assets/raw6-matched-mouth.png": "data/40-comparison/geometry/raw6-matched/region1-mouth-corner-target-off-on.png",
    "assets/raw6-matched-sections.png": "data/44-report-figures-v2/sections/raw6-matched.png",
    "assets/learned-common-mouth.png": "data/40-comparison/geometry/learned-axis-common-update/region1-mouth-corner-target-off-on.png",
    "assets/learned-residual-off.png": "data/40-comparison/highpass/learned-axis-matched/axis-off-residual.png",
    "assets/learned-residual-on.png": "data/40-comparison/highpass/learned-axis-matched/axis-on-residual.png",
    "assets/raw6-residual-off.png": "data/40-comparison/highpass/raw6-matched/raw6-off-residual.png",
    "assets/raw6-residual-on.png": "data/40-comparison/highpass/raw6-matched/raw6-on-residual.png",
    "assets/historical-context.png": "data/44-report-figures-v2/historical-context-tradeoff.png",
    "assets/calibration-divergence.png": "data/44-report-figures-v2/calibration-vs-primary-first16.png",
    "assets/learning-rate-pilots.png": "data/44-report-figures-v2/learning-rate-calibration-trajectories.png",
    "assets/quarter-trajectories.png": "data/52-conservative-rate-comparison/optimizer-trajectories.png",
    "assets/quarter-attained-states.png": "data/52-conservative-rate-comparison/attained-state-comparison.png",
    "assets/quarter-matched-sections.png": "data/52-conservative-rate-comparison/sections/conservative-off-vs-on.png",
    "assets/quarter-off-rate-sections.png": "data/52-conservative-rate-comparison/sections/off-original-vs-conservative.png",
    "assets/quarter-on-rate-sections.png": "data/52-conservative-rate-comparison/sections/on-original-vs-conservative.png",
    "assets/quarter-off-endpoints.png": "data/52-conservative-rate-comparison/geometry/unmatched-off-endpoints.png",
    "assets/quarter-on-endpoints.png": "data/52-conservative-rate-comparison/geometry/unmatched-on-endpoints.png",
    "downloads/quarter-rate-results.md": "docs/49-quarter-rate-results.md",
    "downloads/quarter-rate-execution.md": "docs/48-conservative-rate-execution.md",
    "downloads/quarter-rate-comparison.json": "data/52-conservative-rate-comparison/summary.json",
    "downloads/quarter-rate-states.csv": "data/52-conservative-rate-comparison/selected-states.csv",
    "downloads/quarter-rate-verification.json": "data/53-conservative-rate-verification/summary.json",
    "downloads/sources/52-compare-conservative-rate.py": "data/52-conservative-rate-comparison/sources/52-compare-conservative-rate.py",
    "downloads/sources/53-verify-conservative-rate.py": "data/53-conservative-rate-verification/sources/53-verify-conservative-rate.py",
    "assets/rate03-trajectories.png": "data/59-rate03-report-figures/optimizer-trajectories.png",
    "assets/rate03-attained-states.png": "data/59-rate03-report-figures/global-fit-motion-comparison.png",
    "assets/rate03-original-sections.png": "data/58-rate03-comparison/sections/original-vs-rate03.png",
    "assets/rate03-quarter-sections.png": "data/58-rate03-comparison/sections/quarter-vs-rate03.png",
    "assets/rate03-endpoints.png": "data/58-rate03-comparison/geometry/unmatched-smoothed-endpoints.png",
    "downloads/rate-03-results.md": "docs/56-rate-03-results.md",
    "downloads/rate-03-execution.md": "docs/55-rate-03-execution.md",
    "downloads/rate-03-freeze.json": "data/54-rate-03-settings/summary.json",
    "downloads/rate-03-comparison.json": "data/58-rate03-comparison/summary.json",
    "downloads/rate-03-states.csv": "data/58-rate03-comparison/selected-states.csv",
    "downloads/rate-03-verification.json": "data/57-rate-03-verification/summary.json",
    "downloads/rate-03-plot-receipt.json": "data/59-rate03-report-figures/summary.json",
    "downloads/sources/54-freeze-rate-03.py": "data/54-rate-03-settings/sources/54-freeze-rate-03.py",
    "downloads/sources/57-verify-rate-03.py": "data/57-rate-03-verification/sources/verifier/57-verify-rate-03.py",
    "downloads/sources/58-compare-rate-03.py": "data/58-rate03-comparison/sources/58-compare-rate-03.py",
    "downloads/sources/59-render-rate-03-report-plots.py": "data/59-rate03-report-figures/sources/59-render-rate-03-report-plots.py",
    "downloads/sources/study_runner.py": "data/25-learned-axis-smooth/sources/experiment/study_runner.py",
    # Human-readable evidence and source snapshots.
    "downloads/report.md": "docs/40-results.md",
    "downloads/protocol.md": "docs/12-learned-axis-plan.md",
    "downloads/execution-record.md": "docs/14-execution-record.md",
    "downloads/calibration-revision.md": "docs/15-calibration-revision.md",
    "downloads/comparison-summary.json": "data/40-comparison/summary.json",
    "downloads/selected-states.csv": "data/40-comparison/selected-states.csv",
    "downloads/historical-context.csv": "data/40-comparison/historical-context.csv",
    "downloads/regional-metrics.csv": "data/43-regional-matched-metrics/regional-matched-metrics.csv",
    "downloads/regional-metrics-summary.json": "data/43-regional-matched-metrics/summary.json",
    "downloads/render-receipt.json": "data/42-axis-on-endpoint-v2/summary.json",
    "downloads/verification.json": "data/50-verification-v2/summary.json",
    "downloads/calibration-audit.json": "data/18-calibration-main-divergence-audit/summary.json",
    "downloads/learning-rate-diagnosis.md": "docs/45-learning-rate-diagnosis.md",
    "downloads/implementation-comparison.md": "docs/46-implementation-comparison.md",
    "downloads/conservative-rate-plan.md": "docs/47-conservative-rate-plan.md",
    "downloads/conservative-rate-settings.json": "data/47-conservative-rate-settings/settings.json",
    "downloads/rate-03-plan.md": "docs/54-rate-03-plan.md",
    "downloads/rate-03-settings.json": "data/54-rate-03-settings/settings.json",
    "downloads/psd-rate-switch.json": "../../07/tensor-active-stress/data/102-larger-rate32/resume.json",
    "downloads/psd-final-continuation.json": "../../07/tensor-active-stress/data/102-fit1024/resume.json",
    "downloads/psd-final-summary.json": "../../07/tensor-active-stress/data/102-fit1024/summary.json",
    "downloads/pilot-endpoints.csv": "data/45-learning-rate-evidence/pilot-endpoints.csv",
    "downloads/primary-milestones.csv": "data/45-learning-rate-evidence/primary-milestones.csv",
    "downloads/learning-rate-evidence.json": "data/45-learning-rate-evidence/summary.json",
    "downloads/plot-receipt-v2.json": "data/44-report-figures-v2/summary.json",
    "downloads/sources/study_metrics.py": "src/study_metrics.py",
    "downloads/sources/activation_controls.py": "src/activation_controls.py",
    "downloads/sources/study_physics.py": "src/study_physics.py",
    "downloads/sources/volume_preserving_active.py": "src/volume_preserving_active.py",
    "downloads/sources/20-run-case.py": "src/20-run-case.py",
    "downloads/sources/psd-controls.py": "../../07/tensor-active-stress/src/tensor_controls.py",
    "downloads/sources/psd-muscle-law.py": "../../07/tensor-active-stress/src/tensor_active.py",
    "downloads/sources/psd-runner.py": "../../07/tensor-active-stress/src/20-face-inverse.py",
    "downloads/sources/raw6-refit-runner.py": "../../08/local-skin-prestrain/src/20-run-case.py",
    "downloads/sources/40-compare.py": "src/40-compare.py",
    "downloads/sources/42-render-axis-endpoint.py": "data/42-axis-on-endpoint-v2/sources/42-render-axis-endpoint.py",
    "downloads/sources/43-extract-regional-metrics.py": "data/43-regional-matched-metrics/sources/43-extract-regional-metrics.py",
    "downloads/sources/44-render-report-figures-v2.py": "data/44-report-figures-v2/sources/44-render-report-figures-v2.py",
    "downloads/sources/45-extract-learning-rate-evidence.py": "data/45-learning-rate-evidence/sources/45-extract-learning-rate-evidence.py",
    "downloads/sources/50-verify.py": "src/50-verify.py",
    "downloads/sources/60-build-report-site.py": "src/60-build-report-site.py",
}

for glyph_state in (
    "original-off",
    "original-on",
    "quarter-off",
    "quarter-on",
    "rate03-on-128",
    "rate03-on",
):
    for glyph_view in ("side-context", "region1-mouth-corner"):
        COPIES[f"assets/glyphs/{glyph_state}/{glyph_view}.png"] = (
            f"data/76-deformed-activation-glyphs/geometry/{glyph_state}/{glyph_view}.png"
        )


IDEA_SITE = GROUP / "data/92-overview-comparison/web"
assert (IDEA_SITE / "ideas.html").is_file(), "build the idea comparison package first"
for idea_asset in sorted(IDEA_SITE.rglob("*")):
    if idea_asset.is_file():
        relative = idea_asset.relative_to(IDEA_SITE).as_posix()
        assert relative not in COPIES, relative
        COPIES[relative] = str(idea_asset.relative_to(GROUP))


for story_file in sorted((GROUP / "data/89-baseline-story").glob("*.png")):
    COPIES[f"assets/story/{story_file.name}"] = str(story_file.relative_to(GROUP))
for oblique_file in sorted((GROUP / "data/97-baseline-oblique").glob("*.png")):
    COPIES[f"assets/story/oblique/{oblique_file.name}"] = str(
        oblique_file.relative_to(GROUP)
    )
for target_file in sorted((GROUP / "data/101-oblique-target").glob("*.png")):
    COPIES[f"assets/story/oblique-target/{target_file.name}"] = str(
        target_file.relative_to(GROUP)
    )
COPIES.update(
    {
        "assets/story/muscle-location.png": "data/107-muscle-location/overview.png",
        "downloads/story/muscle-location-summary.json": "data/107-muscle-location/summary.json",
        "downloads/sources/107-render-muscle-location.py": "src/107-render-muscle-location.py",
        "downloads/story/oblique-target-summary.json": "data/101-oblique-target/summary.json",
        "downloads/sources/101-add-oblique-target.py": "src/101-add-oblique-target.py",
        "downloads/story/oblique-summary.json": "data/97-baseline-oblique/summary.json",
        "downloads/sources/97-render-baseline-oblique.py": "src/97-render-baseline-oblique.py",
        "study-details.html": "data/86-report-site-before-ideas/index.html",
        "downloads/activation-investigation.md": "docs/90-activation-investigation.md",
        "downloads/story/summary.json": "data/89-baseline-story/summary.json",
        "assets/story/cell27306-muscle-patch.png": "data/93-focused-muscle-patch/cell-patch.png",
        "downloads/story/muscle-patch-receipt.json": "data/93-focused-muscle-patch/summary.json",
        "downloads/sources/93-render-focused-muscle-patch.py": "src/93-render-focused-muscle-patch.py",
        "downloads/story/math-validation.json": str(
            Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
            / "exp/2026/09/08/physical-volume-baseline/data/10-physical-volume-validation.json"
        ),
        "downloads/correction.md": "../../07/tensor-active-stress/docs/114-physical-volume-correction.md",
        "downloads/sources/89-package-baseline-story.py": "src/89-package-baseline-story.py",
        "downloads/sources/90-report-layout.html": "src/90-report-layout.html",
    }
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


class LocalReferenceParser(HTMLParser):
    """Collect local URL references from the generated page."""

    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        attribute = "src" if tag in {"img", "script"} else "href"
        value = values.get(attribute)
        if value:
            self.references.append(value)


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Saved-state study of activation-space C-smoothness in face actuation.">
<title>Activation-space C-smoothness study</title>
<style>
:root{color-scheme:light;--page:#f4f2eb;--ink:#172322;--muted:#52605d;--panel:#fffef9;--soft:#e8ece7;--line:#c9d1cc;--accent:#176a70;--accent2:#7c982f;--warn:#a34e27;--header:#153a3b;--shadow:0 14px 34px rgba(31,48,45,.08)}
@media(prefers-color-scheme:dark){:root{color-scheme:dark;--page:#10191a;--ink:#e5eeeb;--muted:#acbbb7;--panel:#182526;--soft:#223233;--line:#415958;--accent:#83d5d4;--accent2:#b8cf69;--warn:#f0a078;--header:#0d3031;--shadow:none}}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--page);color:var(--ink);font:16px/1.62 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--accent);text-underline-offset:.16em}a:hover{text-decoration-thickness:2px}:focus-visible{outline:3px solid var(--accent);outline-offset:4px}.skip{position:absolute;left:-9999px}.skip:focus{left:1rem;top:1rem;z-index:10;background:var(--panel);padding:.6rem}
.site-header{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--header) 94%,transparent);color:#fff;border-bottom:1px solid rgba(255,255,255,.16);backdrop-filter:blur(12px)}.header-inner{max-width:1180px;margin:auto;padding:.78rem 1rem;display:flex;gap:1.25rem;align-items:center}.wordmark{color:#fff!important;text-decoration:none;font-weight:750;letter-spacing:.01em}.site-header nav{margin-left:auto;display:flex;gap:.9rem;flex-wrap:wrap}.site-header nav a{color:#e8f3f0;font-size:.88rem;text-decoration:none}.site-header nav a:hover{text-decoration:underline}
main{max-width:1120px;margin:auto;padding:clamp(1.2rem,3vw,3rem) 1rem 5rem}.eyebrow{text-transform:uppercase;letter-spacing:.13em;font-size:.78rem;font-weight:750;color:var(--accent)}h1,h2,h3{line-height:1.15;text-wrap:balance}h1{max-width:900px;font-size:clamp(2.35rem,6vw,5.3rem);letter-spacing:-.045em;margin:.35rem 0 1rem}h2{font-size:clamp(1.65rem,3vw,2.5rem);margin:4.5rem 0 .8rem;letter-spacing:-.025em}h3{font-size:1.15rem;margin:1.6rem 0 .4rem}.lede{font-size:clamp(1.05rem,2vw,1.35rem);max-width:850px;color:var(--muted)}.measure{max-width:780px}.kicker{display:inline-flex;gap:.45rem;align-items:center;padding:.25rem .65rem;border-radius:999px;background:var(--soft);font-size:.83rem;font-weight:700}.dot{width:.55rem;height:.55rem;border-radius:50%;background:var(--warn)}
.metric-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem;margin:2rem 0}.metric-card{background:var(--panel);border:1px solid var(--line);border-radius:1rem;padding:1.25rem;box-shadow:var(--shadow)}.metric-card h3{margin:0;color:var(--muted);font-size:.9rem;text-transform:uppercase;letter-spacing:.08em}.metric-value{font-size:clamp(2.4rem,8vw,4.8rem);line-height:1;margin:.75rem 0 .35rem;font-variant-numeric:tabular-nums;letter-spacing:-.05em}.metric-value span{font-size:.35em;letter-spacing:0}.threshold{display:flex;justify-content:space-between;gap:1rem;padding-top:.7rem;border-top:1px solid var(--line);color:var(--muted);font-size:.88rem}.meter{height:.48rem;border-radius:999px;background:var(--soft);overflow:hidden;margin:.75rem 0}.meter>span{display:block;height:100%;background:var(--accent2);min-width:4px}
.callout{margin:1.5rem 0;padding:1rem 1.15rem;background:var(--panel);border:1px solid var(--line);border-left:5px solid var(--warn);border-radius:.45rem}.callout strong{display:block;margin-bottom:.25rem}.plain-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin:1.2rem 0}.plain-grid>div{border-top:3px solid var(--accent);padding-top:.7rem}.plain-grid strong{display:block}.plain-grid p{margin:.25rem 0;color:var(--muted);font-size:.93rem}
.table-scroll{overflow-x:auto;border:1px solid var(--line);border-radius:.7rem;background:var(--panel)}table{border-collapse:collapse;width:100%;min-width:660px;font-variant-numeric:tabular-nums}th,td{padding:.65rem .7rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:var(--soft);font-size:.82rem;line-height:1.35}td:not(:first-child),th:not(:first-child){text-align:right}tbody tr:last-child td{border-bottom:0}.table-note{color:var(--muted);font-size:.82rem;margin:.35rem 0 .7rem}
.figure{margin:1.5rem 0 2.4rem}.figure img{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:.65rem;background:#fff}.figure figcaption{color:var(--muted);font-size:.9rem;margin:.55rem .15rem 0}.figure figcaption strong{color:var(--ink)}.figure a{display:block}.figure-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}.figure-grid .figure{margin:.5rem 0 1.4rem}.pair-images{display:grid;grid-template-columns:1fr 1fr;gap:.7rem}.pair-images img{min-width:0}
.status-grid{display:grid;grid-template-columns:1.1fr .9fr;gap:1rem}.status-card{padding:1.15rem;background:var(--panel);border:1px solid var(--line);border-radius:.7rem}.status-card h3{margin-top:0}.status-card ul{padding-left:1.2rem}.pass{color:var(--accent2);font-weight:750}.fail{color:var(--warn);font-weight:750}.downloads{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.7rem}.download{display:block;padding:.9rem;background:var(--panel);border:1px solid var(--line);border-radius:.55rem;text-decoration:none}.download:hover{border-color:var(--accent);transform:translateY(-1px)}.download strong{display:block;color:var(--ink)}.download span{color:var(--muted);font-size:.83rem}
details{margin:1rem 0;border:1px solid var(--line);border-radius:.65rem;background:var(--panel);padding:.75rem 1rem}summary{cursor:pointer;font-weight:700}code{font-size:.9em;background:var(--soft);padding:.1em .3em;border-radius:.25em}.site-footer{border-top:1px solid var(--line);color:var(--muted);font-size:.85rem;padding:1.4rem 1rem;text-align:center}
@media(max-width:760px){.site-header{position:static}.header-inner{align-items:flex-start;flex-direction:column;gap:.5rem}.site-header nav{margin-left:0;gap:.65rem}.metric-grid,.figure-grid,.status-grid,.plain-grid,.downloads{grid-template-columns:1fr}.pair-images{grid-template-columns:1fr}main{padding-inline:.8rem}h2{margin-top:3.4rem}.table-scroll{margin-right:-.8rem;border-radius:.7rem 0 0 .7rem}th,td{white-space:nowrap}}
.implementation-table{min-width:880px;table-layout:fixed}.implementation-table th,.implementation-table td{text-align:left!important;white-space:normal;overflow-wrap:break-word}.implementation-table td{font-size:.88rem;line-height:1.5}.implementation-table th:first-child{width:17%}section{scroll-margin-top:5rem}
@media print{.site-header{position:static}.downloads,.skip{display:none}main{max-width:none}.figure{break-inside:avoid}.metric-card,.status-card{box-shadow:none}}
</style>
</head>
<body>
<a class="skip" href="#main">Skip to report</a>
<header class="site-header"><div class="header-inner">
<a class="wordmark" href="#top">Activation-space smoothness</a>
<nav aria-label="Report sections"><a href="ideas.html">Shape + activation</a><a href="#activation-glyphs">Glyphs</a><a href="#methods">Methods</a><a href="#learning-rate">Learning rate</a><a href="#rate-03">Rate 0.3</a><a href="#quarter-rate">Quarter rate</a><a href="#matched">Matched result</a><a href="#endpoint">Endpoint</a><a href="#regions">Regions</a><a href="#evidence">Evidence</a></nav>
</div></header>
<main id="main">
<section id="top" aria-labelledby="title">
<p class="eyebrow">Saved-state study · 9 September 2026</p>
<div class="callout"><strong>Compare learned axis, spatial smoothness, and active stress visually.</strong><a href="ideas.html">Open matching shape and activation views</a>, switch between three selected pairs, and inspect the tensor modes omitted by a single line.</div>
<!-- conservative-rate-followup:start -->
<div class="callout"><strong>Rate 0.3 is verified: smaller updates, but inversions remain.</strong>The smoothed arm completed 256 updates with 2.88 mm fit error and 73 inverted tetrahedra. It avoided the quarter-rate run's sharp late deterioration, but its selected match against the original rate has a 9.35% higher surface residual. <a href="#rate-03">Read the final three-rate comparison</a>. The original and quarter-rate study results remain below.</div>
<!-- conservative-rate-followup:end -->
<h1 id="title">The matched surface threshold was not met.</h1>
<p class="lede">In the original study, C-smoothness lowered activation variation in both tested parameterizations. At comparable saved states, the primary surface-residual reduction was <strong>0.40%</strong> for the learned-axis model and <strong>0.20%</strong> for corrected Raw6—both below the predeclared <strong>10%</strong> threshold.</p>
<div class="metric-grid">
<article class="metric-card"><h3>Learned-axis match · updates 11 / 11</h3><p class="metric-value">0.40<span>%</span></p><div class="meter" aria-hidden="true"><span style="width:4%"></span></div><div class="threshold"><span>Residual reduction</span><span>10% required</span></div></article>
<article class="metric-card"><h3>Corrected Raw6 match · updates 200 / 200</h3><p class="metric-value">0.20<span>%</span></p><div class="meter" aria-hidden="true"><span style="width:2%"></span></div><div class="threshold"><span>Residual reduction</span><span>10% required</span></div></article>
</div>
<div class="callout"><strong>Scope of the conclusion</strong>The tested weights and available matched states did not meet the criterion. Axis-off failed before later learned-axis matches could be tested, so the study does not rule out different weights or later matched fits.</div>
<p class="measure">Original-study plot key: learned-axis off is vermillion with circles, on is blue with squares; Raw6 off is magenta with triangles, on is green with diamonds. Off curves are solid and on curves are dashed. Exact-section targets are charcoal and dotted. Calibration plots label each rate separately.</p>
</section>

<!-- activation-glyphs:start -->
<section id="activation-glyphs" aria-labelledby="activation-glyphs-title">
<h2 id="activation-glyphs-title">Muscle activation axes and strength</h2>
<p class="measure"><strong>Each active muscle tetrahedron has one centered line on its saved deformed shape.</strong> The full 3D field contains all 288,235 cells across 103 activation regions. Each line follows the spatial direction <strong>normalize(F n<sub>rest</sub>)</strong> of its learned contraction axis; its length and color both encode commanded shortening, <strong>a = s / (1 + s)</strong>, where s = ‖v‖² and B = I + vvᵀ. These are learned controls, not measured tissue strain.</p>
<p class="measure">Line length is <strong>4.5 mm × a</strong>, using one common scale for every tetrahedron and state. More contraction therefore always means a longer line. There is no minimum length, cell-size scaling, or per-state normalization. All images also use the same linear 0–100% color scale; very weak commands produce correspondingly tiny lines. Line length is a display scale, not tissue displacement.</p>
<p class="measure">Centers are mean(X + u) for each tetrahedron, the faint skin context uses the same saved deformation, and occlusion is recomputed for every saved state and camera. Each view uses the frontmost muscle-region label at the projected deformed cell center. Cells of that muscle are retained throughout its depth; cells belonging to a different muscle behind it are hidden. This keeps internal tetrahedra of the visible muscles, with no spatial sampling. The full 3D exports retain every cell, including those hidden in the images.</p>
<p class="measure">Muscle shapes are shown through the distribution and spatial direction of their tetrahedron glyphs. No muscle surfaces or outline curves are drawn. The two ends of each activation line have equal meaning because v and −v produce the same tensor. The saved states have different fits and update budgets.</p>
<h3>Rate 0.3 · smoothed · update 256</h3>
<div class="figure-grid">
<figure class="figure"><a href="assets/glyphs/rate03-on/side-context.png"><img src="assets/glyphs/rate03-on/side-context.png" alt="Deformed learned activation lines of front-visible muscles at rate 0.3 update 256, with internal tetrahedra retained" loading="lazy"></a><figcaption>Rate 0.3, update 256. Line length and color show commanded shortening; overlapping muscles behind the front muscle are hidden.</figcaption></figure>
<figure class="figure"><a href="assets/glyphs/rate03-on/region1-mouth-corner.png"><img src="assets/glyphs/rate03-on/region1-mouth-corner.png" alt="Deformed mouth-corner close-up of learned muscle activation axes at rate 0.3 update 256" loading="lazy"></a><figcaption>Mouth-corner close-up on the saved deformed geometry. Glyphs show spatially transformed learned controls; they are not displacement vectors.</figcaption></figure>
</div>
<details><summary>Earlier rate-0.3 state · update 128</summary><div class="figure-grid">
<figure class="figure"><a href="assets/glyphs/rate03-on-128/side-context.png"><img src="assets/glyphs/rate03-on-128/side-context.png" alt="Rate 0.3 smoothed activation glyph overview at update 128" loading="lazy"></a><figcaption>Rate 0.3, smoothed, update 128.</figcaption></figure>
<figure class="figure"><a href="assets/glyphs/rate03-on-128/region1-mouth-corner.png"><img src="assets/glyphs/rate03-on-128/region1-mouth-corner.png" alt="Rate 0.3 smoothed activation glyph mouth close-up at update 128" loading="lazy"></a><figcaption>The same mouth camera and strength scale as update 256.</figcaption></figure>
</div></details>
<details><summary>Original rate · smoothing off and on</summary><div class="figure-grid">
<figure class="figure"><a href="assets/glyphs/original-off/side-context.png"><img src="assets/glyphs/original-off/side-context.png" alt="Original-rate unsmoothed learned-axis activation glyph overview at its latest full checkpoint, update 16" loading="lazy"></a><figcaption>Original rate, smoothing off, update 16: the latest full checkpoint. This is earlier than its last accepted surface at update 28.</figcaption></figure>
<figure class="figure"><a href="assets/glyphs/original-on/side-context.png"><img src="assets/glyphs/original-on/side-context.png" alt="Original-rate smoothed learned-axis activation glyph overview at update 128" loading="lazy"></a><figcaption>Original rate, smoothing on, update 128. These endpoints are unmatched.</figcaption></figure>
<figure class="figure"><a href="assets/glyphs/original-off/region1-mouth-corner.png"><img src="assets/glyphs/original-off/region1-mouth-corner.png" alt="Original-rate unsmoothed learned-axis activation glyph mouth close-up" loading="lazy"></a><figcaption>Original off · mouth close-up.</figcaption></figure>
<figure class="figure"><a href="assets/glyphs/original-on/region1-mouth-corner.png"><img src="assets/glyphs/original-on/region1-mouth-corner.png" alt="Original-rate smoothed learned-axis activation glyph mouth close-up" loading="lazy"></a><figcaption>Original on · mouth close-up.</figcaption></figure>
</div></details>
<details><summary>Quarter rate · smoothing off and on · update 64</summary><div class="figure-grid">
<figure class="figure"><a href="assets/glyphs/quarter-off/side-context.png"><img src="assets/glyphs/quarter-off/side-context.png" alt="Quarter-rate unsmoothed learned-axis activation glyph overview at update 64" loading="lazy"></a><figcaption>Quarter rate, smoothing off, update 64.</figcaption></figure>
<figure class="figure"><a href="assets/glyphs/quarter-on/side-context.png"><img src="assets/glyphs/quarter-on/side-context.png" alt="Quarter-rate smoothed learned-axis activation glyph overview at update 64" loading="lazy"></a><figcaption>Quarter rate, smoothing on, update 64. Equal update count does not imply equal fit or motion.</figcaption></figure>
<figure class="figure"><a href="assets/glyphs/quarter-off/region1-mouth-corner.png"><img src="assets/glyphs/quarter-off/region1-mouth-corner.png" alt="Quarter-rate unsmoothed learned-axis activation glyph mouth close-up at update 64" loading="lazy"></a><figcaption>Quarter off · mouth close-up.</figcaption></figure>
<figure class="figure"><a href="assets/glyphs/quarter-on/region1-mouth-corner.png"><img src="assets/glyphs/quarter-on/region1-mouth-corner.png" alt="Quarter-rate smoothed learned-axis activation glyph mouth close-up at update 64" loading="lazy"></a><figcaption>Quarter on · mouth close-up.</figcaption></figure>
</div></details>
<div class="downloads">
<a class="download" href="downloads/rate03-full-activation-glyphs.vtp"><strong>Rate 0.3 full line field</strong><span>288,235 lines · VTP for ParaView</span></a>
<a class="download" href="downloads/activation-glyph-data.zip"><strong>All six fields for ParaView</strong><span>1.18 GB · full VTP fields, deformed skins, and view masks</span></a>
<a class="download" href="downloads/activation-glyphs.md"><strong>Glyph methods and states</strong><span>Per-cell line construction, exact checkpoints, and limitations</span></a>
<a class="download" href="downloads/activation-glyphs.json"><strong>Glyph receipt</strong><span>Input, source, and output hashes</span></a>
<a class="download" href="downloads/activation-glyphs-verification.json"><strong>Independent field checks</strong><span>Cell mapping, axes, lengths, visibility, and archive bytes</span></a>
<a class="download" href="downloads/sources/76-render-deformed-activation-glyphs.py"><strong>Deformed-glyph renderer</strong><span>Reproduce these saved-shape views from controls and displacement</span></a>
<a class="download" href="downloads/sources/77-verify-deformed-activation-glyphs.py"><strong>Independent verifier</strong><span>Check deformed centers, spatial axes, visibility, and archive bytes</span></a>
<a class="download" href="downloads/sources/muscle_glyph_context.py"><strong>Muscle visibility source</strong><span>Region surfaces and camera-based occlusion</span></a>
</div>
</section>
<!-- activation-glyphs:end -->

<!-- rate-03-results:start -->
<section id="rate-03" aria-labelledby="rate-03-title">
<p class="eyebrow">Requested rate 0.3 · completed and verified</p><h2 id="rate-03-title">A smaller step is useful, but it is not an inversion remedy</h2>
<p class="measure">The learned-axis smoothed arm completed <strong>256 updates at rate 0.3</strong>, retaining the original initialization, Adam epsilon and betas, smoothness coefficient, physics, and solvers. A continuation at update 128 restored the exact saved controls, displacement, gradient, moments, and counter. There is <strong>no unsmoothed rate-0.3 arm</strong> in this targeted test.</p>
<div class="table-scroll"><table><thead><tr><th>Smoothed-arm rate</th><th>Endpoint update</th><th>Fit RMS, mm</th><th>Motion RMS, mm</th><th>Inverted tetrahedra</th><th>Minimum det(F)</th><th>Surface residual HP, mm</th></tr></thead><tbody>
<tr><td>7.857986</td><td>128</td><td>0.68109</td><td>5.17958</td><td>94</td><td>−1.52803</td><td>0.16652</td></tr>
<tr><td>1.964496</td><td>64</td><td>5.15210</td><td>7.30812</td><td>2,189</td><td>−4.18661</td><td>0.60969</td></tr>
<tr><td>0.3</td><td>256</td><td>2.87680</td><td>4.04751</td><td>73</td><td>−1.02653</td><td>0.20455</td></tr>
</tbody></table></div>
<p class="table-note">These endpoints have different budgets, fit, and motion. They are descriptive, not matched comparisons. Surface residual HP is the fixed local surface-residual score; lower is better. The original and rate-0.3 endpoints are their best-fit states; quarter-rate on reached its best fit of 3.37838 mm at update 45, before worsening.</p>
<p class="measure">Rate 0.3 avoids that sharp late deterioration through its tested budget, but is slower to fit: the original smoothed run reaches lower error in fewer updates. No further extension was launched because measured runtime would exceed the frozen fitting cutoff. Completion does not establish stationarity or physical validity.</p>
<figure class="figure"><a href="assets/rate03-trajectories.png"><img src="assets/rate03-trajectories.png" alt="Three smoothed learning-rate trajectories showing smaller rate-0.3 physical updates, continuing fit progress and remaining inversions" loading="lazy"></a><figcaption><strong>Rate key:</strong> original 7.857986 is blue with squares; quarter 1.964496 is green with diamonds; 0.3 is purple with dotted lines and triangles. Inversion count, S(C), and Z-step panels use explicitly labeled symmetric-log scales so outliers do not hide the smaller curves.</figcaption></figure>
<h3>First inversion occurs in overlapping motion intervals</h3>
<div class="table-scroll"><table><thead><tr><th>Rate</th><th>Last non-inverted → first inverted update</th><th>Motion RMS interval, mm</th><th>Fit at first inversion, mm</th><th>Maximum commanded shortening</th></tr></thead><tbody>
<tr><td>7.857986</td><td>7 → 8</td><td>0.707708 → 1.166659</td><td>4.697891</td><td>97.51%</td></tr>
<tr><td>1.964496</td><td>24 → 25</td><td>0.981400 → 1.142112</td><td>4.714298</td><td>97.74%</td></tr>
<tr><td>0.3</td><td>110 → 111</td><td>0.966093 → 1.000948</td><td>4.793328</td><td>97.09%</td></tr>
</tbody></table></div>
<p class="table-note">The sampled intervals overlap. Smaller updates locate the first inversion more finely; the later update number does not establish a higher continuous inversion threshold.</p>
<p class="measure">Commanded shortening is a control-space quantity: with s = ‖v‖², it is s/(1+s), derived from A = (I + vvᵀ)⁻¹. It is not observed strain from F. A roughly 1 mm global surface-motion RMS can coexist with a large command in part of the mesh. The summaries do not identify whether the maximally commanded cell is the inverted cell. The unchanged update policy records inversions but does not reject them. <a href="downloads/sources/study_runner.py">Metric definition</a>.</p>
<h3>Matched comparisons show a tradeoff</h3>
<div class="table-scroll"><table><thead><tr><th>Comparison, left → right</th><th>Updates</th><th>Fit gap, mm</th><th>Motion gap, mm</th><th>Inversions</th><th>Surface HP change</th><th>Low-frequency projection ratio</th></tr></thead><tbody>
<tr><td>Original → 0.3</td><td>11 → 158</td><td>0.040135</td><td>0.027378</td><td>18 → 11</td><td>+9.35%</td><td>0.796</td></tr>
<tr><td>Quarter → 0.3</td><td>43 → 184</td><td>0.048882</td><td>0.027298</td><td>205 → 20</td><td>−0.66%</td><td>0.686</td></tr>
</tbody></table></div>
<p class="table-note">Actual saved states, excluding initialization, with ≤0.05 mm gaps in both global fit and motion. Select the lowest mean fit, then normalized mismatch; no interpolation. Positive HP change is worse. <a href="downloads/rate-03-states.csv">Exact selected states</a>.</p>
<p class="measure">At the original-rate match, the latest face displacement update falls from <strong>0.8782 to 0.0344 mm RMS</strong>, but surface residual rises by 9.35% and minimum det(F) is slightly more negative. At the quarter-rate match, the face update falls from 0.2614 to 0.0405 mm RMS, inversion count falls substantially, and the surface residual improves by only 0.66%.</p>
<p class="measure">Both rate-0.3 matches have low-frequency normal target-projection ratios below 0.90. Their local expression structure therefore differs despite the global RMS match. The result supports smaller discrete steps and reduced inversion counts at these states, <strong>not a general surface-quality improvement</strong> or an inversion-free solution.</p>
<figure class="figure"><a href="assets/rate03-attained-states.png"><img src="assets/rate03-attained-states.png" alt="Three-rate surface and inversion diagnostics against global fit and motion, showing the different attained trajectories" loading="lazy"></a><figcaption>The attained-state curves complement the exact matches; they do not relax the matching tolerances or equate local motion patterns.</figcaption></figure>
<div class="figure-grid">
<figure class="figure"><a href="assets/rate03-original-sections.png"><img src="assets/rate03-original-sections.png" alt="Exact skin sections for original-rate update 11 and rate-0.3 update 158 with target" loading="lazy"></a><figcaption>Original versus 0.3: exact selected sections.</figcaption></figure>
<figure class="figure"><a href="assets/rate03-quarter-sections.png"><img src="assets/rate03-quarter-sections.png" alt="Exact skin sections for quarter-rate update 43 and rate-0.3 update 184 with target" loading="lazy"></a><figcaption>Quarter versus 0.3: exact selected sections.</figcaption></figure>
</div>
<figure class="figure"><a href="assets/rate03-endpoints.png"><img src="assets/rate03-endpoints.png" alt="Target and exact saved endpoints for all three smoothed learning rates in fixed face and mouth-corner views" loading="lazy"></a><figcaption><strong>Unmatched endpoints.</strong> Fixed cameras, flat shading, actual displacement, no exaggeration. The rate-0.3 endpoint remains distinct from the target and contains inverted tetrahedra.</figcaption></figure>
<div class="callout"><strong>Practical interpretation</strong>Rate 0.3 is a useful conservative baseline for further diagnostics. It should not be treated as the fix for inversion. The next diagnostic should examine permitted activation magnitude and the acceptance of inverted states explicitly. This experiment did not change either policy, and it does not establish a best learning rate.</div>
<p class="measure"><a href="downloads/rate-03-results.md">Full rate-0.3 results</a> · <a href="downloads/rate-03-execution.md">Commands and run records</a> · <a href="downloads/rate-03-comparison.json">Comparison receipt</a> · <a href="downloads/rate-03-verification.json">Independent verification</a> · <a href="downloads/rate-03-plot-receipt.json">Plot receipt</a>. Verification covers 257 trace/solver/surface states, 18 full checkpoints, unchanged sources, initialization, and the 128→256 saved-Adam lineage. All evidence checks passed; the 73 endpoint inversions remain recorded physical defects.</p>
</section>
<!-- rate-03-results:end -->

<section id="quarter-rate" aria-labelledby="quarter-rate-title">
<p class="eyebrow">Completed rate-only follow-up</p><h2 id="quarter-rate-title">Smaller updates did not remove the initial inversions</h2>
<p class="measure">Both learned-axis arms completed 64 updates at <strong>learning rate 1.964496</strong>, one quarter of the original 7.857986. Initialization, Adam epsilon and betas, smoothness weights, physics, and solvers were unchanged. First inversion moved from update 8 to update 25, but still appeared near <strong>1.14 mm motion</strong>, close to the original 1.17 mm.</p>
<div class="table-scroll"><table><thead><tr><th>Quarter-rate state</th><th>Update</th><th>Fit RMS, mm</th><th>Motion RMS, mm</th><th>Inverted tetrahedra</th><th>Surface residual HP, mm</th></tr></thead><tbody>
<tr><td>Off endpoint / best fit</td><td>64</td><td>2.5878</td><td>4.3589</td><td>409</td><td>0.21883</td></tr>
<tr><td>On best fit</td><td>45</td><td>3.3784</td><td>3.6745</td><td>231</td><td>0.23766</td></tr>
<tr><td>On endpoint</td><td>64</td><td>5.1521</td><td>7.3081</td><td>2,189</td><td>0.60969</td></tr>
</tbody></table></div>
<p class="table-note">Endpoint and best-fit states are descriptive; they are not matched to each other. All configured inner solves succeeded. The on trajectory nevertheless worsened after update 45 and reached minimum det(F) −27.58 at update 63.</p>
<figure class="figure"><a href="assets/quarter-trajectories.png"><img src="assets/quarter-trajectories.png" alt="Original and quarter learning-rate trajectories for fit, motion, inversions, minimum determinant, surface residual and inner solve effort" loading="lazy"></a><figcaption>Rate-comparison key: original off/on are vermillion/blue; quarter off/on are magenta/green. Off curves are solid and on curves dashed. The smoothed quarter-rate run develops a late increase in fit error, inversions, and surface residual.</figcaption></figure>
<h3>What changes at matched fit and motion</h3>
<div class="table-scroll"><table><thead><tr><th>Comparison, left → right</th><th>Updates</th><th>Fit gap, mm</th><th>Motion gap, mm</th><th>Surface HP change</th><th>Inversions</th></tr></thead><tbody>
<tr><td>Quarter off → quarter on</td><td>41 → 45</td><td>0.0220</td><td>0.0456</td><td>+2.25%</td><td>90 → 231</td></tr>
<tr><td>Original off → quarter off</td><td>11 → 34</td><td>0.0080</td><td>0.0372</td><td>+4.12%</td><td>19 → 18</td></tr>
<tr><td>Original on → quarter on</td><td>13 → 44</td><td>0.0239</td><td>0.0094</td><td>+1.68%</td><td>81 → 214</td></tr>
</tbody></table></div>
<p class="table-note">Actual saved states only; both gaps must be ≤0.05 mm. Positive HP change means a worse surface residual. Select the lowest mean fit among eligible pairs; exclude initialization and do not interpolate. <a href="downloads/quarter-rate-states.csv">Exact selected states</a>.</p>
<p class="measure">The quarter-rate off/on pair <strong>also misses the 10% surface improvement threshold</strong>. At the cross-rate matches, the most recent face displacement update falls from 0.868 to 0.279 mm off and from 0.463 to 0.245 mm on. Smaller physical steps are therefore measured, but neither matched surface score improves.</p>
<figure class="figure"><a href="assets/quarter-attained-states.png"><img src="assets/quarter-attained-states.png" alt="Surface residual and inverted tetrahedra plotted against attained fit and attained motion across the two learning rates" loading="lazy"></a><figcaption>Attained-state plots separate progress in deformation from the number of Adam updates.</figcaption></figure>
<figure class="figure"><a href="assets/quarter-matched-sections.png"><img src="assets/quarter-matched-sections.png" alt="Exact saved skin sections for quarter-rate off at update 41 and on at update 45 with target" loading="lazy"></a><figcaption>Exact quarter-rate matched sections. The target is charcoal and dotted; neither curve is spatially smoothed for display.</figcaption></figure>
<details><summary>Cross-rate sections and endpoint geometry</summary>
<figure class="figure"><a href="assets/quarter-off-rate-sections.png"><img src="assets/quarter-off-rate-sections.png" alt="Exact matched off-arm sections at the original and quarter rates" loading="lazy"></a><figcaption>Off arm: original update 11 versus quarter-rate update 34.</figcaption></figure>
<figure class="figure"><a href="assets/quarter-on-rate-sections.png"><img src="assets/quarter-on-rate-sections.png" alt="Exact matched on-arm sections at the original and quarter rates" loading="lazy"></a><figcaption>On arm: original update 13 versus quarter-rate update 44.</figcaption></figure>
<figure class="figure"><a href="assets/quarter-off-endpoints.png"><img src="assets/quarter-off-endpoints.png" alt="Target and exact original and quarter-rate off endpoints in face and mouth-corner views" loading="lazy"></a><figcaption>Unmatched off endpoints, shown at their actual saved deformation with fixed cameras and flat shading.</figcaption></figure>
<figure class="figure"><a href="assets/quarter-on-endpoints.png"><img src="assets/quarter-on-endpoints.png" alt="Target and exact original and quarter-rate on endpoints, showing large mouth-corner distortion at the quarter-rate endpoint" loading="lazy"></a><figcaption>Unmatched on endpoints. Fit, motion, and budgets differ; these views are descriptive and do not substitute for the matched comparison.</figcaption></figure>
</details>
<p class="measure"><a href="downloads/quarter-rate-results.md">Full quarter-rate results</a> · <a href="downloads/quarter-rate-execution.md">Commands and run records</a> · <a href="downloads/quarter-rate-comparison.json">Comparison receipt</a> · <a href="downloads/quarter-rate-verification.json">Independent verification</a>. The evidence checks passed for 130 evaluated states and 12 full checkpoints; inversions remain recorded physical defects.</p>
</section>

<section id="methods" aria-labelledby="methods-title">
<p class="eyebrow">Implementation audit</p><h2 id="methods-title">What changes between experiments</h2>
<p>The off/on comparisons change smoothness within a model. Comparisons across learned axis, corrected Raw6, and historical PSD also change the control space, initialization, optimizer scaling, and sometimes budget. Raw learning rates, penalty coefficients, and endpoint errors therefore do not provide a controlled ranking across models.</p>
<p>The matrix below records the original study settings. The later learned-axis tests retain that implementation and change only the scalar rate to 1.964496 or 0.3, with their separately reported run budgets.</p>
<p>Here B is the inverse active-strain matrix, C = B − I, and Z = BBᵀ − I is the common dimensionless effective field. Historical PSD instead represents additive stress Q = μZ directly through a normalized matrix M. S denotes the spatial variation penalty, evaluated on the specified tensor field.</p>
<div class="table-scroll"><table class="implementation-table">
<thead>
<tr>
<th>Implementation</th>
<th>Learned axis</th>
<th>Corrected Raw6</th>
<th>Historical PSD</th>
</tr>
</thead>
<tbody>
<tr>
<td>Per-cell controls</td>
<td>3 values v; 864,705 scalars in total</td>
<td>6 unscaled symmetric entries q; 1,729,410 scalars</td>
<td>6 Frobenius-orthonormal symmetric coordinates q; 1,729,410 scalars</td>
</tr>
<tr>
<td>Map to effective field</td>
<td>C = vvᵀ; B = I + C; Z = (2 + ‖v‖²)vvᵀ</td>
<td>C = sym_unscaled(q); B = I + C; Z = 2C + C²</td>
<td>M = sym_orthonormal(q); Q = Q_ref M; Q_ref = 3μ, so Z = 3M</td>
</tr>
<tr>
<td>Admissible tensors</td>
<td>C and Z are PSD of rank at most 1; B is positive definite; no magnitude cap</td>
<td>C and B are unconstrained symmetric; Z ≽ −I; no magnitude cap; different signs of B can produce the same Z</td>
<td>Q is PSD of rank 0–3 after spectral projection; eigenvalues limited to 0–0.302013 MPa</td>
</tr>
<tr>
<td>Muscle implementation</td>
<td>Physical-volume active strain: shear term uses FB, volume terms use det(F)</td>
<td>Same physical-volume active-strain law as learned axis</td>
<td>Passive stable law plus additive ½ Q:(FᵀF − I)</td>
</tr>
<tr>
<td>Smooth arm</td>
<td>0.0004450069704 × S(C)</td>
<td>0.003214147722 × S(C)</td>
<td>5.905171468 × S(M), equivalent to 0.6561301632 × S(Z) because Z = 3M</td>
</tr>
<tr>
<td>Initialization</td>
<td>Seed 20260909; ‖v‖² = 0.001; one randomly sampled axis per each of 103 labels, copied to its cells; each cell then optimized independently; zero displacement seed</td>
<td>Exact canonical step-200 controls and saved no-skin displacement; optimizer moments reset for each off/on re-fit</td>
<td>Off/on-64 start at q = 0 and rest displacement; off-1024 continues saved controls, displacement, moments, and counter</td>
</tr>
<tr>
<td>Adam settings</td>
<td>Fixed rate 7.857985795; ε = 0.01; β = (0.9, 0.999)</td>
<td>Fixed rate 0.3; ε = 0.01; β = (0.9, 0.999)</td>
<td>Off/on-64: rate 0.3; off-1024: 0.3 through global update 512, then 0.6; ε = 0.01; β = (0.9, 0.999)</td>
</tr>
<tr>
<td>Update constraints</td>
<td>No control projection, outer backtracking, physical-step cap, or inversion rejection</td>
<td>Same absence of outer constraints as learned axis</td>
<td>Spectral projection after Adam, outside the differentiation graph; upper cap never bound in the recorded runs, but the lower PSD constraint did</td>
</tr>
</tbody>
</table></div>
<p>For the same mapped Z, these muscle laws produce the same equilibrium force and deformation Hessian: setting Q = μ(BBᵀ − I) makes their energy difference independent of F. This correspondence does not make their inverse optimization equivalent. The parameterization, allowable Z, initialization, smoothing field, and Adam/projection history still differ. In particular, “learned axis” does not impose one fixed anatomical fiber direction per muscle.</p>
<h3>Run protocol and comparison role</h3>
<div class="table-scroll"><table class="implementation-table">
<thead>
<tr>
<th>Run</th>
<th>Start and optimizer</th>
<th>Smoothness term</th>
<th>Executed result</th>
<th>Valid comparison role</th>
</tr>
</thead>
<tbody>
<tr>
<td>Axis-off</td>
<td>Shared seed-20260909 controls; zero displacement seed; fresh Adam at 7.858</td>
<td>None</td>
<td>Accepted/evaluated states through 28; fit 6.04 mm; attempted 29 fails. Best fit 3.539 mm at 15</td>
<td>Common prefix with Axis-on; selected actual-state match is 11/11</td>
</tr>
<tr>
<td>Axis-on</td>
<td>Same controls and displacement seed; fresh Adam at 7.858</td>
<td>4.45007e−4 S(C)</td>
<td>Completed 128 updates; fit 0.681 mm</td>
<td>Update 128 is unpaired because Axis-off has no corresponding state</td>
</tr>
<tr>
<td>Raw6-off</td>
<td>Canonical step-200 controls plus saved no-skin displacement; fresh Adam at 0.3</td>
<td>None</td>
<td>Completed 200 re-fit updates; fit 1.3978 mm</td>
<td>Equal-start, equal-budget off/on pair</td>
</tr>
<tr>
<td>Raw6-on</td>
<td>Exact same controls and displacement; fresh Adam at 0.3</td>
<td>0.00321415 S(C)</td>
<td>Completed 200 re-fit updates; fit 1.4047 mm</td>
<td>Selected actual-state match is endpoint 200/200</td>
</tr>
<tr>
<td>PSD-off-64</td>
<td>Zero controls, rest displacement, fresh Adam at 0.3</td>
<td>None</td>
<td>Completed 64 updates; fit 3.82 mm</td>
<td>Historical equal-start, equal-budget pair with PSD-on-64</td>
</tr>
<tr>
<td>PSD-on-64</td>
<td>Same zero/rest/fresh start at 0.3</td>
<td>5.90517 S(M)</td>
<td>Completed 64 updates; fit 4.03 mm</td>
<td>Historical controlled smoothness comparison</td>
</tr>
<tr>
<td>PSD-off-1024</td>
<td>Continues saved controls, displacement, moments, and counter; rate 0.3 then 0.6 after 512</td>
<td>None</td>
<td>Completed global update 1024; fit 1.54 mm</td>
<td>Longer-run context; no matched smooth continuation</td>
</tr>
</tbody>
</table></div>
<p>The learned-axis initial q/C/Z arrays are bitwise identical across off/on, and equilibrated initial displacement differs by 1.38e−14 mm RMS. The first-update q difference is 1.97e−6, exceeding the predeclared 1e−6 gate. The report retains that failed verification gate rather than treating the pair as numerically identical throughout.</p>
<h3>Shared physical and numerical settings</h3>
<p>The fixture hashes agree across these comparisons: 288,235 active tetrahedra, 15,302 finite fitted IsFace vertices, the same fixed constraints, and the same 501,409-edge graph connecting neighboring cells within the same muscle. All use zero skin energy and no contact. The data term is uniform Cartesian coordinate MSE over the finite fitted face vertices, multiplied by 10⁶; the reported vector RMS is a separate diagnostic.</p>
<p>The common penalty is S(X) = (ℓ² / V_a) Σ_(i,j ∈ E) w_ij ‖X_i − X_j‖²_F, with ℓ = 0.005 m, V_a = Σ_k V_k f_k, and w_ij = (A_ij / d_ij) · 2f_i f_j / (f_i + f_j). Here f is muscle fraction, A is shared-face area, and d is cell-centroid distance. Edges connect only cells with the same muscle label. The norm is Frobenius; the PSD coordinate norm is identical because its six coordinates are orthonormal. This is a spatial difference penalty, not a magnitude penalty.</p>
<div class="table-scroll"><table class="implementation-table">
<thead>
<tr>
<th>Shared setting</th>
<th>Recorded value</th>
</tr>
</thead>
<tbody>
<tr>
<td>Passive materials, E in MPa</td>
<td>Aponeurosis: stable, E = 0.1, ν = 0.35; fat: stable, E = 0.003, ν = 0.49; muscle: stable, E = 0.03, ν = 0.49</td>
</tr>
<tr>
<td>Forward equilibrium</td>
<td>PNCG; maximum 5,000 iterations; relative tolerance 5e−4; absolute tolerance 1e−10; implementation-default line search, recorded maximum 10</td>
</tr>
<tr>
<td>Implicit adjoint</td>
<td>CG followed by MinRes when needed; each maximum 10,000 iterations and relative tolerance 5e−4</td>
</tr>
<tr>
<td>Failure and geometry policy</td>
<td>Failed or nonfinite forward/adjoint evaluations stop the run. Inverted tetrahedra and stress spectra are recorded diagnostics; inversions alone do not reject the state</td>
</tr>
</tbody>
</table></div>
<p><a href="downloads/implementation-comparison.md">Full implementation comparison</a> · <a href="downloads/sources/activation_controls.py">Current controls</a> · <a href="downloads/sources/study_physics.py">Common physics</a> · <a href="downloads/sources/psd-runner.py">Historical PSD runner</a> · <a href="downloads/psd-rate-switch.json">PSD rate-switch receipt</a> · <a href="downloads/psd-final-continuation.json">PSD continuation receipt</a> · <a href="downloads/psd-final-summary.json">PSD final state and cap</a></p>
</section>

<section id="learning-rate" aria-labelledby="learning-rate-title">
<p class="eyebrow">Distorted intermediate states</p><h2 id="learning-rate-title">The selected rate was aggressive; its long-run safety was not checked</h2>
<p class="measure">Learning rate <strong>7.86 is a strong suspect in the later Axis-off blow-up</strong>, but it has not been isolated as the cause of all distortion. The calibration selected the fastest of four 16-update pilots. Its “stable” flag checked finite successful solves and short-term fit progress; it did <strong>not</strong> require inversion-free geometry.</p>
<div class="table-scroll"><table><thead><tr><th>Fit-only pilot rate</th><th>Update-16 fit RMS (mm)</th><th>Motion RMS (mm)</th><th>Inverted tetrahedra</th><th>First inverted update</th></tr></thead><tbody>
<tr><td>0.982</td><td>5.302</td><td>0.015</td><td>0</td><td>None through 16</td></tr>
<tr><td>1.964</td><td>5.233</td><td>0.155</td><td>0</td><td>None through 16</td></tr>
<tr><td>3.929</td><td>4.376</td><td>1.760</td><td>1</td><td>14</td></tr>
<tr><td><strong>7.858, selected</strong></td><td><strong>3.028</strong></td><td>4.210</td><td><strong>225</strong></td><td>8</td></tr>
</tbody></table></div><p class="table-note">Same initial controls, fresh Adam in each pilot. <a href="downloads/pilot-endpoints.csv">Exact pilot values</a> and <a href="downloads/learning-rate-evidence.json">source receipt</a>.</p>
<figure class="figure"><a href="assets/learning-rate-pilots.png"><img src="assets/learning-rate-pilots.png" alt="Four learning-rate pilots: faster early fitting at the selected rate accompanies many more inverted tetrahedra" loading="lazy"></a><figcaption><strong>Existing calibration traces.</strong> Smaller rates also attain much less motion during these 16 updates.</figcaption></figure>
<div class="callout"><strong>Halving the rate delays the first inversion, but reaches it at similar motion.</strong>The 3.929 pilot first inverts at update 14 and 1.122 mm motion RMS; the 7.858 pilot first inverts at update 8 and 1.167 mm. The smaller rates have not been continued to comparable final fit or motion. Their early counts do not establish that lowering the rate alone removes distortion.</div>
<p class="measure">In the primary Axis-off run, fit worsens from its best <strong>3.539 mm at update 15</strong> to <strong>6.038 mm at update 28</strong>, while inverted cells grow from <strong>541 to 12,657</strong>; attempted update 29 fails. These shapes are saved optimizer iterates, not a physical motion sequence. <a href="downloads/primary-milestones.csv">Primary milestones</a>.</p>
<p class="measure">Adam has a fixed global rate with β = (0.9, 0.999), ε = 0.01, and adaptive moment normalization. This runner has no outer backtracking, loss-decrease test, physical-step cap, or inversion rejection. The inner equilibrium line search works at fixed activation. Moreover, Z = (2 + ‖v‖²)vvᵀ grows nonlinearly, while S(C) limits spatial variation rather than activation magnitude or det(F). Axis-on completes 128 updates at the same rate but still has 94 inverted tetrahedra.</p>
<p class="measure">The completed <a href="#quarter-rate">quarter-rate follow-up</a> changes only the rate and compares attained fit, motion, and inversion counts. The pilot diagnosis itself used existing traces; its original evidence is retained below the new experiment results. See the <a href="downloads/learning-rate-diagnosis.md">full diagnosis</a> for the evidence and limits.</p>
</section>

<section id="matched" aria-labelledby="matched-title">
<p class="eyebrow">Primary decision</p><h2 id="matched-title">Two exact saved-state comparisons</h2>
<p class="measure">The search used actual post-update states only. It required global fit and motion RMS to differ by no more than 0.05 mm, without interpolation. A useful effect also required lower S(C), at least 90% low-frequency target-motion retention, and at least 10% reduction in the local surface residual.</p>
<div class="table-scroll"><table><thead><tr><th>Pair</th><th>Updates<br>off / on</th><th>Fit RMS<br>off / on (mm)</th><th>Motion RMS<br>off / on (mm)</th><th>Residual HP<br>off / on (mm)</th><th>S(C)<br>off / on</th><th>P<sub>U</sub> ratio</th><th>Result</th></tr></thead><tbody>
<tr><td>Learned axis</td><td>11 / 11</td><td>3.769 / 3.759</td><td>3.065 / 3.068</td><td>0.2469 / 0.2459</td><td>325 / 296</td><td>1.008</td><td><strong>0.40%</strong></td></tr>
<tr><td>Corrected Raw6</td><td>200 / 200</td><td>1.398 / 1.405</td><td>4.818 / 4.807</td><td>0.1799 / 0.1795</td><td>9.40 / 8.37</td><td>0.998</td><td><strong>0.20%</strong></td></tr>
</tbody></table></div><p class="table-note">Residual HP is the primary 5 mm high-pass score on the frozen right-side union. P<sub>U</sub> is the low-frequency target projection.</p>
<div class="figure-grid">
<figure class="figure"><a href="assets/learned-matched-sections.png"><img src="assets/learned-matched-sections.png" alt="Three exact skin sections through the learned-axis matched states" loading="lazy"></a><figcaption><strong>Learned axis, matched updates 11 / 11.</strong> Exact target, off, and on skin-triangle sections.</figcaption></figure>
<figure class="figure"><a href="assets/raw6-matched-sections.png"><img src="assets/raw6-matched-sections.png" alt="Three exact skin sections through the Raw6 matched states" loading="lazy"></a><figcaption><strong>Raw6, matched updates 200 / 200.</strong> The off and on sections closely overlap.</figcaption></figure>
</div>
<div class="figure-grid">
<figure class="figure"><a href="assets/learned-matched-mouth.png"><img src="assets/learned-matched-mouth.png" alt="Target, learned-axis off, and learned-axis on at matched update 11" loading="lazy"></a><figcaption>Target and learned-axis matched saved geometry at the right mouth corner.</figcaption></figure>
<figure class="figure"><a href="assets/raw6-matched-mouth.png"><img src="assets/raw6-matched-mouth.png" alt="Target, Raw6 off, and Raw6 on at matched update 200" loading="lazy"></a><figcaption>Target and corrected Raw6 matched endpoints at the right mouth corner.</figcaption></figure>
</div>
</section>

<section id="endpoint" aria-labelledby="endpoint-title">
<p class="eyebrow">Unpaired endpoint</p><h2 id="endpoint-title">Axis-on reached its best fit at update 128</h2>
<p class="measure">Axis-on completed its 128-update budget with fit RMS 0.681 mm, motion RMS 5.180 mm, primary residual 0.167 mm, S(C) 286, and 94 inverted tetrahedra. Axis-off failed at attempted update 29, so update 128 has no off state and is shown only as an unpaired attained endpoint.</p>
<figure class="figure"><a href="assets/endpoint-side.png"><img src="assets/endpoint-side.png" alt="Target smile and unpaired Axis-on update 128 endpoint" loading="eager"></a><figcaption><strong>Unpaired endpoint.</strong> Target and Axis-on saved surfaces at update 128, shown as an unpaired endpoint.</figcaption></figure>
<div class="figure-grid">
<figure class="figure"><a href="assets/endpoint-mouth.png"><img src="assets/endpoint-mouth.png" alt="Target and unpaired Axis-on update 128 mouth-corner close-up" loading="lazy"></a><figcaption>Right mouth-corner close-up.</figcaption></figure>
<figure class="figure"><a href="assets/endpoint-sections.png"><img src="assets/endpoint-sections.png" alt="Exact target and unpaired Axis-on update 128 sections" loading="lazy"></a><figcaption>Three exact skin sections through the target and endpoint.</figcaption></figure>
</div>
<h3>Why the 62.1% equal-update reduction is descriptive</h3>
<p class="measure">At common update 28, Axis-on had a 62.1% lower primary residual, 99.98% lower S(C), and 12,322 fewer inverted tetrahedra. Fit RMS differed by 4.00 mm and motion RMS by 3.65 mm, so the pair was outside both matching tolerances.</p>
<figure class="figure"><a href="assets/learned-common-mouth.png"><img src="assets/learned-common-mouth.png" alt="Target and learned-axis off and on states at common update 28" loading="lazy"></a><figcaption><strong>Common update 28, not a matched comparison.</strong> Off and on are at the same optimizer count but substantially different fit and motion.</figcaption></figure>
</section>

<section id="regions" aria-labelledby="regions-title">
<p class="eyebrow">Local checks</p><h2 id="regions-title">No frozen support approaches 10%</h2>
<p class="measure">The primary union contains 1,592 unique skin vertices in the right mouth-corner, right lateral-cheek, and right lower-cheek/jaw supports. Negative reduction means the on-state residual is slightly larger.</p>
<div class="table-scroll"><table><thead><tr><th>Frozen support</th><th>Learned-axis residual HP reduction</th><th>Raw6 residual HP reduction</th></tr></thead><tbody>
<tr><td>Right mouth corner</td><td>0.480%</td><td>0.311%</td></tr><tr><td>Right lateral cheek</td><td>−0.108%</td><td>0.052%</td></tr><tr><td>Right lower cheek / jaw</td><td>−0.060%</td><td>0.192%</td></tr>
</tbody></table></div><p class="table-note">Scroll horizontally on narrow screens. Full off/on fit, residual, displacement, and low-frequency values are in the <a href="downloads/regional-metrics.csv">regional CSV</a>.</p>
<h3>Protected nose-to-mouth region</h3>
<p class="measure">The protocol assesses this region using fit and low-frequency projection, alongside the exact sections.</p>
<div class="table-scroll"><table><thead><tr><th>Pair</th><th>Off / on fit RMS (mm)</th><th>Off / on low-frequency projection</th></tr></thead><tbody><tr><td>Learned axis, update 11</td><td>6.048 / 6.031</td><td>0.605 / 0.596</td></tr><tr><td>Raw6, update 200</td><td>1.935 / 1.947</td><td>0.823 / 0.820</td></tr></tbody></table></div>
<figure class="figure"><a href="assets/fit-motion-tradeoff.png"><img src="assets/fit-motion-tradeoff.png" alt="Surface residual versus fit and motion for both parameterizations" loading="lazy"></a><figcaption>Surface score against global fit and motion across the saved trajectories.</figcaption></figure>
</section>

<section id="limits" aria-labelledby="limits-title">
<p class="eyebrow">Execution record</p><h2 id="limits-title">Evidence integrity passed; the full study did not</h2>
<div class="status-grid">
<article class="status-card"><h3 class="pass">Retained evidence verified</h3><ul><li>Saved source snapshots and inputs agree.</li><li>Raw6 begins from the archived controls exactly; re-equilibrated geometry differs by 8.89e−18 mm RMS.</li><li>Axis-on completed 128 updates and Raw6-on completed 200.</li><li>Failed controls were not rendered as equilibrated states.</li></ul></article>
<article class="status-card"><h3 class="fail">Study checks not met</h3><ul><li>Axis-off stopped after accepted update 28 when attempted update 29 hit the 5,000-iteration forward limit.</li><li>The first-update q maximum difference was 1.97e−6 against the predeclared 1e−6 gate.</li><li>Other relative, C/Z, and geometry first-update gates passed.</li></ul></article>
</div>
<p class="measure">The learned-axis result uses one initialization seed, one target, one mesh, and fixed budgets. It does not establish inverse stationarity, anatomical fiber recovery, physiological validity, or robustness to other seeds and regularization weights.</p>
<figure class="figure"><a href="assets/trajectories.png"><img src="assets/trajectories.png" alt="Fit, motion, primary surface score, and low-frequency projection trajectories" loading="lazy"></a><figcaption>All fresh saved-state trajectories. The failed Axis-off prefix ends at update 28.</figcaption></figure>
<figure class="figure"><a href="assets/activation-variation.png"><img src="assets/activation-variation.png" alt="C and Z spatial variation over optimizer updates" loading="lazy"></a><figcaption>S(C) and S(Z) quantify spatial variation in different tensor fields. The two models use separately calibrated penalty weights.</figcaption></figure>
<details><summary>Calibration-to-primary divergence</summary><p>The revised learned-axis calibration selected learning rate 7.857985794554741 and λ<sub>C</sub> = 0.0004450069704277614 within its tested grid. The primary Axis-off run later diverged from that pilot along an increasingly inverted, nonconvex path. A read-only audit found no archived implementation or configuration mismatch, but no deterministic GPU replay identified the initiating perturbation.</p><figure class="figure"><a href="assets/calibration-divergence.png"><img src="assets/calibration-divergence.png" alt="Selected calibration pilot and primary Axis-off through update 16" loading="lazy"></a><figcaption>Selected calibration pilot and primary Axis-off through update 16.</figcaption></figure></details>
<details><summary>Matched residual maps</summary><p>Each off/on pair shares one symmetric color range. These signed maps display the saved 5 mm high-pass residual field; the decision uses its area-weighted RMS over the frozen union.</p><div class="figure-grid"><figure class="figure"><div class="pair-images"><a href="assets/learned-residual-off.png"><img src="assets/learned-residual-off.png" alt="Learned-axis off signed residual map" loading="lazy"></a><a href="assets/learned-residual-on.png"><img src="assets/learned-residual-on.png" alt="Learned-axis on signed residual map" loading="lazy"></a></div><figcaption>Learned-axis match, off then on.</figcaption></figure><figure class="figure"><div class="pair-images"><a href="assets/raw6-residual-off.png"><img src="assets/raw6-residual-off.png" alt="Raw6 off signed residual map" loading="lazy"></a><a href="assets/raw6-residual-on.png"><img src="assets/raw6-residual-on.png" alt="Raw6 on signed residual map" loading="lazy"></a></div><figcaption>Raw6 match, off then on.</figcaption></figure></div></details>
</section>

<section aria-labelledby="context-title"><p class="eyebrow">Earlier protocols</p><h2 id="context-title">Historical PSD states are context only</h2><p class="measure">These reused states have earlier protocols and different penalty settings. They are not an equal-start or consistently equal-budget ranking across models.</p>
<div class="table-scroll"><table><thead><tr><th>Context state</th><th>Fit RMS (mm)</th><th>Motion RMS (mm)</th><th>Union residual HP (mm)</th><th>P<sub>U</sub></th><th>S(Z)</th></tr></thead><tbody><tr><td>PSD-off, update 64</td><td>3.82</td><td>1.81</td><td>0.284</td><td>0.264</td><td>7.88</td></tr><tr><td>PSD-on, update 64</td><td>4.03</td><td>1.56</td><td>0.306</td><td>0.227</td><td>0.357</td></tr><tr><td>PSD-off, update 1024</td><td>1.54</td><td>4.43</td><td>0.146</td><td>0.822</td><td>196</td></tr></tbody></table></div>
<figure class="figure"><a href="assets/historical-context.png"><img src="assets/historical-context.png" alt="Historical PSD context plotted with current endpoints" loading="lazy"></a><figcaption>Attained fit and local surface error. Historical PSD markers provide context; these states are not matched across models.</figcaption></figure></section>

<section id="evidence" aria-labelledby="evidence-title"><p class="eyebrow">Definitions and files</p><h2 id="evidence-title">How to read and reproduce the report</h2>
<div class="plain-grid"><div><strong>Fit and motion</strong><p>Uniform vector RMS over all finite fitted face vertices. These are global scores.</p></div><div><strong>Primary residual HP</strong><p>Area-weighted rest-normal target residual after a 5 mm high-pass operator, on the three-support local union.</p></div><div><strong>C and Z</strong><p>C is the activation tensor whose spatial variation is penalized. For learned axis, C = vvᵀ; for corrected Raw6, C = B − I. In both cases, Z = BBᵀ − I is the effective activation field.</p></div></div>
<p class="measure">The surface comparison panels use exact saved displacements. The full glyph exports contain one line for every active tetrahedron, constructed at its saved deformed center with its saved shortening and spatially transformed learned axis; the panels hide overlapping muscles behind the front muscle. There is no interpolation, deformation exaggeration, surface smoothing, or reconstruction of failed states. The copied files below are hashed in the <a href="manifest.json">site manifest</a>.</p>
<div class="downloads">
<a class="download" href="downloads/report.md"><strong>Concise report</strong><span>Markdown · final scientific narrative</span></a>
<a class="download" href="downloads/protocol.md"><strong>Frozen protocol</strong><span>Matching rules and study scope</span></a>
<a class="download" href="downloads/execution-record.md"><strong>Execution record</strong><span>Calibration and run history</span></a>
<a class="download" href="downloads/selected-states.csv"><strong>Selected states</strong><span>Exact primary comparison rows</span></a>
<a class="download" href="downloads/regional-metrics.csv"><strong>Regional metrics</strong><span>Eight saved-state region rows</span></a>
<a class="download" href="downloads/historical-context.csv"><strong>PSD context</strong><span>Reused historical endpoint metrics</span></a>
<a class="download" href="downloads/comparison-summary.json"><strong>Comparison receipt</strong><span>Selections, hashes, and outputs</span></a>
<a class="download" href="downloads/verification.json"><strong>Verification receipt</strong><span>Independent retained-evidence checks</span></a>
<a class="download" href="downloads/render-receipt.json"><strong>Render receipt</strong><span>Endpoint and exact-section sources</span></a>
<a class="download" href="downloads/calibration-audit.json"><strong>Divergence audit</strong><span>Read-only source and state comparison</span></a>
<a class="download" href="downloads/implementation-comparison.md"><strong>Implementation tables</strong><span>Control spaces, starts, solvers, and run roles</span></a>
<a class="download" href="downloads/learning-rate-diagnosis.md"><strong>Learning-rate diagnosis</strong><span>Pilot evidence and causal limits</span></a>
<a class="download" href="downloads/plot-receipt-v2.json"><strong>Updated plot receipt</strong><span>Colors, source hashes, and nine figures</span></a>
<a class="download" href="downloads/sources/44-render-report-figures-v2.py"><strong>Plot source</strong><span>Trace curves and exact-section renderer</span></a>
<a class="download" href="downloads/sources/45-extract-learning-rate-evidence.py"><strong>Learning-rate extraction</strong><span>Existing traces; no numerical replay</span></a>
<a class="download" href="downloads/sources/40-compare.py"><strong>Comparison source</strong><span>CPU metrics and exact-state rendering</span></a>
<a class="download" href="downloads/sources/60-build-report-site.py"><strong>Site build source</strong><span>Rebuild copied assets and manifest</span></a>
</div></section>
</main>
<footer class="site-footer">Static, dependency-free report · exact saved-state evidence · <a href="manifest.json">SHA-256 manifest</a></footer>
</body></html>
"""


HTML = (GROUP / "src/90-report-layout.html").read_text()


def build() -> dict[str, object]:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    copied: list[dict[str, object]] = []
    for destination_name, source_name in sorted(COPIES.items()):
        source = GROUP / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = STAGING / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_hash = digest(source)
        copied_hash = digest(destination)
        if copied_hash != source_hash:
            raise ValueError(f"copied file differs from source: {source}")
        copied.append(
            {
                "site_path": destination_name,
                "source_path": str(source.resolve()),
                "bytes": destination.stat().st_size,
                "sha256": copied_hash,
            }
        )
    index = STAGING / "index.html"
    index.write_text(HTML)
    manifest = {
        "schema_version": 1,
        "scope": (
            "Self-contained static report built from exact saved-state figures and "
            "copied evidence; no experiment, metric, or geometry recomputation."
        ),
        "index": {
            "site_path": "index.html",
            "bytes": index.stat().st_size,
            "sha256": digest(index),
        },
        "copied_files": copied,
    }
    (STAGING / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    parser = LocalReferenceParser()
    parser.feed(HTML)
    missing = []
    external = []
    for reference in parser.references:
        if reference.startswith("#"):
            continue
        if "://" in reference or reference.startswith("//"):
            external.append(reference)
            continue
        target = (STAGING / reference.split("#", 1)[0]).resolve()
        if not target.is_relative_to(STAGING.resolve()) or not target.is_file():
            missing.append(reference)
    if external:
        raise ValueError(f"external page dependencies are forbidden: {external}")
    if missing:
        raise FileNotFoundError(f"missing generated page references: {missing}")
    if SITE.exists():
        shutil.rmtree(SITE)
    STAGING.rename(SITE)
    return {
        "status": "completed_static_report_build",
        "site": str(SITE),
        "files": 2 + len(copied),
        "bytes": sum(path.stat().st_size for path in SITE.rglob("*") if path.is_file()),
        "index_sha256": manifest["index"]["sha256"],
        "manifest_sha256": digest(SITE / "manifest.json"),
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
