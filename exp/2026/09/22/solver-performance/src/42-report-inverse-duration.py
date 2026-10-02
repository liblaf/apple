"""Build a readable report from validated complete-update timing trees."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/current-model-report-001"


def markdown_table(head: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        "| " + " | ".join(row) + " |" for row in [head, ["---"] * len(head), *rows]
    )


def web_table(head: list[str], rows: list[list[str]]) -> str:
    th = "<tr>" + "".join(f"<th>{html.escape(x)}</th>" for x in head) + "</tr>"
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-scroll"><table>{th}{body}</table></div>'


def count_hvp(tree: dict, counts: dict[str, int], path: tuple[str, ...] = ()) -> None:
    for name, node in tree.items():
        branch = (*path, name)
        if name == "model/hess_prod":
            if "newton_cg" in branch:
                counts["newton"] += node["count"]
            if "adjoint_linear" in branch:
                counts["adjoint"] += node["count"]
        count_hvp(node["children"], counts, branch)


def main(cfg: Config) -> None:  # noqa: PLR0915
    summary = json.loads(
        (GROUP / "data/inverse-duration-analysis-001/summary.json").read_text()
    )
    assert summary["success"]
    profiles = summary["profiles"]
    assert len(profiles) == 4
    heads = ["Case / adjoint", "Full update", "Forward", "Adjoint", "Other"]
    rows, ccd_rows, stage_rows = [], [], []
    labels = []
    for p in profiles:
        label = ("Regularized" if p["label"] == "regularized" else "Zero smoothing") + (
            " / previous" if p["variant"] == "historical" else " / optimized"
        )
        labels.append(label)
        top = p["top_level_exclusive_seconds"]
        total = p["root_inclusive_seconds"]
        rows.append(
            [
                label,
                f"{total:.2f} s",
                *[
                    f"{top[k]:.2f} s ({100 * top[k] / total:.1f}%)"
                    for k in ("forward", "adjoint", "other")
                ],
            ]
        )
        ccd = p["ccd_exact"]
        remainder = (
            ccd["inclusive_seconds"] - ccd["broad_seconds"] - ccd["narrow_seconds"]
        )
        ccd_rows.append(
            [
                label,
                str(ccd["count"]),
                f"{ccd['inclusive_seconds']:.3f} s",
                f"{100 * ccd['inclusive_seconds'] / top['forward']:.1f}%",
                f"{ccd['broad_seconds']:.3f} s",
                f"{ccd['narrow_seconds']:.3f} s",
                f"{remainder:.3f} s",
                f"{1000 * ccd['seconds_per_call']:.1f} ms",
            ]
        )
        f = p["forward_receipt"]
        stages = f["tree_stage_inclusive_seconds"]
        stage_rows.append(
            [
                label,
                f"{f['coarse_steps']} / {stages['coarse_pncg']:.2f} s",
                f"{f['newton_steps']} / {stages['newton']:.2f} s",
                f"{stages['newton_cg']:.2f} s",
                f"{f['trace_regularization_retry_seconds']:.2f} s",
            ]
        )
    ccd_heads = [
        "Case / adjoint",
        "Queries",
        "CCD total",
        "% forward",
        "Broad phase",
        "Narrow phase",
        "Transfer / wrapper",
        "Mean/query",
    ]
    stage_heads = [
        "Case / adjoint",
        "PNCG steps / time",
        "Newton steps / time",
        "Newton CG incl. retries",
        "Recorded shift retries",
    ]
    detail_heads = ["Exclusive cost (s)", *labels]

    def details(key: str) -> list[list[str]]:
        return [
            [name, *[f"{p[key][name]:.3f}" for p in profiles]]
            for name in profiles[0][key]
        ]

    forward_rows = details("forward_exclusive_seconds")
    adjoint_rows = details("adjoint_exclusive_seconds")
    extras = []
    raw_paths = [
        (
            GROUP
            / "data"
            / (
                "inverse-duration-regularized-002"
                if p["label"] == "regularized"
                else "inverse-duration-hard-001"
            )
            / p["case"]
            / p["variant"]
            / "timing.json"
        )
        for p in profiles
    ]

    def inclusive_by_name(tree: dict, wanted: str) -> float:
        return sum(
            (node["inclusive_seconds"] if name == wanted else 0)
            + inclusive_by_name(node["children"], wanted)
            for name, node in tree.items()
        )

    raw = [json.loads(path.read_text()) for path in raw_paths]
    endpoints = json.loads(
        (GROUP / "data/inverse-duration-endpoint-checks.json").read_text()
    )
    hvp_counts = []
    for i, s in enumerate(raw):
        counts = {"newton": 0, "adjoint": 0}
        count_hvp(s["profiling"]["tree"], counts)
        stage_rows[i].extend([str(counts["newton"]), str(counts["adjoint"])])
        hvp_counts.append({"label": labels[i], **counts})
    stage_heads.extend(["Newton HVPs", "Adjoint HVPs"])
    summary["report_detail"] = {
        "linear_solve_hvp_counts": hvp_counts,
        "endpoint_checks": endpoints,
    }
    small_heads = ["Selected operation (inclusive s)", *labels]
    for key, label in [
        ("shape_metrics", "Shape diagnostics"),
        ("adam_step", "Adam.step (including dispatch)"),
        ("stress_projection", "Stress PSD projections (update + stationarity)"),
        ("checkpoint_read", "Update checkpoint read"),
        ("checkpoint_write", "Checkpoint serialization/write"),
        ("regularizers", "Stress regularizer evaluation"),
        ("skin_loss", "Skin-position loss"),
    ]:
        extras.append(
            [
                label,
                *[f"{inclusive_by_name(s['profiling']['tree'], key):.3f}" for s in raw],
            ]
        )

    report = f"""# Where inverse-physics time goes

Measured complete-update profiles, 22 September 2026. All four numerical measurements ran sequentially on the same Paratera RTX 4090, with eight IPC threads.

## Main findings

The dominant cost changes with the state. In the regularized Smile fit, speeding up the adjoint exposes forward Newton-CG as the main remaining cost. In the difficult zero-smoothing case, nonlinear forward convergence remains the bottleneck. CCD is significant but does not explain most of the runtime. Rebuilding discrete contact, PNCG directional curvature and repeated Hessian products also consume substantial time.

The optimized variant changes only the adjoint: relative tolerance `1e-4` and CUDA contact products during its linear solve. The previous variant uses `1e-7` and CPU contact products. Both retain the same hybrid forward solver, shift reset, final force threshold `1e-8`, active stress, full cranium/mandible/eyeball collision, Adam 0.3, and no magnitude/jaw prior or outer rejection. Each case retains its original smoothing weight.

## Full inverse-update duration

{markdown_table(heads, rows)}

Each measurement is the actual `Fitter.fit_step` advancing saved step15 to step16. It includes the operative checkpoint load, projected Adam proposal, objective/gradient evaluation, diagnostics and checkpoint/log writes. Model creation, preparation copies and reconstruction of the initial warm adjoint are excluded. Both variants receive the same reconstructed adjoint at the saved step15 state. These are warm-started update profiles, unlike the previous zero-initial-adjoint microbenchmarks.

“Other” includes optimizer work, parameter projections, regularizers, metrics, I/O and remaining Python/dispatch overhead. Its residual is larger on the first profile in each process, consistent with first-use setup; that residual is not fully attributed. We do not treat differences in this category as an algorithmic speedup.

## CCD measured directly

{markdown_table(ccd_heads, ccd_rows)}

CCD total is measured around the actual contact maximum-step query and includes GPU-to-host position preparation, swept broad-phase candidate construction, narrow-phase continuous testing and wrapper overhead. Counts include the initial boundary-motion check, which earlier inner-solver counters omitted. “Mean/query” is total divided by count, not a sampled median. These replace count-times-MouthOpen estimates for these four Smile updates.

Static candidate construction during contact-state initialization or update is **not** included in the CCD category. It appears separately as contact rebuild. The narrow-phase timer wraps `Candidates.compute_collision_free_stepsize`; its duration includes the native query orchestration, not only a single geometric predicate.

## Forward: additive cost breakdown

{markdown_table(detail_heads, forward_rows)}

All rows above are exclusive allocations and sum to forward time. FEM operators include volumetric tissue and skin work, along with launch/synchronization overhead; these are completed-operation wall times, not isolated GPU-kernel timings. Directional curvature is the combined `model.hess_quad` call used by coarse PNCG; its FEM and IPC portions were not separately instrumented. Contact HVP/transfer includes sparse multiplication, vector movement and gather/scatter, excluding separately counted Hessian assembly. The residual category includes Krylov vector operations, diagonal preconditioning, line-search/control work, material/boundary setup, and timing overhead.

Contact state updates rebuild static candidates in `OwnedContact.update`; CCD separately builds swept candidates. This means broad-phase work occurs in both places. Candidate reuse could reduce this duplication, but it needs proof that reused candidates cover every accepted and backtracked position before it is implemented.

## Forward: nested algorithm stages

{markdown_table(stage_heads, stage_rows)}

This is a second view of the same forward time. Newton-CG is inside Newton, and shift retries are inside its linear work; **do not add these columns together or to the preceding table**. The remainder outside PNCG/Newton covers forward initialization and terminal checks. Successful linear solves and unsuccessful curvature/regularization attempts both cost time.

Both variants use identical forward code and bitwise-identical proposed stress/jaw tensors, yet hard-case nonlinear paths differ. The hard endpoints differ by **{endpoints["cases"]["zero_smoothing"]["skin_rms_difference_mm"]:.5f} mm skin RMS** and **{endpoints["cases"]["zero_smoothing"]["max_node_difference_mm"]:.5f} mm maximum-node displacement**, despite both meeting the force/contact/no-inversion conditions. The regularized difference is only {endpoints["cases"]["regularized"]["skin_rms_difference_mm"]:.7f} mm skin RMS. Previous unchanged-control repeats already showed this variability. The table describes observed paths; it is not evidence that changing an adjoint backend accelerates a forward solve. Forward repeatability deserves attention before attributing timing gains to nonlinear solver changes.

The hard historical solve spent **45.39 s** in unsuccessful Newton regularization attempts and made **17,278 Newton HVPs**. Exact contact-Hessian assembly was just **0.213 s**. This is a repeated-application/convergence bottleneck, not an assembly bottleneck. The shorter hard forward path made 5,714 Newton HVPs; its lower cost is largely explained by different linear-solve work.

## Adjoint: additive cost breakdown

{markdown_table(detail_heads, adjoint_rows)}

The adjoint applies the unshifted physical Hessian. The optimized path uploads the exact IPC contact Hessian once per owned state, applies it on CUDA during CG/MINRES, then restores CPU contact for the residual and boundary derivative checks. No CCD query is part of the adjoint. A fresh discrete contact state is still built for its owned saved displacement. Remaining adjoint control includes state reconstruction, mixed derivatives and boundary gradients. Upload/assembly and sparse matvec work are charged separately where the hooks resolve them.

## Smaller costs inside the update

{markdown_table(small_heads, extras)}

These explanatory timings lie inside the top-level “other” category. Projections occur both for the Adam proposal and the stationarity diagnostic. Checkpoint write excludes preparation copies and includes the timed serialization/write function, but not every preceding CPU-tree conversion. Their sum is not a complete decomposition of “other”; construction, state restoration, cloning, conversion and dispatch remain in its residual.

## Relation to the complete fitting runs

The prior full trajectories provide a less intrusive timing baseline:

| Run | Updates | Total | Forward | Adjoint | Other |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original PNCG, regularized | 20 | 1455.17 s | 85.8% | 12.3% | 1.9% |
| Hybrid, regularized | 20 | 781.43 s | 71.4% | 25.6% | 3.0% |
| Hybrid, zero smoothing | 19 valid | 3013.78 s | 85.7% | 13.5% | 0.8% |

In the regularized hybrid's last five updates, the split was 32.4% forward, 59.5% adjoint and 8.1% other. In the zero-smoothing run's last five valid updates, forward remained 80.2%. The zero-smoothing trajectory has a different objective and is not a matched solver-speed comparison. Its failed proposal20 is excluded. Historical figures use authoritative `iteration-timing.jsonl` ledgers, not differences between intermediate trace timestamps.

## Optimization priorities

1. **Reduce Newton/adjoint Hessian products.** Improve conditioning/preconditioning and reduce repeated failed shift trials. Separately profile FEM kernel execution versus launch/synchronization overhead before deciding which to optimize. The previous scalar shift-reuse experiment did not establish endpoint agreement, so it remains experimental.
2. **Reduce contact rebuild work.** Profile-preserving broad-phase/candidate reuse is a concrete target; retain owned backward states and full collision coverage.
3. **Address expensive PNCG directional curvature.** It is a separate cost from CCD. Alternative curvature evaluation or switching policy needs matched convergence and gradient checks, because changing it changes the nonlinear path.
4. **Accelerate CCD where its measured share warrants it.** Broad phase and narrow phase have different costs; optimizing only the narrow phase cannot remove the full CCD total. Do not skip collision checks.
5. **Reduce diagnostic frequency if needed.** Shape statistics and checkpoint handling matter once updates become short. Adam arithmetic and skin-loss evaluation are not the main bottleneck.

These are priorities informed by the measurements, not additional implemented optimizations.

## Measurement limits and reproduction

Each scoped GPU operation is synchronized so its completed work is charged to the correct phase. These fences perturb overlap and add overhead. Inclusive parent time contains its children; only exclusive allocations are added. Every tree's accounting identity was checked. The reported update total is the `inverse_update` root, not `outer_profile_wall_seconds`, which also includes timing-hook setup and a final fence.

There is one profiled update per case/variant, no statistical confidence interval and no new 20-update optimized trajectory. Forward-path variability prevents interpreting all cross-row timing differences as optimization effects. Parameter gradients were checked in the preceding fixed-state study; these duration probes primarily establish timing and physical validity, not a new long-trajectory accuracy certificate.

Run from `exp/2026/09/22/solver-performance` on the compute host:

```sh
DEBUG=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 CHERRIES_NAME=inverse-duration CHERRIES_TAGS=smile,performance,inverse-profile \\
  python src/40-profile-inverse-updates.py --output-dir data/NEW-RUN \\
  --cases regularized --variants historical,candidate
# Run zero_smoothing separately after the first process exits.
```

Numerical output: `data/inverse-duration-regularized-002` and `data/inverse-duration-hard-001`. Each contains source snapshots, hashes, a common warm-start receipt, full timing trees and independent resulting checkpoints. `regularized-001` is preserved as a failed hook preflight; no inverse update completed there. Analysis: `src/41-analyze-inverse-duration.py`, `data/inverse-duration-analysis-001`. Historical audit: `data/historical-duration-audit-001`. Report: `src/42-report-inverse-duration.py` and this file.

Cherries debug runs retain local logs and receipts; no remote Comet URL is present. Validation includes timer nesting/static/slotted/exception-restoration CPU checks, compilation/Ruff, completed physical updates with no inversions, nonzero GPU-adapter use for candidate runs, and additive timing-tree checks. Profiling wrappers are restored after each update; no production physics source was changed.
"""
    (GROUP / "docs/40-inverse-duration-breakdown.md").write_text(report)
    (cfg.output_dir / "duration-report.md").write_text(report)
    (cfg.output_dir / "duration-evidence.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    shutil.copy2(
        GROUP / "data/inverse-duration-analysis-001/inverse-duration-breakdown.png",
        cfg.output_dir / "inverse-duration-breakdown.png",
    )
    index = cfg.output_dir / "index.html"
    prefix = (
        index.read_text()
        .split("<main>", 1)[0]
        .replace(
            "Facial inverse physics · current model & algorithm",
            "Inverse physics · measured duration breakdown",
        )
    )
    prefix += "<style>.table-scroll{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}.table-scroll table{min-width:620px}.lead{max-width:1000px}</style>"
    page = (
        prefix
        + f"""<main><header><div class="eyebrow">Apple · Full inverse-update profiling</div>
<h1>Where the time goes</h1><p class="lead">CCD is only part of the cost. The complete profiles separate nonlinear convergence, collision rebuilds, Hessian products, adjoints and update overhead.</p>
<div class="meta"><span>22 September 2026 · sequential RTX 4090 measurements</span><a href="duration-report.md">Detailed report</a><a href="optimizations.html">Adjoint optimization</a><a href="index.html">Model &amp; fitting</a></div></header>
<div class="status">One complete step15 → step16 update per case and setting, with a shared reconstructed warm adjoint. GPU timing fences perturb overlap. Hard forward paths vary even with unchanged code; these are cost profiles, not a new trajectory speedup claim.</div>
<section class="section"><span class="num">01 · Full update</span><h2>Forward, adjoint and everything else</h2>{web_table(heads, rows)}<p class="note">Previous: CPU contact, adjoint tolerance 10⁻⁷. Optimized: GPU contact in the adjoint solve only, tolerance 10⁻⁴. Forward settings and physics are identical. Swipe wide tables on a phone.</p></section>
<section class="section"><span class="num">02 · Direct CCD measurements</span><h2>Broad phase and continuous testing cost different amounts</h2>{web_table(ccd_heads, ccd_rows)}<p class="note">Includes the initial boundary-motion query. Static contact rebuilds are separate. Per-query values are measured averages, not the earlier MouthOpen estimates.</p></section>
<section class="section"><span class="num">03 · Cost distribution</span><figure><a href="inverse-duration-breakdown.png"><img src="inverse-duration-breakdown.png" alt="Full update, forward and adjoint cost distributions for regularized and zero-smoothing cases"></a><figcaption>Open the figure for full resolution. Stacked allocations are exclusive, so they sum without double-counting parent and child scopes.</figcaption></figure></section>
<section class="section"><h2>Forward stages explain long updates</h2>{web_table(stage_heads, stage_rows)}<p class="note">Nested view: CG is inside Newton, and shift retries are inside CG work. Do not add these columns together.</p><p>The hard endpoints differ by 0.02856 mm skin RMS and 0.29978 mm at the most changed node, despite identical parameter proposals and both passing physical checks. Treat their runtime difference as solver-path variability.</p><p>Candidate construction is repeated for swept CCD and again for static contact updates. PNCG's directional curvature is another distinct physical computation. Repeated Newton Hessian products and unsuccessful regularization attempts can outweigh CCD.</p></section>
<section class="section"><details open><summary>Full forward allocation, seconds</summary>{web_table(detail_heads, forward_rows)}</details><details><summary>Full adjoint allocation, seconds</summary>{web_table(detail_heads, adjoint_rows)}</details><details><summary>Optimizer, metrics and checkpoint costs, seconds</summary>{web_table(small_heads, extras)}</details></section>
<section class="section"><h2>What to optimize next</h2><ol><li>Reduce Hessian-product counts through better conditioning and preconditioning.</li><li>Reduce repeated contact-candidate construction while preserving collision coverage.</li><li>Measure alternatives to PNCG's costly directional-curvature evaluation.</li><li>Target CCD broad/narrow phases according to their measured shares.</li><li>Reduce expensive diagnostic frequency when updates become short.</li></ol><p class="note">These are measured priorities, not changes made by this profiling task. Shift reuse remains experimental. Collision checks and physical gates remain active.</p></section>
<section class="section"><h2>The bottleneck changes over a fit</h2><p>The original regularized hybrid trajectory spent <strong>71.4% forward / 25.6% adjoint</strong> overall. Its last five updates spent <strong>59.5% in the adjoint</strong>. The zero-smoothing trajectory spent <strong>85.7% forward</strong> overall. Optimizing one fixed percentage for every state would miss this change.</p></section>
<footer class="links"><a href="duration-report.md">Detailed report &amp; reproduction</a><a href="duration-evidence.json">Numerical evidence</a><a href="optimizations.html">Optimization validation</a><a href="index.html">Anatomy &amp; fit comparison</a></footer></main></html>"""
    )
    (cfg.output_dir / "duration.html").write_text(page)
    marker = "<!-- inverse-duration-follow-up -->"
    text = index.read_text()
    if marker not in text:
        banner = (
            marker
            + '<div class="callout"><strong>Measured inverse-update duration:</strong> CCD, contact rebuilds, forward linear solves and adjoints. <a href="duration.html">Open the detailed breakdown →</a></div>'
        )
        index.write_text(text.replace("</header>", "</header>" + banner, 1))
    cherries.log_output(GROUP / "docs/40-inverse-duration-breakdown.md")
    cherries.log_output(cfg.output_dir / "duration.html")


if __name__ == "__main__":
    cherries.main(main)
