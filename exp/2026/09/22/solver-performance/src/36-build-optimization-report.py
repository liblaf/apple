# ruff: noqa: RUF001, S608
"""Publish completed optimization evidence into the existing local report."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/current-model-report-001"


def read(path: str) -> dict:
    return json.loads((GROUP / "data" / path).read_text())


def main(cfg: Config) -> None:
    validation = read("adjoint-scope-validation-001/summary.json")
    figure = read("solver-optimization-visuals-001/summary.json")
    repeat = read("forward-control-hard-repeat-002/cross-run-comparison.json")
    assert validation["success"]
    assert figure["success"]
    rows = []
    accuracy = []
    for arm in validation["arms"]:
        row = next(row for row in arm["rows"] if row["rtol"] == 1e-4)
        metrics = row["comparisons_to_cpu_1e-8"]
        for name, item in metrics.items():
            if name == "next_projected_adam_jaw_update":
                assert item["difference_l2"] == 0
            else:
                assert item["relative_l2"] <= 1e-3
        old = arm["old_cpu_1e-7_adjoint_seconds"]
        new = row["adjoint"]["seconds"]
        label = (
            "Regularized step 20"
            if arm["label"] == "baseline20"
            else "Zero-smoothing step 19"
        )
        rows.append([label, f"{old:.2f} s", f"{new:.2f} s", f"{old / new:.2f}×"])
        accuracy.append(
            [
                label,
                f"{100 * metrics['data_gradient_q']['relative_l2']:.5f}%",
                f"{100 * metrics['total_gradient_q']['relative_l2']:.5f}%",
                f"{100 * metrics['data_gradient_jaw']['relative_l2']:.5f}%",
                f"{100 * metrics['next_projected_adam_q_update']['relative_l2']:.5f}%",
            ]
        )
    timing_heads = [
        "Saved state",
        "CPU contact, rtol 1e-7",
        "GPU adjoint only, rtol 1e-4",
        "Speedup",
    ]
    accuracy_heads = [
        "Saved state",
        "Data stress gradient",
        "Total stress gradient",
        "Jaw gradient",
        "Projected Adam stress update",
    ]

    def md_table(heads: list[str], body: list[list[str]]) -> str:
        return "\n".join(
            "| " + " | ".join(row) + " |"
            for row in [heads, ["---"] * len(heads), *body]
        )

    def web_table(heads: list[str], body: list[list[str]]) -> str:
        head = "<tr>" + "".join(f"<th>{html.escape(x)}</th>" for x in heads) + "</tr>"
        content = "".join(
            "<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in row) + "</tr>"
            for row in body
        )
        return f'<div class="table-scroll"><table>{head}{content}</table></div>'

    flags = "--adjoint-rtol 1e-4 --gpu-contact true --gpu-contact-scope adjoint --newton-shift-policy reset"
    report = f"""# Smile solver optimization results

22 September 2026. Sequential runs on the same Paratera RTX 4090, eight IPC threads.

The selected opt-in candidate is **adjoint relative tolerance `1e-4` with GPU contact products scoped to the adjoint linear solve**. It leaves the forward solver unchanged. It passed direct gradient and projected-Adam comparisons on two saved states. Newton shift reuse and GPU contact during forward solving remain experimental.

## Measured adjoint improvement

{md_table(timing_heads, rows)}

These are single cold-start adjoint measurements with identical zero initial adjoints, excluding model construction and transfer. They are not whole inverse-update speedups or new 20-update fits. Existing production fits warm-start adjoints; their speedup may differ.

The CPU-only tolerance sweep already gave 1.67× and 1.80× at `1e-4` versus `1e-7`. Moving contact matvecs to the GPU gives the additional gain above. The complete five-point CPU and three-point whole-adjoint GPU sweep is plotted separately below.

## Direct accuracy against CPU rtol 1e-8

Each percentage is `100 * ||candidate-reference|| / ||reference||`. The reference and candidate use the identical saved displacement, active stress, material fields, boundary conditions and objective. No forward solve is run during these checks.

{md_table(accuracy_heads, accuracy)}

All values pass the provisional 0.1% engineering comparison gate. The next projected jaw update is exactly zero in both saved states because of its bound; raw jaw-gradient error is reported separately. This does not validate future jaw motion or accumulated trajectory error. The reference is tight differentiation at the saved approximate equilibrium, not a newly tightened primal equilibrium.

## Implementation and model

The physical adjoint solves the original unshifted Hessian system. The exact IPC contact Hessian is assembled on CPU once per owned state, uploaded as CSR, then applied on CUDA during CG/MINRES. CPU contact is restored before recomputing the true residual and fixed-boundary derivative. Installation and cleanup are exception-safe. This does not move broad phase, collision detection, CCD or all IPC operations onto the GPU.

The cached complete-model HVP microbenchmark fell from 3.779 ms to 2.938 ms (1.286×, five samples per route); first GPU upload plus HVP cost 11.564 ms. Three deterministic directions and a perturbed state agreed with CPU within 7e-17 relative L2. Cache refreshes were checked on state updates and state switches; the old-state replay tested upload invalidation, without a separate numerical oracle for that final replay.

The model remains active-stress volumetric tissue plus skin mechanics, with complete cranium, mandible and fixed eyeball collision. Collision surface counts: 34,245 soft vertices / 65,580 triangles; cranium 17,575 / 35,162; mandible 9,476 / 18,948; eyes 1,298 / 2,560. Contact is frictionless soft-versus-rigid IPC; soft-soft and rigid-rigid pairs are excluded. Forward force tolerance stays `1e-8`. Adam stays at 0.3, with no magnitude or jaw prior and no outer step rejection. The two saved states retain their respective original smoothing weights, including the zero-smoothing diagnostic.

## Why forward changes were not selected

The hybrid keeps PNCG until the switch threshold, then uses safeguarded Newton-CG with diagonal preconditioning, curvature-dependent shifts, CCD and Armijo. Shift reuse carries a normalized successful shift to the next Newton iteration; a revised guard resets it near convergence. The unchanged `reset` policy remains selected.

Easy forward replay: CPU reset 7.374 s; CPU reuse 7.775 s; GPU reset 5.755 s; GPU reuse 5.804 s. All passed the endpoint gate (skin RMS <= 0.001 mm and maximum full-node difference <= 0.01 mm).

Hard replay: CPU reset 75.933 s; CPU reuse 62.319 s; GPU reset 78.759 s; GPU reuse 76.065 s. Every solve passed force/contact/inversion checks, but every non-control variant failed endpoint agreement. Maximum node differences were 0.0311, 0.2260 and 0.1422 mm respectively. The earlier unguarded shift-reuse trial also failed; its apparent 2.44× speedup is not an accepted result.

An unchanged CPU-reset repeat took **{repeat["seconds"]:.3f} s**, versus 75.933 s, with **{repeat["skin_rms_difference_mm"]:.7f} mm skin RMS** and **{repeat["maximum_node_difference_mm"]:.7f} mm maximum-node** difference. It too exceeded the maximum-node gate. PNCG coarse steps changed from 271 to 213 under identical parameters. Thus hard-case failures and timing differences cannot be assigned solely to the new variants. Sensitivity to floating-point reductions/contact branching is a plausible explanation, not a proven root cause. Restricting GPU changes to the adjoint preserves the existing forward implementation.

## Bottleneck and practical use

Late regularized-fit updates previously spent about 59.5% of their time in the adjoint, so this is a useful target. The zero-smoothing diagnostic spent about 85.7% in forward solving; its main bottleneck remains nonlinear equilibrium/contact behavior. A forward convergence and repeatability study is more valuable there than claiming a speedup from one trajectory.

Use these explicit options with `src/20-fit-smile.py`:

```sh
{flags}
```

Historical defaults remain unchanged (`1e-7`, GPU off, shift reset). No new full inverse trajectory has been run with the candidate settings.

## Commands and evidence

Run from `exp/2026/09/22/solver-performance`. Remote numerical runs used `/root/codex-apple-performance/apple/.venv/bin/python`, `DEBUG=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8`, readable `CHERRIES_NAME`, and task tags. All GPU jobs were serialized. Local debug Cherries receipts were retained; there is no remote Comet run URL.

```sh
# Fixed-state sweep; set CHECKPOINT and OUTPUT to an unused run directory.
DEBUG=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 CHERRIES_NAME=adjoint-scope CHERRIES_TAGS=smile,adjoint,scoped-gpu \\
  python src/30-adjoint-tolerance-sweep.py --checkpoint "$CHECKPOINT" --output-dir "$OUTPUT" \\
  --tolerances 1e-8,1e-4 --reference-rtol 1e-8 --forward-atol 1e-8 --ipc-threads 8 \\
  --gpu-contact true --gpu-contact-scope adjoint
```

CPU reference runs used `--tolerances 1e-8 --gpu-contact false`. The earlier CPU ladder was `1e-8,1e-7,1e-6,1e-5,1e-4`; whole-adjoint GPU sweeps used `1e-8,1e-7,1e-4`. Forward replay used `src/32-replay-forward-optimizations.py` on each step-15 checkpoint; the unchanged repeat used `--variants reset_cpu`. Complete configurations, input/source hashes and raw vectors are retained per run.

Evidence directories under `data/`: `adjoint-tolerance-*-001`, `adjoint-tolerance-*-gpu-002`, `adjoint-reference-*-cpu-003`, `adjoint-scope-*-gpu-003`, `adjoint-scope-validation-001`, `gpu-contact-benchmark-002`, `forward-optimizations-easy-002`, `forward-optimizations-hard-001`, `forward-control-hard-repeat-002`, and `solver-optimization-visuals-001`. Failed `gpu-contact-benchmark-001` (driver state-selection error, corrected) and `shift-reuse-easy-001` remain preserved.

Validation: compilation and Ruff; CPU SPD/nonconvex solver checks; adapter install/uninstall, cache and exception-cleanup checks; actual GPU HVP equivalence; fixed-state gradient/Adam comparisons; forward force, collision and inversion checks. The changes are experiment-local and retain the existing dirty workspace. No commit or publication outside the temporary tailnet report was made.
"""
    (GROUP / "docs/31-solver-optimization-results.md").write_text(report)
    cfg.output_dir.mkdir(exist_ok=True)
    (cfg.output_dir / "optimization-report.md").write_text(report)
    evidence = {
        "validation": validation,
        "sweep_and_forward": figure,
        "unchanged_forward_repeat": repeat,
    }
    (cfg.output_dir / "optimization-evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    shutil.copy2(
        GROUP / "data/solver-optimization-visuals-001/solver-optimizations.png",
        cfg.output_dir / "solver-optimizations.png",
    )
    index = cfg.output_dir / "index.html"
    prefix = index.read_text().split("<main>", 1)[0]
    prefix = prefix.replace(
        "Facial inverse physics · current model & algorithm",
        "Smile solver · optimization results",
    )
    prefix += "<style>.table-scroll{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}td:first-child{font-weight:600}</style>"
    page = (
        prefix
        + f"""<main><header>
<div class="eyebrow">Apple · Measured solver follow-up</div>
<h1>Faster adjoints.<br>Forward path preserved.</h1>
<p class="lead">Use <strong>adjoint tolerance 10⁻⁴</strong> and GPU contact products only inside the adjoint linear solve. Direct CPU-reference checks pass on both saved Smile states.</p>
<div class="meta"><span>22 September 2026 · RTX 4090 · sequential runs</span><a href="index.html">Model &amp; original fit comparison</a><a href="optimization-report.md">Full report</a></div></header>
<div class="status">These are fixed-state adjoint timings. A full inverse trajectory with the candidate settings has not been run. Shift reuse and GPU contact in forward solving remain experimental.</div>
<section class="section"><span class="num">01 · Measured result</span><h2>Less work in each adjoint solve</h2>{web_table(timing_heads, rows)}
<p class="note">Identical zero initial adjoints; model setup excluded. Historical fits use warm starts, so these ratios do not directly predict whole-fit speed.</p></section>
<section class="section"><span class="num">02 · Accuracy</span><h2>Compared directly with CPU tolerance 10⁻⁸</h2>{web_table(accuracy_heads, accuracy)}<p class="note">On narrow screens, swipe the table horizontally.</p>
<p class="note">Relative L2 errors; all below the 0.1% comparison gate. Both projected jaw updates are zero at their bound. The saved forward state stays fixed, so this does not establish accumulated trajectory accuracy.</p></section>
<section class="section"><span class="num">03 · What changed</span><div class="grid"><div><h2>Upload once, multiply on GPU</h2><p>The exact contact Hessian is assembled on CPU and uploaded once per owned state. CG/MINRES reuses its CUDA sparse matrix. CPU contact is restored for the true residual and boundary derivative checks.</p><p class="note">Cached whole-model Hessian products: 3.779 → 2.938 ms. First upload plus product: 11.564 ms. No Hessian damping is introduced into the physical adjoint.</p></div><div><h2>Same model and optimization</h2><p>Active stress, full skull/mandible/eyeball collision, Adam 0.3, no magnitude or jaw prior, no outer step rejection, and forward force tolerance 10⁻⁸ remain in place.</p><p class="note">GPU contact defaults to off. When enabled, its default scope is the adjoint solve.</p></div></div></section>
<section class="section"><span class="num">04 · Full sweep</span><h2>Accuracy, cost and forward limitations</h2><figure><a href="solver-optimizations.png"><img src="solver-optimizations.png" alt="Adjoint timings and errors across tolerances, plus easy and hard forward timings with failed agreement marked"></a><figcaption>Open the figure for full resolution. The plotted GPU sweep uses the earlier whole-adjoint adapter. The tables above use the final adjoint-solver-only option. Hatched forward bars passed physical checks but failed endpoint agreement against their first control.</figcaption></figure></section>
<section class="section"><h2>Why shift reuse stays experimental</h2><p>Easy forward GPU replay improved from 7.37 s to 5.75 s and passed endpoint agreement. The hard case did not establish a reliable gain. Even an unchanged CPU control varied from 75.93 s to {repeat["seconds"]:.2f} s, with a {repeat["maximum_node_difference_mm"]:.4f} mm maximum-node difference.</p><p>That variability prevents attributing the hard-case differences solely to new code. The selected candidate therefore changes only the adjoint. Forward convergence and repeatability remain the next useful optimization target.</p></section>
<section class="section"><h2>Candidate options</h2><pre>{html.escape(flags)}</pre><p class="note">Pass to <code>src/20-fit-smile.py</code>. Existing defaults and historical results are preserved.</p></section>
<footer class="links"><a href="optimization-report.md">Report &amp; reproduction commands</a><a href="optimization-evidence.json">Numerical evidence</a><a href="zero-smoothness.html">Zero-smoothing diagnostic</a><a href="index.html">Model &amp; fit comparison</a></footer></main></html>"""
    )
    (cfg.output_dir / "optimizations.html").write_text(page)
    marker = "<!-- solver-optimization-follow-up -->"
    text = index.read_text()
    if marker not in text:
        banner = (
            marker
            + '<div class="callout"><strong>Solver optimization results:</strong> looser adjoint tolerance plus GPU contact in the adjoint only. <a href="optimizations.html">See measured speedups, accuracy and forward limitations →</a></div>'
        )
        index.write_text(text.replace("</header>", "</header>" + banner, 1))
    cherries.log_output(cfg.output_dir / "optimizations.html")
    cherries.log_output(GROUP / "docs/31-solver-optimization-results.md")


if __name__ == "__main__":
    cherries.main(main)
