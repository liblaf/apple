"""Report the completed, bounded Newton-only versus hybrid cold-start probe."""

from __future__ import annotations

import json
from pathlib import Path

from markdown_it import MarkdownIt

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent


class Config(cherries.BaseConfig):
    source: Path = GROUP / "data/cold-forward-comparison-001"
    output: Path = GROUP / "data/current-model-report-001"


def summarize(row: dict) -> dict:
    forward = row["forward"]
    trace = forward.get("trace", forward.get("solver", {}).get("trace", []))
    retries = [r for step in trace for r in step.get("regularization_retries", [])]
    return {
        "method": row["method"],
        "success": row["success"],
        "seconds": row["forward_wall_seconds"],
        "initial_force": row["prewarm_force_norm"],
        "final_force": forward.get(
            "grad_norm", forward.get("solver", {}).get("grad_norm")
        ),
        "pncg_steps": forward.get(
            "coarse_steps",
            0
            if row["method"] == "newton_diag"
            else forward.get("solver", {}).get("optimizer_step"),
        ),
        "pncg_curvature_calls": forward["counts"].get("hess_quad", 0),
        "newton_steps": len(trace),
        "hvp_calls": forward["counts"].get("hess_prod", 0),
        "inner_ccd_calls": forward["counts"].get("ccd", 0),
        "shifted_newton_steps": sum(step["shift"] > 0 for step in trace),
        "linear_retry_count": len(retries),
        "linear_retry_seconds": sum(r.get("seconds", 0) for r in retries),
        "successful_linear_solve_seconds": sum(
            step["linear"]["seconds"] for step in trace
        ),
        "failure": row.get("failure"),
        "shape": row.get("shape"),
        "collision": row.get("collision"),
        "saved_target_endpoint": row.get("saved_target_endpoint"),
    }


def main(cfg: Config) -> None:
    raw = json.loads((cfg.source / "summary.json").read_text())
    status = json.loads((cfg.source / "status.json").read_text())
    assert not status["running"]
    assert len(raw["results"]) == 2
    assert raw["results"][0]["initial"] == raw["results"][1]["initial"]
    rows = [summarize(row) for row in raw["results"]]
    headers = [
        "Method",
        "Forward time",
        "PNCG counter",
        "Newton steps",
        "Last recorded force",
        "Valid equilibrium",
        "HVP calls",
    ]
    values = []
    for row in rows:
        force = row["final_force"]
        values.append(
            [
                row["method"],
                f"{row['seconds']:.2f} s",
                str(row["pncg_steps"]),
                str(row["newton_steps"]),
                "unavailable" if force is None else f"{force:.8g}",
                "Yes" if row["success"] else "No",
                f"{row['hvp_calls']:,}",
            ]
        )
    table = "\n".join(
        "| " + " | ".join(row) + " |"
        for row in [headers, ["---"] * len(headers), *values]
    )
    details = []
    for row in rows:
        details.append(
            f"- **{row['method']}**: {row['shifted_newton_steps']} shifted Newton steps; {row['linear_retry_count']} linear retries costing {row['linear_retry_seconds']:.2f} s; successful linear solves cost {row['successful_linear_solve_seconds']:.2f} s. These times are included in total forward time. There were {row['pncg_curvature_calls']} PNCG curvature evaluations and {row['inner_ccd_calls']} inner CCD queries, excluding the initial boundary query."
        )
        if row["failure"]:
            details.append(
                f"  Failure: `{row['failure']['message']}`. This is failure within the declared budget, not proof that the method cannot converge."
            )
        if row["success"]:
            shape, contact = row["shape"], row["collision"]
            assert shape["inverted_tetrahedra"] == 0
            assert contact["state_feasible"]
            endpoint = row["saved_target_endpoint"]
            details.append(
                f"  Endpoint: zero inverted tetrahedra; full bone/eye contact audit passed. Difference from the saved warm-start endpoint: {endpoint['skin_weighted_rms_mm']:.6f} mm skin weighted RMS and {endpoint['maximum_node_mm']:.6f} mm maximum-node displacement. The saved endpoint was not used as the seed."
            )
    comparison = raw["results"][1].get("comparison_to_newton_diag", {})
    conclusion = (
        "Both arms reached valid equilibria; endpoint agreement must be considered alongside their timings."
        if all(row["success"] for row in rows)
        else "At least one arm did not reach a valid equilibrium within its declared budget. No matched-convergence speedup ratio is claimed."
    )
    report = f"""# One cold-start forward solve: Newton-CG-only versus hybrid

{conclusion}

{table}

On this case, direct Newton made substantially more force-residual progress in less elapsed time. Hybrid spent its entire budget in coarse PNCG and never reached Newton. This revises the expectation that the PNCG phase necessarily helps a difficult cold start. The stopping causes differ, so these durations do not measure time to the same converged solution. More Newton iterations could change its outcome, but that extension was not run.

## Fixed problem and timing

Both arms use the saved zero-smoothing Smile stress at accepted update 16, with jaw angle zero. They start from the same contact-valid, loaded neutral displacement from update 0. There is no new Adam proposal, expression displacement warm start, adjoint solve, load continuation or outer optimization update. This is a substantial activation jump from neutral, not the earlier small first-proposal probe.

The initial free-force norms were {rows[0]["initial_force"]:.10g} and {rows[1]["initial_force"]:.10g}. Activation, jaw and neutral-seed tensor hashes are identical. The physical model retains fixed neutral prestress, skin membrane and complete cranium, mandible and eyeball contact.

Newton-CG-only ran first, followed by hybrid, sequentially on the same Paratera RTX 4090 with eight IPC threads. Both use force tolerance `1e-8`, CG relative tolerance `1e-3`, scalar diagonal preconditioning, shift reset, CCD and Armijo. Hybrid switches at `max(1e-8, 0.001 * initial_force, 1e-7)`. Each arm has a 600-second forward wall budget and a 100-Newton-step limit; hybrid additionally performs coarse PNCG. A failed arm is retained and never silently replaced.

This is a cold displacement start with prewarmed operators, not a cold process. Model/input construction, fixed-state kernel prewarm and post-solve metrics/audits are excluded. Complete forward solves are CUDA-synchronized at their boundaries. Each primal reconstructs its own contact state after prewarm. This is one observation per method, in fixed order, without a confidence interval.

## Solver work and endpoint checks

{chr(10).join(details)}

Comparison receipt: `{json.dumps(comparison, sort_keys=True)}`.

For an interrupted PNCG solve, the force and iteration counter come from the last computed optimizer gradient in its failure receipt. They are not an independent force recomputation on a saved terminal displacement. Failed arms do not produce validated endpoint geometry or adjoint gradients. PNCG force norms need not decrease monotonically even when its energy line search succeeds.

## Reproduction and evidence

Run from `exp/2026/09/22/solver-performance` on the compute host, using a new output directory:

```sh
DEBUG=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 CHERRIES_NAME=cold-smile-newton-vs-hybrid CHERRIES_TAGS=smile,performance,cold-forward,newton,hybrid \\
  python -u src/43-compare-cold-forward.py \\
  --checkpoint data/inverse-duration-hard-001/zero_smoothing/historical/expressions/Smile/latest.pt \\
  --neutral-checkpoint data/smile-fit-adam03-no-smoothness-005/arms/hybrid_diag/expressions/Smile/initial.pt \\
  --output-dir data/NEW-RUN
```

Raw outputs, full traces, source archives and input hashes: `data/cold-forward-comparison-001`. Local Cherries logging was enabled with remote Comet disabled. The process can complete successfully while a solver arm fails its declared convergence budget; read the per-arm results. No production solver or physics setting was changed.

All 282 copied compute-host files passed SHA-256 verification. The log also contains three missing legacy `data/simple-skin-forward` asset-registration warnings; the explicit numerical/source receipts are present. These logging warnings were not the solver stopping causes.
"""
    (GROUP / "docs/43-cold-forward-comparison.md").write_text(report)
    (cfg.output / "cold-forward-report.md").write_text(report)
    evidence = {"rows": rows, "comparison": comparison, "protocol": raw["protocol"]}
    (cfg.output / "cold-forward-evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    prefix = (cfg.output / "index.html").read_text().split("<main>", 1)[0]
    body = MarkdownIt().enable("table").render(report)
    body = body.replace("<table>", '<div style="overflow-x:auto"><table>').replace(
        "</table>", "</table></div>"
    )
    page = (
        prefix
        + '<style>pre{white-space:pre-wrap;overflow-wrap:anywhere}table{min-width:620px}h2{margin-top:2em}</style><main><header><div class="eyebrow">Apple · Cold-start forward comparison</div><div class="meta"><a href="duration.html">Duration breakdown</a><a href="cold-forward-evidence.json">Numerical evidence</a></div></header>'
        + body
        + "</main></html>"
    )
    (cfg.output / "cold-forward.html").write_text(page)
    marker = "<!-- cold-forward-comparison -->"
    banner = (
        marker
        + '<div class="callout"><strong>Cold-start comparison:</strong> <a href="cold-forward.html">Newton-CG-only versus hybrid on the same Smile stress →</a></div>'
    )
    for name in ("index.html", "duration.html"):
        path = cfg.output / name
        content = path.read_text()
        if marker not in content:
            path.write_text(content.replace("</header>", "</header>" + banner, 1))
    cherries.log_output(GROUP / "docs/43-cold-forward-comparison.md")


if __name__ == "__main__":
    cherries.main(main)
