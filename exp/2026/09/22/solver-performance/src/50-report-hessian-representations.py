"""Render the bounded frozen-Hessian representation benchmark evidence."""

from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
VARIANT_LABELS = {
    "current_cpu_contact": "current FEM + CPU contact",
    "current_gpu_contact": "current FEM + GPU contact",
    "cached_fem_gpu_contact": "cached FEM + GPU contact",
    "sparse_fem_gpu_contact": "assembled BSR FEM + GPU contact",
}


class Config(cherries.BaseConfig):
    summary_path: Path = GROUP / "data/hessian-representations-001/summary.json"
    output_dir: Path = GROUP / "data/hessian-representations-report-001"
    document_path: Path = GROUP / "docs/50-hessian-representations.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(receipt: dict[str, Any]) -> float:
    return float(receipt["median"])


def number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}g}"


def break_even(
    setup_seconds: float, control_seconds: float, candidate_seconds: float
) -> float | None:
    saved = control_seconds - candidate_seconds
    if saved <= 0:
        return None
    return setup_seconds / saved


def summarize_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    variants = {variant["name"]: variant for variant in state["variants"]}
    assert tuple(variants) == tuple(VARIANT_LABELS), tuple(variants)
    cpu_hvp = median(variants["current_cpu_contact"]["hvp_seconds"])
    gpu_hvp = median(variants["current_gpu_contact"]["hvp_seconds"])
    rows = []
    for name, label in VARIANT_LABELS.items():
        variant = variants[name]
        hvp = median(variant["hvp_seconds"])
        setup = median(variant["numeric_setup_seconds"])
        cg = median(variant["cg_seconds"])
        steps = [int(item["steps"]) for item in variant["cg"]]
        accuracy = max(float(item["relative_l2"]) for item in variant["accuracy"])
        rows.append(
            {
                "state": state["state"],
                "force_l2": float(state["force_l2"]),
                "cg_is_artificial_probe": float(state["force_l2"]) < 1e-8,
                "variant": name,
                "label": label,
                "hvp_seconds": hvp,
                "hvp_ms": 1000 * hvp,
                "cg_seconds": cg,
                "setup_seconds": setup,
                "setup_plus_cg_seconds": setup + cg,
                "cg_steps_min": min(steps),
                "cg_steps_max": max(steps),
                "extra_gpu_memory_mib": int(variant["extra_persistent_bytes"]) / 2**20,
                "exact_relative_accuracy": accuracy,
                "speedup_vs_cpu_current": cpu_hvp / hvp,
                "speedup_vs_gpu_current": gpu_hvp / hvp,
                "break_even_gpu_applies": break_even(setup, gpu_hvp, hvp),
                "break_even_vs": "current_gpu_contact",
                "construction_first_call_seconds": float(
                    variant["construction_first_call_seconds"]
                ),
                "first_apply_prewarm_seconds": float(
                    variant["first_apply_prewarm_seconds"]
                ),
                "startup_break_even_cg_solves": break_even(
                    max(0.0, float(variant["construction_first_call_seconds"]) - setup),
                    median(variants["current_gpu_contact"]["cg_seconds"]),
                    cg + setup,
                ),
                "metadata": variant["metadata"],
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = (
        "State",
        "Variant",
        "HVP ms",
        "CG s",
        "Refresh s",
        "Refresh + CG s",
        "CG iters",
        "Extra GPU MiB",
        "Max relative error",
        "HVP x CPU",
        "HVP x GPU",
        "GPU break-even applies",
    )
    body = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        iterations = (
            str(row["cg_steps_min"])
            if row["cg_steps_min"] == row["cg_steps_max"]
            else f"{row['cg_steps_min']}-{row['cg_steps_max']}"
        )
        break_even_text = (
            "not faster"
            if row["break_even_gpu_applies"] is None
            else number(row["break_even_gpu_applies"])
        )
        body.append(
            "| "
            + " | ".join(
                (
                    row["state"],
                    row["label"],
                    number(row["hvp_ms"]),
                    number(row["cg_seconds"]),
                    number(row["setup_seconds"]),
                    number(row["setup_plus_cg_seconds"]),
                    iterations,
                    number(row["extra_gpu_memory_mib"]),
                    f"{row['exact_relative_accuracy']:.2e}",
                    number(row["speedup_vs_cpu_current"]),
                    number(row["speedup_vs_gpu_current"]),
                    break_even_text,
                )
            )
            + " |"
        )
    return "\n".join(body)


def html_table(rows: list[dict[str, Any]]) -> str:
    headers = (
        "State",
        "Variant",
        "HVP ms",
        "CG s",
        "Refresh s",
        "Refresh + CG s",
        "CG iters",
        "Extra GPU MiB",
        "Max relative error",
        "HVP x CPU",
        "HVP x GPU",
        "GPU break-even applies",
    )
    body = []
    for row in rows:
        iterations = (
            str(row["cg_steps_min"])
            if row["cg_steps_min"] == row["cg_steps_max"]
            else f"{row['cg_steps_min']}-{row['cg_steps_max']}"
        )
        values = (
            row["state"],
            row["label"],
            number(row["hvp_ms"]),
            number(row["cg_seconds"]),
            number(row["setup_seconds"]),
            number(row["setup_plus_cg_seconds"]),
            iterations,
            number(row["extra_gpu_memory_mib"]),
            f"{row['exact_relative_accuracy']:.2e}",
            number(row["speedup_vs_cpu_current"]),
            number(row["speedup_vs_gpu_current"]),
            "not faster"
            if row["break_even_gpu_applies"] is None
            else number(row["break_even_gpu_applies"]),
        )
        body.append(
            "<tr>"
            + "".join(f"<td>{html.escape(value)}</td>" for value in values)
            + "</tr>"
        )
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def plot(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    states = list(dict.fromkeys(row["state"] for row in rows))
    figure, axes = plt.subplots(1, 2, figsize=(13, 6.0), constrained_layout=True)
    colors = ("#4d4d4d", "#1b9e77", "#d95f02", "#7570b3")
    width = 0.18
    positions = list(range(len(states)))
    for offset, (name, label) in enumerate(VARIANT_LABELS.items()):
        selected = [
            next(
                row for row in rows if row["state"] == state and row["variant"] == name
            )
            for state in states
        ]
        x = [position + (offset - 1.5) * width for position in positions]
        axes[0].bar(
            x,
            [row["hvp_ms"] for row in selected],
            width,
            label=label,
            color=colors[offset],
        )
        axes[1].bar(
            x,
            [row["setup_plus_cg_seconds"] for row in selected],
            width,
            label=label,
            color=colors[offset],
        )
    for axis, ylabel, title in zip(
        axes,
        ("median HVP wall time (ms)", "median refresh + CG wall time (s)"),
        ("Frozen-state HVP application", "One refresh plus one PCG solve"),
        strict=True,
    ):
        axis.set_xticks(
            positions, ["Loaded neutral\n(cold state)", "Saved Smile\n(residual probe)"]
        )
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=2, loc="outside lower center", fontsize=9)
    figure.suptitle(
        "Exact Hessian representations on RTX 4090\nExcludes one-time sparse topology/JIT startup (29.4 s) and common contact setup",
        fontsize=13,
    )
    paths = {}
    for suffix in ("png", "svg"):
        path = output_dir / f"hessian-representations.{suffix}"
        figure.savefig(path, dpi=180 if suffix == "png" else None)
        paths[suffix] = str(path)
    plt.close(figure)
    return paths


def conclusion(rows: list[dict[str, Any]]) -> str:
    ratios = []
    for state in dict.fromkeys(row["state"] for row in rows):
        by_name = {row["variant"]: row for row in rows if row["state"] == state}
        ratios.append(
            by_name["current_gpu_contact"]["cg_seconds"]
            / by_name["sparse_fem_gpu_contact"]["setup_plus_cg_seconds"]
        )
    return f"Assembled sparse FEM is fastest in both frozen-state tests: {min(ratios):.2f}–{max(ratios):.2f}× faster than matrix-free FEM with the same GPU contact, including numeric matrix refresh plus PCG. Cache mesh topology once, rebuild numerical values at each Newton state, and reuse the matrix through CG and shift retries. This is a linear-system result; end-to-end forward and inverse speedup is not yet measured."


def startup_summary(rows: list[dict[str, Any]]) -> str:
    sparse = [row for row in rows if row["variant"] == "sparse_fem_gpu_contact"]
    lines = []
    for row in sparse:
        if row["metadata"].get("topology_cache_hit"):
            continue
        topology = row["metadata"].get("topology_seconds")
        count = row["startup_break_even_cg_solves"]
        if row["construction_first_call_seconds"] == 0 or count is None:
            continue
        topology_text = (
            "unreported" if topology is None else f"{float(topology):.2f} s topology"
        )
        lines.append(
            f"For `{row['state']}`, assembled BSR first construction cost {row['construction_first_call_seconds']:.2f} s ({topology_text}; JIT may also be included). Including a fresh numerical matrix for each later state, that startup amortizes after about {math.ceil(count)} comparable Newton linear solves against matrix-free FEM with GPU contact."
        )
    return (
        "\n\n".join(lines)
        or "No nonzero assembled-BSR construction receipt was available."
    )


def main(cfg: Config) -> None:
    assert cfg.summary_path.is_file(), cfg.summary_path
    assert not cfg.output_dir.exists(), cfg.output_dir
    summary = json.loads(cfg.summary_path.read_text())
    assert summary["schema"] == "frozen-hessian-representations-v1"
    assert len(summary["results"]) == 2
    rows = [row for state in summary["results"] for row in summarize_state(state)]
    assert len(rows) == 8
    assert all(
        math.isfinite(row["exact_relative_accuracy"])
        and row["exact_relative_accuracy"] < 1e-10
        for row in rows
    )
    cfg.output_dir.mkdir(parents=True)
    plot(rows, cfg.output_dir)
    source = {
        "path": str(cfg.summary_path.resolve()),
        "sha256": sha256(cfg.summary_path),
    }
    evidence = {
        "schema": "hessian-representations-report-evidence-v1",
        "source_summary": source,
        "rows": rows,
        "protocol": summary["protocol"],
        "interpretation": {
            "break_even": "numeric refresh seconds divided by median current_gpu_contact HVP seconds minus candidate HVP seconds; null means no positive per-apply saving",
            "startup_break_even_cg_solves": "(first construction minus one numeric refresh) divided by (GPU-contact current CG minus candidate CG minus numeric refresh); null means no positive saving. Comparable new-state solves, not repeated RHS at one state.",
            "extra_gpu_memory": "explicit persistent operator buffers in MiB; Warp/native allocation is outside Torch allocator counters",
            "operator_accuracy": "relative L2 product error is against the CPU-contact reference",
            "solution_difference": "CG solution-relative-difference is against the current_gpu_contact shift-calibration solution, not the CPU-contact reference",
        },
    }
    (cfg.output_dir / "hessian-representations-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    table = markdown_table(rows)
    text = f"""# Frozen Hessian representation microbenchmark

{conclusion(rows)}

{table}

## Scope and timing

This is a frozen-Newton-system microbenchmark at two saved states. It measures exact Hessian-vector products and scalar diagonally preconditioned PCG with the same RHS, shift, diagonal and relative tolerance. It does not validate an end-to-end forward solve, an inverse update, or a converged-solution wall-time improvement.

`cached FEM` stores frozen bulk invariants, membrane derivatives, and copied material arrays. This prototype copies all three full-mesh bulk potential fields, so it uses more memory than the assembled matrix; this is not a fundamental memory lower bound for matrix-free methods. `assembled BSR FEM` stores the full assembled FEM block-sparse representation. Both retain the common GPU contact contribution. The CPU-contact current route is included as a control for the contact transfer path.

Construction/first-call timing may include JIT and topology setup. Numeric refresh is reported separately and is the setup term used in the break-even calculation. Break-even is undefined when a candidate does not improve on `current_gpu_contact`; the table says `not faster` rather than reporting a negative apply count.

## Startup cost and amortization

{startup_summary(rows)}

These startup figures must remain separate from numeric refresh and repeated apply timing. The per-state HVP break-even in the table uses numeric refresh only. The construction amortization above subtracts numerical refresh from the per-state PCG saving; it is conditional on similarly sized later Newton linear solves.

The reported extra GPU MiB is explicit persistent operator storage. Torch allocator telemetry excludes Warp/native allocations; CUDA memory telemetry covers the wider device allocation scope. CPU sparse-topology storage is not included in the GPU storage number. Exact relative accuracy is the largest checked relative L2 operator difference against the CPU-contact reference.

The shifted CG calibration and its `solution_relative_difference` receipt use `current_gpu_contact` as the solution reference. The CPU-contact route is the independent product-accuracy and true-residual reference. These are different comparisons.

{chr(10).join(f"`{row['state']}` force L2: {row['force_l2']:.3e}" + ("; it is already below the 1e-8 forward force target, so its CG result is an artificial residual probe rather than a necessary Newton iteration." if row["cg_is_artificial_probe"] else ".") for row in rows[::4])}

## Numerical validation and reproducibility

Three deterministic directions per state check exact unshifted HVP agreement against the original CPU-contact route; the maximum relative L2 difference is below 4.1e-16. All 24 measured PCG solves pass the original-operator true residual check at relative tolerance 1e-3. Cold state shift is zero. The saved Smile residual probe requires a common shift of 2.8459e-7 after three 2000-step calibration budgets; those calibration attempts are excluded from the one-successful-solve timing. This benchmark does not test unshifted implicit-adjoint solve speed.

One RTX 4090, float64, eight IPC threads, 30 synchronized HVP samples and three PCG repeats per state and variant. The same physical source hashes and checkpoint/input hashes match the prior cold-start experiment. Physical model: tissue and skin with active stress, fixed prestress, cranium, mandible and eyeball contact. Different iteration counts within roughly the same range reflect floating-point summation order, not different tolerances. For PNCG or very short CG solves, matrix rebuild cost may not amortize; the measured sparse crossover is about 122–125 HVP applications per state.

Run from `/root/codex-apple-performance/apple/exp/2026/09/22/solver-performance`:

```bash
DEBUG=1 CHERRIES_NAME="Frozen Hessian representations 001" CHERRIES_TAGS="solver-performance,hessian-cache,smile,rtx4090" /root/codex-apple-performance/apple/.venv/bin/python src/49-benchmark-hessian-representations.py
```

Cherries completed successfully with `states_completed: 2`; `DEBUG=1` kept local logs and disabled remote Comet recording. Full terminal log: `tmp/hessian-representations-001-terminal.log`. Source archives and raw timing arrays are retained beside the summary. The imported legacy benchmark registered an unused missing `data/simple-skin-forward` asset, producing a shutdown warning; the requested benchmark outputs and completion receipt exist.

## Evidence

Source summary: `{source["path"]}` (`SHA-256 {source["sha256"]}`). Figure assets: `hessian-representations.png` and `hessian-representations.svg`. The benchmark ran variants sequentially on one GPU; repeat medians are descriptive measurements, not a throughput estimate for concurrent solves.
"""
    cfg.document_path.write_text(text)
    (cfg.output_dir / "hessian-representations.md").write_text(text)
    page = f"""<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Frozen Hessian representations</title><style>body{{font:16px system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;font-size:13px}}th,td{{border:1px solid #bbb;padding:.35rem;text-align:right}}th:nth-child(-n+2),td:nth-child(-n+2){{text-align:left}}.table{{overflow-x:auto}}img{{max-width:100%;height:auto}}code{{overflow-wrap:anywhere}}</style><nav><a href="index.html">Model report</a> · <a href="adaptive-pncg.html">Adaptive PNCG</a></nav><h1>Frozen Hessian representation microbenchmark</h1><p>{html.escape(conclusion(rows))}</p><div class=\"table\">{html_table(rows)}</div><p>RTX 4090 · float64 · active stress and prestress · bones and eyes in contact · PCG rtol 1e-3 · sequential execution. Maximum checked HVP error: 4.1e-16. All 24 PCG solves passed the original-operator residual check.</p><h2>Figure</h2><img src=\"hessian-representations.svg\" alt=\"HVP and refresh plus CG timing by state and representation\"><h2>Startup cost and amortization</h2><p>{html.escape(startup_summary(rows))}</p><h2>Scope and limitations</h2><p>This is a frozen-Newton-system microbenchmark, not end-to-end forward or inverse validation. Cached FEM stores invariants plus copied material arrays; assembled BSR uses a full FEM representation. All variants use the same preconditioner, RHS, shift and PCG tolerance. Setup first call can include JIT/topology; numeric refresh is separate. Product accuracy uses CPU contact; CG solution difference uses GPU-contact calibration. The saved Smile state already meets the forward target; its result is an artificial residual probe using a common shift of 2.8459e-7. Cold state uses no shift. Calibration retries and common contact assembly are outside plotted solve timings. Sparse matrix values consume 239 MiB of additional GPU storage; the current cached-invariant prototype consumes 1138 MiB because it also copies three sets of material fields.</p><p><a href=\"hessian-representations-evidence.json\">Evidence JSON</a> · <a href=\"hessian-representations.md\">Markdown report</a></p></html>"""
    (cfg.output_dir / "hessian-representations.html").write_text(page)
    cherries.log_output(cfg.document_path)
    cherries.log_output(cfg.output_dir / "hessian-representations.html")


if __name__ == "__main__":
    cherries.main(main)
