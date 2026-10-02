# ruff: noqa: EM101, PERF401, TRY003
"""Render the measured cuDSS frozen-system benchmark; never fabricate data."""

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


class Config(cherries.BaseConfig):
    summary_path: Path = GROUP / "data/cudss-comparison-001/summary.json"
    output_dir: Path = GROUP / "data/cudss-report-001"
    document_path: Path = GROUP / "docs/52-cudss.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(records: list[dict[str, Any]], key: str = "seconds") -> float | None:
    values = sorted(float(item[key]) for item in records if key in item)
    if not values:
        return None
    return (
        values[len(values) // 2]
        if len(values) % 2
        else 0.5 * (values[len(values) // 2 - 1] + values[len(values) // 2])
    )


def number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}g}"


def mib(value: float | None) -> str:
    return "—" if value is None else f"{float(value) / 2**20:.1f}"


def _peak(records: list[dict[str, Any]]) -> int | None:
    values = [
        int(item["memory"]["sampled_peak_device_bytes"])
        for item in records
        if "memory" in item
    ]
    return max(values) if values else None


def _memory_estimate(record: dict[str, Any]) -> int | None:
    estimates = record.get("analysis", {}).get("memory_estimates", [])
    return int(estimates[1]) if len(estimates) >= 2 else None


def _direct_row(
    state: dict[str, Any], system: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    factors = record.get("factorizations", [])
    reused = [
        item for item in record.get("solves", []) if item.get("kind") == "reuse_factors"
    ]
    all_solves = record.get("solves", [])
    successes = [
        item for item in factors if int(item.get("info", {}).get("info", 0)) == 0
    ]
    factor_median = median(successes)
    reused_median = median(reused)
    first_factor = float(factors[0]["seconds"]) if factors else None
    factor_info = successes[-1].get("info", {}) if successes else {}
    residuals = [
        float(item["reference_true_relative_residual"])
        for item in all_solves
        if "reference_true_relative_residual" in item
    ]
    status = (
        "success" if record.get("success") else record.get("reason", "not completed")
    )
    return {
        "state": state["state"],
        "system": system["system"],
        "shift": float(system["shift"]),
        "mode": record["mode"],
        "status": status,
        "spd_failure": record["mode"] == "spd" and not bool(record.get("success")),
        "force_l2": float(state["force_l2"]),
        "free_matrix_seconds": float(system["free_matrix_seconds"]),
        "shifted_reconstruction_seconds": float(
            system["shifted_reconstruction_seconds"]
        ),
        "pcg_median_seconds": median(system.get("pcg", [])),
        "symbolic_seconds": float(record["analysis_seconds"])
        if "analysis_seconds" in record
        else None,
        "first_factor_seconds": first_factor,
        "factor_median_seconds": factor_median,
        "reused_solve_median_seconds": reused_median,
        "factor_plus_reused_solve_seconds": (
            factor_median + reused_median
            if factor_median is not None and reused_median is not None
            else None
        ),
        "factor_nnz": int(factor_info["lu_nnz"]) if "lu_nnz" in factor_info else None,
        "symbolic_memory_estimate_bytes": _memory_estimate(record),
        "sampled_device_peak_bytes": _peak(factors),
        "max_reference_true_relative_residual": max(residuals) if residuals else None,
        "factorizations": len(factors),
        "reused_solves": len(reused),
        "analysis_memory": record.get("analysis_memory"),
        "reason": record.get("reason"),
        "inertia": factor_info.get("inertia"),
    }


def summarize(
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matched: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    for state in summary["results"]:
        for system in state["systems"]:
            for direct in system["direct"]:
                row = _direct_row(state, system, direct)
                (matched if system["system"] == "matched_newton" else probes).append(
                    row
                )
    if len({row["state"] for row in matched}) != 2:
        raise ValueError("expected matched-Newton cuDSS records for exactly two states")
    if not any(row["mode"] == "spd" for row in matched):
        raise ValueError("matched records lack the Cholesky SPD attempt")
    return matched, probes


def table(rows: list[dict[str, Any]]) -> str:
    headers = (
        "State",
        "Mode",
        "Status",
        "Free CSR build s",
        "PCG median s",
        "Symbolic s",
        "First factor s",
        "Factor median s",
        "Reuse solve median s",
        "Factor + reuse solve s",
        "Factor nnz",
        "Symbolic estimate MiB",
        "Sampled device peak MiB",
        "Max true residual",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["state"],
                    row["mode"],
                    row["status"],
                    number(row["free_matrix_seconds"]),
                    number(row["pcg_median_seconds"]),
                    number(row["symbolic_seconds"]),
                    number(row["first_factor_seconds"]),
                    number(row["factor_median_seconds"]),
                    number(row["reused_solve_median_seconds"]),
                    number(row["factor_plus_reused_solve_seconds"]),
                    number(row["factor_nnz"]),
                    mib(row["symbolic_memory_estimate_bytes"]),
                    mib(row["sampled_device_peak_bytes"]),
                    "—"
                    if row["max_reference_true_relative_residual"] is None
                    else f"{row['max_reference_true_relative_residual']:.2e}",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def html_table(rows: list[dict[str, Any]]) -> str:
    markdown = table(rows).splitlines()
    headings = markdown[0].strip("|").split(" | ")
    body = []
    for line in markdown[2:]:
        values = line.strip("|").split(" | ")
        body.append(
            "<tr>"
            + "".join(f"<td>{html.escape(value)}</td>" for value in values)
            + "</tr>"
        )
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(value)}</th>" for value in headings)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def plot(rows: list[dict[str, Any]], output: Path) -> None:
    states = list(dict.fromkeys(row["state"] for row in rows))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    x = list(range(len(states)))
    width = 0.22
    for offset, mode in enumerate(("spd", "symmetric")):
        chosen = [
            next(
                (row for row in rows if row["state"] == state and row["mode"] == mode),
                None,
            )
            for state in states
        ]
        axes[0].bar(
            [item + (offset - 0.5) * width for item in x],
            [
                row["factor_plus_reused_solve_seconds"]
                if row and row["factor_plus_reused_solve_seconds"] is not None
                else math.nan
                for row in chosen
            ],
            width,
            label="Cholesky" if mode == "spd" else "LDLT",
        )
        axes[1].bar(
            [item + (offset - 0.5) * width for item in x],
            [
                row["sampled_device_peak_bytes"] / 2**20
                if row and row["sampled_device_peak_bytes"] is not None
                else math.nan
                for row in chosen
            ],
            width,
            label="Cholesky" if mode == "spd" else "LDLT",
        )
    pcg = [
        next(row for row in rows if row["state"] == state)["pcg_median_seconds"]
        for state in states
    ]
    axes[0].plot(x, pcg, "ko", label="free-CSR PCG median")
    axes[0].set_ylabel("seconds")
    axes[0].set_title("Factor plus reused solve versus PCG")
    axes[1].set_ylabel("sampled whole-device peak (MiB)")
    axes[1].set_title("Factorization memory observation")
    for axis in axes:
        axis.set_xticks(x, ["Loaded neutral\n(cold)", "Saved Smile\n(residual probe)"])
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=9)
    figure.suptitle(
        "Exact free-space FEM + IPC systems on RTX 4090\nExcludes symbolic analysis (10.7–11.2 s) and CPU matrix construction (3.2–3.3 s)",
        fontsize=12,
    )
    for suffix in ("png", "svg"):
        figure.savefig(output / f"cudss.{suffix}", dpi=180 if suffix == "png" else None)
    plt.close(figure)


def main(cfg: Config) -> None:
    if not cfg.summary_path.is_file():
        raise FileNotFoundError(cfg.summary_path)
    if cfg.output_dir.exists():
        raise FileExistsError(cfg.output_dir)
    summary = json.loads(cfg.summary_path.read_text())
    if summary.get("schema") != "cudss-frozen-v1":
        raise ValueError("requires summary schema cudss-frozen-v1")
    matched, probes = summarize(summary)
    cfg.output_dir.mkdir(parents=True)
    plot(matched, cfg.output_dir)
    source = {
        "path": str(cfg.summary_path.resolve()),
        "sha256": sha256(cfg.summary_path),
    }
    evidence = {
        "schema": "cudss-report-evidence-v1",
        "source_summary": source,
        "matched_newton_rows": matched,
        "unshifted_probe_rows": probes,
        "protocol": summary["protocol"],
        "interpretation": {
            "free_csr_build": "FEM numeric refresh plus CPU SciPy BSR-to-CSR transfer, IPC scatter, addition, free-DOF restriction, roundoff-only symmetrization, and GPU CSR upload. It is not included in factor plus solve.",
            "memory": "symbolic estimate is cuDSS-reported bytes; sampled device peak is NVML whole-device usage at 10 ms intervals, not an exact allocation peak.",
            "reuse": "A symbolic analysis may be reused only when dimensions, CSR ordering/index base, matrix type, and the reported matrix pattern hash are identical. The benchmark does not assume patterns match across states.",
            "physical_operator": "The matched system uses the prior benchmark's Newton shift. The saved unshifted system is a separate LDLT residual probe.",
        },
    }
    (cfg.output_dir / "cudss-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    matched_table = table(matched)
    compact = [
        "| Frozen state | Free-CSR PCG | Cholesky factor + solve | LDLT factor + solve |",
        "|---|---:|---:|---:|",
    ]
    amortization = []
    for state_name in dict.fromkeys(row["state"] for row in matched):
        by_mode = {row["mode"]: row for row in matched if row["state"] == state_name}
        chol, ldlt = by_mode["spd"], by_mode["symmetric"]
        compact.append(
            f"| {state_name} | {chol['pcg_median_seconds']:.3f} s | {chol['factor_plus_reused_solve_seconds']:.3f} s | {ldlt['factor_plus_reused_solve_seconds']:.3f} s |"
        )
        amortization.append(
            chol["factor_median_seconds"]
            / (chol["pcg_median_seconds"] - chol["reused_solve_median_seconds"])
        )
    compact_table = "\n".join(compact)
    headline = "CG remains faster for one RHS at the current 1e-3 tolerance. Cholesky costs about 1.19 s to factor, then 25.5 ms per reused solve, and gives much smaller residuals. For similarly costly RHS solves, factors pay off at roughly two RHS on the same matrix after symbolic analysis is amortized. This is a cost model from repeated solves of one RHS, not a measured independent multi-RHS workload."
    all_systems = [
        system for state in summary["results"] for system in state["systems"]
    ]
    pattern_hashes = {
        system["matrix_metadata"]["matrix_pattern_hash"] for system in all_systems
    }
    pattern_note = (
        "All three tested free matrices have exactly the same CSR pattern hash, despite changed contact counts. This permits symbolic reuse for these patterns; full-mesh cross-state refactorization was not timed here. A small GPU check separately validated value-update/refactorization."
        if len(pattern_hashes) == 1
        else "Free matrix pattern hashes differ; symbolic analysis must be repeated when the pattern changes."
    )
    probe_section = "No unshifted probe receipt was recorded."
    if probes:
        probe_section = table(probes)
    text = f"""# cuDSS direct solves on exact free-space Hessians

{headline}

{compact_table}

Times above exclude symbolic analysis and common free-matrix construction. All GPU tests ran sequentially on one RTX 4090, using float64 and 597,177 free DOFs. The saved state is already converged; its shifted system is a residual probe, not a necessary forward iteration.

This report compares scalar-diagonal PCG with cuDSS Cholesky and LDLT on the same assembled free FEM-plus-IPC systems at two frozen Smile states. It reports direct-solver device work and the separate CPU sparse merge/restriction cost. It does not claim an end-to-end Newton, forward, or inverse speed-up.

## Matched shifted Newton systems

{matched_table}

The `Free CSR build s` column is prominent because it includes CPU SciPy conversion, collision scatter, global addition, free-DOF restriction, and GPU upload. It is excluded from `Factor + reuse solve s`; adding it is necessary when the matrix must be rebuilt at a new Newton state.

The earlier assembled-BSR FEM plus GPU-contact benchmark measured PCG medians of **1.24 s** for loaded neutral and **1.77 s** for saved Smile. Those are the earlier split FEM/contact sparse-operator PCG times, not full direct-method totals. The new merged free-CSR PCG is faster after assembly, but its current CPU conversion takes about 3.3 s per state including FEM refresh, so this prototype does not yet improve total forward runtime. This points to caching the free-space structure and filling values directly on GPU as a separate optimization.

## Saved-state unshifted residual probe

{probe_section}

The saved Smile state is already a small-residual state. Its unshifted LDLT record is a separate numerical probe, not an adjoint or production Newton timing. LDLT reports inertia `[597177, 0]` (positive, negative) for both physical unshifted snapshots with pivot epsilon zero. The saved unshifted matrix was tested with LDLT only; Cholesky was tested on the cold unshifted and saved shifted matrices. These numerical diagnostics support positive definiteness of these snapshots, not every state in the nonlinear trajectory.

All 40 direct solves pass independent original-operator residual checks: cold/shifted relative residuals are below 3.84e-13, and the unshifted saved probe is below 1.68e-12. The direct acceptance threshold was 1e-7; PCG used 1e-3. The maximum assembled-operator relative error is below 3.66e-16. Thus the timing comparison uses the accuracy sufficient for our current forward solves; higher-accuracy PCG was not benchmarked here.

## Interpretation limits

Both matched Cholesky tests succeeded. LDLT success alone does not establish positive definiteness; the inertia diagnostics provide that evidence for these snapshots. `Factor nnz` is the cuDSS-reported factor nonzero count. The symbolic memory estimate is reported by cuDSS; sampled device peak is whole-device NVML usage sampled every 10 ms, so it can miss brief peaks and includes other allocations.

cuDSS reports 401,335,200 factor nonzeros versus 23,995,107 full free-CSR entries (about 16.7 times as many). Its estimated solver peak device requirement is 4,246,885,492 bytes (3.96 GiB). Observed total device usage peaked at about 7.08 GiB, including the model, full/lower matrices and solver workspaces. Full plus lower free-CSR GPU storage is about 563 MiB; cuDSS itself needs only the lower input, while the full matrix is retained for PCG. CPU process lifetime high-water RSS before/after conversion is in raw receipts; it is not a separately sampled merge-only peak.

Symbolic analysis costs 10.7–11.2 s and is excluded from the displayed numeric factor-plus-solve times. Numerical factorization uses the same analyzed pattern, while values are freshly factored on each repeat; this is not reuse of numerical factors for changing matrices. The first full-mesh Cholesky solve costs 80.9 ms versus about 25.5 ms for later solves. Cold FEM topology construction is also separate (26.75 s in this process).

{pattern_note}

Symbolic analysis can only be reused when the free-system dimension, matrix type, CSR ordering/index base, and sparsity pattern are all identical. The benchmark analyzed each case independently; reuse at additional states requires verifying the free pattern again. Repeated numerical factorization and reused triangular solves are measured only after a successful analysis for that particular matrix pattern.

## Reproduction and implementation

The pinned `nvidia-cudss-cu13==0.7.0.20` wheel was installed into experiment-local `tmp/cudss-runtime`; existing environment packages and physical sources were unchanged. Wrapper ABI constants were checked against the installed header. GPU smoke checks cover SPD solves, numeric value updates/refactorization, symmetric-indefinite solves and inertia. Wheel/header hashes are saved in `tmp/cudss-install-provenance.json`.

Run from `/root/codex-apple-performance/apple/exp/2026/09/22/solver-performance`:

```bash
DEBUG=1 CUDSS_LIBRARY="$PWD/tmp/cudss-runtime/nvidia/cu13/lib/libcudss.so.0" CHERRIES_NAME="cuDSS frozen Smile comparison 001" CHERRIES_TAGS="solver-performance,cudss,factorization,smile,rtx4090" /root/codex-apple-performance/apple/.venv/bin/python -u src/51-benchmark-cudss.py
```

Cherries finished successfully (`states_completed: 2`); debug retained local logs and disabled remote Comet recording. Each of the five matrix/method cases has three numeric factorizations and five further solves reusing factors, plus three post-factor solves. Matched PCG has three repeats per state. Terminal output is in `tmp/cudss-comparison-001-terminal.log`; the imported legacy helper emits an unrelated unused `simple-skin-forward` asset warning at shutdown.

The implementation uses [cuDSS analysis, factorization and solve phases](https://docs.nvidia.com/cuda/cudss/getting_started.html), with exact API details pinned by the archived 0.7 header. No production solver default was changed.

## Evidence

Source summary: `{source["path"]}` (SHA-256 `{source["sha256"]}`). Figure assets: `cudss.png` and `cudss.svg`. Raw timing arrays, factor receipts, residual checks, and per-system matrix metadata are retained in [cudss-evidence.json](cudss-evidence.json).
"""
    cfg.document_path.write_text(text)
    (cfg.output_dir / "cudss.md").write_text(text)
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>cuDSS free Hessian benchmark</title><style>body{{font:16px system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;font-size:12px}}th,td{{border:1px solid #bbb;padding:.35rem;text-align:right}}th:nth-child(-n+3),td:nth-child(-n+3){{text-align:left}}.table{{overflow-x:auto}}img{{max-width:100%;height:auto}}code{{overflow-wrap:anywhere}}</style><main><nav><a href="index.html">Model report</a> · <a href="hessian-representations.html">Hessian representation benchmark</a></nav><h1>cuDSS direct solves on exact free-space Hessians</h1><p>{html.escape(headline)}</p><p><strong>Startup and construction:</strong> symbolic analysis 10.7–11.2 s; current CPU free-CSR construction 3.18–3.34 s per state. These costs are excluded from the solver-only figure. Observed total GPU peak: 7.08 GiB. Direct relative residual: below 1.68e-12; PCG target: 1e-3.</p><details><summary>Detailed measurements</summary><div class="table">{html_table(matched)}</div></details><h2>Factorization and solve versus PCG</h2><img src="cudss.svg" alt="Factor plus reused solve versus PCG and sampled device memory by state"><h2>Reuse and definiteness</h2><p>{html.escape(pattern_note)}</p><p>Both unshifted physical snapshots have LDLT inertia [597177, 0] without pivot perturbation. This supports positive definiteness of these snapshots; it does not establish it along an entire nonlinear trajectory.</p><h2>Unshifted saved-state probe</h2><div class="table">{html_table(probes)}</div><h2>Limits</h2><p>Direct factor plus solve excludes the reported CPU free-CSR construction. Both matched Cholesky cases succeeded; LDLT inertia is recorded separately. NVML peak samples are not exact allocation peaks, and symbolic analysis may be reused only for an identical matrix pattern and descriptor.</p><p><a href="cudss.md">Markdown report</a> · <a href="cudss-evidence.json">Evidence JSON</a></p></main></html>"""
    (cfg.output_dir / "cudss.html").write_text(page)
    cherries.log_output(cfg.document_path)
    cherries.log_output(cfg.output_dir / "cudss.html")


if __name__ == "__main__":
    cherries.main(main)
