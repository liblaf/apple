# ruff: noqa: PERF401
"""Render measured CPU versus cached-GPU free-CSR assembly receipts."""

from __future__ import annotations

import hashlib
import html
import json
import shlex
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent


class Config(cherries.BaseConfig):
    summary_path: Path = GROUP / "data/gpu-free-assembly-001/summary.json"
    output_dir: Path = GROUP / "data/gpu-free-assembly-report-001"
    document_path: Path = GROUP / "docs/54-gpu-free-assembly.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}g}"


def mib(value: float | None) -> str:
    return "—" if value is None else f"{float(value) / 2**20:.1f}"


def median(values: list[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    return (
        values[middle]
        if len(values) % 2
        else 0.5 * (values[middle - 1] + values[middle])
    )


def seconds(value: Any) -> float | None:
    """Accept a benchmark scalar, stats receipt, or sample list."""
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, dict):
        if "median" in value:
            return float(value["median"])
        if "seconds" in value:
            return float(value["seconds"])
    if isinstance(value, list):
        return median(
            [
                float(item["seconds"] if isinstance(item, dict) else item)
                for item in value
            ]
        )
    message = f"unsupported timing receipt: {type(value).__name__}"
    raise TypeError(message)


def max_error(receipt: Any) -> float | None:
    if not receipt:
        return None
    if isinstance(receipt, (float, int)):
        return float(receipt)
    values = []
    for item in receipt:
        if isinstance(item, dict):
            for key in (
                "relative_l2",
                "relative_error",
                "reference_true_relative_residual",
                "error",
            ):
                if key in item:
                    values.append(float(item[key]))
                    break
        else:
            values.append(float(item))
    return max(values) if values else None


def _first_setup(receipt: dict[str, Any]) -> float | None:
    """The first constructor/changed-state setup is retained, not averaged."""
    return seconds(receipt.get("first_state_setup_seconds"))


def _pcg(receipt: dict[str, Any]) -> float | None:
    return seconds(receipt.get("pcg"))


def _direct(receipt: dict[str, Any]) -> dict[str, Any]:
    direct = receipt.get("cudss_spd") or {}
    factors = direct.get("factorizations", [])
    solves = direct.get("solves", [])
    factor = seconds(factors)
    solve = seconds(solves)
    residual = max_error(solves)
    status = direct.get("status")
    if status is None:
        if not direct:
            status = "not measured"
        elif direct.get("success", False):
            status = "success"
        else:
            status = direct.get("reason", "failed")
    return {
        "status": str(status),
        "analysis_seconds": seconds(direct.get("analysis_seconds")),
        "factor_seconds": factor,
        "solve_seconds": solve,
        "residual": residual,
    }


def summarize(summary: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for state in summary["results"]:
        old, gpu = state["old"], state["gpu"]
        old_prepare = seconds(old.get("setup_refresh_seconds"))
        gpu_prepare = seconds(gpu.get("setup_refresh_seconds"))
        if old_prepare is None or gpu_prepare is None:
            message = "each route requires repeated setup_refresh_seconds samples"
            raise ValueError(message)
        old_pcg, gpu_pcg = _pcg(old), _pcg(gpu)
        old_total = (
            old_prepare + old_pcg
            if old_prepare is not None and old_pcg is not None
            else None
        )
        gpu_total = (
            gpu_prepare + gpu_pcg
            if gpu_prepare is not None and gpu_pcg is not None
            else None
        )
        gpu_minus_old = (
            gpu_total - old_total
            if old_total is not None and gpu_total is not None
            else None
        )
        gpu_first_setup = _first_setup(gpu)
        amortization = (
            (gpu_first_setup - gpu_prepare) / (old_total - gpu_total)
            if state["state"] == "cold_loaded_neutral"
            and gpu_first_setup is not None
            and old_total is not None
            and gpu_total is not None
            and gpu_first_setup > gpu_prepare
            and old_total > gpu_total
            else None
        )
        metadata = gpu.get("metadata", {})
        rows.append(
            {
                "state": state["state"],
                "force_l2": float(state["force_l2"]),
                "contact_state_seconds": seconds(state.get("contact_state_seconds")),
                "contact_cpu_assembly_seconds": seconds(
                    state.get("contact_cpu_assembly_seconds")
                ),
                "fem_preparation_seconds": seconds(
                    state.get("fem_preparation_seconds")
                ),
                "fem_refresh_seconds": seconds(metadata.get("fem_setup_seconds")),
                "old_prepare_seconds": old_prepare,
                "old_pcg_seconds": old_pcg,
                "old_total_seconds": old_total,
                "gpu_prepare_seconds": gpu_prepare,
                "gpu_pcg_seconds": gpu_pcg,
                "gpu_total_seconds": gpu_total,
                "gpu_minus_old_total_seconds": gpu_minus_old,
                "gpu_first_state_setup_seconds": gpu_first_setup,
                "constructor_amortization_refresh_solve_cycles": amortization,
                "old_accuracy": max_error(old.get("matrix_relative_errors")),
                "gpu_accuracy": max_error(gpu.get("matrix_relative_errors")),
                "gpu_lower_accuracy": max_error(gpu.get("lower_relative_errors")),
                "old_bytes": old.get("persistent_bytes"),
                "gpu_bytes": gpu.get("persistent_bytes"),
                "symbolic_seconds": seconds(metadata.get("symbolic_seconds")),
                "contact_upload_seconds": seconds(
                    metadata.get("contact_upload_seconds")
                ),
                "gpu_numeric_seconds": seconds(metadata.get("gpu_numeric_seconds")),
                "metadata_numeric_setup_seconds": seconds(
                    metadata.get("numeric_setup_seconds")
                ),
                "symbolic_cache_hit": metadata.get(
                    "symbolic_cache_hit", gpu.get("reused_analysis")
                ),
                "contact_pattern_changed": metadata.get("contact_pattern_changed"),
                "pattern_hash": metadata.get("pattern_hash")
                or metadata.get("matrix_pattern_hash"),
                "direct": _direct(gpu),
            }
        )
    if [row["state"] for row in rows] != ["cold_loaded_neutral", "saved_smile"]:
        message = "results must preserve chronological cold_loaded_neutral then saved_smile order"
        raise ValueError(message)
    return rows, summary["transition_back_to_cold"]


def report_table(rows: list[dict[str, Any]]) -> str:
    headings = (
        "State",
        "Old refresh median s",
        "Old PCG median s",
        "Old refresh + solve s",
        "GPU refresh median s",
        "GPU PCG median s",
        "GPU refresh + solve s",
        "GPU minus old s",
        "GPU first setup s",
        "Cold startup-excess amortization cycles",
        "Matrix error",
        "Lower error",
        "GPU persistent MiB",
        "SPD status",
        "SPD residual",
    )
    lines = [
        "| " + " | ".join(headings) + " |",
        "|" + "|".join(["---"] * len(headings)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["state"],
                    number(row["old_prepare_seconds"]),
                    number(row["old_pcg_seconds"]),
                    number(row["old_total_seconds"]),
                    number(row["gpu_prepare_seconds"]),
                    number(row["gpu_pcg_seconds"]),
                    number(row["gpu_total_seconds"]),
                    number(row["gpu_minus_old_total_seconds"]),
                    number(row["gpu_first_state_setup_seconds"]),
                    number(row["constructor_amortization_refresh_solve_cycles"]),
                    "—"
                    if row["gpu_accuracy"] is None
                    else f"{row['gpu_accuracy']:.2e}",
                    "—"
                    if row["gpu_lower_accuracy"] is None
                    else f"{row['gpu_lower_accuracy']:.2e}",
                    mib(row["gpu_bytes"]),
                    row["direct"]["status"],
                    "—"
                    if row["direct"]["residual"] is None
                    else f"{row['direct']['residual']:.2e}",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def html_table(rows: list[dict[str, Any]]) -> str:
    lines = report_table(rows).splitlines()
    headings = lines[0].strip("|").split(" | ")
    body = []
    for line in lines[2:]:
        body.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(value)}</td>"
                for value in line.strip("|").split(" | ")
            )
            + "</tr>"
        )
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(value)}</th>" for value in headings)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def headline(rows: list[dict[str, Any]]) -> str:
    preparation = [
        row["old_prepare_seconds"] / row["gpu_prepare_seconds"] for row in rows
    ]
    totals = [row["old_total_seconds"] / row["gpu_total_seconds"] for row in rows]
    error = max(row["gpu_accuracy"] for row in rows)
    lower = max(row["gpu_lower_accuracy"] for row in rows)
    return (
        f"Across the two frozen states, cached GPU numeric refresh is {min(preparation):.1f}-{max(preparation):.1f}x faster than CPU SciPy free-CSR rebuild; "
        f"refresh plus PCG is {min(totals):.2f}-{max(totals):.2f}x faster. Maximum matrix and lower-storage relative errors are {error:.2e} and {lower:.2e}."
    )


def startup_memory(rows: list[dict[str, Any]]) -> str:
    cold = rows[0]
    excess = cold["gpu_first_state_setup_seconds"] - cold["gpu_prepare_seconds"]
    increment = cold["gpu_bytes"] - cold["old_bytes"]
    return (
        f"The cold GPU-route constructor is {cold['gpu_first_state_setup_seconds']:.2f} s; its excess over the warm refresh is {excess:.2f} s. "
        f"Both arms also share {cold['fem_preparation_seconds']:.2f} s of initial FEM topology/assembly preparation, excluded from these free-CSR timings. "
        f"Persistent buffers are {mib(cold['old_bytes'])} MiB for CPU-assembled free CSR and {mib(cold['gpu_bytes'])} MiB for the GPU route, an incremental {mib(increment)} MiB. "
        "The GPU route stores values and cached static index maps; IPC Hessian evaluation remains CPU-owned, while its sparse contact contribution is remapped and uploaded for the GPU numeric assembly."
    )


def transition_summary(transition: dict[str, Any]) -> str:
    metadata = transition["metadata"]
    matrix = max_error(transition.get("matrix_relative_errors"))
    lower = max_error(transition.get("lower_relative_errors"))
    return (
        f"Return to cold took {float(transition['setup_seconds']):.3f} s. Contact remap took {float(metadata['contact_remap_seconds']):.3g} s; "
        f"contact pattern changed={metadata['contact_pattern_changed']}, union reused={metadata['union_reused_for_contact']}, "
        f"symbolic cache hit={metadata['symbolic_cache_hit']}. Matrix/lower errors were {matrix:.2e}/{lower:.2e}; pattern hash `{metadata['pattern_hash']}`."
    )


def plot(rows: list[dict[str, Any]], output: Path) -> None:
    labels = ["Loaded neutral\n(cold)", "Saved Smile\n(residual probe)"]
    x = list(range(len(rows)))
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    width = 0.34
    for offset, prefix, color in (
        (-width / 2, "old", "#777777"),
        (width / 2, "gpu", "#1b9e77"),
    ):
        preparation = [row[f"{prefix}_prepare_seconds"] for row in rows]
        solve = [row[f"{prefix}_pcg_seconds"] for row in rows]
        positions = [value + offset for value in x]
        axes[0].bar(
            positions,
            preparation,
            width,
            color=color,
            alpha=0.55,
            label=f"{prefix.upper()} preparation",
        )
        axes[0].bar(
            positions,
            solve,
            width,
            bottom=preparation,
            color=color,
            label=f"{prefix.upper()} PCG",
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("seconds")
    axes[0].set_title("One free-CSR preparation plus PCG")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    for offset, key, label, color in (
        (-0.22, "gpu_first_state_setup_seconds", "GPU first setup", "#7570b3"),
        (0.0, "old_prepare_seconds", "old median refresh", "#777777"),
        (0.22, "gpu_prepare_seconds", "GPU median refresh", "#1b9e77"),
    ):
        values = [row[key] if row[key] is not None else float("nan") for row in rows]
        axes[1].bar(
            [value + offset for value in x], values, 0.22, label=label, color=color
        )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("seconds")
    axes[1].set_title("First setup versus repeated numeric refresh")
    axes[1].legend(fontsize=7)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Exact frozen free-space assembly: CPU CSR rebuild versus cached GPU numeric refresh",
        fontsize=12,
    )
    for suffix in ("png", "svg"):
        figure.savefig(
            output / f"gpu-assembly.{suffix}", dpi=180 if suffix == "png" else None
        )
    plt.close(figure)


def main(cfg: Config) -> None:
    if not cfg.summary_path.is_file():
        raise FileNotFoundError(cfg.summary_path)
    if cfg.output_dir.exists():
        raise FileExistsError(cfg.output_dir)
    summary = json.loads(cfg.summary_path.read_text())
    if summary.get("schema") != "gpu-free-assembly-v1":
        message = "requires gpu-free-assembly-v1 benchmark summary"
        raise ValueError(message)
    rows, transition = summarize(summary)
    cfg.output_dir.mkdir(parents=True)
    plot(rows, cfg.output_dir)
    source = {
        "path": str(cfg.summary_path.resolve()),
        "sha256": sha256(cfg.summary_path),
    }
    evidence = {
        "schema": "gpu-free-assembly-report-evidence-v1",
        "source_summary": source,
        "rows": rows,
        "transition_back_to_cold": transition,
        "protocol": summary["protocol"],
        "interpretation": {
            "old": "CPU SciPy free-CSR rebuild and upload, timed separately from PCG.",
            "gpu": "cached topology/symbolic representation with GPU numeric refresh; initial constructor and later refresh are distinct receipts.",
            "total": "setup plus one PCG solve at a frozen state, excluding contact-state construction and contact CPU Hessian assembly shown separately in evidence.",
            "direct": "lower-triangle validation and cuDSS SPD receipt verify the assembled operator; a successful factorization does not establish end-to-end forward or inverse performance.",
        },
    }
    (cfg.output_dir / "gpu-assembly-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    table = report_table(rows)
    result_headline = headline(rows)
    startup_text = startup_memory(rows)
    return_text = transition_summary(transition)
    config = summary["protocol"]["config"]
    reproduce = (
        'DEBUG=1 CUDSS_LIBRARY="$PWD/tmp/cudss-runtime/nvidia/cu13/lib/libcudss.so.0" '
        'CHERRIES_NAME="GPU free Hessian assembly" CHERRIES_TAGS="solver-performance,gpu-assembly,smile,rtx4090" '
        "/root/codex-apple-performance/apple/.venv/bin/python -u src/53-benchmark-gpu-free-assembly.py "
        f"--output-dir {shlex.quote(str(config['output_dir']))}"
    )
    text = f"""# GPU free-space Hessian assembly

{result_headline}

This frozen-state benchmark compares the previous CPU SciPy free-CSR rebuild with a GPU-resident numeric assembly. Both use the same exact FEM plus IPC Hessian restricted to free DOFs. The GPU path caches static FEM/contact index maps and updates numeric values on GPU; collision Hessian evaluation remains CPU-owned, then its exact sparse contribution is remapped and uploaded. It measures matrix preparation and one PCG solve, not a complete forward or inverse solve.

## Measured cold-to-saved sequence

{table}

The sequence is chronological: loaded neutral is constructed first, then saved Smile refreshes the cached representation. The primary comparison is median repeated numeric refresh plus median PCG. `GPU first setup s` separately retains the initial constructor or first changed-state setup before those repeats. Each arm has three refresh and three PCG samples, run sequentially on one GPU.

## What the totals include

`Old refresh + solve s` is the old CPU CSR rebuild/upload median refresh plus median PCG. `GPU refresh + solve s` is the cached GPU median refresh plus median PCG. The first setup is not hidden: it is retained separately. The cold startup-excess amortization uses only initial GPU setup minus its warm numeric refresh, divided by a positive same-state refresh-plus-solve saving. The second panel does not combine inclusive sub-timings. Contact-state construction and one-time process startup remain separate.

{startup_text}

The earlier [cuDSS comparison](cudss.html) used the CPU-built combined free CSR. The [Hessian representation comparison](hessian-representations.html) used split FEM/contact sparse operators. This run repeats the CPU baseline on the same GPU; it does not infer a full-method speed-up by combining timings from different runs.

## Accuracy, lower storage, and direct validation

Matrices and diagonal-inclusive lower storage are independently compared with the original matrix-free FEM plus CPU IPC operator. The SPD receipt is an additional direct lower-storage residual check. These independent checks validate the frozen linear operator; they do not validate a full forward trajectory.

{return_text}

Symbolic analysis may be reused only when the recorded matrix pattern and descriptor agree. Contact entries inside the cached structure only need new scatter destinations; off-pattern entries rebuild the union. The sparse union, sorting, and large index maps are constructed on CUDA; initial FEM/free-DOF mapping and small contact remapping still use CPU. The unchanged material-model FEM topology preparation is a separate shared startup cost. The GPU assembler is an experiment-local backend; the production forward/inverse defaults have not been changed.

## Reproduction

The receipt records source hashes and archives benchmark sources before execution; it also checks checkpoint and input hashes against the established baseline. Run from this experiment directory with local Cherries logging:

```bash
{reproduce}
```

The recorded output path was `{config["output_dir"]}`. Use a new output directory for another run; the harness refuses to overwrite receipts. `DEBUG=1` keeps Cherries local. The cuDSS path refers to the previously staged 0.7 runtime. Cherries completed both states; raw terminal output and source-copy verification are retained under `tmp/`.

## Evidence

Source summary: `{source["path"]}` (SHA-256 `{source["sha256"]}`). Assets: `gpu-assembly.png`, `gpu-assembly.svg`, and [gpu-assembly-evidence.json](gpu-assembly-evidence.json). Results are sequential single-GPU measurements; medians describe the recorded samples and do not estimate concurrent throughput.
"""
    cfg.document_path.write_text(text)
    (cfg.output_dir / "gpu-assembly.md").write_text(text)
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>GPU free-space Hessian assembly</title><style>body{{font:16px system-ui,sans-serif;max-width:1450px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;font-size:12px}}th,td{{border:1px solid #bbb;padding:.35rem;text-align:right}}th:nth-child(-n+2),td:nth-child(-n+2){{text-align:left}}.table{{overflow-x:auto}}img{{max-width:100%;height:auto}}code{{overflow-wrap:anywhere}}</style><main><nav><a href="cudss.html">cuDSS</a> · <a href="hessian-representations.html">Hessian representations</a></nav><h1>GPU free-space Hessian assembly</h1><p>{html.escape(result_headline)}</p><p>{html.escape(startup_text)}</p><div class="table">{html_table(rows)}</div><img src="gpu-assembly.svg" alt="Stacked preparation and PCG timing bars for CPU and cached GPU free-space assembly"><p>{html.escape(return_text)}</p><p>Matrix and lower-triangle checks use the independent original operator; cuDSS is an additional lower-storage residual check. These frozen-system timings do not establish a full forward or inverse speed-up.</p><p><a href="gpu-assembly.md">Markdown report</a> · <a href="gpu-assembly-evidence.json">Evidence JSON</a></p></main></html>"""
    (cfg.output_dir / "gpu-assembly.html").write_text(page)
    cherries.log_output(cfg.document_path)
    cherries.log_output(cfg.output_dir / "gpu-assembly.html")


if __name__ == "__main__":
    cherries.main(main)
