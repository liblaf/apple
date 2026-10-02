"""Compare measured V100 replay with the immutable archived RTX 4090 record."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

LABELS = {
    "current_cpu_contact": "Matrix-free FEM + CPU contact",
    "current_gpu_contact": "Matrix-free FEM + GPU contact",
    "cached_fem_gpu_contact": "Cached FEM + GPU contact",
    "sparse_fem_gpu_contact": "Assembled FEM + GPU contact",
}


def record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def compare(baseline: dict, candidate: dict) -> list[dict]:
    for key in ("checkpoint_sha256", "neutral_checkpoint_sha256", "seed"):
        assert baseline["protocol"][key] == candidate["protocol"][key], key
    for key in (
        "linear_rtol",
        "ipc_threads",
        "hvp_repeats",
        "cg_repeats",
        "setup_repeats",
    ):
        assert (
            baseline["protocol"]["config"][key] == candidate["protocol"]["config"][key]
        ), key
    rows = []
    original_states = {s["state"]: s for s in baseline["results"]}
    for state in candidate["results"]:
        old = original_states[state["state"]]
        for key in ("full_points", "free_dofs", "displacement_sha256", "shift"):
            assert old[key] == state[key], (state["state"], key, old[key], state[key])
        old_variants = {v["name"]: v for v in old["variants"]}
        for variant in state["variants"]:
            reference = old_variants[variant["name"]]
            row = {"state": state["state"], "variant": variant["name"]}
            for gpu, data in (("rtx4090", reference), ("v100", variant)):
                assert len(data["hvp_seconds"]["samples"]) == 30
                assert len(data["cg"]) == 3
                assert max(x["relative_l2"] for x in data["accuracy"]) < 1e-10
                assert (
                    max(x["reference_true_relative_residual"] for x in data["cg"])
                    <= 1.05e-3
                )
                row[gpu] = {
                    "hvp_ms": 1000 * statistics.median(data["hvp_seconds"]["samples"]),
                    "hvp_ms_range": [
                        1000 * min(data["hvp_seconds"]["samples"]),
                        1000 * max(data["hvp_seconds"]["samples"]),
                    ],
                    "cg_seconds": statistics.median(x["seconds"] for x in data["cg"]),
                    "cg_seconds_range": [
                        min(x["seconds"] for x in data["cg"]),
                        max(x["seconds"] for x in data["cg"]),
                    ],
                    "refresh_seconds": statistics.median(
                        data["numeric_setup_seconds"]["samples"]
                    ),
                    "cg_steps_range": [
                        min(x["steps"] for x in data["cg"]),
                        max(x["steps"] for x in data["cg"]),
                    ],
                    "cg_ms_per_hvp": 1000
                    * statistics.median(
                        x["seconds"] / x["hvp_calls"] for x in data["cg"]
                    ),
                    "max_true_relative_residual": max(
                        x["reference_true_relative_residual"] for x in data["cg"]
                    ),
                    "max_hvp_relative_error": max(
                        x["relative_l2"] for x in data["accuracy"]
                    ),
                    "extra_persistent_bytes": data["extra_persistent_bytes"],
                }
                row[gpu]["refresh_plus_cg_seconds"] = (
                    row[gpu]["refresh_seconds"] + row[gpu]["cg_seconds"]
                )
            row["speedup_v100"] = {
                metric: row["rtx4090"][metric] / row["v100"][metric]
                for metric in (
                    "hvp_ms",
                    "cg_seconds",
                    "refresh_plus_cg_seconds",
                    "cg_ms_per_hvp",
                )
            }
            rows.append(row)
    assert len(rows) == 8
    return rows


def plot(rows: list[dict], output: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), layout="constrained", sharey=True)
    labels = [
        f"{'Cold neutral' if r['state'] == 'cold_loaded_neutral' else 'Saved Smile'}\n{LABELS[r['variant']]}"
        for r in rows
    ]
    y = np.arange(len(rows))
    for ax, metric, title in zip(
        axes,
        ("hvp_ms", "refresh_plus_cg_seconds"),
        ("Hessian-vector product (ms)", "Numeric refresh + PCG (s)"),
        strict=True,
    ):
        for gpu, offset, color, label in (
            ("rtx4090", -0.18, "#5b7691", "RTX 4090 · archived CUDA 13"),
            ("v100", 0.18, "#258774", "V100 PCIe · CUDA 12"),
        ):
            bars = ax.barh(
                y + offset,
                [r[gpu][metric] for r in rows],
                height=0.33,
                color=color,
                label=label,
            )
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_title(title)
        ax.set_xlim(
            0, max(r[g][metric] for r in rows for g in ("rtx4090", "v100")) * 1.2
        )
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Full-face FP64 frozen-system benchmark · 597,177 free DOFs\nMedians; 30 HVP samples and 3 PCG repeats · lower is faster",
        fontsize=12,
    )
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    rows = compare(baseline, candidate)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "baseline": record(args.baseline),
        "candidate": record(args.candidate),
        "speedup_definition": "RTX 4090 seconds divided by V100 seconds; greater than one favors V100",
        "rows": rows,
    }
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = [
        "| State / implementation | 4090 HVP ms | V100 HVP ms | V100 HVP speedup | 4090 CG s | V100 CG s | V100 CG speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        a, b, speed = row["rtx4090"], row["v100"], row["speedup_v100"]
        table.append(
            f"| {row['state']} / {LABELS[row['variant']]} | {a['hvp_ms']:.3f} | {b['hvp_ms']:.3f} | {speed['hvp_ms']:.2f}x | {a['cg_seconds']:.3f} | {b['cg_seconds']:.3f} | {speed['cg_seconds']:.2f}x |"
        )
    (args.output / "table.md").write_text("\n".join(table) + "\n")
    plot(rows, args.output)
    print("\n".join(table))


if __name__ == "__main__":
    main()
