# ruff: noqa: E402, PLR0915, RUF001
"""Render measured performance and write a report from saved profile receipts."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
JOINT = GROUP.parents[4] / "exp/2026/09/21/joint-activation-material-mandible"
sys.path.insert(0, str(JOINT / "src"))
from joint_common import ProfileJoint, sha256, write_json


class Config(cherries.BaseConfig):
    profile_dir: Path = GROUP / "data/profile-001"
    run_dir: Path = GROUP / "data/forward-003"
    output_dir: Path = GROUP / "data/profile-plots-001"
    report: Path = GROUP / "docs/12-performance.md"


def main(cfg: Config) -> None:
    profile = json.loads((cfg.profile_dir / "summary.json").read_text())
    summary = json.loads((cfg.run_dir / "summary.json").read_text())
    trace = [
        json.loads(line)
        for line in (cfg.run_dir / "trace.jsonl").read_text().splitlines()
    ]
    assert profile["success"]
    assert sha256(cfg.run_dir / "trace.jsonl") == profile["production_trace"]["sha256"]
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, cfg.output_dir / Path(__file__).name)
    segments = []
    for start, stop, wall in (
        (0, 100, summary["forward_seconds"] - summary["segment_forward_seconds"]),
        (100, 200, summary["segment_forward_seconds"]),
    ):
        rows = trace[start + 1 : stop + 1]
        accepted = sum(r["newton"]["linear"]["seconds"] for r in rows)
        rejected = sum(
            t["seconds"] for r in rows for t in r["newton"]["regularization_retries"]
        )
        segments.append(
            {
                "start": start,
                "stop": stop,
                "wall_seconds": wall,
                "accepted_pcg_seconds": accepted,
                "rejected_pcg_seconds": rejected,
                "other_seconds": wall - accepted - rejected,
                "accepted_pcg_iterations": sum(
                    r["newton"]["linear"]["steps"] for r in rows
                ),
                "rejected_systems": sum(
                    len(r["newton"]["regularization_retries"]) for r in rows
                ),
            }
        )
    windows = profile["windows"]
    synced = [
        next(r for r in w["runs"] if r["mode"] == "synchronized") for w in windows
    ]
    totals = {
        key: sum(r[key] for r in segments)
        for key in (
            "wall_seconds",
            "accepted_pcg_seconds",
            "rejected_pcg_seconds",
            "other_seconds",
            "accepted_pcg_iterations",
            "rejected_systems",
        )
    }
    cg_pct = (
        100
        * (totals["accepted_pcg_seconds"] + totals["rejected_pcg_seconds"])
        / totals["wall_seconds"]
    )
    rejected_pct = 100 * totals["rejected_pcg_seconds"] / totals["wall_seconds"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.15]}
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.30, top=0.76, wspace=0.42)
    fig.suptitle(
        "Neutral Newton-CG · performance profile",
        x=0.06,
        y=0.96,
        ha="left",
        fontsize=19,
        weight="bold",
    )
    fig.text(
        0.06,
        0.87,
        f"200 iterations: {totals['wall_seconds']:.2f} s   ·   linear solves: {cg_pct:.1f}%   ·   rejected systems: {rejected_pct:.1f}%",
        fontsize=12,
        color="#425268",
    )
    bottom = np.zeros(2)
    for key, label, color in (
        ("accepted_pcg_seconds", "Accepted linear systems", "#25769A"),
        ("rejected_pcg_seconds", "Rejected linear systems", "#D07B45"),
        ("other_seconds", "Other work + reporting", "#B5BEC8"),
    ):
        value = np.array([r[key] for r in segments])
        axes[0].bar([0, 1], value, bottom=bottom, label=label, color=color, width=0.58)
        bottom += value
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["0–100", "100–200"],
        ylabel="Seconds",
        title="Original run receipts",
    )
    for i, v in enumerate(bottom):
        axes[0].text(i, v + 0.7, f"{v:.2f}", ha="center", fontsize=10)
    axes[0].set_ylim(0, max(bottom) * 1.18)
    axes[0].legend(
        loc="upper left", bbox_to_anchor=(-0.06, -0.19), frameon=False, fontsize=9
    )
    med = np.array([w["baseline_median_seconds"] * 100 for w in windows])
    low = np.array([w["baseline_min_seconds"] * 100 for w in windows])
    high = np.array([w["baseline_max_seconds"] * 100 for w in windows])
    axes[1].bar(
        range(3),
        med,
        color="#25769A",
        width=0.58,
        yerr=np.vstack((med - low, high - med)),
        capsize=5,
    )
    for i, v in enumerate(med):
        axes[1].text(i, high[i] + 9, f"{v:.0f}", ha="center")
    labels = ["0–10", "100–110", "190–200"]
    axes[1].set(
        xticks=range(3),
        xticklabels=labels,
        ylabel="Milliseconds / Newton step",
        title="Replay without instrumentation",
        ylim=(0, max(high) * 1.18),
    )
    axes[1].text(
        0.5,
        -0.24,
        "Median of 3 repeats; whiskers = min/max",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=9,
    )
    components = []
    for run in synced:
        t = run["timing_totals"]
        total = t["window"]["inclusive_seconds"]
        fem = t["fem/hess_prod"]["inclusive_seconds"]
        contact = t["contact/hess_prod"]["inclusive_seconds"]
        pcg = t["pcg"]["inclusive_seconds"]
        components.append([fem, contact, pcg - fem - contact, total - pcg])
    fractions = np.array(components)
    fractions = fractions / fractions.sum(axis=1)[:, None] * 100
    left = np.zeros(3)
    for j, (label, color) in enumerate(
        (
            ("FEM Hessian products", "#25769A"),
            ("Contact products + transfers", "#8C67B1"),
            ("Remaining PCG work", "#8CB6BC"),
            ("Outside PCG", "#B5BEC8"),
        )
    ):
        axes[2].barh(
            range(3), fractions[:, j], left=left, color=color, label=label, height=0.58
        )
        left += fractions[:, j]
    axes[2].set(
        yticks=range(3),
        yticklabels=labels,
        xlabel="% of synchronized replay",
        title="Attribution with synchronization",
        xlim=(0, 100),
    )
    axes[2].invert_yaxis()
    axes[2].legend(
        loc="upper left", bbox_to_anchor=(-0.05, -0.19), frameon=False, fontsize=9
    )
    fig.text(
        0.06,
        0.035,
        "RTX 4090 shared with another compute job. Replay excludes setup, checkpoint I/O and logging. Nested synchronization changes overlap.",
        fontsize=9,
        color="#596777",
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(cfg.output_dir / f"performance.{suffix}", dpi=180)
    plt.close(fig)
    write_json(
        cfg.output_dir / "analysis.json",
        {
            "production_segments": segments,
            "production_totals": totals,
            "pcg_percent": cg_pct,
            "rejected_percent": rejected_pct,
            "synchronized_components_seconds": components,
            "profile_sha256": sha256(cfg.profile_dir / "summary.json"),
            "forward_solves": 0,
        },
    )
    table = []
    for w, r in zip(windows, synced, strict=True):
        t = r["timing_totals"]
        table.append(
            f"| {w['start_iteration']}–{w['start_iteration'] + 10} | {w['baseline_median_seconds'] / 10:.4f} ({w['baseline_min_seconds'] / 10:.4f}–{w['baseline_max_seconds'] / 10:.4f}) | {r['operation_counts']['hess_prod']} | {r['accepted_pcg_iterations']} | {r['rejected_linear_systems']} | {100 * t['pcg']['inclusive_seconds'] / t['window']['inclusive_seconds']:.1f}% |"
        )
    detail = []
    for label, key in (
        ("All PCG attempts", "pcg"),
        ("FEM Hessian products", "fem/hess_prod"),
        ("Contact HVP including transfers", "contact/hess_prod"),
        ("CPU contact SpMV (inside contact HVP)", "cpu/contact_spmv"),
        ("Contact Hessian assembly", "ipc/hessian_assembly"),
        ("Broad phase (CCD and updates)", "ipc/broad_phase"),
        ("Native CCD", "ipc/ccd"),
    ):
        detail.append(
            "| "
            + label
            + " | "
            + " | ".join(
                f"{r['timing_totals'][key]['inclusive_seconds']:.3f}" for r in synced
            )
            + " |"
        )
    errors = [r["gradient_replay_absolute_error"] for w in windows for r in w["runs"]]
    report = f"""# Neutral Newton forward performance

PCG is the largest measured cost: linear-system attempts occupy **{cg_pct:.1f}%** of the actual 200-iteration forward loop. Rejected systems alone take **{totals["rejected_pcg_seconds"]:.2f} s ({rejected_pct:.1f}%)**. Both GPU FEM products and the CPU contact path with device transfers contribute; CPU sparse multiplication alone is a smaller component.

![Performance measurements](../data/{cfg.output_dir.name}/performance.png)

## Actual trajectory

The original 0–100 segment took {segments[0]["wall_seconds"]:.6f} s; continuation 100–200 took {segments[1]["wall_seconds"]:.6f} s, totaling **{totals["wall_seconds"]:.6f} s**. The stored receipts attribute {totals["accepted_pcg_seconds"]:.6f} s to accepted linear systems and {totals["rejected_pcg_seconds"]:.6f} s to rejected attempts. These attempt timers include tiny setup/descent checks surrounding PCG. The remaining {totals["other_seconds"]:.6f} s includes diagonal assembly, energy/gradient evaluation, collision work, updates, logging and checkpoint I/O. Setup and shutdown are excluded from that production loop.

There were {totals["accepted_pcg_iterations"]} accepted-system CG iterations and {totals["rejected_systems"]} rejected linear systems. All 200 steps accepted their first Armijo trial: backtracking is not the cause of the measured cost. Resetting the shift and retrying is the requested policy; this profiling experiment leaves it unchanged.

## Matched replay

Three ten-step windows were replayed from the actual saved checkpoints, each with three baseline repetitions, one separate cProfile pass, and one separate synchronized hierarchical pass. One warmup step per window and all state reconstruction/preflight checks are excluded. This is a diagnostic replay, not a continuation beyond 200 or a new adopted neutral. Every replay matches its saved initial/final energy and gradient (relative tolerance 1e-8), accepted PCG iteration counts, shift-retry counts, and line-search trial counts. Largest absolute endpoint gradient discrepancy across measured passes: {max(errors):.3e} MPa·m².

| Window | Baseline seconds / step: median (range) | HVP calls | Accepted CG iterations | Rejected systems | Synchronized PCG share |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

Later windows need more HVPs. From 100–110 to 190–200, accepted CG iterations rise only 270→280, while total HVPs rise 384→470. After subtracting accepted CG iterations and their ten true-residual checks, remaining HVPs rise 104→180; these include rejected systems and any extra true-residual checks. The greater amount of linear-solver work is consistent with the late-window slowdown.

## Where PCG time goes

These are **inclusive seconds under forced synchronization**, not additive rows or uninstrumented percentages. CPU SpMV is inside contact HVP; FEM/contact HVP are inside PCG. The complete timing JSON also records additive exclusive times.

| Scope | 0–10 | 100–110 | 190–200 |
| --- | ---: | ---: | ---: |
{chr(10).join(detail)}

The contact HVP gathers a GPU direction, calls `.numpy(force=True)`, multiplies the cached SciPy sparse Hessian on CPU, converts the result back to a GPU tensor, and scatters it into the result. Its time therefore combines transfers, synchronization, indexing, Python/native glue and sparse multiplication. The cProfile outputs identify `.numpy()` and scalar tensor conversions as hot boundaries, but their waiting time cannot be interpreted as pure transfer bandwidth cost. FEM timing is completed Warp adapter work, not kernel-only GPU timing.

Contact Hessians are cached within each accepted state. Assembly happens once after a state change; the ten-step windows have 9, 9 and 10 assemblies because production preflight primed the start-state Hessian at 0 and 100. Assembly must not be multiplied by the number of HVPs. Broad-phase construction and CCD remain measurable even when no Armijo backtracking occurs.

## Performance work suggested by this profile

1. Benchmark a GPU-resident contact Hessian matvec that uploads the exact physical Hessian once per accepted state, preserving the current matrix, diagonal and residual checks. This targets repeated host/device round trips; the attainable speedup is unmeasured.
2. Inspect FEM HVP kernel and launch costs with a GPU kernel profiler before selecting a kernel optimization. This wall-clock profile establishes the boundary cost but not the kernel-level cause.
3. Reduce rejected-system work only in a separate solver-policy comparison. Shift reuse or a different preconditioner changes the user's declared method and has not been evaluated here.

No optimizer, material, collision or tolerance changes were applied. At iteration 200 the endpoint remains nonconverged with 62 inverted tetrahedra, so performance measurements do not establish a valid neutral.

## Reproduction and limits

Run from `the experiment directory`:

```bash
CHERRIES_NAME='Neutral Newton matched window performance profile' \\
CHERRIES_TAGS='neutral,newton-cg,profiling,matched-replay' \\
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYVISTA_OFF_SCREEN=true \\
python src/40-profile-forward.py \\
  > data/profile-001.stdout.log 2>&1
```

The profiler completed successfully, exit 0, including Cherries shutdown. [Comet profile](https://www.comet.com/liblaf/apple/3b988c4eb27a4f2fbb90fd48d1fc8c41). The report/plot script is `src/50-report-profile.py` and executes zero forward solves.

Runtime: {profile["runtime"]["gpu"]}, PyTorch {profile["runtime"]["torch"]}, IPC threads {profile["runtime"]["ipc_threads"]}. Verified numerical source hashes match the saved production run; the current working tree has unrelated changes. Material/model setup took {profile["model_setup_seconds_excluded"]:.3f} s, excluded. This setup metric excludes Python imports and CUDA initialization. Input lineage and replay checkpoints are hash-bound in [profile receipts](../data/{cfg.profile_dir.name}/summary.json); the executed instrumentation source is copied beside the receipts.

The GPU was shared with another compute job throughout observed before/after snapshots; these are workload timings, not isolated hardware benchmarks. Baseline trials are sequential, and changing shared load can affect both variation and comparisons. Synchronization overhead (and load variation) changes replay wall time: baseline medians are {", ".join(f"{w['baseline_median_seconds']:.3f}" for w in windows)} s; synchronized passes are {", ".join(f"{r['wall_seconds']:.3f}" for r in synced)} s. No statistical speedup claim is made.

Raw results: [summary](../data/{cfg.profile_dir.name}/summary.json), [window 190 timing tree](../data/{cfg.profile_dir.name}/window-190/synchronized.json), [window 190 Python hotspots](../data/{cfg.profile_dir.name}/window-190/cprofile.txt), and binary `.prof` files beside each text dump. [Plot SVG](../data/{cfg.output_dir.name}/performance.svg), [plot PDF](../data/{cfg.output_dir.name}/performance.pdf), [200-iteration energy and gradient](../data/convergence-plots-200/energy-gradient.png).
"""
    cfg.report.write_text(report)
    cherries.log_output(cfg.output_dir)
    cherries.log_output(cfg.report)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
