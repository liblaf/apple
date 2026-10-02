# ruff: noqa: E402, RUF001
"""Compare saved legacy PNCG and Newton runs without another forward solve."""

from __future__ import annotations

import json
import re
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
    run_dir: Path = GROUP / "data/forward-pncg-200-001"
    comparison_dir: Path = GROUP / "data/forward-contact-200-001"
    output_dir: Path = GROUP / "data/pncg-review-002"
    report: Path = GROUP / "docs/14-legacy-pncg.md"


def load(directory: Path) -> tuple[dict, dict, list]:
    summary = json.loads((directory / "summary.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    assert sha256(directory / "protocol.json") == summary["protocol"]["sha256"]
    assert sha256(directory / "terminal.npz") == summary["terminal"]["sha256"]
    trace = [
        json.loads(line)
        for line in (directory / "trace.jsonl").read_text().splitlines()
    ]
    assert [r["iteration"] for r in trace] == list(range(summary["accepted_steps"] + 1))
    assert all(r["accepted"] for r in trace)
    assert np.all(np.diff([r["energy"] for r in trace]) <= 0)
    np.testing.assert_allclose(
        trace[-1]["grad_norm"], summary["final_grad_norm"], rtol=1e-10, atol=1e-15
    )
    np.testing.assert_allclose(
        trace[-1]["energy"], summary["final_energy"], rtol=1e-10, atol=1e-20
    )
    return summary, protocol, trace


def main(cfg: Config) -> None:
    p, protocol, pt = load(cfg.run_dir)
    n, reference, nt = load(cfg.comparison_dir)
    with (
        np.load(cfg.run_dir / "initial.npz") as a,
        np.load(cfg.comparison_dir / "initial.npz") as b,
    ):
        assert np.array_equal(a["displacement_m"], b["displacement_m"])
    for key in ("materials", "mandible", "collision", "diagonal_policy"):
        assert protocol[key] == reference[key]
    np.testing.assert_allclose(
        p["effective_grad_threshold"], n["effective_grad_threshold"], rtol=1e-14, atol=0
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, cfg.output_dir / Path(__file__).name)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), layout="constrained")
    fig.suptitle(
        "Neutral reference forward · legacy PNCG vs Newton-CG",
        fontsize=17,
        weight="bold",
    )
    for label, trace, color in (
        ("Legacy PNCG", pt, "#aa623b"),
        ("Newton-CG", nt, "#197a69"),
    ):
        for row, xkey in enumerate(("iteration", "elapsed_seconds")):
            x = [r[xkey] for r in trace]
            axes[row, 0].plot(
                x, [r["energy"] for r in trace], color=color, label=label, lw=1.8
            )
            axes[row, 1].semilogy(
                x, [r["grad_norm"] for r in trace], color=color, label=label, lw=1.8
            )
    for row, xlabel in enumerate(
        ("Accepted solver iteration", "Recorded loop elapsed time [s]")
    ):
        axes[row, 0].set(xlabel=xlabel, ylabel="Energy [MPa m³]")
        axes[row, 1].set(xlabel=xlabel, ylabel="Gradient norm [MPa m²]")
        axes[row, 1].axhline(
            p["effective_grad_threshold"],
            ls="--",
            lw=1,
            color="#718090",
            label="Force threshold",
        )
        for ax in axes[row]:
            ax.grid(alpha=0.16)
            ax.legend(frameon=False, fontsize=9)
        axes[row, 1].legend(
            frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 0.08)
        )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(cfg.output_dir / f"energy-gradient-comparison.{suffix}", dpi=180)
    plt.close(fig)
    stdout_path = cfg.run_dir.with_suffix(".stdout.log")
    output = stdout_path.read_text()
    comet = re.search(r"https://www\.comet\.com/liblaf/apple/[a-f0-9]+", output)
    assert comet is not None
    run_url = comet.group()
    result = {
        "schema": "legacy-pncg-neutral-review-v1",
        "success": True,
        "forward_solves": 0,
        "run_summary": {
            "path": str(cfg.run_dir / "summary.json"),
            "sha256": sha256(cfg.run_dir / "summary.json"),
        },
        "comparison_summary": {
            "path": str(cfg.comparison_dir / "summary.json"),
            "sha256": sha256(cfg.comparison_dir / "summary.json"),
        },
        "same_initial_coordinates": True,
        "same_materials_and_physical_derivative_policy": True,
        "force_threshold": p["effective_grad_threshold"],
        "pncg_gradient_over_threshold": p["final_grad_norm"]
        / p["effective_grad_threshold"],
        "pncg_backtracks": sum(r["pncg"]["backtracks"] for r in pt[1:]),
        "pncg_line_search_trials": sum(r["pncg"]["line_search_trials"] for r in pt[1:]),
        "maximum_accepted_coordinate_step_m": max(
            (r["pncg"]["max_coordinate_displacement_m"] for r in pt[1:]), default=0
        ),
        "comet_url": run_url,
    }
    assert result["maximum_accepted_coordinate_step_m"] <= protocol["solver"][
        "line_search"
    ]["max_coordinate_displacement_m"] * (1 + 1e-12)
    write_json(cfg.output_dir / "summary.json", result)
    rows = [
        f"| {label} | {s['accepted_steps']} | {s['forward_seconds']:.3f} | {s['final_grad_norm']:.6e} | {s['final_energy']:.6e} | {s['geometry']['inverted_tetrahedra']} |"
        for label, s in (("Legacy PNCG", p), ("Newton-CG, GPU contact", n))
    ]
    cfg.report.write_text(
        "\n".join(
            [
                "# Legacy PNCG neutral forward",
                "",
                f"Ran one legacy PNCG forward from the same repaired constitutive-reference seed as the current Newton comparison. It saved **{p['accepted_steps']} accepted steps** in **{p['forward_seconds']:.3f} s**. Status: **{p['status']}**.",
                "",
                "## Result",
                "",
                "| Solver | Accepted steps | Loop seconds | Final gradient norm | Final energy | Inverted tetrahedra |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                *rows,
                "",
                f"PNCG's final gradient is **{result['pncg_gradient_over_threshold']:.2f}×** the shared stopping threshold **{p['effective_grad_threshold']:.6e}**. Soft-rigid contact feasible: **{p['collision']['state_feasible']}**. Minimum det(F): **{p['geometry']['detF_min']:.6g}**. Termination: `{p['optimizer_result']}`; receipt: `{p['failure']}`.",
                "",
                f"![Energy and gradient comparison](../data/{cfg.output_dir.name}/energy-gradient-comparison.png)",
                "",
                "Iteration budgets are equal but work per iteration differs. The time-axis curves provide additional context; these single runs do not establish a universal solver ranking or equal-accuracy speedup. Timers include trace logging/checkpoint writes and exclude model construction, preflight and terminal geometry audits. Energy uses MPa m³ and gradient norm uses MPa m²; multiply by 1e6 for J and N.",
                "",
                "## Solver and model",
                "",
                "This uses the old strict Dai-Kou+ PNCG implementation through `AcceptedForcePncg`, not the experimental adaptive-PNCG method. Its legacy eye-neutral settings are Armijo 0.25, 0.5 mm coordinate cap, up to 60 halvings (61 trials), initial damping 0.001 with bounds 1e-6–1, conjugacy restart interval 200, and CCD safety 0.95. PNCG computes one physical Hessian quadratic form per step and applies adaptive damping to its search model. It does not invoke inner PCG or Newton shift escalation.",
                "",
                "Materials, zero activation, prescribed neutral jaw pose, constitutive reference, repaired initial coordinates, exact physical derivative policy, and bone/eye collision geometry match the current Newton run. The current exact diagonal and GPU contact HVP implementation are retained. Thus this is the legacy solver on the current model, not a replay of the older clamped-derivative backend. Newton and PNCG retain their respective line-search/damping/CCD settings, explicitly recorded in the protocols.",
                "",
                "The force criterion is evaluated at accepted states, with both primary and secondary thresholds set to max(1e-8, 1e-3 times the original initial gradient). Strict line-search failure stops the solve; the runner restores the last accepted state before saving the terminal artifact. No alternative solver or recovery solve is invoked. Existing selected-neutral assets and the Newton driver are unchanged.",
                "",
                f"Accepted PNCG line-search trials: **{result['pncg_line_search_trials']}**, including **{result['pncg_backtracks']}** backtracks. Largest accepted coordinate displacement: **{result['maximum_accepted_coordinate_step_m'] * 1000:.4f} mm**. Operation counts: `{p['operation_counts']}`.",
                "",
                f"The objective recorded **{p['rejected_contact_trials']} contact-infeasible trial rejections**. Accepted energy decreases throughout, but the force norm has pronounced spikes and plateaus. Fewer inverted elements than Newton does not imply better volume geometry: PNCG's minimum det(F) is {p['geometry']['detF_min']:.3f}, versus {n['geometry']['detF_min']:.3f} for Newton. Neither endpoint passes the force/geometry gate.",
                "",
                "## Command and evidence",
                "",
                "Working directory: `exp/2026/09/22/neutral-newton`.",
                "",
                "```sh",
                "CHERRIES_NAME='Neutral reference legacy PNCG 200' \\",
                "CHERRIES_TAGS='neutral,pncg,legacy-solver,200-iterations' \\",
                "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\",
                "PYVISTA_OFF_SCREEN=true \\",
                "python src/70-run-pncg-neutral.py",
                "```",
                "",
                f"[Comet run]({run_url}); [terminal receipt](../data/{cfg.run_dir.name}/summary.json); [protocol](../data/{cfg.run_dir.name}/protocol.json); [trace](../data/{cfg.run_dir.name}/trace.jsonl); [terminal displacement](../data/{cfg.run_dir.name}/terminal.npz); [source/input provenance](../data/{cfg.run_dir.name}/provenance.json); [stdout](../data/{cfg.run_dir.name}.stdout.log).",
                "",
                "The initial launch stopped at configuration validation before any model/solver work because of a dynamic-module annotation lookup. The module registration was corrected and the Config smoke check passed; that log is retained as `forward-pncg-config-failure.stdout.log`. The actual forward stores all source hashes and the unchanged runtime binding. A nonzero forward exit denotes the saved force/geometry gate failure, not missing artifacts. The report script runs no physics solves.",
                "",
            ]
        )
        + "\n"
    )
    cherries.log_output(cfg.output_dir)
    cherries.log_output(cfg.report)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
