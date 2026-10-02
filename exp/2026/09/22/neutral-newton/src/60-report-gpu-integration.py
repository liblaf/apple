# ruff: noqa: E402, PLR0915, RUF001
"""Report saved matched GPU profiles and full neutral forwards; run no solves."""

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

ARMS = (
    (
        "matrix_free",
        "Reference",
        "profile-reference-002",
        "forward-reference-200-001",
        "#8898a6",
    ),
    (
        "gpu_contact",
        "GPU contact",
        "profile-gpu-contact-002",
        "forward-contact-200-001",
        "#197a69",
    ),
    (
        "gpu_sparse",
        "GPU sparse",
        "profile-gpu-sparse-001",
        "forward-sparse-200-001",
        "#8967a8",
    ),
)
REPEAT = "forward-reference-200-002"


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/gpu-integration-review-003"
    report: Path = GROUP / "docs/13-gpu-integration.md"


def load_run(name: str) -> tuple[dict, dict, list[dict], np.ndarray]:
    directory = GROUP / "data" / name
    summary = json.loads((directory / "summary.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    trace = [
        json.loads(line)
        for line in (directory / "trace.jsonl").read_text().splitlines()
    ]
    assert summary["accepted_steps"] == 200
    assert sha256(directory / "terminal.npz") == summary["terminal"]["sha256"]
    assert sha256(directory / "protocol.json") == summary["protocol"]["sha256"]
    assert [row["iteration"] for row in trace] == list(range(201))
    assert np.all(np.diff([row["energy"] for row in trace]) < 0)
    with np.load(directory / "terminal.npz", allow_pickle=False) as saved:
        u = saved["displacement_m"].copy()
    return summary, protocol, trace, u


def branches(row: dict) -> tuple:
    newton = row["newton"]
    return (
        newton["linear"]["steps"],
        tuple(r["reason"] for r in newton["regularization_retries"]),
        newton["backtracks"],
        newton["line_search_trials"],
    )


def trajectory_difference(
    trace: list, u: np.ndarray, reference: list, reference_u: np.ndarray
) -> dict:
    result = {
        "max_terminal_coordinate_difference_m": float(np.max(np.abs(u - reference_u))),
        "terminal_node_rms_difference_m": float(
            np.sqrt(np.mean(np.sum((u - reference_u) ** 2, axis=1)))
        ),
        "branch_mismatch_iterations": [
            a["iteration"]
            for a, b in zip(trace[1:], reference[1:], strict=True)
            if branches(a) != branches(b)
        ],
    }
    for key in ("energy", "grad_norm"):
        actual, expected = (
            np.array([r[key] for r in trace]),
            np.array([r[key] for r in reference]),
        )
        result[key] = {
            "maximum_absolute_difference": float(np.max(np.abs(actual - expected))),
            "maximum_relative_difference": float(
                np.max(np.abs(actual - expected) / np.abs(expected))
            ),
            "terminal_absolute_difference": float(abs(actual[-1] - expected[-1])),
        }
    for key in ("alpha", "ccd"):
        result[key] = {
            "maximum_absolute_difference": max(
                abs(a["newton"][key] - b["newton"][key])
                for a, b in zip(trace[1:], reference[1:], strict=True)
            ),
            "differing_iterations": [
                a["iteration"]
                for a, b in zip(trace[1:], reference[1:], strict=True)
                if a["newton"][key] != b["newton"][key]
            ],
        }
    return result


def check_protocol(actual: dict, expected: dict) -> None:
    for key in (
        "materials",
        "mandible",
        "collision",
        "diagonal_policy",
        "initial_geometry",
    ):
        assert actual[key] == expected[key], key
    # Derived norms can differ by one floating-point ULP; solver inputs must match exactly.
    a, b = actual["solver"].copy(), expected["solver"].copy()
    for key in ("initial_grad_norm", "effective_grad_threshold"):
        np.testing.assert_allclose(a.pop(key), b.pop(key), rtol=1e-14, atol=0)
    assert a == b


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, cfg.output_dir / Path(__file__).name)
    _, reference_protocol, reference_trace, reference_u = load_run(ARMS[0][3])
    results, histories = [], []
    for backend, label, profile_name, forward_name, color in ARMS:
        profile_dir, forward_dir = (
            GROUP / "data" / profile_name,
            GROUP / "data" / forward_name,
        )
        profile = json.loads((profile_dir / "summary.json").read_text())
        summary, protocol, trace, u = load_run(forward_name)
        assert profile["success"]
        assert protocol["config"]["hessian_backend"] == backend
        check_protocol(protocol, reference_protocol)
        histories.append(trace)
        runs = [r for w in profile["windows"] for r in w["runs"]]
        original_residuals = [
            v
            for r in runs
            if r["mode"] == "validation"
            for v in r["reference_true_relative_residuals"]
        ]
        assert len(original_residuals) == 30
        assert max(original_residuals) <= 1e-3
        operator_errors = [v for r in runs for v in r["operator_relative_errors"]]
        assert not operator_errors or max(operator_errors) < 1e-10
        difference = trajectory_difference(trace, u, reference_trace, reference_u)
        assert not difference["branch_mismatch_iterations"]
        first_setup = summary["hessian_backend"]["first_setup_seconds"]
        results.append(
            {
                "backend": backend,
                "label": label,
                "color": color,
                "forward_dir": forward_name,
                "profile_dir": profile_name,
                "profile_summary_sha256": sha256(profile_dir / "summary.json"),
                "forward_summary_sha256": sha256(forward_dir / "summary.json"),
                "loop_seconds": summary["forward_seconds"],
                "backend_first_setup_seconds": first_setup,
                "loop_plus_backend_setup_seconds": summary["forward_seconds"]
                + first_setup,
                "profile_medians": [
                    w["baseline_median_seconds"] for w in profile["windows"]
                ],
                "profile_min": [w["baseline_min_seconds"] for w in profile["windows"]],
                "profile_max": [w["baseline_max_seconds"] for w in profile["windows"]],
                "maximum_sampled_backend_bytes": max(
                    r["backend_before_timing"]["persistent_bytes"] for r in runs
                ),
                "late_synchronized_timing": next(
                    r
                    for r in profile["windows"][-1]["runs"]
                    if r["mode"] == "synchronized"
                )["timing_totals"],
                "trajectory_difference": difference,
                "maximum_operator_relative_error": max(operator_errors)
                if operator_errors
                else 0.0,
                "maximum_original_operator_relative_residual": max(original_residuals),
                "final_gradient": summary["final_grad_norm"],
                "final_energy": summary["final_energy"],
                "inversions": summary["geometry"]["inverted_tetrahedra"],
                "collision_feasible": summary["collision"]["state_feasible"],
                "hessian_backend": summary["hessian_backend"],
                "operation_counts": summary["operation_counts"],
                "accepted_pcg_steps": sum(
                    r["newton"]["linear"]["steps"] for r in trace[1:]
                ),
                "shift_rejections": sum(
                    len(r["newton"]["regularization_retries"]) for r in trace[1:]
                ),
            }
        )
    repeat_summary, repeat_protocol, repeat_trace, repeat_u = load_run(REPEAT)
    check_protocol(repeat_protocol, reference_protocol)
    repeat = {
        "forward_dir": REPEAT,
        "loop_seconds": repeat_summary["forward_seconds"],
        "summary_sha256": sha256(GROUP / "data" / REPEAT / "summary.json"),
        "trajectory_difference": trajectory_difference(
            repeat_trace, repeat_u, reference_trace, reference_u
        ),
    }
    historical_summary, _, historical_trace, historical_u = load_run("forward-003")
    historical = trajectory_difference(
        historical_trace, historical_u, reference_trace, reference_u
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(left=0.075, right=0.97, top=0.77, bottom=0.23, wspace=0.32)
    fig.suptitle(
        "Neutral solver · integrated GPU backends",
        x=0.075,
        y=0.97,
        ha="left",
        fontsize=19,
        weight="bold",
    )
    fig.text(
        0.075,
        0.86,
        "Same model and Newton policy · matched replay plus full 200-step runs",
        color="#526575",
    )
    x = np.arange(3)
    for index, row in enumerate(results):
        med = np.array(row["profile_medians"]) * 100
        low, high = (
            np.array(row["profile_min"]) * 100,
            np.array(row["profile_max"]) * 100,
        )
        axes[0].bar(
            x + (index - 1) * 0.24,
            med,
            0.23,
            color=row["color"],
            label=row["label"],
            yerr=np.vstack([med - low, high - med]),
            capsize=3,
        )
    axes[0].set(
        xticks=x,
        xticklabels=["0–10", "100–110", "190–200"],
        ylabel="Milliseconds / Newton step",
        title="Matched replay · median of 3",
    )
    axes[0].set_ylim(0, 330)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", ncols=3)
    for i, row in enumerate(results):
        axes[1].barh(i, row["loop_seconds"], color=row["color"], height=0.6)
        axes[1].barh(
            i,
            row["backend_first_setup_seconds"],
            left=row["loop_seconds"],
            color=row["color"],
            alpha=0.35,
            hatch="///",
            height=0.6,
        )
        axes[1].text(
            row["loop_plus_backend_setup_seconds"] + 1,
            i,
            f"{row['loop_plus_backend_setup_seconds']:.1f} s",
            va="center",
            fontsize=10,
        )
    axes[1].set(
        yticks=x,
        yticklabels=[r["label"] for r in results],
        xlabel="Seconds · loop + backend construction",
        title="Actual 200-step entrypoint",
    )
    axes[1].set_xlim(
        0, max(r["loop_plus_backend_setup_seconds"] for r in results) * 1.2
    )
    axes[1].invert_yaxis()
    fig.text(
        0.075,
        0.075,
        "Whiskers: min/max. Hatched bar: sparse construction before the loop. Model loading and preflight checks are excluded.",
        fontsize=9,
        color="#526575",
    )
    fig.text(
        0.075,
        0.035,
        "Sequential RTX 4090 measurements on a shared desktop. Contact upload during preflight was not separately timed.",
        fontsize=9,
        color="#526575",
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(cfg.output_dir / f"gpu-backends.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for row, trace in zip(results, histories, strict=True):
        axes[0].plot(
            [r["iteration"] for r in trace],
            [r["energy"] for r in trace],
            label=row["label"],
            color=row["color"],
            alpha=0.8,
        )
        axes[1].semilogy(
            [r["iteration"] for r in trace],
            [r["grad_norm"] for r in trace],
            label=row["label"],
            color=row["color"],
            alpha=0.8,
        )
    for ax, key in zip(axes, ("energy", "grad_norm"), strict=True):
        ax.plot(
            [r["iteration"] for r in historical_trace],
            [r[key] for r in historical_trace],
            color="#c48e57",
            ls="--",
            lw=1,
            label="Earlier 100 + 100 run",
        )
    axes[0].set(
        xlabel="Accepted Newton iteration",
        ylabel="Energy [MPa m³]",
        title="Fresh backend curves visually overlap",
    )
    axes[1].set(
        xlabel="Accepted Newton iteration",
        ylabel="Gradient norm [MPa m²]",
        title="Historical path differs after iteration 44",
    )
    axes[1].axhline(
        reference_protocol["solver"]["effective_grad_threshold"],
        ls=":",
        color="#a55b39",
        label="Stopping threshold",
    )
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9)
    fig.savefig(cfg.output_dir / "trajectory-validation.png", dpi=180)
    plt.close(fig)
    write_json(
        cfg.output_dir / "summary.json",
        {
            "success": True,
            "default_backend": "gpu_contact",
            "results": results,
            "reference_repeat": repeat,
            "historical_trajectory_difference": historical,
            "forward_solves_in_report": 0,
        },
    )
    rows = [
        f"| {r['label']} | {r['loop_seconds']:.3f} | {r['backend_first_setup_seconds']:.3f} | {r['loop_plus_backend_setup_seconds']:.3f} | {r['maximum_sampled_backend_bytes'] / 2**20:.1f} |"
        for r in results
    ]
    replay_rows = [
        f"| {window} | "
        + " | ".join(f"{r['profile_medians'][i] / 10:.4f}" for r in results)
        + " |"
        for i, window in enumerate(("0–10", "100–110", "190–200"))
    ]
    work_rows = [
        f"| {r['label']} | {r['operation_counts']['hess_prod']} | {r['accepted_pcg_steps']} | {r['shift_rejections']} |"
        for r in results
    ]
    contact_speedup = results[0]["loop_seconds"] / results[1]["loop_seconds"]
    window_speedups = np.array(results[0]["profile_medians"]) / np.array(
        results[1]["profile_medians"]
    )
    trajectory_rows = [
        f"| {r['label']} | {r['trajectory_difference']['max_terminal_coordinate_difference_m']:.3e} | {r['trajectory_difference']['energy']['maximum_relative_difference']:.3e} | {r['trajectory_difference']['grad_norm']['maximum_relative_difference']:.3e} |"
        for r in results[1:]
    ]
    component_rows = [
        f"| {label} | "
        + " | ".join(
            "—"
            if key not in r["late_synchronized_timing"]
            else f"{r['late_synchronized_timing'][key]['inclusive_seconds']:.3f}"
            for r in results
        )
        + " |"
        for key, label in (
            ("pcg", "PCG, including HVPs"),
            ("fem/hess_prod", "FEM HVPs, within PCG"),
            ("backend/prepare", "Sparse matrix refresh/cache checks"),
            ("ipc/broad_phase", "IPC broad phase, across CCD/updates"),
            ("ipc/ccd", "CCD kernel"),
        )
    ]
    lines = [
        "# Integrated GPU neutral solver",
        "",
        "The GPU implementations are reusable library backends wired into the neutral Newton-CG driver. **GPU contact is the default** because it is faster for this workload. Full GPU sparse assembly is integrated and validated as `gpu_sparse`; its assembly and startup costs do not pay off for these short PCG solves.",
        "",
        f"![Backend comparison](../data/{cfg.output_dir.name}/gpu-backends.png)",
        "",
        "## Actual main-solver runs",
        "",
        "All three backends ran 200 accepted iterations from the same repaired constitutive reference through `10-run-neutral.py`. Materials, constraints, energy, gradient, exact diagonal, shift schedule, PCG tolerance, line search and CCD use the same policy. Each run saved a complete trace, protocol, checkpoints, source archive, and geometry/contact diagnostics.",
        "",
        "| Backend | Forward loop s | Sparse startup s | Sum s | Sampled backend MiB |",
        "| --- | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        f"GPU contact's observed full loop was **{contact_speedup:.3f}× faster**, a **{100 * (1 - 1 / contact_speedup):.1f}% time reduction**, than the fresh reference. The reference repeat took **{repeat['loop_seconds']:.3f} s**. These are sequential observations, not statistical throughput estimates. Sparse construction happens during preflight and is added back explicitly. Contact upload during preflight was not separately timed. Common model construction, source/input validation, coordinate-HVP checks, terminal geometry auditing and Comet shutdown are excluded.",
        "",
        "Memory values are the largest sampled live backend buffers at profile preflight, not total/peak GPU memory. Contact buffers are released after updates, so the terminal receipt reports zero resident contact bytes; that is not its working memory usage. Sparse buffers persist across states.",
        "",
        "| Backend | HVP calls | Accepted PCG iterations | Rejected shift attempts |",
        "| --- | ---: | ---: | ---: |",
        *work_rows,
        "",
        "The 200-step discrete branch records agree, but floating-point residual recomputations can change HVP counts. Accepted-PCG totals exclude work in rejected systems. Use the replay windows below for a controlled timing comparison.",
        "",
        "## Matched-window profile",
        "",
        "Each ten-step window has three uninstrumented repetitions, a separate residual-validation pass, a cProfile pass, and a CUDA-synchronized hierarchical pass. Initialization is excluded. The late window refreshes its first state's values inside timing. Initial sparse topology/union construction is measured separately from per-step work.",
        "",
        "| Window | Reference s/step | GPU contact s/step | GPU sparse s/step |",
        "| --- | ---: | ---: | ---: |",
        *replay_rows,
        "",
        f"GPU contact is **{min(window_speedups):.3f}–{max(window_speedups):.3f}× faster** across these matched windows. Use these fresh controls instead of the older profile recorded while another GPU compute job was active. Instrumented timings change GPU/CPU overlap and are for attribution. Contact HVPs reuse GPU CSR products instead of transferring every Krylov vector to CPU; IPC contact construction and CCD remain on CPU.",
        "",
        "Late-window synchronized attribution (seconds for ten steps; nested rows must not be added):",
        "",
        "| Component | Reference | GPU contact | GPU sparse |",
        "| --- | ---: | ---: | ---: |",
        *component_rows,
        "",
        "With GPU contact, PCG accounts for 59.7% of the 2.424 s instrumented window and FEM Hessian products alone account for 49.6%. IPC broad phase takes 18.9% and the CCD kernel 6.0%. Contact-product time falls from 0.336 s to 0.105 s. Full sparse assembly reduces PCG to 0.296 s but spends 1.353 s refreshing matrices/checking the cache, 51.5% of its 2.628 s window. This explains why faster sparse products do not make this solve faster overall.",
        "",
        "## Numerical validation",
        "",
        f"Maximum checked GPU-HVP relative error against the original matrix-free FEM + CPU contact operator: **{max(r['maximum_operator_relative_error'] for r in results):.3e}**, below the declared 1e-10 gate. All 30 accepted PCG solutions per backend in the validation windows were independently checked using that original operator. Worst relative residual: **{max(r['maximum_original_operator_relative_residual'] for r in results):.6g}**, below 1e-3. Each replay also passed its saved endpoint energy/gradient tolerance (rtol 1e-8) and recorded PCG-iteration, shift-retry and line-search checks.",
        "",
        "Full trajectories are not bitwise identical. All 200 accepted PCG counts, retry reasons/counts, backtracks and trial counts match the fresh reference. Continuous CCD/step-length factors differ slightly and are recorded separately. The following differences are diagnostics, not an assertion of exact trajectory equality:",
        "",
        "| Backend vs fresh reference | Max terminal coordinate difference m | Max energy relative difference | Max gradient-norm relative difference |",
        "| --- | ---: | ---: | ---: |",
        *trajectory_rows,
        "",
        f"The same-code reference repeat differs by at most **{repeat['trajectory_difference']['max_terminal_coordinate_difference_m']:.3e} m** at the terminal coordinates, with **{len(repeat['trajectory_difference']['branch_mismatch_iterations'])}** differing discrete branch records. Derived initial norm/threshold values are checked to rtol 1e-14 (the sparse run differs by one ULP); solver input settings match exactly.",
        "",
        f"Maximum absolute CCD/step-length factor difference is **{max(r['trajectory_difference']['ccd']['maximum_absolute_difference'] for r in results):.3e}** for the GPU backends versus **{repeat['trajectory_difference']['ccd']['maximum_absolute_difference']:.3e}** for the reference repeat. These differences occur at iteration 14 for the GPU runs and iteration 28 for the repeated reference.",
        "",
        f"The earlier `forward-003` 100 + 100 continuation follows a different path from the fresh reference: its first discrete difference is iteration **{historical['branch_mismatch_iterations'][0]}**. At state 44 the fresh run rejects one additional shifted system for nonpositive curvature. Tiny prior numerical differences therefore change a discrete decision and amplify. This occurs in the original backend too, so it is not evidence of a GPU-backend policy change. The exact source of floating-point drift has not been isolated. Historical endpoint inversions: {historical_summary['geometry']['inverted_tetrahedra']}; fresh endpoint inversions: {results[0]['inversions']}.",
        "",
        f"All fresh endpoints remain **nonconverged**, with gradient about **{results[0]['final_gradient']:.3e}**, versus the **{reference_protocol['solver']['effective_grad_threshold']:.3e}** threshold, **{results[0]['inversions']} inverted tetrahedra**, and feasible soft-rigid contact. These results validate the backend comparison, not a physically valid neutral equilibrium. Selected-neutral assets remain unchanged.",
        "",
        f"![Trajectory comparison](../data/{cfg.output_dir.name}/trajectory-validation.png)",
        "",
        "## Integration and cache contract",
        "",
        "`src/liblaf/apple/forward/hessian` contains the promoted FEM BSR assembler, cached free-DOF GPU CSR assembler, GPU contact cache and `HessianProblem`. The adapter belongs to one solve and does not patch a model class or change implicit-adjoint execution. Numerical values refresh after state updates; the unshifted physical matrix is reused during PCG and shift retries. Energy, gradient, exact diagonal and CCD delegate to the original problem. Explicit backends fail visibly rather than falling back.",
        "",
        "The promoted code checks free-DOF/topology mutation tokens. A global topology-cache regression was fixed so a new operator cannot inherit stale slots after Torch connectivity changes. Warp connectivity has no cheap mutation version and must remain immutable for the model lifetime. Unsupported FEM types fail explicitly. This validation covers the neutral model's StableNeoHookeanStress/StableNeoHookeanMembrane registry. The sparse profile predates the topology-cache guard fix; the full sparse run uses the final code. The fix does not change numerical assembly for fixed topology.",
        "",
        "## Reproduction and evidence",
        "",
        "Run from this experiment group with `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYVISTA_OFF_SCREEN=true` and descriptive `CHERRIES_NAME`/`CHERRIES_TAGS`. Every run requires a fresh output directory.",
        "",
        "```sh",
        "python src/10-run-neutral.py \\",
        "  --max-steps 200 --hessian-backend gpu_contact --output-dir data/new-forward",
        "python src/40-profile-forward.py \\",
        "  --hessian-backend gpu_contact --output-dir data/new-profile",
        "```",
        "",
        "Replace `gpu_contact` with `matrix_free` or `gpu_sparse` for the other arms. Forwards intentionally exit 1 after saving 200 steps when the force/geometry gate fails; profiles exit 0. All runs must complete Cherries shutdown. The early `profile-gpu-contact-001.stdout.log` records an enum spelling rejection before solver work; the successful contact profile is `002`.",
        "",
        "Regression validation: `pytest -q tests/forward tests/solvers --no-cov` passed all 12 tests, including GPU contact cache behavior, sparse mapping/topology contracts, static forward mechanics and existing solver tests. Targeted Ruff checks passed. The first report attempt stopped because it treated continuous CCD factors as discrete branch labels; the corrected report preserves those differences as diagnostics and changes no solver output.",
        "",
        *[
            f"- {r['label']}: [forward receipt](../data/{r['forward_dir']}/summary.json), [profile receipt](../data/{r['profile_dir']}/summary.json), [Python profile](../data/{r['profile_dir']}/window-190/cprofile.txt), [timing tree](../data/{r['profile_dir']}/window-190/synchronized.json)."
            for r in results
        ],
        f"- [Reference repeat](../data/{REPEAT}/summary.json).",
        "",
        "- [Independent audit](../data/gpu-backend-audit.json), [test log](../data/gpu-integration-tests.log), [source validation](../data/gpu-integration-code-validation.json). Library changes after benchmarking are formatting only, with identical parsed ASTs against the archived full-run sources.",
        "",
        f"[Machine-readable comparison](../data/{cfg.output_dir.name}/summary.json) retains hashes and numerical differences. This report/plot script performs zero forward solves. Source session: ‘What’s the main performance bottleneck? Can we improve performance? Related works include VBD, GIPC, etc.’ (`01a0c4dd-55f2-7270-959b-3852514f93e1`); original prototypes remain intact.",
    ]
    cfg.report.write_text("\n".join(lines) + "\n")
    cherries.log_output(cfg.output_dir)
    cherries.log_output(cfg.report)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
