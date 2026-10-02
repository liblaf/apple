"""Render the completed cold hybrid-forward timing profile without rerunning it."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

EXPERIMENT = Path(__file__).resolve().parent.parent


class Config(cherries.BaseConfig):
    run_dir: Path = EXPERIMENT / "data/hybrid-first-profile-001"
    output_dir: Path = EXPERIMENT / "data/hybrid-first-profile-report-001"
    document_path: Path = EXPERIMENT / "docs/57-hybrid-profile.md"
    baseline_dir: Path | None = None
    run_log: Path | None = None


class ProfileReport(profiles.Profile):
    """Performance evidence profile that never invokes a Git plugin."""

    def init(self) -> core.Run:
        os.environ.setdefault("COMET_AUTO_LOG_GIT_METADATA", "false")
        os.environ.setdefault("COMET_AUTO_LOG_GIT_PATCH", "false")
        run = core.run
        run.plugins.register(
            plugins.Comet(run=run, disabled=os.environ.get("DEBUG") == "1")
        )
        run.plugins.register(plugins.Logging(run=run))
        run.plugins.register(plugins.Local(run=run))
        return run


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def run_receipt(log_path: Path | None) -> dict[str, str] | None:
    """Extract the command and Comet receipt from a completed Cherries log."""
    if log_path is None:
        return None
    text = log_path.read_text()
    command = re.findall(r"cherries/cmd\s+: (.+)", text)
    comet_url = re.findall(r"cherries/comet/url\s+: (https://\S+)", text)
    if not command or not comet_url:
        message = f"missing command or Comet receipt in {log_path}"
        raise AssertionError(message)
    return {"path": str(log_path), "command": command[-1], "comet_url": comet_url[-1]}


def walk(
    tree: dict[str, Any], path: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    for name, node in tree.items():
        current = (*path, name)
        yield current, node
        yield from walk(node["children"], current)


def find_node(tree: dict[str, Any], name: str) -> dict[str, Any] | None:
    for path, node in walk(tree):
        if path[-1] == name:
            return node
    return None


def exclusive_category(path: tuple[str, ...]) -> str:  # noqa: C901, PLR0912
    """Name an exclusive leaf by its most specific instrumented operation."""
    joined = "/".join(path)
    assembly = (
        "hessian/fem_constructor",
        "hessian/fem_numeric",
        "hessian/csr_constructor",
        "hessian/csr_refresh",
    )
    # Trace and validation deliberately instrument post-solve diagnostics. They
    # must never be presented as solver work.
    if "trace/" in joined or "validation/" in joined:
        label = "Trace/validation"
    # Only explicit CCD scopes belong here.  Other collision/IPC work can be
    # force, energy, or curvature evaluation and remains separate.
    elif "max_step_size" in joined or "ipc/ccd" in joined:
        label = "CCD"
    # A setup scope can contain other Hessian calls. Charge it to assembly;
    # otherwise an apply scope is the sparse matrix-vector product.
    elif any(scope in joined for scope in assembly):
        label = "Hessian assembly/cache"
    elif "ipc/barrier_hessian" in joined:
        label = "IPC Hessian assembly"
    elif "hessian/cache_prepare" in joined:
        label = "Cache preparation overhead"
    elif "hessian/diagonal" in joined:
        label = "Hessian diagonal"
    elif "hessian/spmv" in joined:
        label = "Sparse SpMV"
    elif "pcg" in path:
        label = "PCG control/vector work"
    elif any(name.endswith("/grad") for name in path):
        label = "Force evaluation"
    elif any(name.endswith("/fun") for name in path):
        label = "Energy evaluation"
    elif any(name.endswith(("/update", "/state_at")) for name in path):
        label = "Contact/state update"
    elif any(name.endswith("/hess_diag") for name in path):
        label = "PNCG diagonal"
    elif any(name.endswith("/hess_quad") for name in path):
        label = "PNCG directional curvature"
    elif "collision" in joined or "ipc/" in joined:
        label = "Contact/IPC non-CCD"
    else:
        label = "Phase controller/other"
    return label


def subtree_breakdown(node: dict[str, Any] | None) -> dict[str, float]:
    """Partition one parent into its own and descendants' exclusive times."""
    totals: Counter[str] = Counter()
    if node is None:
        return dict(totals)
    for path, entry in walk({"phase": node}):
        seconds = max(0.0, float(entry["exclusive_seconds"]))
        totals[exclusive_category(path)] += seconds
    return dict(totals)


def phase_rows(tree: dict[str, Any], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "phase": "Model construction",
            "seconds": float(protocol["model_setup_seconds"]),
            "kind": "outside forward timing",
        },
        {
            "phase": "Operator prewarm/JIT",
            "seconds": float(protocol["operator_prewarm_seconds"]),
            "kind": "excluded from forward timing",
        },
    ]
    for name in ("forward", "pncg", "newton"):
        node = find_node(tree, name)
        if node is not None:
            rows.append(
                {
                    "phase": name.upper() if name == "pncg" else name.title(),
                    "seconds": float(node["inclusive_seconds"]),
                    "kind": "inclusive phase; do not add with children",
                }
            )
    return rows


def call_counts(tree: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for path, node in walk(tree):
        if node["count"]:
            rows.append(
                {
                    "scope": "/".join(path),
                    "calls": int(node["count"]),
                    "inclusive_seconds": float(node["inclusive_seconds"]),
                    "exclusive_seconds": float(node["exclusive_seconds"]),
                }
            )
    return sorted(rows, key=lambda row: (-row["inclusive_seconds"], row["scope"]))


def plot(  # noqa: C901
    trace: list[dict[str, Any]],
    *,
    initial_force: float,
    force_target: float,
    pncg: dict[str, float],
    newton: dict[str, float],
    failed_endpoint: tuple[float, float] | None,
    output: Path,
) -> None:
    pncg_rows = [row for row in trace if row.get("kind") in {"initial", "pncg"}]
    newton_rows = [row for row in trace if row.get("kind") == "newton"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for rows, color, label, marker in (
        (pncg_rows, "#1f77b4", "PNCG accepted state", "."),
        (newton_rows, "#d62728", "Newton accepted state", "o"),
    ):
        if rows:
            axes[0].plot(
                [row["elapsed_seconds"] for row in rows],
                [row["force"] for row in rows],
                color=color,
                marker=marker,
                linewidth=1.2,
                label=label,
            )
            axes[1].plot(
                [row["elapsed_seconds"] for row in rows],
                [row["energy"] for row in rows],
                color=color,
                marker=marker,
                linewidth=1.2,
                label=label,
            )
    axes[0].set_yscale("log")
    if pncg_rows and newton_rows:
        start = pncg_rows[-1]["elapsed_seconds"]
        stop = newton_rows[0]["elapsed_seconds"]
        for axis in axes[:2]:
            axis.axvspan(start, stop, color="#eeeeee", zorder=-1)
            axis.text(
                (start + stop) / 2,
                0.05,
                "Sparse setup +\nfirst Newton step",
                transform=axis.get_xaxis_transform(),
                ha="center",
                fontsize=8,
                color="#555555",
            )
    axes[0].axhline(
        force_target, color="#333", linestyle="--", linewidth=1, label="force target"
    )
    axes[0].scatter(
        [0.0],
        [initial_force],
        color="#111",
        marker="s",
        s=28,
        label="cold initial state",
    )
    if failed_endpoint is not None:
        endpoint_time, endpoint_force = failed_endpoint
        axes[0].scatter(
            [endpoint_time],
            [endpoint_force],
            color="#7f0000",
            marker="x",
            s=42,
            linewidths=1.4,
            label="failed endpoint (contact-feasibility guard)",
        )
    axes[0].set_xlabel("Elapsed forward time (s)")
    axes[0].set_ylabel("Force norm")
    axes[1].set_xlabel("Elapsed forward time (s)")
    axes[1].set_ylabel("Mechanical energy")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    def plot_group(name: str) -> str:
        if name in {"Hessian assembly/cache", "IPC Hessian assembly"}:
            return "Hessian assembly"
        if name in {"PCG control/vector work", "Sparse SpMV"}:
            return "Sparse CG"
        if name in {
            "Force evaluation",
            "Energy evaluation",
            "Contact/state update",
            "Contact/IPC non-CCD",
        }:
            return "State, force, energy"
        if name.startswith("PNCG "):
            return "PNCG curvature/diagonal"
        if name in {"CCD", "Trace/validation"}:
            return name
        return "Controller/cache/other"

    plot_phases = []
    for phase in (pncg, newton):
        grouped = Counter()
        for label, seconds in phase.items():
            grouped[plot_group(label)] += seconds
        plot_phases.append(grouped)
    labels = list(dict.fromkeys(label for phase in plot_phases for label in phase)) or [
        "No timed solver phase"
    ]
    left = [0.0, 0.0]
    colors = plt.get_cmap("tab10").colors
    for index, label in enumerate(labels):
        values = [phase.get(label, 0.0) for phase in plot_phases]
        axes[2].barh(
            ["PNCG", "Newton"], values, left=left, label=label, color=colors[index]
        )
        left = [prior + value for prior, value in zip(left, values, strict=True)]
    axes[2].set_xlabel("Exclusive seconds")
    axes[2].set_title("Solver phases: additive exclusive allocations")
    axes[2].legend(fontsize=7, loc="best")
    axes[0].set_title("Accepted-state residual")
    axes[1].set_title("Accepted-state energy")
    figure.suptitle("Cold Smile hybrid first profile", fontsize=13)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def solver_receipt(result: dict[str, Any]) -> dict[str, Any]:
    """Recover the solver receipt from either a success or failure result."""
    forward = result.get("forward")
    if isinstance(forward, dict):
        receipt = forward.get("solver", forward)
        if isinstance(receipt, dict):
            return receipt
    receipt = result.get("failure_receipt")
    return receipt if isinstance(receipt, dict) else {}


def step_receipts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row["accepted_step"]
        for row in rows
        if isinstance(row.get("accepted_step"), dict)
    ]


def pncg_finding(receipt: dict[str, Any], steps: int) -> str:
    pncg = receipt.get("pncg")
    if not isinstance(pncg, dict):
        return (
            f"PNCG recorded {steps} finite accepted updates, then the forward solve failed "
            "before producing a structured handoff or convergence receipt."
        )
    reason = str(pncg.get("reason", "not recorded")).replace("_", " ")
    windows = pncg.get("windows", [])
    if reason == "converged":
        return f"PNCG converged after {steps} accepted updates; sparse Newton was not entered."
    if reason == "stalled":
        return (
            f"PNCG handed off after {steps} accepted updates because its configured "
            f"force-window stall criterion fired ({len(windows)} recorded windows)."
        )
    return f"PNCG handed off after {steps} accepted updates for {reason}."


def optional_hvp_finding(result: dict[str, Any]) -> str | None:
    validation = result.get("hvp_validation")
    if not isinstance(validation, dict):
        return None
    error = validation.get("handoff_relative_error")
    if error is None:
        return "No sparse/reference HVP validation was recorded because sparse Newton was not entered."
    return (
        f"The sparse/reference HVP relative error at the Newton handoff was {float(error):.3g}; "
        "this validates that handoff state only."
    )


def comparable_baseline(
    run_dir: Path, protocol: dict[str, Any], baseline_dir: Path
) -> dict[str, Any]:
    """Load a baseline only after checking the shared workload and Newton settings."""
    baseline_summary = read_json(baseline_dir / "summary.json")
    baseline_timing = read_json(baseline_dir / "timing.json")
    baseline_trace = read_trace(baseline_dir / "trace.jsonl")
    baseline_protocol = baseline_summary["protocol"]
    mismatches = [
        key
        for key in (
            "checkpoint_sha256",
            "neutral_checkpoint_sha256",
            "activation_sha256",
            "seed_displacement_sha256",
        )
        if protocol.get(key) != baseline_protocol.get(key)
    ]
    mismatches.extend(
        f"config.{key}"
        for key in (
            "forward_atol",
            "linear_rtol",
            "max_newton_steps",
            "wall_seconds",
            "ipc_threads",
        )
        if protocol["config"].get(key) != baseline_protocol["config"].get(key)
    )
    if protocol.get("newton") != baseline_protocol.get("newton"):
        mismatches.append("newton")
    if protocol["versions"].get("ipctk") != baseline_protocol["versions"].get("ipctk"):
        mismatches.append("ipctk")
    if protocol.get("gpu_device") != baseline_protocol.get("gpu_device"):
        mismatches.append("gpu_device")
    if bool(read_json(run_dir / "timing.json").get("cuda_sync")) != bool(
        baseline_timing.get("cuda_sync")
    ):
        mismatches.append("timing.cuda_sync")
    if mismatches:
        raise AssertionError(
            "baseline does not share the required workload/Newton/timing settings: "
            + ", ".join(mismatches)
        )
    return {
        "directory": str(baseline_dir),
        "summary": baseline_summary,
        "timing": baseline_timing,
        "trace": baseline_trace,
        "pncg_changed": protocol.get("pncg") != baseline_protocol.get("pncg"),
    }


def baseline_rows(
    result: dict[str, Any], trace: list[dict[str, Any]], baseline: dict[str, Any]
) -> list[list[str]]:
    baseline_result = baseline["summary"]["result"]
    baseline_trace = baseline["trace"]
    current_newton = [row for row in trace if row.get("kind") == "newton"]
    old_newton = [row for row in baseline_trace if row.get("kind") == "newton"]
    metrics = (
        (
            "Forward wall time (s)",
            result["forward_wall_seconds"],
            baseline_result["forward_wall_seconds"],
        ),
        ("Terminal force", result["terminal_force"], baseline_result["terminal_force"]),
        (
            "PNCG accepted updates",
            len([row for row in trace if row.get("kind") == "pncg"]),
            len([row for row in baseline_trace if row.get("kind") == "pncg"]),
        ),
        ("Newton accepted steps", len(current_newton), len(old_newton)),
        (
            "Successful CG iterations",
            sum(
                int(item.get("linear", {}).get("steps", 0))
                for item in step_receipts(current_newton)
            ),
            sum(
                int(item.get("linear", {}).get("steps", 0))
                for item in step_receipts(old_newton)
            ),
        ),
    )
    rows = []
    for label, current, old in metrics:
        current_float, old_float = float(current), float(old)
        delta = current_float - old_float
        rows.append(
            [label, f"{current_float:.6g}", f"{old_float:.6g}", f"{delta:+.6g}"]
        )
    return rows


def termination_text(result: dict[str, Any]) -> str:
    collision = result.get("collision", {})
    feasible = collision.get("state_feasible") if isinstance(collision, dict) else None
    if result.get("success"):
        return f"converged; endpoint feasible={feasible}"
    return f"failed: {result.get('failure', 'reason not recorded')}; endpoint feasible={feasible}"


def main(cfg: Config) -> None:  # noqa: C901, PLR0915
    status = read_json(cfg.run_dir / "status.json")
    assert status["running"] is False, (
        "profile is still running; do not render partial evidence"
    )
    assert not cfg.output_dir.exists(), cfg.output_dir
    summary_path = cfg.run_dir / "summary.json"
    timing_path = cfg.run_dir / "timing.json"
    trace_path = cfg.run_dir / "trace.jsonl"
    summary = read_json(summary_path)
    timing = read_json(timing_path)
    trace = read_trace(trace_path)
    protocol = summary["protocol"]
    result = summary["result"]
    tree = timing["tree"]
    assert timing["schema"] == "inverse-hierarchical-timing-v1"
    assert not timing["missing_hooks"]
    assert trace
    receipt_from_log = run_receipt(cfg.run_log)
    baseline = (
        comparable_baseline(cfg.run_dir, protocol, cfg.baseline_dir)
        if cfg.baseline_dir is not None
        else None
    )
    cfg.output_dir.mkdir(parents=True)
    phases = phase_rows(tree, protocol)
    pncg_node = find_node(tree, "pncg")
    newton_node = find_node(tree, "newton")
    pncg_breakdown = subtree_breakdown(pncg_node)
    newton_breakdown = subtree_breakdown(newton_node)
    if pncg_node is not None:
        assert (
            abs(sum(pncg_breakdown.values()) - float(pncg_node["inclusive_seconds"]))
            <= 1e-6
        )
    if newton_node is not None:
        assert (
            abs(
                sum(newton_breakdown.values()) - float(newton_node["inclusive_seconds"])
            )
            <= 1e-6
        )
    counts = call_counts(tree)
    last = trace[-1]
    receipt = solver_receipt(result)
    pncg_rows = [row for row in trace if row.get("kind") == "pncg"]
    newton_rows = [row for row in trace if row.get("kind") == "newton"]
    newton_receipts = step_receipts(newton_rows)
    cg_iterations = sum(
        int(item.get("linear", {}).get("steps", 0)) for item in newton_receipts
    )
    retries = sum(
        len(item.get("regularization_retries", [])) for item in newton_receipts
    )
    shifts = [float(item["shift"]) for item in newton_receipts if "shift" in item]
    backtracked = sum(int(item.get("backtracks", 0)) > 0 for item in newton_receipts)
    ccd_limited = sum(float(item.get("ccd", 1.0)) < 1.0 for item in newton_receipts)
    attempted_updates = int(
        result.get("forward", {}).get("counts", {}).get("update", len(pncg_rows))
    )
    evidence = {
        "schema": "hybrid-profile-report-v1",
        "inputs": {
            "summary_sha256": sha256(summary_path),
            "timing_sha256": sha256(timing_path),
            "trace_sha256": sha256(trace_path),
        },
        "result": {
            "success": bool(result["success"]),
            "failure": result.get("failure"),
            "forward_wall_seconds": result["forward_wall_seconds"],
            "terminal_force": result["terminal_force"],
            "last_trace_force": last["force"],
            "last_trace_energy": last["energy"],
            "force_target": protocol["config"]["forward_atol"],
        },
        "phase_timing": phases,
        "pncg_exclusive_breakdown_seconds": pncg_breakdown,
        "newton_exclusive_breakdown_seconds": newton_breakdown,
        "call_counts": counts,
        "protocol": {
            "pncg": protocol["pncg"],
            "newton": protocol["newton"],
            "newton_hessian": protocol["newton_hessian"],
            "ipctk": protocol["versions"]["ipctk"],
            "scope": protocol["scope"],
        },
        "observed_solver": {
            "pncg_updates": len(pncg_rows),
            "pncg_reason": receipt.get("pncg", {}).get("reason"),
            "newton_accepted_steps": len(newton_rows),
            "successful_cg_iterations": cg_iterations,
            "regularization_retries": retries,
            "backtracked_steps": backtracked,
            "ccd_limited_newton_steps": ccd_limited,
            "accepted_shift_min": min(shifts) if shifts else None,
            "accepted_shift_max": max(shifts) if shifts else None,
            "attempted_pncg_updates": attempted_updates,
        },
        "baseline": None
        if baseline is None
        else {
            "directory": baseline["directory"],
            "pncg_changed": baseline["pncg_changed"],
            "comparison_rows": baseline_rows(result, trace, baseline),
        },
        "run_receipt": receipt_from_log,
    }
    (cfg.output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    chart = cfg.output_dir / cfg.document_path.stem
    failed_endpoint = None
    if not result["success"] and attempted_updates > len(pncg_rows):
        failed_endpoint = (
            float(result["forward_wall_seconds"]),
            float(result["terminal_force"]),
        )
    plot(
        trace,
        initial_force=float(protocol["initial_force"]),
        force_target=float(protocol["config"]["forward_atol"]),
        pncg=pncg_breakdown,
        newton=newton_breakdown,
        failed_endpoint=failed_endpoint,
        output=chart,
    )
    phase_table = markdown_table(
        ["Timing", "Seconds", "Interpretation"],
        [[row["phase"], f"{row['seconds']:.3f}", row["kind"]] for row in phases],
    )
    breakdown_table = markdown_table(
        ["Newton exclusive allocation", "Seconds"],
        [[name, f"{seconds:.3f}"] for name, seconds in newton_breakdown.items()],
    )
    pncg_breakdown_table = markdown_table(
        ["PNCG exclusive allocation", "Seconds"],
        [[name, f"{seconds:.3f}"] for name, seconds in pncg_breakdown.items()],
    )
    count_table = markdown_table(
        ["Scope", "Calls", "Inclusive s", "Exclusive s"],
        [
            [
                row["scope"],
                str(row["calls"]),
                f"{row['inclusive_seconds']:.3f}",
                f"{row['exclusive_seconds']:.3f}",
            ]
            for row in counts
        ],
    )
    conclusion = (
        f"The instrumented forward solve took {result['forward_wall_seconds']:.3f} s and "
        f"{'converged' if result['success'] else 'did not converge'}; terminal force was "
        f"{result['terminal_force']:.6g} against {protocol['config']['forward_atol']:.1e}. "
        "This is one cold fixture profile and does not establish a general solver speedup."
    )
    first_setup = sum(
        row["inclusive_seconds"]
        for row in counts
        if row["scope"].endswith(("hessian/fem_constructor", "hessian/csr_constructor"))
    )
    cg_seconds = sum(
        row["inclusive_seconds"] for row in counts if row["scope"].endswith("/pcg")
    )
    ccd_seconds = sum(
        row["inclusive_seconds"]
        for row in counts
        if row["scope"].endswith("collision/max_step_size")
    )
    total_assembly = newton_breakdown.get(
        "Hessian assembly/cache", 0.0
    ) + newton_breakdown.get("IPC Hessian assembly", 0.0)
    findings = [pncg_finding(receipt, len(pncg_rows))]
    if newton_rows:
        shift_text = (
            f"Accepted Newton shifts ranged from {min(shifts):.6g} to {max(shifts):.6g}."
            if shifts
            else "Accepted Newton shifts were not recorded."
        )
        findings.extend(
            [
                f"Newton accepted {len(newton_rows)} steps with {cg_iterations} successful CG iterations and {retries} regularization retries; {backtracked} accepted steps used backtracking and {ccd_limited} were CCD-limited. {shift_text}",
                f"Newton's exclusive FEM/CSR plus IPC Hessian assembly was {total_assembly:.3f} s, including {first_setup:.3f} s in first-construction scopes. All CG attempts together cost {cg_seconds:.3f} s; CCD wrappers across the forward solve cost {ccd_seconds:.3f} s. These are nested timings and must not be added to phase totals.",
            ]
        )
    else:
        findings.append(
            "Sparse Newton was not entered, so the Newton timing and CG tables are empty by design."
        )
    if isinstance(result.get("shape"), dict) and isinstance(
        result.get("collision"), dict
    ):
        findings.append(
            f"Endpoint validation found {result['shape'].get('inverted_tetrahedra', 'unknown')} inverted tetrahedra and contact feasibility {result['collision'].get('state_feasible', 'unknown')}."
        )
    if not result["success"] and attempted_updates > len(pncg_rows):
        minimum_gap = float(
            result["collision"].get("minimum_active_distance_m", float("nan"))
        )
        required_gap = float(
            result["collision"].get("minimum_required_distance_m", float("nan"))
        )
        findings.append(
            f"The accepted-state trace ends after PNCG step {len(pncg_rows)}, but the saved endpoint is after attempted update {attempted_updates}, where evaluation failed. Its terminal force and collision audit describe that failed endpoint, not the last finite trace state."
        )
        findings.append(
            f"The nonfinite energy is the contact-feasibility guard returning +inf at a minimum gap of {minimum_gap:.6g} m, below its required {required_gap:.6g} m clearance; it does not by itself demonstrate mechanical-energy divergence. The one-shot PNCG update has no rollback. The 60-step earliest stall handoff could not fire before this failure."
        )
    if pncg_rows:
        minimum = min(pncg_rows, key=lambda row: float(row["force"]))
        last_pncg = pncg_rows[-1]
        curvature_seconds = pncg_breakdown.get("PNCG directional curvature", 0.0)
        ccd_phase_seconds = pncg_breakdown.get("CCD", 0.0)
        pncg_seconds = float(pncg_node["inclusive_seconds"]) if pncg_node else 0.0
        ccd_curvature_share = (
            (curvature_seconds + ccd_phase_seconds) / pncg_seconds
            if pncg_seconds > 0
            else 0.0
        )
        findings.append(
            f"The finite trace reached its lowest force {float(minimum['force']):.6g} at PNCG step {minimum['step']}, then ended at {float(last_pncg['force']):.6g} at step {last_pncg['step']}. Directional-curvature evaluation used {curvature_seconds:.3f} s and CCD used {ccd_phase_seconds:.3f} s, together {ccd_curvature_share:.1%} of timed PNCG."
        )
    hvp = optional_hvp_finding(result)
    if hvp is not None:
        findings.append(hvp)
    if baseline is not None:
        findings.append(
            "The baseline shares checkpoint, seed, activation, Newton settings, GPU model, IPC version, and synchronized timing. Its PNCG curvature policy differs by design. Sequential runs may have different compiled-kernel cache state, and the timing tree does not isolate JIT; first-construction and total-wall deltas therefore are descriptive, not a clean cold-JIT speed comparison. The runs also have different termination reasons, so their wall-time ratio is not a completed-solve speed comparison."
        )
    findings_markdown = "\n\n".join(findings)
    findings_html = "".join(f"<p>{html.escape(item)}</p>" for item in findings)
    document = f"""# Hybrid-first cold forward profile

{conclusion}

{findings_markdown}

![Residual, energy, and additive solver timing](../data/{cfg.output_dir.name}/{chart.name}.png)

## Protocol

The run starts from the recorded cold displacement with saved Smile activation and fixed prestress. It uses undamped PNCG with per-contribution-clamped curvature, followed only when needed by exact GPU free-CSR Newton with an absolute exact Jacobi diagonal. IPC is PyPI {protocol["versions"]["ipctk"]}. Model construction and ordinary-operator prewarm are excluded; first sparse Newton construction is included. The profiler adds synchronization and changes overlap, so values are instrumented completed-operation wall times. Kernel JIT is not separately timed.

{phase_table}

Parent phase timings are inclusive explanatory values and must not be added to their children. The phase allocations below use exclusive nodes only, so each sums to its own phase without nested double counting.

{pncg_breakdown_table}

{breakdown_table}

{"" if baseline is None else "## Comparison with baseline\n\n" + markdown_table(["Metric", "Clamped run", "Baseline", "Delta"], baseline_rows(result, trace, baseline)) + "\n\nThe PNCG policy intentionally differs. The comparison is therefore an observed fixture comparison, not an attribution of every delta to clamping."}

## Calls

{count_table}

## Interpretation

Residual and energy are sampled only at accepted trace states. The plotted trace shows the actual recorded behavior, including any residual decrease followed by stall. It excludes an outer inverse update and adjoint, and makes no universal optimizer or end-to-end speed claim. Startup/JIT prewarm remains outside the forward timer; sparse initial construction is charged inside its first Newton phase. See [evidence JSON](../data/{cfg.output_dir.name}/evidence.json) for hashes, exact receipts, and call counts.

## Reproduce and audit

Exact executed sources, protocol, timing tree, accepted-state trace, endpoint, and runtime-binding receipt are in `{cfg.run_dir}`. The runtime binding verifies historical input arrays and records runtime source differences. This renderer performs no forward iterations.

{"" if receipt_from_log is None else "The measured run used:\n\n```bash\n" + receipt_from_log["command"] + "\n```\n\nComet receipt: " + receipt_from_log["comet_url"] + "."}

{"" if baseline is None else "The clamped run terminated as `" + termination_text(result) + "`. The baseline terminated as `" + termination_text(baseline["summary"]["result"]) + "`. These are not two completed solves, so the wall-time delta is not a solver speedup."}
"""
    cfg.document_path.write_text(document)
    svg = (chart.with_suffix(".svg")).read_text()
    encoded_png = base64.b64encode(chart.with_suffix(".png").read_bytes()).decode()
    baseline_html = (
        ""
        if baseline is None
        else (
            "<h2>Comparison with baseline</h2><pre>"
            + html.escape(
                markdown_table(
                    ["Metric", "Clamped run", "Baseline", "Delta"],
                    baseline_rows(result, trace, baseline),
                )
            )
            + "</pre>"
        )
    )
    receipt_html = (
        ""
        if receipt_from_log is None
        else (
            "<h2>Reproduce and audit</h2><pre>"
            + html.escape(receipt_from_log["command"])
            + '</pre><p>Run receipt: <a href="'
            + html.escape(receipt_from_log["comet_url"], quote=True)
            + '">Comet</a></p>'
        )
    )
    page = f"""<!doctype html><meta charset=\"utf-8\"><title>Hybrid-first cold profile</title><style>body{{font:16px system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:.35rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}svg{{max-width:100%;height:auto}}</style><h1>Hybrid-first cold forward profile</h1><p>{html.escape(conclusion)}</p><section>{svg}</section><h2>Phase timing</h2><pre>{html.escape(phase_table)}</pre><h2>PNCG exclusive allocation</h2><pre>{html.escape(pncg_breakdown_table)}</pre><h2>Newton exclusive allocation</h2><pre>{html.escape(breakdown_table)}</pre>{baseline_html}<h2>Call counts</h2><pre>{html.escape(count_table)}</pre>{receipt_html}<p>Instrumented timing uses hierarchical exclusive accounting for additive breakdowns. Parent inclusive phase time is explanatory only. Model setup and JIT prewarm are outside the forward timer; first sparse topology construction is inside it.</p><p><a download=\"{chart.name}.png\" href=\"data:image/png;base64,{encoded_png}\">Download PNG</a> · <a href=\"evidence.json\">Evidence JSON</a></p>"""
    page = page.replace(
        "<h2>Phase timing</h2>", findings_html + "<h2>Phase timing</h2>"
    )
    page = page.replace(
        "<h2>Call counts</h2><pre>",
        "<details><summary>Detailed call counts</summary><pre>",
    )
    page = page.replace("</pre><p>Instrumented", "</pre></details><p>Instrumented")
    page = page.replace(
        "</style>",
        "pre{overflow-x:auto;font-size:13px}summary{cursor:pointer;font-weight:600}</style>",
    )
    html_path = cfg.output_dir / f"{chart.name}.html"
    html_path.write_text(page)
    for output in (
        cfg.output_dir / "evidence.json",
        chart.with_suffix(".png"),
        chart.with_suffix(".svg"),
        html_path,
        cfg.document_path,
    ):
        cherries.log_output(output)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileReport)
