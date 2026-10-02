"""Render CPU-only evidence for the adaptive cold-start PNCG diagnostic."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from liblaf import cherries

EXPERIMENT = Path(__file__).resolve().parent.parent


class Config(cherries.BaseConfig):
    summary_path: Path = EXPERIMENT / "data/adaptive-pncg-cold-003/summary.json"
    first_trial_path: Path = EXPERIMENT / "data/adaptive-pncg-cold-002/summary.json"
    output_dir: Path = EXPERIMENT / "data/adaptive-pncg-report-001"
    document_path: Path = EXPERIMENT / "docs/48-adaptive-pncg-cold.md"
    launch_command: str = "DEBUG=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 CHERRIES_NAME=cold-smile-adaptive-pncg-window20 CHERRIES_TAGS=smile,performance,cold-forward,pncg,newton,adaptive /root/codex-apple-performance/apple/.venv/bin/python -u src/47-test-adaptive-pncg.py --checkpoint data/inverse-duration-hard-001/zero_smoothing/historical/expressions/Smile/latest.pt --neutral-checkpoint data/smile-fit-adam03-no-smoothness-005/arms/hybrid_diag/expressions/Smile/initial.pt --window-steps 20 --output-dir data/adaptive-pncg-cold-003"


MARKER_LABELS = {
    "newton_diag": "historical Newton-only endpoint",
    "hybrid_diag": "historical hybrid last-recorded point",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solver_receipt(result: dict[str, Any]) -> dict[str, Any]:
    """Read the direct success receipt or the nested failure receipt schema."""
    forward = result.get("forward")
    if forward is None:
        forward = result["failure"]["receipt"]
    return forward.get("solver", forward)


def force(row: dict[str, Any]) -> float:
    return float(row["force"])


def correction_effects(
    trace: list[dict[str, Any]], corrections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    effects = []
    for index, correction in enumerate(corrections, start=1):
        pncg_steps = correction["pncg_steps"]
        before = next(
            row
            for row in reversed(trace)
            if row["kind"] == "pncg" and row["pncg_steps"] == pncg_steps
        )
        after = next(
            row
            for row in trace
            if row["kind"] == "newton" and row["pncg_steps"] == pncg_steps
        )
        effects.append(
            {
                "correction": index,
                "pncg_steps": pncg_steps,
                "measured_seconds": correction["seconds"],
                "force_before": force(before),
                "force_after": force(after),
                "force_ratio": force(after) / force(before),
                "energy_before": float(before["energy"]),
                "energy_after": float(after["energy"]),
                "energy_change": float(after["energy"] - before["energy"]),
            }
        )
    return effects


def baseline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        forward = row.get("forward") or {}
        solver = forward.get("solver") or {}
        counts = row.get("operation_counts") or {}
        steps = counts.get("steps")
        if steps is None:
            steps = forward.get(
                "steps", solver.get("optimizer_step", len(solver.get("trace", ())))
            )
        result.append(
            {
                "method": row.get("method"),
                "success": row.get("success"),
                "forward_seconds": row.get("forward_wall_seconds"),
                "force": row.get(
                    "terminal_force_norm",
                    forward.get("grad_norm", solver.get("grad_norm")),
                ),
                "steps": steps,
                "hess_prod": (counts.get("counts") or forward.get("counts") or {}).get(
                    "hess_prod", 0
                ),
                "pncg_steps": solver.get("optimizer_step", 0)
                if row.get("method") == "hybrid_diag"
                else 0,
                "newton_steps": len(solver.get("trace", ()))
                if row.get("method") == "newton_diag"
                else solver.get("newton_steps", 0),
                "current_best_force": None,
                "final_energy": None,
                "failure": (row.get("failure") or {}).get("message"),
            }
        )
    return result


def pilot_row(summary: dict[str, Any]) -> dict[str, Any]:
    result = summary["result"]
    solver = solver_receipt(result)
    trace = solver["trace"]
    last = trace[-1]
    return {
        "method": "adaptive_diag pilot 002",
        "success": result["success"],
        "forward_seconds": result["forward_wall_seconds"],
        "force": solver.get("last_observed_accepted_force", last["force"]),
        "steps": solver.get("steps", last["accepted_step"]),
        "pncg_steps": solver.get("pncg_steps", last["pncg_steps"]),
        "newton_steps": solver.get("newton_steps", last["newton_steps"]),
        "hess_prod": (result.get("forward") or {})
        .get("counts", {})
        .get("hess_prod", 0),
        "current_best_force": min(force(row) for row in trace),
        "final_energy": last["energy"],
        "failure": (result.get("failure") or {}).get("message"),
    }


def historical_markers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    markers = []
    for method, label in MARKER_LABELS.items():
        row = next(item for item in rows if item["method"] == method)
        assert row["forward_seconds"] is not None
        assert row["force"] is not None
        markers.append(
            {"label": label, "seconds": row["forward_seconds"], "force": row["force"]}
        )
    return markers


def adaptive_row(
    result: dict[str, Any], solver: dict[str, Any], trace: list[dict[str, Any]]
) -> dict[str, Any]:
    last = trace[-1]
    return {
        "method": "adaptive_diag",
        "success": result["success"],
        "forward_seconds": result.get("forward_wall_seconds"),
        "force": solver.get("last_observed_accepted_force", last["force"]),
        "steps": solver.get("steps", last["accepted_step"]),
        "hess_prod": (result.get("forward") or {}).get("counts", {}).get("hess_prod"),
        "pncg_steps": solver.get("pncg_steps"),
        "newton_steps": solver.get("newton_steps"),
        "current_best_force": min(force(row) for row in trace),
        "final_energy": last["energy"],
        "failure": (result.get("failure") or {}).get("message"),
    }


def plot(
    trace: list[dict[str, Any]], markers: list[dict[str, Any]], output_dir: Path
) -> dict[str, str]:
    seconds = [float(row["seconds"]) for row in trace]
    forces = [force(row) for row in trace]
    energies = [float(row["energy"]) for row in trace]
    newton = [row["kind"] == "newton" for row in trace]
    figure, (force_axis, energy_axis) = plt.subplots(
        1, 2, figsize=(12, 4.8), constrained_layout=True
    )
    force_axis.plot(
        seconds,
        forces,
        color="#2166ac",
        linewidth=1.2,
        label="accepted adaptive states",
    )
    force_axis.scatter(
        [seconds[index] for index, value in enumerate(newton) if value],
        [forces[index] for index, value in enumerate(newton) if value],
        color="#b2182b",
        marker="D",
        s=30,
        zorder=3,
        label="Newton correction",
    )
    for marker, color in zip(markers, ("#4d4d4d", "#8073ac"), strict=True):
        force_axis.scatter(
            marker["seconds"],
            marker["force"],
            color=color,
            marker="X",
            s=65,
            zorder=4,
            label=marker["label"],
        )
    force_axis.axhline(
        1e-8, color="#000", linestyle="--", linewidth=0.8, label="force target 1e-8"
    )
    force_axis.set(
        xlabel="controller elapsed seconds",
        ylabel="Force norm",
        yscale="log",
        title="Accepted-state force history",
    )
    force_axis.grid(visible=True, which="both", alpha=0.25)
    force_axis.legend(fontsize=7)
    energy_axis.plot(
        seconds,
        energies,
        color="#1b7837",
        linewidth=1.2,
        label="accepted adaptive states",
    )
    energy_axis.scatter(
        [seconds[index] for index, value in enumerate(newton) if value],
        [energies[index] for index, value in enumerate(newton) if value],
        color="#b2182b",
        marker="D",
        s=30,
        zorder=3,
        label="Newton correction",
    )
    energy_axis.set(
        xlabel="controller elapsed seconds",
        ylabel="mechanical energy (model units)",
        title="Accepted-state mechanical energy",
    )
    energy_axis.grid(visible=True, alpha=0.25)
    energy_axis.legend(fontsize=7)
    figure.text(
        0.5,
        -0.02,
        "Adaptive points use the controller clock; historical markers use full-forward endpoint times. No arm converged.",
        ha="center",
        fontsize=8,
    )
    paths = {}
    for suffix in ("png", "svg", "pdf"):
        path = output_dir / f"adaptive-pncg-cold.{suffix}"
        figure.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight")
        paths[suffix] = str(path)
    plt.close(figure)
    return paths


def stop_reason(row: dict[str, Any]) -> str:
    if row["success"]:
        return "converged"
    failure = row["failure"] or "failed"
    return failure.split(":", maxsplit=1)[0]


def table(rows: list[dict[str, Any]]) -> str:
    headers = ("Method", "Time (s)", "PNCG", "Newton", "Last force", "Stop reason")
    body = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    body.extend(
        "| "
        + " | ".join(
            (
                row["method"],
                f"{row['forward_seconds']:.2f}",
                str(row["pncg_steps"]),
                str(row["newton_steps"]),
                f"{row['force']:.3e}",
                stop_reason(row),
            )
        )
        + " |"
        for row in rows
    )
    return "\n".join(body)


def html_table(rows: list[dict[str, Any]]) -> str:
    headers = ("Method", "Time (s)", "PNCG", "Newton", "Last force", "Stop reason")
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(value)}</td>"
            for value in (
                row["method"],
                f"{row['forward_seconds']:.2f}",
                str(row["pncg_steps"]),
                str(row["newton_steps"]),
                f"{row['force']:.3e}",
                stop_reason(row),
            )
        )
        + "</tr>"
        for row in rows
    )
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{header}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table>"
    )


def main(cfg: Config) -> None:
    assert cfg.summary_path.is_file(), cfg.summary_path
    assert cfg.first_trial_path.is_file(), cfg.first_trial_path
    assert not cfg.output_dir.exists(), cfg.output_dir
    summary = json.loads(cfg.summary_path.read_text())
    first_trial = json.loads(cfg.first_trial_path.read_text())
    assert summary["schema"] == "cold-smile-adaptive-pncg-v1"
    result = summary["result"]
    solver = solver_receipt(result)
    trace = solver["trace"]
    windows = solver["windows"]
    corrections = solver["corrections"]
    assert trace
    assert all(row["kind"] in {"initial", "pncg", "newton"} for row in trace)
    assert all(float(row["seconds"]) >= 0 for row in trace)
    effects = correction_effects(trace, corrections)
    baselines = baseline_rows(summary["historical_baselines"])
    markers = historical_markers(baselines)
    adaptive = adaptive_row(result, solver, trace)
    first_pilot = pilot_row(first_trial)
    cfg.output_dir.mkdir(parents=True)
    plots = plot(trace, markers, cfg.output_dir)
    evidence = {
        "schema": "adaptive-pncg-cold-report-evidence-v1",
        "source_summary": {
            "path": str(cfg.summary_path.resolve()),
            "sha256": sha256(cfg.summary_path),
        },
        "result_success": result["success"],
        "adaptive_solver": {
            "trace": trace,
            "windows": windows,
            "corrections": corrections,
            "correction_effects": effects,
        },
        "historical_baselines": baselines,
        "adaptive_row": adaptive,
        "first_pilot": first_pilot,
        "historical_force_markers": markers,
        "protocol": summary["protocol"],
        "plot_paths": plots,
    }
    (cfg.output_dir / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    first = effects[0] if effects else None
    last = effects[-1] if effects else None
    switch = summary["protocol"]["switch_rule"]
    correction_seconds = sum(item["measured_seconds"] for item in effects)
    correction_fraction = correction_seconds / result["forward_wall_seconds"]
    speed_statement = (
        "No matched-convergence speedup is reported because the adaptive run did not converge."
        if not result["success"]
        else "A matched-convergence time ratio may be assessed only against a successful baseline with the same gate."
    )
    correction_text = (
        "No Newton correction was accepted."
        if not effects
        else (
            f"First correction: PNCG step {first['pncg_steps']}, force {first['force_before']:.3e} to {first['force_after']:.3e}, "
            f"measured {first['measured_seconds']:.3f}s. Last correction: PNCG step {last['pncg_steps']}, "
            f"force {last['force_before']:.3e} to {last['force_after']:.3e}, measured {last['measured_seconds']:.3f}s. "
            f"All {len(effects)} corrections reduced force immediately; their measured total was {correction_seconds:.3f}s ({correction_fraction:.1%} of full forward wall time)."
        )
    )
    rows = [*baselines, first_pilot, adaptive]
    newton_force = next(
        row["force"] for row in baselines if row["method"] == "newton_diag"
    )
    conclusion = (
        f"Seven Newton corrections each reduced force immediately, but this cold adaptive trajectory ended at {adaptive['force']:.3e}, "
        f"above the Newton-only endpoint {newton_force:.3e}. The adaptive arm stopped at an Armijo exhaustion and neither compared arm converged."
    )
    report_relative = f"../data/{cfg.output_dir.name}"
    markdown = f"""# Adaptive PNCG cold-start diagnostic

{conclusion} {speed_statement}

The exact switch protocol was: {switch}

{correction_text}

The plots use accepted-state measurements from the adaptive controller. Their elapsed seconds start after boundary setup, so they are distinct from full forward wall time. Red diamonds mark accepted Newton corrections. The force panel contains two standalone historical endpoint markers read from their raw receipts; they are not time curves. Historical mechanical energy is intentionally absent because those runs do not provide exact accepted-state timestamps.

## Historical baseline evidence

{table(rows)}

## Preserved pilot 002

Pilot 002 stopped after `{first_pilot["pncg_steps"]}` accepted PNCG updates and `{first_pilot["newton_steps"]}` Newton corrections in `{first_pilot["forward_seconds"]:.3f}` seconds. Its last accepted force was `{first_pilot["force"]:.6e}` and its best observed accepted force was `{first_pilot["current_best_force"]:.6e}`. It exhausted Armijo before completing the 100-update window, so the two-poor-window trigger could not fire. The result is preserved as a failed trial; it is not overwritten or used as a speedup baseline.

## Source and input equivalence

The adaptive protocol records physical-source changes as `{summary["protocol"]["physical_source_changes"]}` and binds the checkpoint, neutral checkpoint, input hashes, and baseline protocol in its evidence. Best force, final recorded energy, HVP counts, and full traces remain in [evidence.json]({report_relative}/evidence.json).

## Recorded command

```bash
{cfg.launch_command}
```

The working directory was `/root/codex-apple-performance/apple/exp/2026/09/22/solver-performance`. This was a local Cherries run; `DEBUG=1` disables Comet in its profile, and this report has no Comet upload. Reproduction must use a new output directory.

## Timing interpretation

The controller trace includes accepted-state diagnostics and explicit Newton post-update energy evaluation. JIT and fixed-model operator prewarm are excluded. It starts after boundary setup, whereas the historical endpoint markers use full forward wall time. The candidate has dense JSON diagnostics; timed historical arms were not rerun. The plot therefore supports trajectory diagnosis, not directly comparable wall-time speedups; it does not establish a matched-convergence speedup when an arm fails.
"""
    cfg.document_path.write_text(markdown)
    svg = Path(plots["svg"]).read_text()
    html_page = f"""<!doctype html><meta charset=utf-8><title>Adaptive PNCG cold diagnostic</title>
<style>body{{font:15px sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:.35rem;text-align:left}}svg{{max-width:100%;height:auto}}</style>
<h1>Adaptive PNCG cold-start diagnostic</h1><p>{html.escape(conclusion)} {html.escape(speed_statement)}</p>
<p>{html.escape(correction_text)}</p><p>Switch protocol: {html.escape(switch)}</p>{svg}
<h2>Evidence table</h2>{html_table(rows)}
<p>Method/timing caveat: adaptive points use the controller clock after boundary setup; historical markers use full-forward endpoint times. No arm converged, so no speed ratio is claimed. Historical force points are standalone markers; historical energy is not plotted because exact timestamps are unavailable.</p>
<p><a href="evidence.json">Evidence JSON</a> · <a href="adaptive-pncg-cold.pdf">PDF</a></p>"""
    (cfg.output_dir / "index.html").write_text(html_page)
    cherries.log_metrics(
        {
            "adaptive_report/corrections": len(corrections),
            "adaptive_report/success": float(result["success"]),
        }
    )


if __name__ == "__main__":
    cherries.main(main)
