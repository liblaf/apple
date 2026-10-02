"""Plot snapshot no-contact residuals without joining separate objectives."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint  # noqa: E402


class Config(cherries.BaseConfig):
    runs: tuple[Path, ...] = (
        GROUP / "data/pose-rigid-001",
        GROUP / "data/pose-rigid-diagnostic-001",
        GROUP / "data/pose-rigid-resume-001",
        GROUP / "data/pose-rigid-resume-002",
    )
    output_dir: Path = GROUP / "data/review-repaired-reference-005/rigid-forward"


@dataclass(frozen=True)
class Snapshot:
    path: Path
    sha256: str
    bytes: int
    payload: Any


@dataclass(frozen=True)
class Segment:
    label: str
    run: str
    phase: str
    pose: str
    trace: Snapshot
    rows: list[dict[str, Any]]
    threshold_n: float
    contact_force_n: float | None
    partial: bool
    failure: str | None


def snapshot_json(path: Path) -> Snapshot:
    raw = path.read_bytes()
    return Snapshot(
        path=path.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw),
        payload=json.loads(raw),
    )


def snapshot_lines(path: Path) -> Snapshot:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return Snapshot(
        path=path.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw),
        payload=rows,
    )


def pose_label(pose: list[float] | None) -> str:
    if pose is None:
        return "pose unavailable"
    value = np.asarray(pose, dtype=float)
    assert value.shape == (6,)
    assert np.isfinite(value).all()
    return (
        f"rotation {np.linalg.norm(value[:3]) * 180 / math.pi:.3f}°; "
        f"translation {np.linalg.norm(value[3:]) * 1000:.3f} mm"
    )


def trace_segment(
    *,
    run: Path,
    phase: str,
    pose: list[float] | None,
    trace: Snapshot,
    threshold_n: float,
    contact_force_n: float | None,
    run_status: str,
) -> Segment:
    rows = trace.payload
    assert isinstance(rows, list)
    assert rows
    for row in rows:
        assert {"seconds", "force_norm"} <= row.keys(), row
        assert math.isfinite(float(row["seconds"]))
        assert math.isfinite(float(row["force_norm"]))
    last = rows[-1]
    failure = str(last["failure"]) if last.get("kind") == "failure" else None
    partial = failure is None and float(last["force_norm"]) > threshold_n / 1e6
    label = f"{run.name} · {phase}\n{pose_label(pose)} · {run_status}"
    return Segment(
        label=label,
        run=run.name,
        phase=phase,
        pose=pose_label(pose),
        trace=trace,
        rows=rows,
        threshold_n=threshold_n,
        contact_force_n=contact_force_n,
        partial=partial,
        failure=failure,
    )


def collect_run(run: Path) -> tuple[list[Segment], list[Snapshot], dict[str, Any]]:
    protocol = snapshot_json(run / "protocol.json")
    config = protocol.payload["config"]
    threshold_n = float(config["forward_atol"]) * 1e6
    assert threshold_n > 0
    inputs: list[Snapshot] = [protocol]
    segments: list[Segment] = []
    run_state: dict[str, Any] = {"run": run.name, "protocol": str(protocol.path)}
    summary_path = run / "continuation/summary.json"
    if summary_path.is_file():
        summary = snapshot_json(summary_path)
        inputs.append(summary)
        run_state["continuation_status"] = summary.payload["status"]
        for step in summary.payload["steps"]:
            index = int(step["index"])
            trace_path = run / "continuation" / f"step-{index:03d}" / "force.jsonl"
            if not trace_path.is_file():
                continue
            trace = snapshot_lines(trace_path)
            inputs.append(trace)
            proposal = step.get("proposal")
            contact_force_n = (
                float(proposal["forward"]["grad_norm"]) * 1e6
                if proposal is not None
                else None
            )
            segments.append(
                trace_segment(
                    run=run,
                    phase=f"continuation step {index} no-contact relaxation",
                    pose=step.get("new_pose_rad_m"),
                    trace=trace,
                    threshold_n=threshold_n,
                    contact_force_n=contact_force_n,
                    run_status=str(step["status"]),
                )
            )
    source_trace = run / "source-relaxation/force.jsonl"
    if source_trace.is_file():
        trace = snapshot_lines(source_trace)
        inputs.append(trace)
        source_status = "running"
        source_partial = True
        interruption_path = run / "interruption.json"
        if interruption_path.is_file():
            interruption = snapshot_json(interruption_path)
            inputs.append(interruption)
            assert interruption.payload["status"] == "intentionally_interrupted"
            source_status = "intentionally interrupted"
        source_summary_path = run / "source-relaxation/summary.json"
        if source_summary_path.is_file():
            source_summary = snapshot_json(source_summary_path)
            inputs.append(source_summary)
            source_partial = False
            source_status = (
                "converged no-contact"
                if source_summary.payload["success"]
                else "failed"
            )
        contact_force_n = None
        source_corrector_path = run / "source-corrector.json"
        if source_corrector_path.is_file():
            source_corrector = snapshot_json(source_corrector_path)
            inputs.append(source_corrector)
            contact_force_n = (
                float(source_corrector.payload["forward"]["grad_norm"]) * 1e6
            )
        segments.append(
            trace_segment(
                run=run,
                phase="source no-contact relaxation resume",
                pose=protocol.payload["old_pose_rad_m"],
                trace=trace,
                threshold_n=threshold_n,
                contact_force_n=contact_force_n,
                run_status=source_status,
            )
        )
        run_state["source_relaxation_status"] = source_status
        run_state["source_relaxation_partial"] = source_partial
    return segments, inputs, run_state


def plot(segments: list[Segment], output: Path) -> None:
    assert segments
    figure, axes = plt.subplots(
        len(segments), 1, figsize=(11, 3.2 * len(segments)), constrained_layout=True
    )
    if len(segments) == 1:
        axes = [axes]
    for axis, segment in zip(axes, segments, strict=True):
        seconds = np.asarray([row["seconds"] for row in segment.rows], dtype=float)
        force_n = (
            np.asarray([row["force_norm"] for row in segment.rows], dtype=float) * 1e6
        )
        kinds = [str(row["kind"]) for row in segment.rows]
        axis.semilogy(
            seconds,
            force_n,
            color="#2a6f97",
            marker=".",
            lw=1.2,
            label="accepted no-contact state",
        )
        failed = np.asarray([kind == "failure" for kind in kinds])
        if failed.any():
            axis.scatter(
                seconds[failed],
                force_n[failed],
                marker="x",
                s=56,
                color="#b2182b",
                label="wall-budget snapshot",
            )
        if segment.contact_force_n is not None:
            axis.text(
                0.99,
                0.92,
                f"contact endpoint: {segment.contact_force_n:.3g} N\n(time not shown)",
                ha="right",
                va="top",
                transform=axis.transAxes,
                color="#4d9221",
                fontsize=8,
            )
        axis.axhline(
            segment.threshold_n, color="#7f3b08", ls="--", lw=1, label="force threshold"
        )
        axis.set_title(segment.label, loc="left", fontsize=10)
        axis.set_xlabel("wall time within this saved segment (s)")
        axis.set_ylabel("force (N)")
        axis.grid(visible=True, which="both", alpha=0.25)
        axis.legend(fontsize=8, loc="best")
        if segment.failure:
            axis.text(
                0.99,
                0.06,
                segment.failure,
                ha="right",
                transform=axis.transAxes,
                color="#b2182b",
                fontsize=8,
            )
    figure.suptitle(
        "Rigid-pose no-contact residual snapshots; segments are not one continuous solve",
        fontsize=13,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(cfg: Config) -> None:
    output = cfg.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    segments: list[Segment] = []
    inputs: list[Snapshot] = []
    runs: list[dict[str, Any]] = []
    for item in cfg.runs:
        run = item.resolve()
        assert run.is_dir(), run
        found, captured, state = collect_run(run)
        segments.extend(found)
        inputs.extend(captured)
        runs.append(state)
    assert segments, "no saved force traces"
    image = output / "rigid-forward-force.png"
    plot(segments, image)
    receipt = {
        "schema": "rigid-forward-force-snapshot-v1",
        "scope": "Each panel is an independently saved no-contact segment; no energy or objective continuity is implied across panels.",
        "runs": runs,
        "segments": [
            {
                "run": item.run,
                "phase": item.phase,
                "pose": item.pose,
                "trace": {
                    "path": str(item.trace.path),
                    "sha256": item.trace.sha256,
                    "bytes": item.trace.bytes,
                },
                "samples": len(item.rows),
                "threshold_n": item.threshold_n,
                "contact_force_n": item.contact_force_n,
                "partial": item.partial,
                "failure": item.failure,
            }
            for item in segments
        ],
        "captured_inputs": [
            {"path": str(item.path), "sha256": item.sha256, "bytes": item.bytes}
            for item in inputs
        ],
        "plot": {
            "path": str(image),
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        },
    }
    receipt_path = output / "rigid-forward-force-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    cherries.log_output(image)
    cherries.log_output(receipt_path)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
