# ruff: noqa: C901, EM101, EM102, TRY003
"""Capture an immutable optimizer checkpoint for interim visual QA."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.append(str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))

from experiment import Profile  # noqa: E402


class Config(cherries.BaseConfig):
    source: Path = Path("56-mouthopen-fit-reuse")
    output: Path = Path("57-checkpoint50-preview")
    attempt: int = 50


def record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def rows_from_bytes(raw: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in raw.decode().splitlines() if line]


def main(cfg: Config) -> None:
    source = cherries.input(cfg.source)
    output = cherries.output(cfg.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preview snapshot: {output}")
    summary_path = source / "summary.json"
    source_summary = json.loads(summary_path.read_text())
    if source_summary.get("status") != "running":
        raise RuntimeError("snapshot request expected live stage 56 to remain running")
    checkpoint_source = source / f"step-{cfg.attempt:04d}.npz"
    trace_path = source / "trace.jsonl"
    receipt_path = source / "solver-receipts.jsonl"
    source_manifest_path = source / "source-manifest.json"
    if not all(
        path.is_file()
        for path in (checkpoint_source, trace_path, receipt_path, source_manifest_path)
    ):
        raise FileNotFoundError(
            "step-50 checkpoint or its required provenance is missing"
        )

    trace_bytes = trace_path.read_bytes()
    solver_bytes = receipt_path.read_bytes()
    matching_trace = [
        row
        for row in rows_from_bytes(trace_bytes)
        if int(row["attempt"]) == cfg.attempt
    ]
    matching_solver = [
        row
        for row in rows_from_bytes(solver_bytes)
        if int(row["attempt"]) == cfg.attempt
    ]
    if len(matching_trace) != 1 or len(matching_solver) != 1:
        raise ValueError(
            "step 50 must have exactly one matching trace and solver receipt"
        )
    trace = matching_trace[0]
    solver = matching_solver[0]
    if not solver["forward"]["success"] or not solver["adjoint"]["success"]:
        raise ValueError(
            "step-50 preview checkpoint lacks successful primal/adjoint receipts"
        )
    if int(trace["optimizer_updates"]) != cfg.attempt - 1:
        raise ValueError("attempt-50 trace does not record 49 accepted updates")
    with np.load(checkpoint_source, allow_pickle=False) as checkpoint:
        if int(checkpoint["step"]) != cfg.attempt:
            raise ValueError(
                "checkpoint step index differs from requested immutable attempt"
            )
        if (
            not np.isfinite(checkpoint["u"]).all()
            or not np.isfinite(checkpoint["B"]).all()
        ):
            raise ValueError("step-50 checkpoint has nonfinite state")

    output.mkdir(parents=True, exist_ok=False)
    checkpoint_output = output / "last.npz"
    shutil.copy2(checkpoint_source, checkpoint_output)
    (output / "trace-snapshot.jsonl").write_bytes(trace_bytes)
    (output / "solver-receipts-snapshot.jsonl").write_bytes(solver_bytes)
    source_receipts = {
        "live_summary": record(summary_path),
        "step_checkpoint": record(checkpoint_source),
        "trace_snapshot_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "solver_receipt_snapshot_sha256": hashlib.sha256(solver_bytes).hexdigest(),
        "source_manifest": record(source_manifest_path),
    }
    for name, receipt in source_summary["inputs"].items():
        path = Path(receipt["path"])
        if record(path)["sha256"] != receipt["sha256"]:
            raise ValueError(f"stage-56 input changed before snapshot: {name}")

    preview_summary = {
        "schema": "mouthopen-fit-interim-snapshot-v1",
        "status": "immutable_interim_checkpoint",
        "mode": source_summary["mode"],
        "activation_model": source_summary["activation_model"],
        "config": source_summary["config"],
        "initialization": source_summary["initialization"],
        "solver_policy": source_summary["solver_policy"],
        "material_spec": source_summary["material_spec"],
        "attempted_updates": int(trace["attempt"]),
        "optimizer_updates": int(trace["optimizer_updates"]),
        "skipped_updates": int(trace["attempt"]) - int(trace["optimizer_updates"]),
        "last_metrics": trace,
        "final_checkpoint": record(checkpoint_output),
        "source_snapshot": {
            "path": str(source_manifest_path.resolve()),
            "sha256": source_receipts["source_manifest"]["sha256"],
        },
        "source_stage": str(source.resolve()),
        "source_stage_status_at_capture": source_summary["status"],
        "source_receipts": source_receipts,
        "selected_solver_receipt": solver,
        "is_interim_only": True,
        "finalization_must_require_completed_attempt_budget": True,
    }
    (output / "summary.json").write_text(json.dumps(preview_summary, indent=2) + "\n")
    (output / "trace-row.json").write_text(json.dumps(trace, indent=2) + "\n")
    (output / "solver-receipt.json").write_text(json.dumps(solver, indent=2) + "\n")
    (output / "source.py").write_text(Path(__file__).read_text())
    cherries.log_metrics(
        {
            "attempt": trace["attempt"],
            "optimizer_updates": trace["optimizer_updates"],
            "fit_rms_mm": trace["fit_rms_mm"],
            "normal_angle_rms_deg": trace["normal_angle_rms_deg"],
            "solver_valid": int(trace["solver_valid"]),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
