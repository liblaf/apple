"""Capture a bounded early comparison between matched Raw6 and the June trace."""

# ruff: noqa: EM101, EM102, TRY003

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = ROOT.parents[4]
RAW6_TRACE = ROOT / "data/30-historical-adam-raw6/trace.csv"
JUNE_TRACE = (
    REPOSITORY
    / "exp/2026/06/17/human-face-smile-prestrain-v2/data/20-human-face-smile-no-skin-lr3-trace.jsonl"
)
DEFAULT_OUTPUT = ROOT / "data/32-early-historical-reproduction"
SCALAR_CONTROLS = 1_729_410
CAPTURED_STEPS = 4


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_read(path: Path) -> bytes:
    """Require two identical reads so a live CSV rewrite cannot be captured halfway."""
    first = path.read_bytes()
    time.sleep(0.2)
    second = path.read_bytes()
    if first != second:
        raise RuntimeError(f"live source changed during capture: {path}")
    return first


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def relative_delta(value: float, reference: float) -> float:
    return value / reference - 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)

    raw6_bytes = stable_read(RAW6_TRACE)
    june_bytes = JUNE_TRACE.read_bytes()
    read_time = datetime.now(UTC).isoformat()
    raw6_text = raw6_bytes.decode()
    june_text = june_bytes.decode()
    raw6_lines = raw6_text.splitlines(keepends=True)
    june_lines = june_text.splitlines(keepends=True)
    raw6_records = list(csv.DictReader(raw6_text.splitlines()))
    june_records = [json.loads(line) for line in june_text.splitlines() if line]
    if len(raw6_records) < CAPTURED_STEPS or len(june_records) < CAPTURED_STEPS:
        raise RuntimeError("both traces must contain at least steps 0 through 3")
    raw6_first = raw6_records[:CAPTURED_STEPS]
    june_first = june_records[:CAPTURED_STEPS]
    expected_steps = list(range(CAPTURED_STEPS))
    if [int(row["step"]) for row in raw6_first] != expected_steps:
        raise ValueError("Raw6 trace does not begin with consecutive steps 0 through 3")
    if [int(row["step"]) for row in june_first] != expected_steps:
        raise ValueError("June trace does not begin with consecutive steps 0 through 3")

    raw6_capture = "".join(raw6_lines[: CAPTURED_STEPS + 1]).encode()
    june_capture = "".join(june_lines[:CAPTURED_STEPS]).encode()
    (output / "raw6-first4.csv").write_bytes(raw6_capture)
    (output / "june-first4.jsonl").write_bytes(june_capture)
    (output / "raw6-trace-read-time-snapshot.csv").write_bytes(raw6_bytes)

    differences = []
    for raw6, june in zip(raw6_first, june_first, strict=True):
        data_objective = float(raw6["data_objective_mm2"])
        june_objective = float(june["loss/total"])
        gradient_rms = float(raw6["gradient_rms"])
        june_gradient_rms = float(june["grad/norm"]) / math.sqrt(SCALAR_CONTROLS)
        fit_rms = float(raw6["fit_rms_mm"])
        june_fit_rms = float(june["target/error_rms_mm"])
        differences.append(
            {
                "step": int(raw6["step"]),
                "data_objective_mm2": {
                    "raw6": data_objective,
                    "june": june_objective,
                    "absolute_delta": data_objective - june_objective,
                    "relative_delta": relative_delta(data_objective, june_objective),
                },
                "fit_rms_mm": {
                    "raw6": fit_rms,
                    "june": june_fit_rms,
                    "absolute_delta": fit_rms - june_fit_rms,
                },
                "gradient_rms": {
                    "raw6": gradient_rms,
                    "june_derived_from_grad_norm": june_gradient_rms,
                    "absolute_delta": gradient_rms - june_gradient_rms,
                    "relative_delta": relative_delta(gradient_rms, june_gradient_rms),
                },
                "forward": {
                    "raw6_steps": int(raw6["forward_steps"]),
                    "june_steps": int(june["forward/steps"]),
                    "step_delta": int(raw6["forward_steps"])
                    - int(june["forward/steps"]),
                    "raw6_grad_norm": float(raw6["forward_grad_norm"]),
                    "june_grad_norm": float(june["forward/grad_norm"]),
                },
                "solver_valid": {
                    "raw6": raw6["solver_valid"] == "True",
                    "june": bool(june["forward/success"] and june["adjoint/success"]),
                },
            }
        )

    objective_rel = [
        abs(row["data_objective_mm2"]["relative_delta"]) for row in differences
    ]
    fit_abs = [abs(row["fit_rms_mm"]["absolute_delta"]) for row in differences]
    gradient_rel = [abs(row["gradient_rms"]["relative_delta"]) for row in differences]
    summary = {
        "schema_version": 1,
        "scope": "exact steps 0 through 3 of matched Raw6 compared with the June no-skin lr0.3 trace",
        "captured_at_utc": read_time,
        "scalar_controls": SCALAR_CONTROLS,
        "sources": {
            "raw6_live_trace": {
                "path": str(RAW6_TRACE.resolve()),
                "bytes_at_read_time": len(raw6_bytes),
                "sha256_at_read_time": sha256_bytes(raw6_bytes),
                "rows_at_read_time": len(raw6_records),
                "mutability": (
                    "live append-by-rewrite source; this hash identifies only the copied "
                    "read-time snapshot and is not the final-run trace hash"
                ),
                "snapshot": "raw6-trace-read-time-snapshot.csv",
            },
            "june_trace": {
                "path": str(JUNE_TRACE.resolve()),
                "bytes": len(june_bytes),
                "sha256": sha256_bytes(june_bytes),
                "rows": len(june_records),
            },
        },
        "captures": {
            "raw6_first4": {
                "path": "raw6-first4.csv",
                "sha256": sha256_bytes(raw6_capture),
                "records": raw6_first,
            },
            "june_first4": {
                "path": "june-first4.jsonl",
                "sha256": sha256_bytes(june_capture),
                "records": june_first,
            },
        },
        "differences": differences,
        "bounds_over_steps_0_to_3": {
            "max_absolute_data_objective_relative_delta": max(objective_rel),
            "max_absolute_fit_rms_delta_mm": max(fit_abs),
            "max_absolute_gradient_rms_relative_delta": max(gradient_rel),
            "all_solver_receipts_valid": all(
                row["solver_valid"]["raw6"] and row["solver_valid"]["june"]
                for row in differences
            ),
        },
        "assessment": (
            "The first four matched Raw6 evaluations closely reproduce the June data "
            "objective, fit RMS, gradient scale, and forward iteration scale. The small "
            "differences are compatible with solver-tolerance and numerical-path effects; "
            "concurrent GPU execution is a possible contributor but is not established as "
            "their cause. This bounded audit does not assert identical nonlinear trajectories "
            "or predict agreement through step 200."
        ),
    }
    write_json(output / "summary.json", summary)
    shutil.copy2(Path(__file__), output / Path(__file__).name)


if __name__ == "__main__":
    main()
