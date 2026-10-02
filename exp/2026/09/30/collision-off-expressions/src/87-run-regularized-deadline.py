# Copyright (c) 2026 liblaf
"""Run a fair, serial regularized screen before the 06:00 UTC deadline.

The driver calibrates each expression from its fixed audited parent, then runs
at most one positive-smoothness fit per expression.  It never chains a new
candidate from a screen result.  The run summary explicitly records that this
is a deadline allocation and calibration screen, not a convergence result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GROUP = Path(__file__).resolve().parent.parent
FIT = GROUP / "src/86-fit-regularized.py"
AUDIT = GROUP / "src/90-audit-regularized.py"
OBJECTIVE = GROUP / "src/face_shape_activation_objective.py"
PREFLIGHT = GROUP / "data/face-shape-activation-objective-cpu-preflight-001.json"
FIT_SHA256 = "ec8e973cfa434057ea6287a4958d6c7b381c5219cc70aea379a181447ace2adf"
OBJECTIVE_SHA256 = "9366c6b6da00ed0355200c37c4de6e206c2f2ba9b70c550a33a834769531f9b3"
AUDIT_SHA256 = "9b3caf6eb7bb9ce8c8c75b778ab933ecce7bf9896aa90151de2d855c79ed7b4b"
PREFLIGHT_SHA256 = "348f0672697806a8d32b1a9ddc59d3c811df0e573ef418e5c60f90fa348a1e88"
FIT_STOP_UTC = dt.datetime(2026, 9, 30, 5, 45, tzinfo=dt.UTC)
HARD_STOP_UTC = dt.datetime(2026, 9, 30, 6, 0, tzinfo=dt.UTC)
# Leave two minutes for the owned-child termination grace before the hard stop.
CHILD_STOP_UTC = HARD_STOP_UTC - dt.timedelta(minutes=2)


@dataclass(frozen=True)
class Expression:
    name: str
    parent_kind: str
    parent: str
    refinement_archive: str | None
    initial_alpha: float


EXPRESSIONS = (
    Expression(
        "MouthOpen", "strict_mouthopen", "mouthopen-strict-continuation-001", None, 0.25
    ),
    Expression(
        "Smile",
        "smile_refined",
        "smile-004",
        "smile-baseline-resolution-001/refined-baseline.npz",
        0.04,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def normal_weight(expression: Expression) -> float:
    """Use the exact CPU-preflight normal anchor, never a rounded literal."""
    preflight = json.loads(PREFLIGHT.read_text())
    assert sha256(PREFLIGHT) == PREFLIGHT_SHA256
    assert preflight["source_sha256"] == OBJECTIVE_SHA256
    value = float(
        preflight["expressions"][expression.name]["normal_anchor"]["normal_coefficient"]
    )
    assert value > 0
    return value


def gpu_empty() -> None:
    result = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    assert not result, result


def seconds_until(when: dt.datetime, at: dt.datetime) -> int:
    assert when.tzinfo is not None
    assert at.tzinfo is not None
    return max(0, int((when - at).total_seconds()))


def worker_command(
    expression: Expression,
    output: Path,
    *,
    candidate_id: str,
    smooth_weight: float,
    calibration: bool,
    wall_seconds: int,
) -> list[str]:
    assert sha256(FIT) == FIT_SHA256
    assert sha256(OBJECTIVE) == OBJECTIVE_SHA256
    assert sha256(AUDIT) == AUDIT_SHA256
    assert sha256(PREFLIGHT) == PREFLIGHT_SHA256
    assert smooth_weight > 0 or calibration
    assert wall_seconds > 0
    command = [
        sys.executable,
        "-u",
        str(FIT),
        "--fit-worker",
        "--pilot-source-sha256",
        FIT_SHA256,
        "--expression-name",
        expression.name,
        "--parent-kind",
        expression.parent_kind,
        "--output-dir",
        str(output),
        "--initialization-checkpoint",
        str(GROUP / "data" / expression.parent / "checkpoint.pt"),
        "--continue-optimizer-state",
        "true",
        "--candidate-id",
        candidate_id,
        "--objective-epoch",
        "regularized",
        "--normal-weight",
        repr(normal_weight(expression)),
        "--smooth-weight",
        str(smooth_weight),
        "--maximum-iterations",
        "25",
        "--forward-atol",
        "1e-12",
        "--adjoint-relative-shift",
        "0",
        "--predictor-relative-shift",
        "0",
        "--predictor-rtol",
        "1e-7",
        "--adjoint-rtol",
        "1e-7",
        "--initial-trial-alpha",
        str(expression.initial_alpha),
        "--minimum-trial-alpha",
        "0.001",
        "--wall-seconds",
        str(wall_seconds),
        "--deadline-unix-s",
        str(CHILD_STOP_UTC.timestamp()),
    ]
    if expression.refinement_archive is not None:
        command.extend(
            (
                "--refinement-archive",
                str(GROUP / "data" / expression.refinement_archive),
            )
        )
    if calibration:
        command.extend(("--calibration-only", "true", "--calibration-control", "true"))
    return command


def verify_calibration(expression: Expression, output: Path) -> float:
    """Return the positive calibrated eta after checking the zero-update contract."""
    artifact = output / "objective-calibration.json"
    summary_path = output / "summary.json"
    protocol_path = output / "protocol.json"
    assert artifact.is_file(), artifact
    assert summary_path.is_file(), summary_path
    assert protocol_path.is_file(), protocol_path
    calibration = json.loads(artifact.read_text())
    summary = json.loads(summary_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    beta = normal_weight(expression)
    assert summary["status"] == "calibration_complete_no_optimizer_updates"
    assert summary["optimizer_updates"] == 0
    assert calibration["optimizer_updates"] == 0
    assert calibration["parameters_unchanged"] is True
    assert calibration["normal_coefficient"] == beta
    eta = float(calibration["smooth_coefficient"])
    assert eta > 0
    assert (
        protocol["objective"]["kind"]
        == "l2_position_plus_oriented_normal_plus_activation_smoothness"
    )
    assert protocol["objective"]["weights"] == {"normal": beta, "smooth": 0.0}
    assert protocol["objective_epoch"] == "regularized"
    return eta


def fit_budget(at: dt.datetime, remaining_positive: int) -> int:
    """Reserve ten minutes per remaining audit and split remaining fit time fairly."""
    assert remaining_positive > 0
    audit_reserve = 600 * remaining_positive
    available = seconds_until(FIT_STOP_UTC, at) - audit_reserve
    return max(0, min(3600, available // remaining_positive))


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(
    command: list[str],
    log: Path,
    *,
    deadline: dt.datetime,
    receipt_path: Path,
    receipt: dict[str, Any],
    result: dict[str, Any],
    label: str,
) -> int:
    deadline = min(deadline, CHILD_STOP_UTC)
    remaining = seconds_until(deadline, now())
    assert remaining > 0, "deadline reached before subprocess start"
    assert sha256(FIT) == FIT_SHA256
    assert sha256(OBJECTIVE) == OBJECTIVE_SHA256
    assert sha256(AUDIT) == AUDIT_SHA256
    assert sha256(PREFLIGHT) == PREFLIGHT_SHA256
    gpu_empty()
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="4",
        CUDA_VISIBLE_DEVICES="0",
        CHERRIES_NAME=f"{label} regularized deadline screen",
        CHERRIES_TAGS=f"collision-off,regularized,deadline,{label.lower()}",
    )
    with log.open("x") as stream:
        process = subprocess.Popen(
            command,
            cwd=GROUP,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        result["active"] = {
            "pid": process.pid,
            "command": command,
            "log": str(log),
            "started_at": now().isoformat(),
            "deadline": deadline.isoformat(),
        }
        write_json(receipt_path, receipt)
        try:
            exit_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            # This process is owned by this driver. Preserve accepted checkpoints and logs.
            result["timeout_cleanup"] = "SIGTERM sent to owned child before hard cutoff"
            process.terminate()
            try:
                exit_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                result["timeout_cleanup"] = (
                    "SIGKILL sent after owned-child SIGTERM grace"
                )
                process.kill()
                exit_code = process.wait(timeout=30)
        gpu_empty()
        result["active"]["ended_at"] = now().isoformat()
        result["active"]["exit_code"] = exit_code
        return exit_code


def main() -> None:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audited_parent, output = args.run_dir.resolve(), args.output_dir.resolve()
    assert audited_parent == GROUP / "data/mouthopen-strict-continuation-001"
    assert output.is_relative_to(GROUP / "data")
    assert not output.exists(), output
    assert now() < HARD_STOP_UTC
    assert (audited_parent / "independent-audit.json").is_file()
    assert PREFLIGHT.is_file()
    assert sha256(FIT) == FIT_SHA256
    assert sha256(OBJECTIVE) == OBJECTIVE_SHA256
    assert sha256(AUDIT) == AUDIT_SHA256
    assert sha256(PREFLIGHT) == PREFLIGHT_SHA256
    output.mkdir(parents=True)
    (output / "sources").mkdir(exist_ok=True)
    for source in (Path(__file__), FIT, AUDIT, OBJECTIVE, PREFLIGHT):
        destination = output / "sources" / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
        assert sha256(destination) == sha256(source)
    receipt: dict[str, Any] = {
        "schema": "collision-off-regularized-deadline-driver-v1",
        "status": "running",
        "started_at": now().isoformat(),
        "fit_stop_utc": FIT_STOP_UTC.isoformat(),
        "hard_stop_utc": HARD_STOP_UTC.isoformat(),
        "owned_child_stop_utc": CHILD_STOP_UTC.isoformat(),
        "policy": "deadline allocation, calibration screen incomplete; no convergence claim",
        "fixed_parents": {
            item.name: record(GROUP / "data" / item.parent / "checkpoint.pt")
            for item in EXPRESSIONS
        },
        "sources": {
            source.name: record(source)
            for source in (Path(__file__), FIT, AUDIT, OBJECTIVE, PREFLIGHT)
        },
        "expected_source_sha256": {
            "86-fit-regularized.py": FIT_SHA256,
            "90-audit-regularized.py": AUDIT_SHA256,
            "face_shape_activation_objective.py": OBJECTIVE_SHA256,
            "face-shape-activation-objective-cpu-preflight-001.json": PREFLIGHT_SHA256,
        },
        "normal_weights": {item.name: normal_weight(item) for item in EXPRESSIONS},
        "runs": [],
    }
    write_json(output / "driver.json", receipt)
    calibrations: dict[str, float] = {}
    for item in EXPRESSIONS:
        if now() >= FIT_STOP_UTC:
            receipt["runs"].append(
                {
                    "kind": "calibration_control",
                    "expression": item.name,
                    "status": "skipped",
                    "reason": "fit deadline reached",
                }
            )
            write_json(output / "driver.json", receipt)
            continue
        calibration_dir = output / f"calibration-{item.name.lower()}"
        wall_seconds = min(900, seconds_until(FIT_STOP_UTC, now()))
        result: dict[str, Any] = {
            "kind": "calibration_control",
            "expression": item.name,
            "status": "starting",
            "output": str(calibration_dir),
            "allocated_wall_seconds": wall_seconds,
        }
        receipt["runs"].append(result)
        write_json(output / "driver.json", receipt)
        try:
            command = worker_command(
                item,
                calibration_dir,
                candidate_id=f"deadline-calibration-{item.name.lower()}",
                smooth_weight=0.0,
                calibration=True,
                wall_seconds=wall_seconds,
            )
            result["command"] = command
            result["exit_code"] = run(
                command,
                output / f"calibration-{item.name.lower()}.log",
                deadline=min(
                    FIT_STOP_UTC, now() + dt.timedelta(seconds=wall_seconds + 60)
                ),
                receipt_path=output / "driver.json",
                receipt=receipt,
                result=result,
                label=f"{item.name} calibration",
            )
            assert result["exit_code"] == 0
            eta = verify_calibration(item, calibration_dir)
            calibrations[item.name] = eta
            result["calibration"] = record(
                calibration_dir / "objective-calibration.json"
            )
            result["selected_positive_smooth_weight"] = eta
            result["status"] = "calibration_complete"
        except Exception as error:  # noqa: BLE001 - isolate one expression's screen
            result["status"] = "calibration_failed"
            result["failure"] = {"type": type(error).__name__, "message": str(error)}
            result["artifact_present"] = (
                calibration_dir / "objective-calibration.json"
            ).is_file()
        write_json(output / "driver.json", receipt)
    for index, item in enumerate(EXPRESSIONS):
        if item.name not in calibrations:
            receipt["runs"].append(
                {
                    "kind": "positive_smooth_fit",
                    "expression": item.name,
                    "status": "skipped",
                    "reason": "calibration unavailable",
                }
            )
            write_json(output / "driver.json", receipt)
            continue
        if now() >= FIT_STOP_UTC:
            receipt["runs"].append(
                {
                    "kind": "positive_smooth_fit",
                    "expression": item.name,
                    "status": "skipped",
                    "reason": "fit deadline reached",
                }
            )
            write_json(output / "driver.json", receipt)
            continue
        remaining = sum(
            candidate.name in calibrations for candidate in EXPRESSIONS[index:]
        )
        budget = fit_budget(now(), remaining)
        if budget <= 0:
            receipt["runs"].append(
                {
                    "kind": "positive_smooth_fit",
                    "expression": item.name,
                    "status": "skipped",
                    "reason": "no fair positive-fit budget remains",
                }
            )
            write_json(output / "driver.json", receipt)
            continue
        candidate = output / f"fit-{item.name.lower()}"
        result = {
            "kind": "positive_smooth_fit",
            "expression": item.name,
            "status": "starting",
            "output": str(candidate),
            "allocated_fit_seconds": budget,
            "smooth_weight": calibrations[item.name],
        }
        receipt["runs"].append(result)
        write_json(output / "driver.json", receipt)
        try:
            command = worker_command(
                item,
                candidate,
                candidate_id=f"deadline-positive-{item.name.lower()}",
                smooth_weight=calibrations[item.name],
                calibration=False,
                wall_seconds=budget,
            )
            result["command"] = command
            result["exit_code"] = run(
                command,
                output / f"fit-{item.name.lower()}.log",
                deadline=min(FIT_STOP_UTC, now() + dt.timedelta(seconds=budget + 60)),
                receipt_path=output / "driver.json",
                receipt=receipt,
                result=result,
                label=f"{item.name} positive smooth",
            )
            if (candidate / "endpoint.npz").is_file() and (
                candidate / "summary.json"
            ).is_file():
                audit = [sys.executable, "-u", str(AUDIT), "--run-dir", str(candidate)]
                result["audit_command"] = audit
                result["audit_exit_code"] = run(
                    audit,
                    output / f"audit-{item.name.lower()}.log",
                    deadline=CHILD_STOP_UTC,
                    receipt_path=output / "driver.json",
                    receipt=receipt,
                    result=result,
                    label=f"{item.name} audit",
                )
                artifact = candidate / "independent-audit.json"
                if artifact.is_file():
                    result["audit"] = record(artifact)
            result["status"] = (
                "fit_and_audit_complete"
                if result.get("exit_code") == 0
                and result.get("audit_exit_code") == 0
                and (candidate / "independent-audit.json").is_file()
                and json.loads((candidate / "independent-audit.json").read_text())[
                    "valid_forward"
                ]
                else "fit_or_audit_partial"
            )
        except Exception as error:  # noqa: BLE001 - continue with the other expression
            result["status"] = "fit_or_audit_failed"
            result["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_json(output / "driver.json", receipt)
    receipt["ended_at"] = now().isoformat()
    complete = all(
        any(
            run.get("kind") == "calibration_control"
            and run.get("expression") == item.name
            and run.get("status") == "calibration_complete"
            for run in receipt["runs"]
        )
        and any(
            run.get("kind") == "positive_smooth_fit"
            and run.get("expression") == item.name
            and run.get("status") == "fit_and_audit_complete"
            for run in receipt["runs"]
        )
        for item in EXPRESSIONS
    )
    receipt["status"] = (
        "deadline_screen_complete" if complete else "deadline_screen_complete_partial"
    )
    write_json(output / "driver.json", receipt)
    write_json(
        output / "summary.json",
        {
            "schema": receipt["schema"],
            "status": receipt["status"],
            "driver": record(output / "driver.json"),
            "policy": receipt["policy"],
            "inverse_converged": False,
        },
    )


if __name__ == "__main__":
    main()
