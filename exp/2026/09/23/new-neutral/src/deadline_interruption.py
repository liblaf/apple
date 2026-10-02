"""Create and verify immutable receipt for a deadline-cutoff interruption.

After the numerical PID and its Cherries session have both exited, invoke:
``uv run python src/deadline_interruption.py --run-dir data/inverse-mouthopen-coupled-018 --observed-stop-utc <UTC-ISO-8601> --cherries-shutdown-verified``.
The observed time must be at or after the configured cutoff. Stops during its
five-minute grace window record ``deadline_met: true``; later stops remain
auditable with ``deadline_met: false`` and never permit resumption. The command
refuses any changed endpoint, checkpoint, summary, or progress input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "new-neutral-deadline-interruption-v1"
BINDING_NAMES = (
    "job.json",
    "protocol.json",
    "summary.json",
    "progress.jsonl",
    "endpoint.npz",
    "checkpoint.pt",
)
JOB_IDENTITY_KEYS = (
    "pid",
    "process_start_ticks",
    "boot_id",
    "tool_session_id",
    "command",
    "working_directory",
    "computation_cutoff_utc",
)


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None, value
    return parsed.astimezone(UTC)


def _durable_state(run: Path) -> tuple[dict, dict, dict]:
    import torch

    summary = json.loads((run / "summary.json").read_text())
    progress = (run / "progress.jsonl").read_text().splitlines()
    assert progress
    last = json.loads(progress[-1])
    checkpoint = torch.load(
        run / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    assert summary["status"] == "running"
    assert summary["inverse_converged"] is False
    assert summary["final"] == last
    assert int(last["iteration"]) == int(checkpoint["iteration"])
    assert last["optimizer_steps"] == checkpoint["optimizer_steps"]
    with np.load(run / "endpoint.npz", allow_pickle=False) as endpoint:
        checkpoint_q = checkpoint["activation_inv"].cpu().numpy()
        checkpoint_pose = checkpoint["pose_rad_m"].cpu().numpy()
        checkpoint_u = checkpoint["displacement_m"].cpu().numpy()
        checkpoint_normalized_pose = checkpoint["pose_normalized"].cpu().numpy()
        np.testing.assert_array_equal(endpoint["activation_inv"], checkpoint_q)
        np.testing.assert_array_equal(endpoint["pose_rad_m"], checkpoint_pose)
        np.testing.assert_array_equal(endpoint["displacement_m"], checkpoint_u)
        pose_scale = np.asarray([np.pi / 18] * 3 + [0.01] * 3)
        np.testing.assert_array_equal(
            checkpoint_normalized_pose * pose_scale, checkpoint_pose
        )
    return summary, last, checkpoint


def _process_identity_absent(
    job: dict,
    *,
    proc_root: Path = Path("/proc"),
    boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
) -> None:
    """Accept PID reuse, but reject the still-live saved process identity."""
    process = proc_root / str(int(job["pid"]))
    if not process.exists():
        return
    stat = (process / "stat").read_text()
    _, suffix = stat.rsplit(")", 1)
    fields = suffix.split()
    start_ticks = int(fields[19])  # /proc/<pid>/stat field 22, after state field 3.
    boot_id = boot_id_path.read_text().strip()
    assert not (
        start_ticks == int(job["process_start_ticks"])
        and boot_id == str(job["boot_id"])
    ), "saved numerical process identity is still live"


def _cutoff(protocol: dict, job: dict) -> datetime:
    configured = protocol["config"]["computation_cutoff_utc"]
    deadline = protocol["computation_deadline"]
    assert deadline["computation_cutoff_utc"] == configured
    assert deadline["normalized_cutoff_utc"] == _utc(configured).isoformat()
    assert job["computation_cutoff_utc"] == configured
    return _utc(configured)


def create_deadline_interruption(
    run_dir: Path,
    *,
    observed_stop_utc: str,
    cherries_shutdown_verified: bool,
    output_name: str = "deadline-interruption.json",
) -> Path:
    """Write the one immutable receipt accepted for a running cutoff endpoint."""
    run = run_dir.resolve()
    output = run / output_name
    assert not output.exists(), output
    assert cherries_shutdown_verified
    protocol_path, job_path = run / "protocol.json", run / "job.json"
    protocol, job = (
        json.loads(protocol_path.read_text()),
        json.loads(job_path.read_text()),
    )
    cutoff, observed = _cutoff(protocol, job), _utc(observed_stop_utc)
    assert cutoff <= observed
    _process_identity_absent(job)
    deadline_met = observed < cutoff + timedelta(minutes=5)
    summary, last, checkpoint = _durable_state(run)
    endpoint = run / "endpoint.npz"
    assert summary["endpoint"] == record(endpoint)
    bindings = {name: record(run / name) for name in BINDING_NAMES}
    receipt = {
        "schema": SCHEMA,
        "kind": "computation_cutoff_interruption",
        "run": str(run),
        "bindings": bindings,
        "job_identity": {key: job[key] for key in JOB_IDENTITY_KEYS},
        "cutoff_utc": cutoff.isoformat(),
        "observed_stop_utc": observed.isoformat(),
        "deadline": {
            "deadline_met": deadline_met,
            "grace_window_seconds": 300,
            "seconds_after_cutoff": (observed - cutoff).total_seconds(),
            "seconds_after_user_deadline": max(
                0.0, (observed - cutoff - timedelta(minutes=5)).total_seconds()
            ),
        },
        "shutdown": {
            "numerical_process_identity_absent_verified": True,
            "cherries_shutdown_verified": True,
            "resume_permitted": False,
        },
        "durable_state": {
            "summary_status": summary["status"],
            "inverse_converged": summary["inverse_converged"],
            "last_durable_iteration": int(last["iteration"]),
            "optimizer_steps": last["optimizer_steps"],
            "checkpoint_iteration": int(checkpoint["iteration"]),
            "checkpoint_optimizer_steps": checkpoint["optimizer_steps"],
        },
    }
    temporary = output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(output)
    return output


def verify_deadline_interruption(run_dir: Path, receipt_path: Path) -> dict[str, Any]:
    """Fail unless receipt and all frozen run inputs still agree exactly."""
    run = run_dir.resolve()
    receipt = json.loads(receipt_path.read_text())
    assert receipt["schema"] == SCHEMA
    assert receipt["kind"] == "computation_cutoff_interruption"
    assert receipt["run"] == str(run)
    assert receipt["shutdown"] == {
        "numerical_process_identity_absent_verified": True,
        "cherries_shutdown_verified": True,
        "resume_permitted": False,
    }
    assert set(receipt["bindings"]) == set(BINDING_NAMES)
    for name in BINDING_NAMES:
        assert record(run / name) == receipt["bindings"][name]
    protocol = json.loads((run / "protocol.json").read_text())
    job = json.loads((run / "job.json").read_text())
    cutoff, observed = _cutoff(protocol, job), _utc(receipt["observed_stop_utc"])
    assert receipt["cutoff_utc"] == cutoff.isoformat()
    assert cutoff <= observed
    assert receipt["deadline"] == {
        "deadline_met": observed < cutoff + timedelta(minutes=5),
        "grace_window_seconds": 300,
        "seconds_after_cutoff": (observed - cutoff).total_seconds(),
        "seconds_after_user_deadline": max(
            0.0, (observed - cutoff - timedelta(minutes=5)).total_seconds()
        ),
    }
    expected_identity = {key: job[key] for key in JOB_IDENTITY_KEYS}
    assert receipt["job_identity"] == expected_identity
    _process_identity_absent(job)
    summary, last, checkpoint = _durable_state(run)
    assert summary["endpoint"] == record(run / "endpoint.npz")
    assert receipt["durable_state"] == {
        "summary_status": "running",
        "inverse_converged": False,
        "last_durable_iteration": int(last["iteration"]),
        "optimizer_steps": last["optimizer_steps"],
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "checkpoint_optimizer_steps": checkpoint["optimizer_steps"],
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--observed-stop-utc", required=True)
    parser.add_argument("--cherries-shutdown-verified", action="store_true")
    args = parser.parse_args()
    output = create_deadline_interruption(
        args.run_dir,
        observed_stop_utc=args.observed_stop_utc,
        cherries_shutdown_verified=args.cherries_shutdown_verified,
    )
    print(output)


if __name__ == "__main__":
    main()
