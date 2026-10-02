from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deadline_interruption import (
    _process_identity_absent,
    create_deadline_interruption,
    verify_deadline_interruption,
)

CUTOFF = "2026-09-30T05:55:00+00:00"


def make_run(root: Path, *, mismatch_iteration: bool = False) -> Path:
    run = root / "run"
    run.mkdir()
    pid = os.getpid() + 10_000_000
    (run / "job.json").write_text(
        json.dumps(
            {
                "pid": pid,
                "process_start_ticks": 9,
                "boot_id": "boot",
                "tool_session_id": 7,
                "command": ["python", "runner.py"],
                "working_directory": "/work",
                "computation_cutoff_utc": CUTOFF,
            }
        )
    )
    (run / "protocol.json").write_text(
        json.dumps(
            {
                "config": {"computation_cutoff_utc": CUTOFF},
                "computation_deadline": {
                    "computation_cutoff_utc": CUTOFF,
                    "normalized_cutoff_utc": CUTOFF,
                },
            }
        )
    )
    q = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    normalized_pose = np.asarray([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
    pose = normalized_pose * np.asarray([np.pi / 18] * 3 + [0.01] * 3)
    displacement = np.asarray([[1.0, 2.0, 3.0]])
    np.savez_compressed(
        run / "endpoint.npz",
        activation_inv=q,
        pose_rad_m=pose,
        displacement_m=displacement,
    )
    endpoint_hash = (
        __import__("hashlib").sha256((run / "endpoint.npz").read_bytes()).hexdigest()
    )
    endpoint = {"path": str((run / "endpoint.npz").resolve()), "sha256": endpoint_hash}
    last = {"iteration": 3, "optimizer_steps": {"q": 11, "pose": 11}, "value": 1}
    (run / "progress.jsonl").write_text(json.dumps(last) + "\n")
    summary = {
        "status": "running",
        "inverse_converged": False,
        "endpoint": endpoint,
        "final": {**last, "iteration": 4 if mismatch_iteration else 3},
    }
    (run / "summary.json").write_text(json.dumps(summary))
    torch.save(
        {
            "iteration": 3,
            "optimizer_steps": {"q": 11, "pose": 11},
            "activation_inv": torch.as_tensor(q),
            "pose_normalized": torch.as_tensor(normalized_pose),
            "pose_rad_m": torch.as_tensor(pose),
            "displacement_m": torch.as_tensor(displacement),
        },
        run / "checkpoint.pt",
    )
    return run


def test_valid_receipt_and_verification() -> None:
    with tempfile.TemporaryDirectory() as directory:
        run = make_run(Path(directory))
        receipt = create_deadline_interruption(
            run, observed_stop_utc=CUTOFF, cherries_shutdown_verified=True
        )
        assert (
            verify_deadline_interruption(run, receipt)["durable_state"][
                "last_durable_iteration"
            ]
            == 3
        )


def test_mutated_bound_input_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        run = make_run(Path(directory))
        receipt = create_deadline_interruption(
            run, observed_stop_utc=CUTOFF, cherries_shutdown_verified=True
        )
        (run / "progress.jsonl").write_text('{"mutation":true}\n')
        with pytest.raises(AssertionError):
            verify_deadline_interruption(run, receipt)


def test_mismatched_durable_iteration_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        run = make_run(Path(directory), mismatch_iteration=True)
        with pytest.raises(AssertionError):
            create_deadline_interruption(
                run, observed_stop_utc=CUTOFF, cherries_shutdown_verified=True
            )


def test_late_stop_is_recorded_without_permitting_resumption() -> None:
    with tempfile.TemporaryDirectory() as directory:
        run = make_run(Path(directory))
        receipt = create_deadline_interruption(
            run,
            observed_stop_utc="2026-09-30T06:00:00+00:00",
            cherries_shutdown_verified=True,
        )
        verified = verify_deadline_interruption(run, receipt)
        assert verified["deadline"]["deadline_met"] is False
        assert verified["deadline"]["seconds_after_cutoff"] == 300.0
        assert verified["deadline"]["seconds_after_user_deadline"] == 0.0
        assert verified["shutdown"]["resume_permitted"] is False


def test_reused_pid_does_not_block_absence_of_saved_identity() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "proc"
        process = root / "77"
        process.mkdir(parents=True)
        boot = Path(directory) / "boot_id"
        boot.write_text("same-boot\n")
        # Field 22 is index 19 after the state field.
        (process / "stat").write_text("77 (reused) S " + " ".join(["0"] * 18 + ["44"]))
        job = {"pid": 77, "process_start_ticks": 33, "boot_id": "same-boot"}
        _process_identity_absent(job, proc_root=root, boot_id_path=boot)
        job["process_start_ticks"] = 44
        with pytest.raises(AssertionError):
            _process_identity_absent(job, proc_root=root, boot_id_path=boot)
