# ruff: noqa: PT009, PT027
"""CPU-only tests for the hash-bound Smile disk-full parent clipping contract."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

GROUP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GROUP / "src"))

from smile_recovery_lineage import (  # noqa: E402
    certified_parent_rows,
    verified_recovery_lineage,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SmileRecoveryLineage(unittest.TestCase):
    def make_fixture(self) -> tuple[Path, Path, Path]:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        parent = root / "inverse-smile-coupled-003"
        parent.mkdir()
        rows = [
            {"iteration": index, "optimizer_steps": {"q": index, "pose": index}}
            for index in range(34)
        ]
        (parent / "progress.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        (parent / "endpoint.npz").write_bytes(b"endpoint")
        (parent / "checkpoint.pt").write_bytes(b"checkpoint")
        terms = {"mode": "l2-normal-smooth"}
        (parent / "protocol.json").write_text(
            json.dumps({"objective": "same", "objective_terms": terms})
        )
        (parent / "objective-terms.json").write_text(json.dumps(terms))
        (parent / "summary.json").write_text(
            json.dumps(
                {
                    "endpoint": {"sha256": digest(parent / "endpoint.npz")},
                    "final": rows[32],
                }
            )
        )
        audit = root / "recovery003-independent-audit.json"
        audit.write_text(
            json.dumps(
                {
                    "schema": "expression-coupled-independent-audit-v1",
                    "expression_name": "Smile",
                    "valid_forward": True,
                    "inputs": {
                        key: {"sha256": digest(parent / name)}
                        for name, key in (
                            ("endpoint.npz", "endpoint"),
                            ("protocol.json", "protocol"),
                            ("summary.json", "summary"),
                        )
                    },
                }
            )
        )
        preflight = root / "preflight.json"
        preflight.write_text(
            json.dumps(
                {
                    "schema": "smile-003-disk-full-continuation-preflight-v1",
                    "parent": "/remote/data/inverse-smile-coupled-003",
                    "parent_files_sha256": {
                        **{
                            name: digest(parent / name)
                            for name in (
                                "endpoint.npz",
                                "protocol.json",
                                "summary.json",
                                "progress.jsonl",
                                "checkpoint.pt",
                            )
                        },
                        "independent_audit.json": digest(audit),
                    },
                    "last_committed_iteration": 32,
                    "optimizer_steps": {"q": 32, "pose": 32},
                    "last_progress_iteration": 33,
                    "uncheckpointed_progress_rows": [rows[33]],
                    "uncheckpointed_rows_excluded_from_restart": True,
                }
            )
        )
        return preflight, parent, audit

    def tearDown(self) -> None:
        if hasattr(self, "temporary"):
            self.temporary.cleanup()

    def test_saved_remote_preflight_has_the_expected_contract(self) -> None:
        preflight = GROUP / (
            "data/remote-smile-recovery-sync-006/receipts/"
            "inverse-smile-coupled-006-recovery-control/preflight.json"
        )
        payload = json.loads(preflight.read_text())
        self.assertEqual(
            payload["schema"], "smile-003-disk-full-continuation-preflight-v1"
        )
        self.assertEqual(payload["last_committed_iteration"], 32)
        self.assertEqual(payload["last_progress_iteration"], 33)
        self.assertEqual(
            [row["iteration"] for row in payload["uncheckpointed_progress_rows"]], [33]
        )

    def test_clips_exactly_the_uncheckpointed_row(self) -> None:
        preflight, parent, audit = self.make_fixture()
        rows, receipt = certified_parent_rows(preflight, parent, audit)
        self.assertEqual([row["iteration"] for row in rows], list(range(33)))
        self.assertEqual(receipt["excluded_uncheckpointed_rows"], 1)
        self.assertEqual(receipt["raw_progress_rows"], 34)

    def test_metadata_requires_the_zero_update_boundary(self) -> None:
        preflight, parent, audit = self.make_fixture()
        child = Path(self.temporary.name) / "inverse-smile-coupled-006"
        child.mkdir()
        row = {
            "iteration": 0,
            "local_iteration": 0,
            "optimizer_steps": {"q": 32, "pose": 32},
        }
        terms = {"mode": "l2-normal-smooth"}
        for name, value in (
            ("protocol.json", {"objective": "same", "objective_terms": terms}),
            ("objective-terms.json", terms),
            ("summary.json", {"initial": row}),
        ):
            (child / name).write_text(json.dumps(value))
        (child / "progress.jsonl").write_text(json.dumps(row) + "\n")
        metadata = Path(self.temporary.name) / "lineage.json"
        records = [
            {"path": str(path.resolve()), "sha256": digest(path)}
            for path in (
                parent / "protocol.json",
                parent / "objective-terms.json",
                parent / "summary.json",
                parent / "progress.jsonl",
                preflight,
                audit,
                child / "protocol.json",
                child / "objective-terms.json",
                child / "summary.json",
                child / "progress.jsonl",
            )
        ]
        metadata.write_text(
            json.dumps(
                {
                    "schema": "smile-storage-recovery-lineage-v1",
                    "cuda_initialized": False,
                    "objective_continuity": {
                        "verified": True,
                        "classification": "same_objective_optimizer_continuation_after_storage_failure",
                        "saved_objective_change_label": "continued_objective_change",
                        "saved_label_is_mechanical_not_a_mathematical_change": True,
                    },
                    "lineage_clip": {
                        "parent_last_durable_iteration": 32,
                        "parent_kept_rows": list(range(33)),
                        "parent_uncheckpointed_rows": [
                            json.loads(
                                (parent / "progress.jsonl").read_text().splitlines()[-1]
                            )
                        ],
                        "parent_uncheckpointed_rows_on_trajectory": False,
                        "child_initial_local_iteration": 0,
                        "child_initial_optimizer_steps": {"q": 32, "pose": 32},
                        "child_initial_row_retained_as_zero_update_event": True,
                        "child_initial_displacement_identity_proven": False,
                        "curve_segment_break_at_zero_update": True,
                        "x_axis": "optimizer_step; local_iteration is branch-local",
                        "child_accepted_rows": [],
                    },
                    "input_files": {
                        str(index): value for index, value in enumerate(records)
                    },
                }
            )
        )
        rows, receipt = verified_recovery_lineage(
            metadata, preflight, parent, audit, child
        )
        self.assertEqual(len(rows), 33)
        self.assertTrue(
            receipt["lineage_clip"]["child_initial_row_retained_as_zero_update_event"]
        )

    def test_rejects_a_tampered_metadata_objective_input(self) -> None:
        preflight, parent, audit = self.make_fixture()
        child = Path(self.temporary.name) / "inverse-smile-coupled-006"
        child.mkdir()
        row = {
            "iteration": 0,
            "local_iteration": 0,
            "optimizer_steps": {"q": 32, "pose": 32},
        }
        terms = {"mode": "l2-normal-smooth"}
        for name, value in (
            ("protocol.json", {"objective": "same", "objective_terms": terms}),
            ("objective-terms.json", terms),
            ("summary.json", {"initial": row}),
        ):
            (child / name).write_text(json.dumps(value))
        (child / "progress.jsonl").write_text(json.dumps(row) + "\n")
        metadata = Path(self.temporary.name) / "lineage-tamper.json"
        records = [
            {"path": str(path.resolve()), "sha256": digest(path)}
            for path in (
                parent / "protocol.json",
                parent / "objective-terms.json",
                parent / "summary.json",
                parent / "progress.jsonl",
                preflight,
                audit,
                child / "protocol.json",
                child / "objective-terms.json",
                child / "summary.json",
                child / "progress.jsonl",
            )
        ]
        metadata.write_text(
            json.dumps(
                {
                    "schema": "smile-storage-recovery-lineage-v1",
                    "cuda_initialized": False,
                    "objective_continuity": {
                        "verified": True,
                        "classification": "same_objective_optimizer_continuation_after_storage_failure",
                        "saved_objective_change_label": "continued_objective_change",
                        "saved_label_is_mechanical_not_a_mathematical_change": True,
                    },
                    "lineage_clip": {
                        "parent_last_durable_iteration": 32,
                        "parent_kept_rows": list(range(33)),
                        "parent_uncheckpointed_rows": [
                            json.loads(
                                (parent / "progress.jsonl").read_text().splitlines()[-1]
                            )
                        ],
                        "parent_uncheckpointed_rows_on_trajectory": False,
                        "child_initial_local_iteration": 0,
                        "child_initial_optimizer_steps": {"q": 32, "pose": 32},
                        "child_initial_row_retained_as_zero_update_event": True,
                        "child_initial_displacement_identity_proven": False,
                        "curve_segment_break_at_zero_update": True,
                        "x_axis": "optimizer_step; local_iteration is branch-local",
                        "child_accepted_rows": [],
                    },
                    "input_files": {
                        str(index): value for index, value in enumerate(records)
                    },
                }
            )
        )
        (child / "objective-terms.json").write_text(json.dumps({"mode": "tampered"}))
        with self.assertRaises(AssertionError):
            verified_recovery_lineage(metadata, preflight, parent, audit, child)

    def test_actual_frozen_receipt_passes_and_current_receipts_fail(self) -> None:
        parent = (
            GROUP / "data/remote-smile-old003-lineage-bundle/inverse-smile-coupled-003"
        )
        preflight = (
            GROUP
            / "data/remote-smile-recovery-sync-006/receipts/inverse-smile-coupled-006-recovery-control/preflight.json"
        )
        audit = (
            GROUP
            / "data/remote-smile-recovery-sync-006/receipts/recovery003-independent-audit.json"
        )
        frozen = (
            GROUP
            / "data/recovery006-lineage-preflight-inputs/inverse-smile-coupled-006"
        )
        metadata = GROUP / "data/recovery006-lineage-preflight.json"
        rows, _ = verified_recovery_lineage(metadata, preflight, parent, audit, frozen)
        self.assertEqual(len(rows), 33)
        current = (
            GROUP
            / "data/remote-smile-recovery-sync-006/receipts/inverse-smile-coupled-006"
        )
        with self.assertRaises(AssertionError):
            verified_recovery_lineage(metadata, preflight, parent, audit, current)

    def test_rejects_a_changed_raw_parent_progress_row(self) -> None:
        preflight, parent, audit = self.make_fixture()
        with (parent / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps({"iteration": 34}) + "\n")
        with self.assertRaises(AssertionError):
            certified_parent_rows(preflight, parent, audit)


class TerminalInterruptionEvidence(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.mirror = self.root / "remote-smile-recovery-006"
        self.run = self.mirror / "inverse-smile-coupled-006"
        self.control = self.mirror / "remote-smile-control-006"
        self.run.mkdir(parents=True)
        self.control.mkdir()
        self.rows = [
            {
                "iteration": 0,
                "local_iteration": 0,
                "optimizer_step": 0,
                "optimizer_steps": {"q": 0, "pose": 0},
            },
            {
                "iteration": 1,
                "local_iteration": 1,
                "optimizer_step": 1,
                "optimizer_steps": {"q": 1, "pose": 1},
            },
        ]
        (self.run / "protocol.json").write_text("{}")
        displacement = np.asarray([[1.0, 2.0, 3.0]])
        activation = np.asarray([[0.25, 0.5]])
        pose = np.asarray([0.1, 0.2])
        np.savez(
            self.run / "endpoint.npz",
            displacement_m=displacement,
            activation_inv=activation,
            pose_rad_m=pose,
        )
        torch.save(
            {
                "displacement_m": torch.from_numpy(displacement),
                "activation_inv": torch.from_numpy(activation),
                "pose_rad_m": torch.from_numpy(pose),
                "iteration": 1,
                "local_iteration": 1,
                "optimizer_step": 1,
                "optimizer_steps": {"q": 1, "pose": 1},
            },
            self.run / "checkpoint.pt",
        )
        (self.run / "progress.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows)
        )
        (self.run / "summary.json").write_text(
            json.dumps({"status": "running", "final": self.rows[-1]})
        )
        (self.run / "independent-audit.json").write_text('{"valid_forward": true}')
        self.job = self.control / "job.json"

        def child(exit_code: int) -> dict:
            return {
                "command": ["python", "fit.py"],
                "identity": {"argv": ["python", "fit.py"]},
                "exit_code": exit_code,
                "ended_at_utc": "2026-09-30T05:50:00+00:00",
                "gpu_idle_wait_seconds": 1.0,
                "gpu_apps_after": [],
            }

        self.job.write_text(
            json.dumps(
                {
                    "schema": "collision-on-remote-smile-supervision-v1",
                    "status": "audited_partial_endpoint",
                    "run_dir": str(self.run),
                    "inverse_converged_claim": False,
                    "source_sha256": {
                        "3052-supervise-smile-config-recovery.py": "3d9f2ab21756c619e076fd85be489e425dc13fd11b8ed179bd5807a08c123256"
                    },
                    "deadlines_utc": {"fit": "a", "audit": "b", "delivery": "c"},
                    "interrupted_signals": [],
                    "fit": {
                        **child(1),
                        "deadline_cleanup": [
                            {"signal": "SIGINT", "at_utc": "2026-09-30T05:49:00+00:00"}
                        ],
                    },
                    "audit": child(0),
                    "saved": {
                        name: {
                            "path": str(self.run / name),
                            "sha256": digest(self.run / name),
                        }
                        for name in (
                            "protocol.json",
                            "summary.json",
                            "checkpoint.pt",
                            "endpoint.npz",
                            "progress.jsonl",
                        )
                    },
                    "independent_audit": {
                        "path": str(self.run / "independent-audit.json"),
                        "sha256": digest(self.run / "independent-audit.json"),
                    },
                }
            )
        )
        self.manifest = self.mirror / "sync-manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema": "remote-collision-on-smile-recovery-complete-bundle-v1",
                    "local_root": str(self.mirror),
                    "status": {
                        "terminal_and_gpu_idle": True,
                        "current_processes_all_exited": True,
                        "gpu_compute_apps": [],
                    },
                    "files_sha256": {
                        "remote-smile-control-006/job.json": digest(self.job)
                    },
                }
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def verify(self) -> dict:
        from smile_recovery_lineage import verified_terminal_interruption

        return verified_terminal_interruption(
            self.job,
            self.manifest,
            self.mirror,
            self.run,
            json.loads((self.run / "summary.json").read_text()),
            self.rows,
            self.run / "independent-audit.json",
        )

    def test_terminal_bundle_binds_partial_endpoint_and_signal_evidence(self) -> None:
        receipt = self.verify()
        self.assertEqual(receipt["terminal_status"], "audited_partial_endpoint")
        self.assertEqual(receipt["actual_signal_events"][0]["signal"], "SIGINT")
        self.assertIsNone(receipt["exact_numerical_stop_time_utc"])

    def test_terminal_bundle_rejects_changed_saved_input_or_active_job(self) -> None:
        (self.run / "endpoint.npz").write_bytes(b"changed")
        with self.assertRaises(AssertionError):
            self.verify()
        np.savez(
            self.run / "endpoint.npz",
            displacement_m=np.asarray([[1.0, 2.0, 3.0]]),
            activation_inv=np.asarray([[0.25, 0.5]]),
            pose_rad_m=np.asarray([0.1, 0.2]),
        )
        job = json.loads(self.job.read_text())
        job["saved"]["endpoint.npz"]["sha256"] = digest(self.run / "endpoint.npz")
        self.job.write_text(json.dumps(job))
        manifest = json.loads(self.manifest.read_text())
        manifest["files_sha256"]["remote-smile-control-006/job.json"] = digest(self.job)
        self.manifest.write_text(json.dumps(manifest))
        manifest = json.loads(self.manifest.read_text())
        manifest["status"]["terminal_and_gpu_idle"] = False
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaises(AssertionError):
            self.verify()

    def test_terminal_bundle_rejects_wrong_run(self) -> None:
        job = json.loads(self.job.read_text())
        job["run_dir"] = str(self.root / "other-run")
        self.job.write_text(json.dumps(job))
        manifest = json.loads(self.manifest.read_text())
        manifest["files_sha256"]["remote-smile-control-006/job.json"] = digest(self.job)
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaises(AssertionError):
            self.verify()


if __name__ == "__main__":
    unittest.main()
