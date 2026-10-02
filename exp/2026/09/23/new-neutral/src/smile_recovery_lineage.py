# Copyright (c) 2026 liblaf
"""Fail-closed verification for the Smile 003 disk-full recovery lineage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PREFLIGHT_SCHEMA = "smile-003-disk-full-continuation-preflight-v1"
PARENT_FILES = (
    "endpoint.npz",
    "protocol.json",
    "summary.json",
    "progress.jsonl",
    "checkpoint.pt",
)
AUDIT_KEY = "independent_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def certified_parent_rows(
    preflight_path: Path, parent_dir: Path, parent_audit_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify and return only parent rows committed through the durable checkpoint."""
    preflight_path, parent_dir, parent_audit_path = (
        preflight_path.resolve(),
        parent_dir.resolve(),
        parent_audit_path.resolve(),
    )
    preflight = json.loads(preflight_path.read_text())
    assert preflight["schema"] == PREFLIGHT_SCHEMA
    assert parent_dir.name == Path(preflight["parent"]).name
    bindings = preflight["parent_files_sha256"]
    assert set(PARENT_FILES).issubset(bindings)
    assert AUDIT_KEY in bindings
    for name in PARENT_FILES:
        assert sha256(parent_dir / name) == bindings[name], name
    assert sha256(parent_audit_path) == bindings[AUDIT_KEY]

    audit = json.loads(parent_audit_path.read_text())
    assert audit["schema"] == "expression-coupled-independent-audit-v1"
    assert audit["expression_name"] == "Smile"
    assert audit["valid_forward"] is True
    for name, key in (
        ("endpoint.npz", "endpoint"),
        ("protocol.json", "protocol"),
        ("summary.json", "summary"),
    ):
        assert audit["inputs"][key]["sha256"] == bindings[name]

    rows = [
        json.loads(line)
        for line in (parent_dir / "progress.jsonl").read_text().splitlines()
    ]
    assert rows
    durable = int(preflight["last_committed_iteration"])
    assert durable == 32
    assert preflight["optimizer_steps"] == {"q": durable, "pose": durable}
    assert [int(row["iteration"]) for row in rows] == list(range(len(rows)))
    assert int(preflight["last_progress_iteration"]) == int(rows[-1]["iteration"])
    committed = [row for row in rows if int(row["iteration"]) <= durable]
    uncheckpointed = [row for row in rows if int(row["iteration"]) > durable]
    assert len(committed) == durable + 1
    assert committed[-1]["iteration"] == durable
    assert uncheckpointed == preflight["uncheckpointed_progress_rows"]
    assert preflight["uncheckpointed_rows_excluded_from_restart"] is True
    assert len(uncheckpointed) == 1
    assert uncheckpointed[0]["iteration"] == durable + 1

    summary = json.loads((parent_dir / "summary.json").read_text())
    assert summary["endpoint"]["sha256"] == bindings["endpoint.npz"]
    assert summary["final"] == committed[-1]
    receipt = {
        "schema": "smile-recovery-certified-parent-lineage-v1",
        "preflight": record(preflight_path),
        "parent_files": {name: record(parent_dir / name) for name in PARENT_FILES},
        "parent_audit": record(parent_audit_path),
        "last_committed_iteration": durable,
        "certified_rows": len(committed),
        "raw_progress_rows": len(rows),
        "excluded_uncheckpointed_rows": len(uncheckpointed),
        "uncheckpointed_progress_rows": preflight["uncheckpointed_progress_rows"],
    }
    return committed, receipt


def verified_recovery_lineage(
    metadata_path: Path,
    preflight_path: Path,
    parent_dir: Path,
    parent_audit_path: Path,
    child_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind a final 3090 receipt and return its certified parent trajectory."""
    committed, parent_receipt = certified_parent_rows(
        preflight_path, parent_dir, parent_audit_path
    )
    metadata_path, child_dir = metadata_path.resolve(), child_dir.resolve()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["schema"] == "smile-storage-recovery-lineage-v1"
    assert metadata["cuda_initialized"] is False
    continuity = metadata["objective_continuity"]
    assert continuity["verified"] is True
    assert (
        continuity["classification"]
        == "same_objective_optimizer_continuation_after_storage_failure"
    )
    assert continuity["saved_objective_change_label"] == "continued_objective_change"
    assert continuity["saved_label_is_mechanical_not_a_mathematical_change"] is True

    input_records = list(metadata["input_files"].values())
    assert input_records
    for item in input_records:
        assert set(item) == {"path", "sha256"}
        assert record(Path(item["path"])) == item
    for path in (
        parent_dir / "protocol.json",
        parent_dir / "summary.json",
        parent_dir / "progress.jsonl",
        parent_dir / "objective-terms.json",
        preflight_path,
        parent_audit_path,
        child_dir / "protocol.json",
        child_dir / "summary.json",
        child_dir / "progress.jsonl",
        child_dir / "objective-terms.json",
    ):
        assert record(path) in input_records, path
    parent_protocol = json.loads((parent_dir / "protocol.json").read_text())
    child_protocol = json.loads((child_dir / "protocol.json").read_text())
    parent_terms = json.loads((parent_dir / "objective-terms.json").read_text())
    child_terms = json.loads((child_dir / "objective-terms.json").read_text())
    assert parent_protocol["objective"] == child_protocol["objective"]
    assert parent_protocol["objective_terms"] == child_protocol["objective_terms"]
    assert parent_terms == child_terms == parent_protocol["objective_terms"]

    child_rows = [
        json.loads(line)
        for line in (child_dir / "progress.jsonl").read_text().splitlines()
    ]
    assert child_rows
    child_summary = json.loads((child_dir / "summary.json").read_text())
    assert child_summary["initial"] == child_rows[0]
    clip = metadata["lineage_clip"]
    assert (
        clip["parent_last_durable_iteration"]
        == parent_receipt["last_committed_iteration"]
    )
    assert clip["parent_kept_rows"] == [row["iteration"] for row in committed]
    assert (
        clip["parent_uncheckpointed_rows"]
        == parent_receipt["uncheckpointed_progress_rows"]
    )
    assert clip["parent_uncheckpointed_rows_on_trajectory"] is False
    assert (
        clip["child_initial_local_iteration"] == child_rows[0]["local_iteration"] == 0
    )
    assert clip["child_initial_optimizer_steps"] == child_rows[0]["optimizer_steps"]
    assert clip["child_initial_row_retained_as_zero_update_event"] is True
    assert clip["child_initial_displacement_identity_proven"] is False
    assert clip["curve_segment_break_at_zero_update"] is True
    assert clip["x_axis"] == "optimizer_step; local_iteration is branch-local"
    assert clip["child_accepted_rows"] == [row["iteration"] for row in child_rows[1:]]
    return committed, {
        "schema": "smile-recovery-renderer-lineage-v1",
        "metadata": record(metadata_path),
        "parent": parent_receipt,
        "objective_continuity": continuity,
        "lineage_clip": clip,
    }


def resolve_data_binding(
    item: dict[str, str],
    mirror_root: Path | None,
    local_data_root: Path | None = None,
) -> Path:
    """Resolve a binding only when one declared source has the recorded bytes.

    ``mirror_root`` is the collector root whose children are direct members of
    remote ``data/``.  ``local_data_root`` is explicit because the collector
    deliberately omits immutable neutral and blendshape inputs.
    """
    assert set(item) == {"path", "sha256"}
    original = Path(item["path"])
    candidates = [original]
    roots = [root for root in (mirror_root, local_data_root) if root is not None]
    if roots:
        anchors = [
            index for index, part in enumerate(original.parts) if part == "new-neutral"
        ]
        assert len(anchors) == 1, original
        suffix = Path(*original.parts[anchors[0] + 1 :])
        assert suffix.parts, original
        assert suffix.parts[0] == "data", original
        relative = Path(*suffix.parts[1:])
        candidates.extend(root.resolve() / relative for root in roots)
    unique = list(dict.fromkeys(candidates))
    for path in unique:
        if path.is_file() and sha256(path) == item["sha256"]:
            return path
    raise AssertionError(
        {"binding": item, "candidates": [str(path) for path in unique]}
    )


def verified_terminal_interruption(  # noqa: PLR0915
    terminal_path: Path,
    bundle_manifest_path: Path,
    mirror_root: Path,
    run_dir: Path,
    summary: dict[str, Any],
    progress_rows: list[dict[str, Any]],
    independent_audit_path: Path,
) -> dict[str, Any]:
    """Bind a retained running summary to the immutable terminal collector bundle."""
    terminal_path = terminal_path.resolve()
    bundle_manifest_path = bundle_manifest_path.resolve()
    mirror_root = mirror_root.resolve()
    run_dir = run_dir.resolve()
    assert terminal_path.is_file()
    assert bundle_manifest_path.is_file()
    manifest = json.loads(bundle_manifest_path.read_text())
    assert manifest["schema"] == "remote-collision-on-smile-recovery-complete-bundle-v1"
    assert Path(manifest["local_root"]).resolve() == mirror_root
    status = manifest["status"]
    assert status["terminal_and_gpu_idle"] is True
    assert status["current_processes_all_exited"] is True
    assert not status["gpu_compute_apps"]
    relative_terminal = terminal_path.relative_to(mirror_root).as_posix()
    assert manifest["files_sha256"][relative_terminal] == sha256(terminal_path)

    terminal = json.loads(terminal_path.read_text())
    assert terminal["schema"] == "collision-on-remote-smile-supervision-v1"
    assert terminal["status"] in {"audited_endpoint", "audited_partial_endpoint"}
    assert Path(terminal["run_dir"]).name == run_dir.name
    assert terminal["inverse_converged_claim"] is False
    assert terminal["source_sha256"]["3052-supervise-smile-config-recovery.py"] == (
        "3d9f2ab21756c619e076fd85be489e425dc13fd11b8ed179bd5807a08c123256"
    )
    assert terminal["deadlines_utc"].keys() == {"fit", "audit", "delivery"}

    def completed_child(label: str, *, expected_exit: int | None) -> dict[str, Any]:
        child = terminal[label]
        assert child["identity"]["argv"] == child["command"]
        assert isinstance(child["exit_code"], int)
        if expected_exit is not None:
            assert child["exit_code"] == expected_exit
        assert isinstance(child["ended_at_utc"], str)
        assert isinstance(child["gpu_idle_wait_seconds"], (int, float))
        assert child["gpu_idle_wait_seconds"] >= 0
        assert child["gpu_apps_after"] == []
        return child

    fit = completed_child("fit", expected_exit=None)
    audit = completed_child("audit", expected_exit=0)
    events = fit.get("deadline_cleanup", [])
    assert isinstance(events, list)
    for event in events:
        assert set(event) == {"signal", "at_utc"}
        assert event["signal"] in {"SIGINT", "SIGTERM", "SIGKILL"}
        assert isinstance(event["at_utc"], str)
    assert terminal["interrupted_signals"] == [] or all(
        isinstance(number, int) for number in terminal["interrupted_signals"]
    )

    saved = terminal["saved"]
    assert set(saved) == {
        "protocol.json",
        "summary.json",
        "checkpoint.pt",
        "endpoint.npz",
        "progress.jsonl",
    }
    for name, item in saved.items():
        assert item is not None
        local = run_dir / name
        assert local.is_file(), name
        assert item["sha256"] == sha256(local), name
    assert terminal["independent_audit"]["sha256"] == sha256(independent_audit_path)
    assert progress_rows
    final_row = progress_rows[-1]
    assert summary["final"] == final_row
    final_steps = final_row.get("optimizer_steps")
    if final_steps is not None:
        assert final_steps == summary["final"].get("optimizer_steps")

    import numpy as np
    import torch

    checkpoint = torch.load(
        run_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["iteration"] == final_row["iteration"]
    assert checkpoint["local_iteration"] == final_row["local_iteration"]
    assert checkpoint["optimizer_step"] == final_row["optimizer_step"]
    assert checkpoint["optimizer_steps"] == final_steps
    with np.load(run_dir / "endpoint.npz", allow_pickle=False) as endpoint:
        for key in ("displacement_m", "activation_inv", "pose_rad_m"):
            np.testing.assert_array_equal(endpoint[key], checkpoint[key].numpy())

    return {
        "schema": "smile-renderer-terminal-interruption-evidence-v1",
        "terminal_receipt": record(terminal_path),
        "bundle_manifest": record(bundle_manifest_path),
        "collector_status": {
            "terminal_and_gpu_idle": True,
            "current_processes_all_exited": True,
            "gpu_compute_apps": [],
        },
        "terminal_status": terminal["status"],
        "actual_signal_events": events,
        "fit_process_completion_utc": fit["ended_at_utc"],
        "exact_numerical_stop_time_utc": None,
        "fit_exit_code": fit["exit_code"],
        "audit_process_completion_utc": audit["ended_at_utc"],
        "audit_exit_code": audit["exit_code"],
        "note": (
            "fit ended_at_utc records process and Cherries completion; the exact "
            "numerical stopping time is unknown without a logged numerical event."
        ),
    }
