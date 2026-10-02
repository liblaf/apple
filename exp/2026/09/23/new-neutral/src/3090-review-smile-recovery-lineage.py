# Copyright (c) 2026 liblaf
"""Bind Smile 003→006 objective continuity and renderer lineage on local copies."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

GROUP = Path(__file__).resolve().parent.parent
COMMITTED = 32
OBJECTIVE_SOURCES = (
    "new-neutral/3000-inverse-smile-expression-gradient.py",
    "new-neutral/mouthopen_fit_objective.py",
    "new-neutral/expression_gradient_check.py",
)
INPUT_SOURCES = (
    "blendshapes",
    "blendshape_manifest",
    "neutral_endpoint",
    "neutral_summary",
    "reference_repair",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(path: Path, fingerprints: dict[str, dict[str, str]]) -> bytes:
    assert path.is_file(), path
    data = path.read_bytes()
    fingerprints[str(path.resolve())] = {
        "path": str(path.resolve()),
        "sha256": sha256_bytes(data),
    }
    return data


def load_json(path: Path, fingerprints: dict[str, dict[str, str]]) -> dict:
    return json.loads(read(path, fingerprints))


def load_lines(path: Path, fingerprints: dict[str, dict[str, str]]) -> list[dict]:
    rows = [json.loads(line) for line in read(path, fingerprints).splitlines()]
    assert rows
    return rows


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--child-run", type=Path, required=True)
    parser.add_argument("--recovery-preflight", type=Path, required=True)
    parser.add_argument("--parent-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent = args.parent_run.resolve()
    child = args.child_run.resolve()
    output = args.output.resolve()
    assert parent != child
    assert parent.name == "inverse-smile-coupled-003"
    assert child.name == "inverse-smile-coupled-006"
    assert output not in {parent, child}
    assert not output.exists(), output
    assert not torch.cuda.is_initialized()
    fingerprints: dict[str, dict[str, str]] = {}
    parent_protocol = load_json(parent / "protocol.json", fingerprints)
    child_protocol = load_json(child / "protocol.json", fingerprints)
    parent_terms = load_json(parent / "objective-terms.json", fingerprints)
    child_terms = load_json(child / "objective-terms.json", fingerprints)
    parent_summary = load_json(parent / "summary.json", fingerprints)
    child_summary = load_json(child / "summary.json", fingerprints)
    parent_rows = load_lines(parent / "progress.jsonl", fingerprints)
    child_rows = load_lines(child / "progress.jsonl", fingerprints)
    preflight = load_json(args.recovery_preflight, fingerprints)
    audit = load_json(args.parent_audit, fingerprints)
    parent_checkpoint_path = parent / "checkpoint.pt"
    read(parent_checkpoint_path, fingerprints)
    endpoint_path = parent / "endpoint.npz"
    read(endpoint_path, fingerprints)

    assert parent_protocol["schema"] == child_protocol["schema"]
    assert (
        parent_protocol["expression_name"]
        == child_protocol["expression_name"]
        == "Smile"
    )
    assert parent_protocol["objective"] == child_protocol["objective"]
    assert parent_protocol["objective_terms"] == child_protocol["objective_terms"]
    assert parent_terms == child_terms == parent_protocol["objective_terms"]
    assert parent_protocol["config"]["objective_mode"] == "l2-normal-smooth"
    assert child_protocol["config"]["objective_mode"] == "l2-normal-smooth"
    for name in OBJECTIVE_SOURCES:
        assert (
            parent_protocol["source_sha256"][name]
            == child_protocol["source_sha256"][name]
        ), name
    for name in INPUT_SOURCES:
        assert (
            parent_protocol["sources"][name]["sha256"]
            == child_protocol["sources"][name]["sha256"]
        ), name
    assert (
        child_protocol["initialization"]["objective_initialization"]["kind"]
        == "continued_objective_change"
    )
    assert child_protocol["initialization"]["optimizer_state"]["continued"] is True
    assert (
        child_protocol["initialization"]["checkpoint"]["sha256"]
        == fingerprints[str(parent_checkpoint_path)]["sha256"]
    )
    assert (
        child_protocol["initialization"]["endpoint"]["sha256"]
        == fingerprints[str(endpoint_path)]["sha256"]
    )
    assert preflight["schema"] == "smile-003-disk-full-continuation-preflight-v1"
    assert preflight["last_committed_iteration"] == COMMITTED
    assert preflight["optimizer_steps"] == {"q": COMMITTED, "pose": COMMITTED}
    assert preflight["stationarity_monitor_disabled_across_recovery"] is True
    assert (
        preflight["parent_files_sha256"]["checkpoint.pt"]
        == fingerprints[str(parent_checkpoint_path)]["sha256"]
    )
    assert (
        preflight["parent_files_sha256"]["endpoint.npz"]
        == fingerprints[str(endpoint_path)]["sha256"]
    )
    for name in ("protocol.json", "summary.json", "progress.jsonl"):
        assert (
            preflight["parent_files_sha256"][name]
            == fingerprints[str(parent / name)]["sha256"]
        )
    assert (
        preflight["parent_files_sha256"]["independent_audit.json"]
        == fingerprints[str(args.parent_audit.resolve())]["sha256"]
    )
    assert audit["valid_forward"] is True
    assert (
        audit["inputs"]["endpoint"]["sha256"]
        == fingerprints[str(endpoint_path)]["sha256"]
    )
    assert (
        audit["inputs"]["protocol"]["sha256"]
        == fingerprints[str(parent / "protocol.json")]["sha256"]
    )
    assert (
        audit["inputs"]["summary"]["sha256"]
        == fingerprints[str(parent / "summary.json")]["sha256"]
    )

    state = torch.load(parent_checkpoint_path, map_location="cpu", weights_only=False)
    with np.load(endpoint_path, allow_pickle=False) as archive:
        for name in ("activation_inv", "pose_rad_m", "displacement_m"):
            np.testing.assert_array_equal(state[name].numpy(), archive[name])
    assert len(state["moments"]) == 4
    assert state["optimizer_steps"] == {"q": COMMITTED, "pose": COMMITTED}
    parent_saved = [row for row in parent_rows if row["iteration"] == COMMITTED]
    assert len(parent_saved) == 1
    assert parent_summary["final"] == parent_saved[0]
    uncheckpointed = [row for row in parent_rows if row["iteration"] > COMMITTED]
    assert len(uncheckpointed) <= 1
    if uncheckpointed:
        assert uncheckpointed[0]["iteration"] == COMMITTED + 1
    child_initial = child_rows[0]
    assert child_initial["iteration"] == child_initial["local_iteration"] == 0
    assert child_initial["optimizer_steps"] == {"q": COMMITTED, "pose": COMMITTED}
    assert child_initial["valid_forward"] is True
    assert child_summary["initial"] == child_initial
    assert child_protocol["config"]["convergence_patience"] == 0

    metrics = ("loss", "fit_rms_mm", "force_norm_n", "geometry")
    metric_equal = {
        name: parent_saved[0][name] == child_initial[name] for name in metrics
    }
    for path_text, item in fingerprints.items():
        assert sha256_bytes(Path(path_text).read_bytes()) == item["sha256"], path_text
    result: dict[str, Any] = {
        "schema": "smile-storage-recovery-lineage-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "objective_continuity": {
            "verified": True,
            "classification": "same_objective_optimizer_continuation_after_storage_failure",
            "saved_objective_change_label": "continued_objective_change",
            "saved_label_is_mechanical_not_a_mathematical_change": True,
            "objective_definition": child_protocol["objective"],
            "objective_terms_sha256": fingerprints[str(child / "objective-terms.json")][
                "sha256"
            ],
            "objective_source_sha256": {
                name: child_protocol["source_sha256"][name]
                for name in OBJECTIVE_SOURCES
            },
            "input_sha256": {
                name: child_protocol["sources"][name]["sha256"]
                for name in INPUT_SOURCES
            },
        },
        "lineage_clip": {
            "parent_last_durable_iteration": COMMITTED,
            "parent_kept_rows": [
                row["iteration"] for row in parent_rows if row["iteration"] <= COMMITTED
            ],
            "parent_uncheckpointed_rows": uncheckpointed,
            "parent_uncheckpointed_rows_on_trajectory": False,
            "child_initial_local_iteration": 0,
            "child_initial_optimizer_steps": child_initial["optimizer_steps"],
            "child_initial_row_retained_as_zero_update_event": True,
            "child_initial_displacement_identity_proven": False,
            "child_initial_metric_equal_to_parent": metric_equal,
            "zero_update_marker": (
                "metric_change_at_unchanged_parameters"
                if not all(metric_equal.values())
                else "equilibrium_replay_identity_unresolved"
            ),
            "child_accepted_rows": [row["iteration"] for row in child_rows[1:]],
            "curve_segment_break_at_zero_update": True,
            "x_axis": "optimizer_step; local_iteration is branch-local",
            "stationarity_or_inverse_convergence_claim": False,
        },
        "input_files": fingerprints,
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
