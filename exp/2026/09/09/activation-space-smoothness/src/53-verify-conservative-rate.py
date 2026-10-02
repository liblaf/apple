# ruff: noqa: C901, EM102, PLR0912, PLR0915, TRY003
"""Verify the saved conservative-rate learned-axis pair without re-solving."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pydantic_settings as ps
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
REFERENCE_DIRS = {
    "off": GROUP / "data/24-learned-axis",
    "on": GROUP / "data/25-learned-axis-smooth",
}
CASES = {"off": "learned-axis", "on": "learned-axis-smooth"}
MODEL = "learned-axis"
INITIAL_PHASE_STEP = 64
MAXIMUM_FOLLOWUP_STEP = 128
CHECKPOINT_INTERVAL = 16
PAIR_THRESHOLDS = {
    "field_max_abs": 1.0e-6,
    "field_relative_l2": 1.0e-4,
    "geometry_rms_mm": 1.0e-6,
}
SETTINGS_KEYS = {
    "adam_eps",
    "betas",
    "smooth_length_m",
    "inputs",
    "smoothness_field",
    "initialization_seed",
    "initial_strength",
    "initialization_mode",
    "controls_validation",
    "smoothness_weight",
    "status",
    "learning_rates",
    "followup",
}
UNCHANGED_SETTINGS_KEYS = SETTINGS_KEYS - {"learning_rates", "followup"}
UNCHANGED_PROVENANCE_KEYS = {
    "inputs",
    "controls_validation",
    "diagnostics",
    "materials",
    "forward_tolerances",
    "objective",
    "physical_field",
    "skin_enabled",
    "magnitude_weight",
    "rank_weight",
    "upper_stress_cap",
    "failure_policy",
    "inverse_stationarity_claimed",
}
UNCHANGED_OPTIMIZER_KEYS = {
    "name",
    "eps",
    "betas",
    "weight_decay",
    "amsgrad",
    "maximize",
    "foreach",
    "fused",
    "initialization_seed",
    "initial_strength",
    "archive_initialization",
}


class Config(cherries.BaseConfig):
    """Locations of the final saved pair and its frozen follow-up settings."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    off_dir: Path
    on_dir: Path
    settings: Path
    output_dir: Path


def load_low_level_verifier() -> ModuleType:
    """Load the original verifier despite its numeric filename."""
    path = GROUP / "src/50-verify.py"
    name = "activation_space_saved_state_verifier"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load low-level verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    original_input = cherries.input
    original_output = cherries.output

    def inert_asset(value: str, *_args: Any, **_kwargs: Any) -> Path:
        return GROUP / "data" / value

    try:
        cherries.input = inert_asset
        cherries.output = inert_asset
        spec.loader.exec_module(module)
    finally:
        cherries.input = original_input
        cherries.output = original_output
    return module


VERIFY = load_low_level_verifier()
require = VERIFY.require
require_close = VERIFY.require_close
sha256 = VERIFY.sha256
record = VERIFY.record
write_json = VERIFY.write_json


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_recorded_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (GROUP / path).resolve()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def settings_intervention(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    settings = read_json(path)
    require(set(settings) == SETTINGS_KEYS, "conservative settings schema differs")
    require(
        settings["status"] == "frozen_before_primary_runs", "settings are not frozen"
    )
    followup = settings["followup"]
    original_record = followup["original_settings"]
    original_path = resolve_recorded_path(original_record["path"])
    require(original_path.is_file(), f"original settings are missing: {original_path}")
    require(
        sha256(original_path) == original_record["sha256"],
        "original settings hash differs",
    )
    original = read_json(original_path)
    for key in sorted(UNCHANGED_SETTINGS_KEYS):
        require(settings[key] == original[key], f"settings intervention changed {key}")
    old_rate = float(original["learning_rates"][MODEL])
    new_rate = float(settings["learning_rates"][MODEL])
    require_close(float(followup["original_rate"]), old_rate, "declared original rate")
    require_close(float(followup["rate_fraction"]), 0.25, "declared rate fraction")
    require_close(new_rate, old_rate / 4.0, "quarter learning rate")
    require(
        followup["weight_policy"]
        == "Retain the original calibrated C weight without retuning",
        "follow-up weight policy differs",
    )
    require(
        int(followup["first_phase_steps"]) == INITIAL_PHASE_STEP, "first phase differs"
    )
    require(
        int(followup["conditional_target_steps"]) == MAXIMUM_FOLLOWUP_STEP,
        "conditional target differs",
    )
    plan = followup["plan"]
    plan_path = resolve_recorded_path(plan["path"])
    require(plan_path.is_file(), f"follow-up plan is missing: {plan_path}")
    require(sha256(plan_path) == plan["sha256"], "follow-up plan hash differs")
    validation = settings["controls_validation"]
    validation_path = resolve_recorded_path(validation["path"])
    require(
        sha256(validation_path) == validation["sha256"], "controls validation drifted"
    )
    require(
        read_json(validation_path)["status"] == "passed", "controls validation failed"
    )
    frozen_summary_path = path.parent / "summary.json"
    frozen_summary = read_json(frozen_summary_path)
    require(
        frozen_summary["status"] == "frozen", "follow-up settings summary is not frozen"
    )
    require(
        frozen_summary["settings_sha256"] == sha256(path),
        "settings summary hash differs",
    )
    require_close(frozen_summary["learning_rate"], new_rate, "settings summary rate")
    return (
        settings,
        original,
        {
            "settings": record(path),
            "frozen_summary": record(frozen_summary_path),
            "original_settings": record(original_path),
            "plan": record(plan_path),
            "changed_numerical_field": "learning_rates.learned-axis",
            "old_learning_rate": old_rate,
            "new_learning_rate": new_rate,
            "rate_fraction": new_rate / old_rate,
            "unchanged_required_fields": sorted(UNCHANGED_SETTINGS_KEYS),
            "smoothness_weight_retained": float(settings["smoothness_weight"]),
        },
    )


def normalized_parent_inputs(
    directory: Path, config: dict[str, Any], provenance: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = copy.deepcopy(config)
    provenance = copy.deepcopy(provenance)
    if config["resume"] is None:
        return config, provenance
    resume = resolve_recorded_path(config["resume"])
    parent = provenance["parent"]
    require(parent is not None, f"resume lacks parent receipt: {directory}")
    parent_path = resolve_recorded_path(parent["checkpoint"])
    require(resume == parent_path, f"resume and parent path differ: {directory}")
    config["resume"] = str(resume)
    parent["checkpoint"] = str(parent_path)
    return config, provenance


def require_reference_protocol_equal(
    label: str,
    provenance: dict[str, Any],
    reference: dict[str, Any],
    new_sources: dict[str, Any],
    reference_sources: dict[str, Any],
) -> dict[str, Any]:
    for key in sorted(UNCHANGED_PROVENANCE_KEYS):
        require(
            provenance[key] == reference[key], f"{label} changed provenance field {key}"
        )
    new_hashes = new_sources["snapshot_sha256_by_module"]
    old_hashes = reference_sources["snapshot_sha256_by_module"]
    require(new_hashes == old_hashes, f"{label} numerical source snapshots differ")
    for key in sorted(UNCHANGED_OPTIMIZER_KEYS):
        require(
            provenance["optimizer"][key] == reference["optimizer"][key],
            f"{label} changed optimizer field {key}",
        )
    return {
        "reference_provenance": record(REFERENCE_DIRS[label] / "provenance.json"),
        "unchanged_provenance_fields": sorted(UNCHANGED_PROVENANCE_KEYS),
        "unchanged_optimizer_fields": sorted(UNCHANGED_OPTIMIZER_KEYS),
        "source_module_count": len(new_hashes),
        "source_snapshots_identical_to_original": True,
    }


def load_state_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in ("q", "C", "Z", "u")}


def validate_continuation_lineage(
    directory: Path,
    case: str,
    fixture: Any,
    settings: dict[str, Any],
    settings_sha: str,
    expected_weight: float,
) -> dict[str, Any]:
    """Follow every saved-Adam parent back to the completed fresh phase."""
    expected_lr = float(settings["learning_rates"][MODEL])
    nodes = []
    visited: set[Path] = set()
    current = directory.resolve()
    while True:
        require(
            current not in visited, f"continuation lineage contains a cycle: {current}"
        )
        visited.add(current)
        config = read_json(current / "config.json")
        provenance = read_json(current / "provenance.json")
        summary = read_json(current / "summary.json")
        rows = VERIFY.read_trace(current / "trace.csv")
        require(config["case"] == case, f"lineage config case differs: {current}")
        require(
            provenance["case"] == case, f"lineage provenance case differs: {current}"
        )
        require(provenance["model"] == MODEL, f"lineage model differs: {current}")
        require(
            summary["parent"] == provenance["parent"],
            f"lineage parent differs: {current}",
        )
        require(
            provenance["settings"]["sha256"] == settings_sha,
            f"lineage settings hash differs: {current}",
        )
        require_close(
            provenance["optimizer"]["lr"], expected_lr, f"lineage rate {current}"
        )
        require_close(
            provenance["smoothness_weight"],
            expected_weight,
            f"lineage weight {current}",
        )
        start = int(provenance["optimizer"]["start_step"])
        target = int(provenance["optimizer"]["target_step"])
        require(
            int(summary["last_evaluated_step"]) == int(rows[-1]["step"]),
            f"lineage last step differs: {current}",
        )
        node = {
            "directory": str(current),
            "start_step": start,
            "target_step": target,
            "last_accepted_step": int(rows[-1]["step"]),
            "status": summary["status"],
            "provenance": record(current / "provenance.json"),
            "optimizer": record(current / "optimizer-latest.pt"),
        }
        normalized_config, normalized_provenance = normalized_parent_inputs(
            current, config, provenance
        )
        if provenance["parent"] is None:
            require(
                config["resume"] is None, f"fresh lineage root has resume: {current}"
            )
            require(
                start == 0 and target == INITIAL_PHASE_STEP,
                "lineage root is not fresh 64",
            )
            require(
                summary["status"] == "completed_fixed_budget_not_stationarity_certified"
                and int(rows[-1]["step"]) == INITIAL_PHASE_STEP,
                "lineage root did not complete the fresh 64-step phase",
            )
            require(
                provenance["optimizer"]["initialization"]
                == settings["initialization_mode"],
                "lineage root initialization differs",
            )
            node["role"] = "fresh_64_root"
            nodes.append(node)
            break

        parent_receipt, parent_state = VERIFY.validate_parent_checkpoint(
            normalized_config,
            normalized_provenance,
            case,
            MODEL,
            fixture,
            settings_sha,
            expected_weight,
            start,
            float(rows[start]["gradient_rms"]),
        )
        require(
            parent_receipt is not None and parent_state is not None,
            "missing lineage parent",
        )
        prefix = VERIFY.validate_copied_parent_prefix(
            current,
            Path(normalized_provenance["parent"]["checkpoint"]),
            rows,
            start,
            parent_state,
        )
        node.update(
            {
                "role": "saved_adam_continuation",
                "parent_checkpoint": parent_receipt,
                "copied_parent_prefix": prefix,
            }
        )
        nodes.append(node)
        current = Path(normalized_provenance["parent"]["checkpoint"]).parent.resolve()

    return {
        "status": "passed",
        "node_count": len(nodes),
        "nodes_final_to_root": nodes,
        "fresh_root_step": INITIAL_PHASE_STEP,
        "optimizer_reset_after_root": False,
    }


def verify_run(
    label: str,
    directory: Path,
    fixture: Any,
    settings: dict[str, Any],
    settings_sha: str,
) -> dict[str, Any]:
    case = CASES[label]
    reference_dir = REFERENCE_DIRS[label]
    for name in (
        "config.json",
        "provenance.json",
        "summary.json",
        "trace.csv",
        "solver-receipts.jsonl",
        "optimizer-latest.pt",
        "best.npz",
        "best-objective.npz",
    ):
        require(
            (directory / name).is_file(), f"missing run artifact: {directory / name}"
        )
    config = read_json(directory / "config.json")
    provenance = read_json(directory / "provenance.json")
    summary = read_json(directory / "summary.json")
    rows = VERIFY.read_trace(directory / "trace.csv")
    require(config["case"] == case, f"config case differs: {label}")
    require(provenance["case"] == case, f"provenance case differs: {label}")
    require(provenance["model"] == MODEL, f"model differs: {label}")
    require(summary["case"] == case, f"summary case differs: {label}")
    require(
        summary["parent"] == provenance["parent"], f"summary parent differs: {label}"
    )

    status = summary["status"]
    completed = status == "completed_fixed_budget_not_stationarity_certified"
    interrupted = status == "failed_before_completion"
    require(completed or interrupted, f"unsupported run status: {label}: {status!r}")
    failure = summary["failure"]
    require((failure is None) == completed, f"status/failure receipt differs: {label}")
    administrative_cutoff = bool(
        interrupted
        and isinstance(failure, dict)
        and failure.get("type") == "KeyboardInterrupt"
    )
    numerical_failure = interrupted and not administrative_cutoff

    settings_path = resolve_recorded_path(config["settings"])
    recorded_settings_path = resolve_recorded_path(provenance["settings"]["path"])
    require(
        settings_path.resolve() == recorded_settings_path.resolve(),
        f"settings paths differ: {label}",
    )
    require(
        sha256(settings_path) == settings_sha, f"config settings hash differs: {label}"
    )
    require(
        provenance["settings"]["sha256"] == settings_sha,
        f"provenance settings hash differs: {label}",
    )
    require(
        sha256(recorded_settings_path) == settings_sha,
        f"recorded settings drifted: {label}",
    )

    expected_weight = float(settings["smoothness_weight"]) if label == "on" else 0.0
    require_close(
        provenance["smoothness_weight"], expected_weight, f"provenance weight {label}"
    )
    require_close(
        summary["smoothness_weight"], expected_weight, f"summary weight {label}"
    )
    expected_lr = float(settings["learning_rates"][MODEL])
    require_close(provenance["optimizer"]["lr"], expected_lr, f"learning rate {label}")
    require_close(
        provenance["optimizer"]["eps"], settings["adam_eps"], f"Adam eps {label}"
    )
    require(
        tuple(provenance["optimizer"]["betas"]) == tuple(settings["betas"]),
        f"Adam betas differ: {label}",
    )
    require(
        int(config["steps"]) == int(provenance["optimizer"]["target_step"]),
        f"target differs: {label}",
    )
    require(
        int(config["checkpoint_interval"]) == CHECKPOINT_INTERVAL,
        f"checkpoint interval differs: {label}",
    )
    require(
        int(config["initialization_seed"]) == int(settings["initialization_seed"]),
        f"seed differs: {label}",
    )
    require(
        float(config["smoothness_multiplier"]) == 1.0,
        f"smoothness multiplier differs: {label}",
    )

    start = int(provenance["optimizer"]["start_step"])
    target = int(provenance["optimizer"]["target_step"])
    fresh = provenance["parent"] is None
    require((config["resume"] is None) == fresh, f"resume/parent differs: {label}")
    if fresh:
        require(start == 0, f"fresh run start differs: {label}")
        require(target == INITIAL_PHASE_STEP, f"fresh run target differs: {label}")
        require(
            provenance["optimizer"]["initialization"]
            == settings["initialization_mode"],
            f"fresh initialization mode differs: {label}",
        )
    else:
        require(
            start >= INITIAL_PHASE_STEP, f"continuation starts before phase 64: {label}"
        )
        require(
            start < target <= MAXIMUM_FOLLOWUP_STEP,
            f"continuation target differs: {label}",
        )
        require(
            start % CHECKPOINT_INTERVAL == 0,
            f"continuation start is unscheduled: {label}",
        )
        require(
            provenance["optimizer"]["initialization"]
            == provenance["parent"]["continuation"],
            f"continuation semantics differ: {label}",
        )
    require(target % CHECKPOINT_INTERVAL == 0, f"target is unscheduled: {label}")

    reference_provenance = read_json(reference_dir / "provenance.json")
    require(
        reference_provenance["settings"]["sha256"]
        == settings["followup"]["original_settings"]["sha256"],
        f"reference settings hash differs: {label}",
    )
    new_sources = VERIFY.verify_sources(directory, provenance)
    reference_sources = VERIFY.verify_sources(reference_dir, reference_provenance)
    reference_protocol = require_reference_protocol_equal(
        label, provenance, reference_provenance, new_sources, reference_sources
    )

    normalized_config, normalized_provenance = normalized_parent_inputs(
        directory, config, provenance
    )
    trace_receipt = VERIFY.validate_trace(
        case, rows, normalized_provenance, expected_weight, completed=completed
    )
    parent_receipt, parent_state = VERIFY.validate_parent_checkpoint(
        normalized_config,
        normalized_provenance,
        case,
        MODEL,
        fixture,
        settings_sha,
        expected_weight,
        trace_receipt["start_step"],
        float(rows[trace_receipt["start_step"]]["gradient_rms"]),
    )
    parent_prefix = None
    if parent_state is not None:
        parent_prefix = VERIFY.validate_copied_parent_prefix(
            directory,
            Path(normalized_provenance["parent"]["checkpoint"]),
            rows,
            start,
            parent_state,
        )
    continuation_lineage = validate_continuation_lineage(
        directory,
        case,
        fixture,
        settings,
        settings_sha,
        expected_weight,
    )

    require(
        int(summary["last_evaluated_step"]) == int(rows[-1]["step"]),
        f"summary last step differs: {label}",
    )
    require(
        int(summary["best_step"]) == trace_receipt["best_step"],
        f"summary best differs: {label}",
    )
    require(
        int(summary["best_objective_step"]) == trace_receipt["best_objective_step"],
        f"summary best objective differs: {label}",
    )
    VERIFY.compare_mapping_subset(
        summary["last_metrics"], rows[-1], f"summary last metrics {label}"
    )
    row_by_step = {int(row["step"]): row for row in rows}
    best_row = row_by_step[int(summary["best_step"])]
    best_objective_row = row_by_step[int(summary["best_objective_step"])]
    VERIFY.compare_mapping_subset(
        summary["best_metrics"], best_row, f"summary best metrics {label}"
    )
    VERIFY.compare_mapping_subset(
        summary["best_objective_metrics"],
        best_objective_row,
        f"summary best objective metrics {label}",
    )
    if label == "off":
        require(
            trace_receipt["best_step"] == trace_receipt["best_objective_step"],
            "off best aliases differ",
        )

    last = trace_receipt["last_accepted_step"]
    expected_steps = [
        step
        for step in range(last + 1)
        if step in {0, 1}
        or step % CHECKPOINT_INTERVAL == 0
        or (completed and step == target)
    ]
    checkpoint_paths = {
        step: directory / f"step-{step:04d}.npz" for step in expected_steps
    }
    require(
        {path.name for path in directory.glob("step-*.npz")}
        == {path.name for path in checkpoint_paths.values()},
        f"full checkpoint schedule differs: {label}",
    )
    checkpoint_receipts = []
    checkpoint_arrays: dict[int, dict[str, np.ndarray]] = {}
    for step, path in checkpoint_paths.items():
        receipt, arrays = VERIFY.validate_state(
            path, MODEL, fixture, row_by_step[step], check_volume=True
        )
        checkpoint_receipts.append(receipt)
        VERIFY.validate_surface(
            directory / f"surface-{step:04d}.npz", step, fixture, arrays["u"]
        )
        if step in {0, 1, start, last}:
            checkpoint_arrays[step] = arrays

    aliases: dict[str, tuple[str, dict[str, Any]]] = {
        "best": ("best.npz", best_row),
        "best_objective": ("best-objective.npz", best_objective_row),
    }
    if completed:
        require(
            (directory / "last.npz").is_file(),
            f"completed run lacks last alias: {label}",
        )
        aliases["last"] = ("last.npz", rows[-1])
        require(
            not (directory / "failure.json").exists(),
            f"completed run has failure receipt: {label}",
        )
        require(
            not (directory / "failure-controls.npz").exists(),
            f"completed run has failure controls: {label}",
        )
    alias_receipts = {}
    alias_arrays = {}
    for name, (filename, row) in aliases.items():
        receipt, arrays = VERIFY.validate_state(
            directory / filename, MODEL, fixture, row, check_volume=True
        )
        alias_receipts[name] = receipt
        alias_arrays[name] = arrays
        step = int(row["step"])
        if step in checkpoint_arrays:
            for field in ("q", "C", "Z", "u"):
                require(
                    np.array_equal(arrays[field], checkpoint_arrays[step][field]),
                    f"{name} alias differs from scheduled state: {label}/{field}",
                )
    if label == "off":
        for field in ("q", "C", "Z", "u"):
            require(
                np.array_equal(
                    alias_arrays["best"][field], alias_arrays["best_objective"][field]
                ),
                f"off best aliases differ: {field}",
            )

    expected_surface_names = {f"surface-{int(row['step']):04d}.npz" for row in rows}
    require(
        {path.name for path in directory.glob("surface-*.npz")}
        == expected_surface_names,
        f"surface schedule differs: {label}",
    )
    surface_manifest = hashlib.sha256()
    for row in rows:
        step = int(row["step"])
        arrays = checkpoint_arrays.get(step)
        digest = VERIFY.validate_surface(
            directory / f"surface-{step:04d}.npz",
            step,
            fixture,
            None if arrays is None else arrays["u"],
        )
        surface_manifest.update(f"surface-{step:04d}.npz\0{digest}\n".encode())
    for name, arrays in alias_arrays.items():
        step = int(alias_receipts[name]["step"])
        VERIFY.validate_surface(
            directory / f"surface-{step:04d}.npz", step, fixture, arrays["u"]
        )

    initialization = VERIFY.validate_initialization(
        MODEL, settings, fixture, checkpoint_arrays[0], rows[0]
    )
    solver = VERIFY.validate_solver_receipts(directory / "solver-receipts.jsonl", rows)
    optimizer, optimizer_arrays = VERIFY.validate_optimizer(
        directory / "optimizer-latest.pt",
        case,
        MODEL,
        rows[-1],
        normalized_provenance,
        settings_sha,
        expected_weight,
        fixture,
        trace_receipt["best_step"],
        trace_receipt["best_objective_step"],
    )
    VERIFY.validate_surface(
        directory / f"surface-{last:04d}.npz", last, fixture, optimizer_arrays["u"]
    )
    if completed:
        for field in ("q", "u"):
            require(
                np.array_equal(optimizer_arrays[field], alias_arrays["last"][field]),
                f"optimizer and last alias differ: {label}/{field}",
            )
        for field in ("C", "Z"):
            VERIFY.require_array_close(
                optimizer_arrays[field],
                alias_arrays["last"][field],
                f"optimizer reconstruction and last alias differ: {label}/{field}",
            )

    failure_evidence = None
    if interrupted:
        failure_evidence = VERIFY.validate_failure_evidence(
            directory, failure, MODEL, fixture, last, target
        )
    reference_rows = VERIFY.read_trace(reference_dir / "trace.csv")
    _, reference_initial = VERIFY.validate_state(
        reference_dir / "step-0000.npz",
        MODEL,
        fixture,
        reference_rows[0],
        check_volume=True,
    )
    cross_rate_initial = field_comparison(reference_initial, checkpoint_arrays[0])

    first_inversion = next(
        (row for row in rows if int(row["inverted_all_cells"]) > 0), None
    )
    physical_keys = (
        "step",
        "fit_rms_mm",
        "motion_rms_mm",
        "inverted_all_cells",
        "inverted_active_cells",
        "detF_min",
        "shortening_fraction_p99",
        "smoothness_C",
        "smoothness_Z",
    )
    result = {
        "directory": str(directory.resolve()),
        "case": case,
        "run_status": status,
        "outcome_classification": (
            "completed_declared_budget"
            if completed
            else "administrative_keyboard_interrupt"
            if administrative_cutoff
            else "numerical_or_other_failure"
        ),
        "administrative_cutoff": administrative_cutoff,
        "numerical_failure": numerical_failure,
        "evidence_integrity_status": "passed",
        "declared_budget_completed": completed,
        "reached_step_64": last >= INITIAL_PHASE_STEP,
        "reached_step_128": last >= MAXIMUM_FOLLOWUP_STEP,
        "config": record(directory / "config.json"),
        "provenance": record(directory / "provenance.json"),
        "summary": record(directory / "summary.json"),
        "trace": {**record(directory / "trace.csv"), **trace_receipt},
        "solver_receipts": solver,
        "full_checkpoints": checkpoint_receipts,
        "full_checkpoint_schedule": {
            "expected_steps": expected_steps,
            "schedule_complete": True,
            "last_accepted_has_scheduled_full_state": last in expected_steps,
            "last_accepted_has_optimizer_state": True,
        },
        "surface_states": {
            "count": len(rows),
            "manifest_sha256": surface_manifest.hexdigest(),
        },
        "aliases": alias_receipts,
        "optimizer": optimizer,
        "failure_evidence": failure_evidence,
        "parent_checkpoint": parent_receipt,
        "parent_prefix": parent_prefix,
        "continuation_lineage": continuation_lineage,
        "initialization": initialization,
        "cross_rate_initial_against_original_arm": cross_rate_initial,
        "reference_protocol": reference_protocol,
        "sources": new_sources,
        "physical_diagnostics_non_gating": {
            "endpoint": {key: rows[-1][key] for key in physical_keys},
            "first_inversion": (
                None
                if first_inversion is None
                else {key: first_inversion[key] for key in physical_keys}
            ),
            "maximum_inverted_all_cells": max(
                int(row["inverted_all_cells"]) for row in rows
            ),
        },
        "_initial_arrays": checkpoint_arrays[0],
        "_first_arrays": checkpoint_arrays[1],
        "_initial_row": rows[0],
        "_first_row": rows[1],
    }
    return result


def field_comparison(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> dict[str, Any]:
    fields = {}
    for field in ("q", "C", "Z"):
        difference = left[field] - right[field]
        denominator = max(
            float(np.linalg.norm(left[field])), float(np.linalg.norm(right[field]))
        )
        fields[field] = {
            "bitwise_equal": bool(np.array_equal(left[field], right[field])),
            "max_abs": float(np.max(np.abs(difference))),
            "relative_l2": 0.0
            if denominator == 0.0
            else float(np.linalg.norm(difference) / denominator),
            "left_sha256": array_sha256(left[field]),
            "right_sha256": array_sha256(right[field]),
        }
    u_difference = left["u"] - right["u"]
    return {
        "fields": fields,
        "u_bitwise_equal": bool(np.array_equal(left["u"], right["u"])),
        "geometry_rms_mm": float(
            1000.0 * np.linalg.norm(u_difference) / math.sqrt(len(u_difference))
        ),
        "u_max_abs_mm": float(1000.0 * np.max(np.abs(u_difference))),
    }


def paired_equivalence(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    steps = {}
    for step, arrays_key, row_key in (
        (0, "_initial_arrays", "_initial_row"),
        (1, "_first_arrays", "_first_row"),
    ):
        comparison = field_comparison(
            results["off"][arrays_key], results["on"][arrays_key]
        )
        for metrics in comparison["fields"].values():
            metrics["max_abs_passed"] = (
                metrics["max_abs"] <= PAIR_THRESHOLDS["field_max_abs"]
            )
            metrics["relative_l2_passed"] = (
                metrics["relative_l2"] <= PAIR_THRESHOLDS["field_relative_l2"]
            )
        comparison["geometry_rms_mm_passed"] = (
            comparison["geometry_rms_mm"] <= PAIR_THRESHOLDS["geometry_rms_mm"]
        )
        comparison["fit_rms_difference_mm"] = abs(
            float(results["off"][row_key]["fit_rms_mm"])
            - float(results["on"][row_key]["fit_rms_mm"])
        )
        comparison["motion_rms_difference_mm"] = abs(
            float(results["off"][row_key]["motion_rms_mm"])
            - float(results["on"][row_key]["motion_rms_mm"])
        )
        comparison["passed"] = comparison["geometry_rms_mm_passed"] and all(
            value["max_abs_passed"] and value["relative_l2_passed"]
            for value in comparison["fields"].values()
        )
        if step == 0:
            comparison["bitwise_initial_controls_required"] = True
            comparison["bitwise_initial_controls_passed"] = all(
                value["bitwise_equal"] for value in comparison["fields"].values()
            )
            comparison["passed"] = (
                comparison["passed"] and comparison["bitwise_initial_controls_passed"]
            )
        steps[str(step)] = comparison
    passed = steps["0"]["passed"] and steps["1"]["passed"]
    return {
        "status": "passed" if passed else "failed_original_first_update_thresholds",
        "passed": passed,
        "thresholds": PAIR_THRESHOLDS,
        "steps": steps,
        "gate_scope": "numerical off/on equivalence only; independent of evidence integrity",
    }


def main(cfg: Config) -> None:
    torch = VERIFY.torch
    torch.set_default_device("cpu")
    off_dir = resolve_recorded_path(cfg.off_dir)
    on_dir = resolve_recorded_path(cfg.on_dir)
    settings_path = resolve_recorded_path(cfg.settings)
    output_dir = resolve_recorded_path(cfg.output_dir)
    for path in (off_dir, on_dir):
        require(path.is_dir(), f"run directory is missing: {path}")
    require(settings_path.is_file(), f"settings file is missing: {settings_path}")
    require(
        off_dir != on_dir,
        "off/on directories are identical",
    )
    settings, _original_settings, intervention = settings_intervention(settings_path)
    settings_sha = sha256(settings_path)

    fixture_paths = {
        resolve_recorded_path(read_json(path / "config.json")["fixture"])
        for path in (off_dir, on_dir)
    }
    require(len(fixture_paths) == 1, f"off/on fixtures differ: {fixture_paths}")
    fixture_path = fixture_paths.pop()
    for name, expected in settings["inputs"].items():
        path = fixture_path / name
        require(
            sha256(path) == expected["sha256"], f"fixture input hash differs: {path}"
        )
    fixture = VERIFY.load_fixture(fixture_path)

    results = {
        label: verify_run(label, directory, fixture, settings, settings_sha)
        for label, directory in (("off", off_dir), ("on", on_dir))
    }
    equivalence = paired_equivalence(results)
    common_last = min(
        result["trace"]["last_accepted_step"] for result in results.values()
    )
    declared_targets = {result["trace"]["target_step"] for result in results.values()}
    budgets = {
        "declared_targets_equal": len(declared_targets) == 1,
        "declared_target_steps": {
            label: result["trace"]["target_step"] for label, result in results.items()
        },
        "last_accepted_steps": {
            label: result["trace"]["last_accepted_step"]
            for label, result in results.items()
        },
        "common_last_accepted_step": common_last,
        "both_reached_64": all(
            result["reached_step_64"] for result in results.values()
        ),
        "both_reached_128": all(
            result["reached_step_128"] for result in results.values()
        ),
        "both_completed_declared_budget": all(
            result["declared_budget_completed"] for result in results.values()
        ),
        "administratively_interrupted_arms": [
            label
            for label, result in results.items()
            if result["administrative_cutoff"]
        ],
        "numerically_failed_arms": [
            label for label, result in results.items() if result["numerical_failure"]
        ],
        "gate_scope": "budget and termination outcome only; independent of evidence integrity",
    }
    output_results = {}
    for label, result in results.items():
        output_results[label] = {
            key: value for key, value in result.items() if not key.startswith("_")
        }
    output = {
        "status": "passed_evidence_integrity",
        "evidence_integrity": {
            "status": "passed",
            "scope": "CPU-only saved-state and provenance verification; no equilibrium solve or optimizer update",
            "case_count": 2,
            "all_accepted_trace_and_solver_rows_valid": True,
            "scheduled_states_and_surfaces_valid": True,
            "optimizer_checkpoints_valid": True,
            "summary_aliases_valid": True,
        },
        "settings_intervention": intervention,
        "fixture": {
            "path": str(fixture_path.resolve()),
            "active_cells": len(fixture.active_ids),
            "same_muscle_edges": len(fixture.graph_i),
            "inputs": {
                name: record(fixture_path / name) for name in settings["inputs"]
            },
        },
        "budget_and_termination": budgets,
        "numerical_pair_equivalence": equivalence,
        "physical_diagnostics": {
            "status": "recorded_non_gating",
            "policy": "Inversions, determinant, shortening, fit, motion, and smoothness are diagnostics and are not evidence-integrity gates.",
            "cases": {
                label: result["physical_diagnostics_non_gating"]
                for label, result in results.items()
            },
        },
        "cases": output_results,
        "policy": {
            "administrative_cutoff": "KeyboardInterrupt is classified as an administrative stop after validating its accepted prefix and failure artifacts; it is not numerical instability.",
            "numerical_failure": "A non-administrative failure is reported separately while its accepted solver-valid prefix remains evidence.",
            "resumed_runs": "Parent checkpoint hashes, Adam state/counter, saved gradient, boundary state, copied trace, solver receipts, full checkpoints, and surfaces are checked exactly.",
            "source_authority": "Archived source snapshots and recorded hashes are authoritative; current-source drift is reported without rewriting original evidence.",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    require(
        not any(output_dir.iterdir()),
        f"refuse to overwrite verifier output: {output_dir}",
    )
    sources = output_dir / "sources"
    sources.mkdir()
    for path in (
        Path(__file__),
        GROUP / "src/50-verify.py",
        GROUP / "src/experiment_profile.py",
    ):
        shutil.copy2(path, sources / path.name)
    output["verifier_sources"] = {
        path.name: record(path) for path in sorted(sources.iterdir())
    }
    write_json(output_dir / "summary.json", output)
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            cherries.log_output(path)
    cherries.log_metrics(
        {
            "verification/evaluated_states": sum(
                result["trace"]["evaluated_states"] for result in results.values()
            ),
            "verification/full_checkpoints": sum(
                len(result["full_checkpoints"]) for result in results.values()
            ),
            "verification/common_last_step": common_last,
            "verification/both_reached_64": int(budgets["both_reached_64"]),
            "verification/both_reached_128": int(budgets["both_reached_128"]),
            "verification/equivalence_passed": int(equivalence["passed"]),
        }
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "output_dir": str(output_dir),
                "common_last_step": common_last,
                "equivalence": equivalence["status"],
            }
        )
    )


if __name__ == "__main__":
    cherries.main(
        main,
        profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit,
    )
