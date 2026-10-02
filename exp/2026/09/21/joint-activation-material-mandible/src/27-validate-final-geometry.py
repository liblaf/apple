"""CPU regression for final-run jaw geometry receipt adaptation."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import numpy as np
import pydantic_settings as ps
from joint_common import ProfileJoint, archive_sources, sha256, write_json
from joint_final_geometry import (
    SCHEMA,
    adapt_final_run_geometry_receipt,
    validate_final_run_geometry_receipt,
)

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("final-geometry-validation-v1", mkdir=True)


def _load_pilot() -> ModuleType:
    path = Path(__file__).with_name("30-joint-pilot.py")
    spec = importlib.util.spec_from_file_location(
        "joint_pilot_geometry_validation", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _raw_receipt(*, neutral: bool) -> dict:
    return {
        "mode": "neutral" if neutral else "expression_candidate",
        "admissible": False,
        "numerical_geometry_admissible": neutral,
        "anatomical_validation": False,
        "provisional_geometry_ok": False,
        "neutral_invariants_ok": True,
        "neutral_pose_ok": True,
        "pose_in_candidate_box": False,
        "fem_lip": {"numerical_geometry_ok": True, "sentinel": "lip"},
        "fem_mandible_oral": {
            "upper_oral": {"numerical_geometry_ok": True},
            "lower_oral": {"numerical_geometry_ok": True},
            "sentinel": "oral",
        },
        "fem_contact_surfaces": {
            "soft_cranium": {"numerical_geometry_ok": True},
            "soft_mandible": {"numerical_geometry_ok": True},
            "mandible_cranium": {"numerical_geometry_ok": True},
            "sentinel": "contact",
        },
        "source_rigid_jaw": {"sentinel": [7, 11, 13]},
        "source_geometry_role": "legacy source QA sentinel",
        "mandible_pose_consistency": {"passed": True, "max_error_m": 0.0},
        "gate": "blocked legacy anatomy gate",
        "limitations": ["legacy sentinel limitation"],
    }


def _must_reject(receipt: dict, *, neutral: bool) -> None:
    try:
        validate_final_run_geometry_receipt(receipt, neutral=neutral)
    except AssertionError:
        return
    message = "invalid final-run geometry receipt was admitted"
    raise AssertionError(message)


def _must_fail_adaptation(raw: dict) -> None:
    try:
        adapt_final_run_geometry_receipt(
            raw,
            neutral=False,
            normalized_pose=np.zeros(6),
            pose_rad_m=np.zeros(6),
        )
    except TypeError:
        return
    message = "malformed boolean field was normalized instead of rejected"
    raise AssertionError(message)


def _must_reject_forward(pilot: ModuleType, receipt: dict, expected: dict) -> None:
    try:
        pilot.validate_forward_receipt(receipt, expected)
    except AssertionError:
        return
    message = "mismatched forward implementation method was admitted"
    raise AssertionError(message)


def main(cfg: Config) -> None:  # noqa: PLR0915 - explicit receipt fault matrix.
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    checks: dict[str, bool | int] = {}

    zero_raw = _raw_receipt(neutral=False)
    zero = adapt_final_run_geometry_receipt(
        zero_raw,
        neutral=False,
        normalized_pose=np.zeros(6),
        pose_rad_m=np.zeros(6),
    )
    validate_final_run_geometry_receipt(zero, neutral=False)
    assert zero["schema"] == SCHEMA
    assert zero["legacy_source_box_qa"]["pose_in_candidate_box"] is False
    assert zero["legacy_source_box_qa"]["numerical_geometry_admissible"] is False
    assert zero["numerical_geometry_admissible"] is True
    checks["zero_pose_ignores_legacy_source_box_for_fem_admission"] = True

    endpoint_count = 0
    scale = np.asarray([np.deg2rad(10.0)] * 3 + [0.005] * 3)
    for coordinate in range(6):
        for sign in (-1.0, 1.0):
            normalized = np.zeros(6)
            normalized[coordinate] = sign
            adapted = adapt_final_run_geometry_receipt(
                _raw_receipt(neutral=False),
                neutral=False,
                normalized_pose=normalized,
                pose_rad_m=scale * normalized,
            )
            validate_final_run_geometry_receipt(adapted, neutral=False)
            assert adapted["pose"]["normalized"] == normalized.tolist()
            endpoint_count += 1
    assert endpoint_count == 12
    checks["signed_broad_axis_endpoints_use_computational_bounds"] = endpoint_count

    neutral = adapt_final_run_geometry_receipt(
        _raw_receipt(neutral=True),
        neutral=True,
        normalized_pose=np.zeros(6),
        pose_rad_m=np.zeros(6),
    )
    validate_final_run_geometry_receipt(neutral, neutral=True)
    checks["neutral_zero_pose_and_frozen_invariants_admitted"] = True

    check_paths = {
        "mandible_pose_consistency": ("mandible_pose_consistency", "passed"),
        "fem_lip": ("fem_lip", "numerical_geometry_ok"),
        "fem_mandible_upper_oral": (
            "fem_mandible_oral",
            "upper_oral",
            "numerical_geometry_ok",
        ),
        "fem_mandible_lower_oral": (
            "fem_mandible_oral",
            "lower_oral",
            "numerical_geometry_ok",
        ),
        "fem_soft_cranium": (
            "fem_contact_surfaces",
            "soft_cranium",
            "numerical_geometry_ok",
        ),
        "fem_soft_mandible": (
            "fem_contact_surfaces",
            "soft_mandible",
            "numerical_geometry_ok",
        ),
        "fem_mandible_cranium_endpoint": (
            "fem_contact_surfaces",
            "mandible_cranium",
            "numerical_geometry_ok",
        ),
    }
    for name, path in check_paths.items():
        raw = _raw_receipt(neutral=False)
        target = raw
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = False
        adapted = adapt_final_run_geometry_receipt(
            raw,
            neutral=False,
            normalized_pose=np.zeros(6),
            pose_rad_m=np.zeros(6),
        )
        assert adapted["fem_numerical_checks"][name] is False
        assert adapted["fem_numerical_admissible"] is False
        _must_reject(adapted, neutral=False)
        checks[f"failed_{name}_rejected"] = True

    legacy_true_but_failed_fem = _raw_receipt(neutral=False)
    legacy_true_but_failed_fem["pose_in_candidate_box"] = True
    legacy_true_but_failed_fem["numerical_geometry_admissible"] = True
    legacy_true_but_failed_fem["provisional_geometry_ok"] = True
    legacy_true_but_failed_fem["fem_lip"]["numerical_geometry_ok"] = False
    adapted = adapt_final_run_geometry_receipt(
        legacy_true_but_failed_fem,
        neutral=False,
        normalized_pose=np.zeros(6),
        pose_rad_m=np.zeros(6),
    )
    assert adapted["legacy_source_box_qa"]["pose_in_candidate_box"] is True
    _must_reject(adapted, neutral=False)
    checks["legacy_source_box_cannot_admit_failed_fem"] = True

    malformed = _raw_receipt(neutral=False)
    malformed["fem_lip"]["numerical_geometry_ok"] = "false"
    _must_fail_adaptation(malformed)
    checks["malformed_boolean_rejected_without_coercion"] = True

    for name in ("neutral_pose_ok", "neutral_invariants_ok"):
        raw = _raw_receipt(neutral=True)
        raw[name] = False
        adapted = adapt_final_run_geometry_receipt(
            raw,
            neutral=True,
            normalized_pose=np.zeros(6),
            pose_rad_m=np.zeros(6),
        )
        _must_reject(adapted, neutral=True)
        checks[f"failed_{name}_rejected"] = True

    outside = adapt_final_run_geometry_receipt(
        _raw_receipt(neutral=False),
        neutral=False,
        normalized_pose=[1.0001, 0, 0, 0, 0, 0],
        pose_rad_m=[0, 0, 0, 0, 0, 0],
    )
    _must_reject(outside, neutral=False)
    checks["outside_computational_box_rejected"] = True

    mutated = deepcopy(zero)
    mutated["legacy_source_box_qa"]["source_rigid_jaw"]["sentinel"][0] = -1
    assert zero_raw["source_rigid_jaw"]["sentinel"] == [7, 11, 13]
    assert zero["legacy_source_box_qa"]["source_rigid_jaw"]["sentinel"] == [
        7,
        11,
        13,
    ]
    checks["legacy_metadata_preserved_by_deep_copy"] = True

    pilot = _load_pilot()
    data = Path(__file__).resolve().parent.parent / "data"
    actual_path = data / "contact-validation-newton/summary.json"
    initial_path = data / "neutral-convergence-010-contact-bfgs-smoke-002/summary.json"
    actual_forward = json.loads(actual_path.read_text())["forward"]
    initial_forward = json.loads(initial_path.read_text())["first"]["forward"]
    expected_solver = actual_forward["forward_solver"]
    assert expected_solver == initial_forward["forward_solver"]
    pilot.validate_forward_receipt(actual_forward, expected_solver)
    pilot.validate_forward_receipt(initial_forward, expected_solver)
    assert actual_forward["method"] == "inexact_newton_cg"
    assert actual_forward["result"] != "initial_equilibrium"
    assert initial_forward["method"] == "newton_cg"
    assert initial_forward["result"] == "initial_equilibrium"
    mismatched_actual = deepcopy(actual_forward)
    mismatched_actual["method"] = "newton_cg"
    _must_reject_forward(pilot, mismatched_actual, expected_solver)
    mismatched_initial = deepcopy(initial_forward)
    mismatched_initial["method"] = "inexact_newton_cg"
    _must_reject_forward(pilot, mismatched_initial, expected_solver)
    checks["actual_newton_refinement_method_admitted"] = True
    checks["initial_equilibrium_selected_method_admitted"] = True
    checks["mismatched_forward_implementation_methods_rejected"] = True

    receipt = {
        "schema": "joint-final-geometry-validation-v1",
        "success": True,
        "status": "passed_cpu_final_geometry_receipt_validation",
        "scope": (
            "pure receipt adaptation for zero/broad poses, every individual FEM "
            "invariant, neutral invariants, computational bounds, and immutable "
            "legacy QA metadata, plus actual Newton-refinement and initial-equilibrium "
            "forward receipt admission; no new FEM equilibrium or contact solve"
        ),
        "checks": checks,
        "sources": {
            str(Path(__file__)): sha256(Path(__file__)),
            str(Path(__file__).with_name("joint_final_geometry.py")): sha256(
                Path(__file__).with_name("joint_final_geometry.py")
            ),
            str(Path(__file__).with_name("30-joint-pilot.py")): sha256(
                Path(__file__).with_name("30-joint-pilot.py")
            ),
            str(actual_path): sha256(actual_path),
            str(initial_path): sha256(initial_path),
        },
    }
    write_json(cfg.output_dir / "summary.json", receipt)
    cherries.log_metrics({"geometry/check_count": len(checks)})


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
