"""CPU validation for shared prior centers and admission fingerprints."""

from __future__ import annotations

import copy
import importlib.util
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import ModuleType

import pydantic_settings as ps
import torch
from joint_common import ProfileJoint, archive_sources, sha256, write_json

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path


def _load_pilot() -> ModuleType:
    path = Path(__file__).with_name("30-joint-pilot.py")
    spec = importlib.util.spec_from_file_location("joint_pilot_shared_contract", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _field_contract(basis: str) -> dict:
    count = 80 if basis == "spatial80" else 20
    return {
        "basis": basis,
        "coefficient_count": count,
        "skin_baseline_index": count - 2,
        "skin_log_multiplier_index": count - 1,
    }


def _must_reject(callable_: Callable[[], object]) -> None:
    try:
        callable_()
    except (AssertionError, KeyError, TypeError, ValueError):
        return
    msg = "altered shared lineage was accepted"
    raise AssertionError(msg)


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    pilot = _load_pilot()
    checks: dict[str, bool | float] = {}
    for basis in ("constant20", "spatial80"):
        field = _field_contract(basis)
        initialization = torch.linspace(
            -0.4,
            0.7,
            field["coefficient_count"],
            dtype=torch.float64,
        )
        initialization[field["skin_baseline_index"]] = 1.17676
        initialization[field["skin_log_multiplier_index"]] = 0.8
        reference = pilot.build_shared_prior_reference(initialization, field)
        assert torch.equal(
            reference[: field["skin_log_multiplier_index"]],
            initialization[: field["skin_log_multiplier_index"]],
        )
        assert float(reference[field["skin_baseline_index"]]) == 1.17676
        assert float(reference[field["skin_log_multiplier_index"]]) == 0.0
        initialization_values = initialization.tolist()
        reference_values = reference.tolist()
        initialization_hash = pilot.canonical_sha256(initialization_values)
        reference_hash = pilot.canonical_sha256(reference_values)

        calibration = {
            "shared_initialization": initialization_values,
            "shared_prior_reference": reference_values,
            "shared_initialization_sha256": initialization_hash,
            "shared_prior_reference_sha256": reference_hash,
        }
        pilot.validate_shared_prior_lineage(
            calibration,
            initialization_values=initialization_values,
            prior_reference_values=reference_values,
            initialization_sha256=initialization_hash,
            prior_reference_sha256=reference_hash,
        )
        control = {
            "shared_initialization_sha256": initialization_hash,
            "shared_prior_reference_sha256": reference_hash,
            "protocol": {
                "shared_initialization": initialization_values,
                "shared_prior_reference": reference_values,
            },
        }
        pilot.validate_shared_prior_lineage(
            control,
            initialization_values=initialization_values,
            prior_reference_values=reference_values,
            initialization_sha256=initialization_hash,
            prior_reference_sha256=reference_hash,
        )
        altered_calibration = copy.deepcopy(calibration)
        altered_calibration["shared_prior_reference"][-1] = 0.01
        _must_reject(
            partial(
                pilot.validate_shared_prior_lineage,
                altered_calibration,
                initialization_values=initialization_values,
                prior_reference_values=reference_values,
                initialization_sha256=initialization_hash,
                prior_reference_sha256=reference_hash,
            )
        )
        altered_control = copy.deepcopy(control)
        altered_control["shared_prior_reference_sha256"] = "0" * 64
        _must_reject(
            partial(
                pilot.validate_shared_prior_lineage,
                altered_control,
                initialization_values=initialization_values,
                prior_reference_values=reference_values,
                initialization_sha256=initialization_hash,
                prior_reference_sha256=reference_hash,
            )
        )
        comparison = {
            "shared_basis": basis,
            "shared_initialization_sha256": initialization_hash,
            "shared_prior_reference_sha256": reference_hash,
        }
        fingerprint = pilot.canonical_sha256(comparison)
        altered_comparison = copy.deepcopy(comparison)
        altered_comparison["shared_prior_reference_sha256"] = "0" * 64
        assert pilot.canonical_sha256(altered_comparison) != fingerprint
        checks[f"{basis}_skin_baseline_preserved"] = True
        checks[f"{basis}_study_stiffness_center_zero"] = True
        checks[f"{basis}_calibration_lineage_admitted"] = True
        checks[f"{basis}_control_lineage_admitted"] = True
        checks[f"{basis}_altered_lineage_rejected"] = True
        checks[f"{basis}_joint_fingerprint_sensitive"] = True

    receipt = {
        "schema": "joint-shared-prior-contract-validation-v1",
        "success": True,
        "status": "passed_cpu_shared_prior_and_lineage_validation",
        "scope": (
            "constant20/Spatial80 index handling, nonzero skin-baseline center, "
            "study skin-stiffness center, calibration/control lineage, and joint "
            "comparison-fingerprint sensitivity; no FEM solve"
        ),
        "checks": checks,
        "sources": {
            str(Path(__file__)): sha256(Path(__file__)),
            str(Path(__file__).with_name("30-joint-pilot.py")): sha256(
                Path(__file__).with_name("30-joint-pilot.py")
            ),
        },
    }
    write_json(cfg.output_dir / "summary.json", receipt)
    cherries.log_metrics({"shared_contract/check_count": len(checks)})


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
