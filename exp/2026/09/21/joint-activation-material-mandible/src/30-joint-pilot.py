"""Converged preparation and finite-budget all-field joint optimization.

Calibrate strong smoothness, converge a fixed-shared activation/jaw parent, and
then release every field for the final trend run. Every expression backward is
completed before the next expression is solved.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
import optree
import pydantic_settings as ps
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import (
    PreparedInputs,
    audit_deformed_oral_geometry,
    audit_neutral_oral_geometry,
)
from joint_diagnostics import (
    activation_mapping_summary,
    activation_projected_gradient_mapping,
    activation_spectrum_summary,
    activation_step_summary,
    box_projected_gradient_mapping,
    mass_normalized_activation_gradient_summary,
    objective_stability_summary,
    parameter_block_mapping_summary,
    parameter_step_summary,
    relative_reduction_summary,
    reversible_projected_optimizer_preview,
    shared_prior_summary,
    shared_projected_gradient_mapping,
    shared_reference_departure_quadratic,
)
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from joint_fields import (
    BULK_TISSUES,
    SharedFieldParameters,
    VolumeGraph,
    activation_regularizers,
    activation_stresses_mpa,
    project_activation_,
    symmetric_coordinates,
)
from joint_final_geometry import (
    adapt_final_run_geometry_receipt,
    validate_final_run_geometry_receipt,
)
from joint_physics import JointPhysics
from joint_rigid_bone_collision import RigidBoneCollision
from joint_spatial_fields import (
    EXPECTED_BASIS_SHA256,
    EXPECTED_MESH_CELL_COUNT,
    SpatialSharedFieldParameters,
)

from liblaf import cherries

LOG = logging.getLogger(__name__)
COMPLETED = False
REQUIRED_FINAL_SKIN_PRESTRESS_FRACTION = 1.0
FROZEN_SKIN_PROXY_N_PER_M = 80.6
SPATIAL_SMOOTHNESS_WEIGHT = 100.0
SPATIAL_SMOOTHNESS_FACTOR = 0.5
SharedParameters = SharedFieldParameters | SpatialSharedFieldParameters
EXPECTED_TRIAL_FAILURES = (
    ForwardConvergenceError,
    AssertionError,
    FloatingPointError,
)
EXPECTED_PROJECTION_FAILURES = (ForwardConvergenceError, FloatingPointError)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    prepared_dir: Path = GROUP / "data/prepared"
    neutral_checkpoint: Path
    calibration: Path = GROUP / "data/strong-calibration/calibration.json"
    initial_checkpoint: Path | None = None
    control_receipt: Path | None = None
    contact_spec: Path
    contact_validation: Path
    newton_synthetic_validation: Path = (
        GROUP / "data/contact-validation-newton/summary.json"
    )
    newton_face_gradient_validation: Path = (
        GROUP / "data/face-gradient-validation-contact-newton/summary.json"
    )
    rigid_bone_ccd_validation: Path = (
        GROUP / "data/rigid-bone-ccd-validation-003/summary.json"
    )
    output_dir: Path = cherries.output("joint-pilot", mkdir=True)
    stage: Literal["calibrate", "control_converge", "joint_trend"] = "joint_trend"
    activation_neighbor_rms_budget_dimensionless: float
    preparation_max_updates: int = 1000
    preparation_wall_budget_seconds: float = 86400
    final_updates: int = 100
    final_wall_budget_seconds: float = 43200
    activation_learning_rate: float = 0.003
    shared_learning_rate: float = 0.0003
    jaw_learning_rate: float = 0.003
    adam_eps: float = 1e-8
    magnitude_weight: float = 0.001
    shared_prior_weight: float = 0.01
    jaw_prior_weight: float = 0.01
    seed: int = 20260921
    forward_rtol: float = 1e-6
    forward_atol: float = 1e-12
    adjoint_rtol: float = 1e-7
    max_forward_steps: int = 10000
    forward_method: Literal["pncg", "newton_cg"] = "pncg"
    newton_linear_rtol: float = 1e-3
    newton_max_steps: int = 12
    calibration_probe_scale_dimensionless: float = 0.001
    calibration_relative_tightening_factor: float = 0.5
    calibration_ratio_relative_tolerance: float = 0.25
    convergence_window: int = 5
    convergence_consecutive: int = 5
    projected_gradient_relative_tolerance: float = 0.01
    objective_relative_span_tolerance: float = 1.0e-4
    activation_step_volume_rms_tolerance: float = 1.0e-4
    activation_step_max_tet_tolerance: float = 1.0e-3
    jaw_step_coordinate_rms_tolerance: float = 1.0e-4
    jaw_step_max_row_tolerance: float = 1.0e-3
    outer_backtrack_factor: float = 0.5
    outer_max_backtracks: int = 8
    outer_armijo_coefficient: float = 1.0e-4


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def build_shared_prior_reference(
    initialization: torch.Tensor,
    shared_field: dict[str, Any],
) -> torch.Tensor:
    """Keep neutral bulk/baseline centers and restore study skin stiffness."""
    count = int(shared_field["coefficient_count"])
    baseline_index = int(shared_field["skin_baseline_index"])
    stiffness_index = int(shared_field["skin_log_multiplier_index"])
    assert initialization.shape == (count,)
    assert baseline_index == count - 2
    assert stiffness_index == count - 1
    reference = initialization.detach().clone()
    reference[stiffness_index] = 0.0
    assert torch.equal(reference[:stiffness_index], initialization[:stiffness_index])
    assert torch.equal(reference[baseline_index], initialization[baseline_index])
    return reference


def validate_shared_prior_lineage(
    artifact: dict[str, Any],
    *,
    initialization_values: list[float],
    prior_reference_values: list[float],
    initialization_sha256: str,
    prior_reference_sha256: str,
) -> None:
    """Require exact shared initialization/prior-center admission lineage."""
    assert artifact["shared_initialization_sha256"] == initialization_sha256
    assert artifact["shared_prior_reference_sha256"] == prior_reference_sha256
    vector_source = artifact
    if "shared_initialization" not in vector_source:
        vector_source = artifact["protocol"]
    assert vector_source["shared_initialization"] == initialization_values
    assert vector_source["shared_prior_reference"] == prior_reference_values
    assert canonical_sha256(vector_source["shared_initialization"]) == (
        initialization_sha256
    )
    assert canonical_sha256(vector_source["shared_prior_reference"]) == (
        prior_reference_sha256
    )


def forward_solver_contract(cfg: Config) -> dict[str, Any]:
    return {
        "method": cfg.forward_method,
        "newton_linear_rtol": cfg.newton_linear_rtol,
        "newton_max_steps": cfg.newton_max_steps,
        "fallback": None,
    }


def validate_forward_solver_contract(
    receipt: dict[str, Any], expected: dict[str, Any]
) -> None:
    assert receipt["method"] == expected["method"], (receipt, expected)
    assert math.isclose(
        float(receipt["newton_linear_rtol"]),
        float(expected["newton_linear_rtol"]),
        rel_tol=0,
        abs_tol=0,
    ), (receipt, expected)
    assert int(receipt["newton_max_steps"]) == int(expected["newton_max_steps"]), (
        receipt,
        expected,
    )
    assert receipt.get("fallback") is None, receipt


def activation_neighbor_rms(
    coordinates: torch.Tensor, graph: VolumeGraph
) -> torch.Tensor:
    delta = coordinates[..., graph.i, :] - coordinates[..., graph.j, :]
    weighted = (graph.conductance_m * delta.square().sum(dim=-1)).sum(dim=-1)
    return torch.sqrt(weighted / graph.conductance_m.sum())


def validate_final_neutral_target(initial: dict[str, Any]) -> None:
    protocol = initial["protocol"]
    fraction = float(protocol["skin_prestress_fraction"])
    target = float(protocol["skin_target_n_per_m"])
    assert fraction == REQUIRED_FINAL_SKIN_PRESTRESS_FRACTION, fraction
    assert math.isclose(
        target, FROZEN_SKIN_PROXY_N_PER_M, rel_tol=0.0, abs_tol=1.0e-12
    ), target
    skin = initial["materials"]["materials"]["skin"]
    proxy = float(skin["continuation"]["source_proxy_mean_n_per_m"])
    assert math.isclose(
        proxy, FROZEN_SKIN_PROXY_N_PER_M, rel_tol=0.0, abs_tol=1.0e-12
    ), proxy
    coordinate = int(
        initial["materials"]["parameterization"]["skin_isotropic_resultant_coordinate"]
    )
    shared_basis = protocol.get("shared_basis", "constant20")
    assert shared_basis in {"constant20", "spatial80"}
    assert coordinate == (78 if shared_basis == "spatial80" else 18)
    scale = float(skin["reference_resultant_scale_n_per_m"])
    checkpoint_resultant = float(initial["shared_coefficients"][coordinate]) * scale
    assert math.isclose(
        checkpoint_resultant,
        FROZEN_SKIN_PROXY_N_PER_M,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ), checkpoint_resultant


def validate_shared_basis_contract(  # noqa: PLR0915
    initial: dict[str, Any],
    *,
    contact_spec_sha256: str,
    expected_solver: dict[str, Any],
    expected_tolerances: dict[str, float | int],
    input_arrays_sha256: str,
    input_manifest_sha256: str,
) -> tuple[str, dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """Validate constant20 or opt-in Spatial80 evidence from the neutral parent."""
    protocol = initial["protocol"]
    shared_basis = protocol.get("shared_basis", "constant20")
    assert shared_basis in {"constant20", "spatial80"}
    shared_field = protocol.get("shared_field")
    if shared_field is None:
        assert shared_basis == "constant20"
        shared_field = {
            "schema": "joint-additive-stress-fields-v1",
            "basis": "constant20",
            "coefficient_count": 20,
            "skin_baseline_index": 18,
            "skin_log_multiplier_index": 19,
        }
    if "shared_field" in initial:
        assert initial["shared_field"] == shared_field
    expected_count = 80 if shared_basis == "spatial80" else 20
    assert shared_field["basis"] == shared_basis
    assert int(shared_field["coefficient_count"]) == expected_count
    assert initial["shared_coefficients"].shape == (expected_count,)
    assert int(shared_field["skin_baseline_index"]) == expected_count - 2
    assert int(shared_field["skin_log_multiplier_index"]) == expected_count - 1
    evidence: dict[str, str] = {}
    if shared_basis == "constant20":
        assert shared_field["schema"] == "joint-additive-stress-fields-v1"
        assert protocol.get("spatial_basis") is None
        return shared_basis, shared_field, None, evidence

    assert shared_field["schema"] == "joint-additive-spatial-stress-fields-v1"
    assert shared_field["bulk_anchor_counts"] == {
        "fat": 4,
        "aponeurosis": 4,
        "muscle": 5,
    }
    assert shared_field["bulk_coordinate_slices"] == {
        "fat": [0, 24],
        "aponeurosis": [24, 48],
        "muscle": [48, 78],
    }
    assert shared_field["fixed_coordinate_ids"] == [78]
    assert shared_field["free_coordinate_ids"] == [*range(78), 79]
    smoothness = shared_field["spatial_smoothness"]
    assert float(smoothness["weight"]) == SPATIAL_SMOOTHNESS_WEIGHT
    assert float(smoothness["factor"]) == SPATIAL_SMOOTHNESS_FACTOR
    assert smoothness["regularizer"] == "bulk_spatial_roughness"
    spatial_basis = protocol["spatial_basis"]
    assert spatial_basis["schema"] == "joint-spatial-basis-binding-v1"
    assert spatial_basis["basis_sha256"] == EXPECTED_BASIS_SHA256
    assert int(spatial_basis["mesh_cell_count"]) == EXPECTED_MESH_CELL_COUNT
    assert spatial_basis["input_arrays_sha256"] == input_arrays_sha256
    assert spatial_basis["input_manifest_sha256"] == input_manifest_sha256
    basis_path = Path(spatial_basis["basis_path"])
    audit_path = Path(spatial_basis["audit_summary_path"])
    assert basis_path.is_file()
    assert audit_path.is_file()
    assert sha256(basis_path) == spatial_basis["basis_sha256"]
    assert sha256(audit_path) == spatial_basis["audit_summary_sha256"]
    evidence[str(basis_path)] = spatial_basis["basis_sha256"]
    evidence[str(audit_path)] = spatial_basis["audit_summary_sha256"]

    validation = protocol["spatial_validation"]
    for name in ("cpu", "full_face_directional"):
        item = validation[name]
        path = Path(item["path"])
        assert path.is_file()
        assert sha256(path) == item["sha256"]
        receipt = json.loads(path.read_text())
        assert receipt == item["receipt"]
        assert receipt["success"] is True
        assert receipt["basis"] == spatial_basis
        evidence[str(path)] = item["sha256"]
    cpu = validation["cpu"]["receipt"]
    assert cpu["schema"] == "joint-spatial-field-validation-v1"
    assert cpu["status"] == "passed_cpu_inactive_spatial_field_validation"
    assert cpu["implicit_default_device_probe"].startswith("cuda")
    assert len(cpu["checks"]) == 19
    assert all(row["passed"] is True for row in cpu["checks"].values())
    for path_text, digest in cpu["sources"].items():
        source_path = Path(path_text)
        assert source_path.is_file()
        assert sha256(source_path) == digest
    directional = validation["full_face_directional"]["receipt"]
    assert directional["schema"] == "joint-spatial-face-gradient-validation-v1"
    assert directional["success"] is True
    assert directional["contact_enabled"] is True
    assert directional["contact_spec_sha256"] == contact_spec_sha256
    assert directional["basis"] == spatial_basis
    assert directional["shared_field_schema"] == shared_field["schema"]
    assert directional["input_arrays_sha256"] == input_arrays_sha256
    assert directional["input_manifest_sha256"] == input_manifest_sha256
    assert directional["cpu_validation_sha256"] == validation["cpu"]["sha256"]
    assert (
        Path(directional["cpu_validation_path"]).resolve()
        == Path(validation["cpu"]["path"]).resolve()
    )
    assert len(directional["checks"]) == 16
    assert int(directional["forward_count"]) == 33
    assert float(directional["maximum_relative_error"]) < 0.02
    assert float(directional["maximum_mechanical_relative_error"]) < 0.02
    assert all(
        math.isfinite(float(row[key])) and float(row[key]) < 0.02
        for row in directional["checks"]
        for key in ("relative_error", "mechanical_relative_error")
    )
    assert float(directional["spatial_smoothness_weight"]) == (
        SPATIAL_SMOOTHNESS_WEIGHT
    )
    assert float(directional["spatial_smoothness_factor"]) == (
        SPATIAL_SMOOTHNESS_FACTOR
    )
    validate_forward_solver_contract(directional["forward_solver"], expected_solver)
    assert directional["forward_tolerances"] == expected_tolerances
    assert directional["adjoint"]["success"] is True
    assert float(directional["adjoint"]["relative_residual"]) <= float(
        expected_tolerances["adjoint_rtol"]
    )
    for path_text, digest in directional["implementation_sha256"].items():
        source_path = Path(path_text)
        assert source_path.is_file()
        assert sha256(source_path) == digest
    return shared_basis, shared_field, spatial_basis, evidence


def validate_implementation_sha256(
    observed: dict[str, str], expected: dict[str, str], *, artifact: str
) -> None:
    """Require an admitted artifact to use the exact current implementation."""
    assert observed == expected, (
        f"{artifact} implementation hashes do not match the current runner",
        observed,
        expected,
    )


def backtracked_projected_adam_step(  # noqa: C901, PLR0915
    *,
    optimizer: torch.optim.Optimizer,
    parameters: tuple[torch.nn.Parameter, ...],
    prepare_step: Any,
    project: Any,
    restore_auxiliary: Any,
    evaluate_trial: Any,
    cost_counters: Any,
    accepted_objective: float,
    backtrack_factor: float,
    maximum_trials: int,
    armijo_coefficient: float,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Try one projected Adam direction as an owned-state transaction."""
    assert 0 < backtrack_factor < 1
    assert maximum_trials > 0
    assert 0 < armijo_coefficient < 1
    accepted_parameters = tuple(value.detach().clone() for value in parameters)
    accepted_gradients = tuple(
        None if value.grad is None else value.grad.detach().clone()
        for value in parameters
    )
    optimizer_state = deepcopy(optimizer.state_dict())

    def restore_accepted(*, restore_optimizer: bool) -> None:
        with torch.no_grad():
            for parameter, accepted in zip(
                parameters, accepted_parameters, strict=True
            ):
                parameter.copy_(accepted)
        for parameter, gradient in zip(parameters, accepted_gradients, strict=True):
            parameter.grad = None if gradient is None else gradient.detach().clone()
        if restore_optimizer:
            optimizer.load_state_dict(optimizer_state)
        restore_auxiliary()

    try:
        prepare_step()
        optimizer.step()
        full_projection = project()
    except EXPECTED_PROJECTION_FAILURES as error:
        restore_accepted(restore_optimizer=True)
        return {
            "accepted": False,
            "reason": "projected Adam proposal or projection failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "trials": [],
        }
    proposed_parameters = tuple(value.detach().clone() for value in parameters)
    directions = tuple(
        proposed - accepted
        for proposed, accepted in zip(
            proposed_parameters, accepted_parameters, strict=True
        )
    )
    directional_derivative = sum(
        float(torch.sum(gradient * direction))
        for gradient, direction in zip(accepted_gradients, directions, strict=True)
        if gradient is not None
    )
    trials: list[dict[str, Any]] = []

    if not math.isfinite(directional_derivative) or directional_derivative >= 0:
        restore_accepted(restore_optimizer=True)
        return {
            "accepted": False,
            "reason": "projected Adam direction is not a finite descent direction",
            "directional_derivative": directional_derivative,
            "full_projection": full_projection,
            "trials": trials,
        }

    fraction = 1.0
    for trial_index in range(maximum_trials):
        if deadline is not None and time.perf_counter() >= deadline:
            restore_accepted(restore_optimizer=True)
            return {
                "accepted": False,
                "budget_exhausted": True,
                "reason": "wall-time budget before outer trial",
                "directional_derivative": directional_derivative,
                "full_projection": full_projection,
                "trials": trials,
            }
        restore_accepted(restore_optimizer=False)
        with torch.no_grad():
            for parameter, accepted, direction in zip(
                parameters, accepted_parameters, directions, strict=True
            ):
                parameter.copy_(accepted + fraction * direction)
        started = time.perf_counter()
        cost_before = cost_counters()
        try:
            scaled_projection = project()
        except EXPECTED_PROJECTION_FAILURES as error:
            trials.append(
                {
                    "trial": trial_index,
                    "fraction": fraction,
                    "accepted": False,
                    "reason": "scaled projection failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "seconds": time.perf_counter() - started,
                    "forward_solves": 0,
                    "adjoint_solves": 0,
                }
            )
            fraction *= backtrack_factor
            continue
        trial: dict[str, Any] = {
            "trial": trial_index,
            "fraction": fraction,
            "armijo_rhs": (
                accepted_objective
                + armijo_coefficient * fraction * directional_derivative
            ),
            "projection": scaled_projection,
        }
        try:
            result = evaluate_trial()
            objective = float(result["objective"])
            trial.update(result)
            trial["seconds"] = time.perf_counter() - started
            cost_after = cost_counters()
            trial["forward_solves"] = cost_after[0] - cost_before[0]
            trial["adjoint_solves"] = cost_after[1] - cost_before[1]
            if not math.isfinite(objective):
                trial["accepted"] = False
                trial["reason"] = "nonfinite objective"
            elif objective > trial["armijo_rhs"]:
                trial["accepted"] = False
                trial["reason"] = "Armijo decrease"
            else:
                trial["accepted"] = True
                trial["reason"] = "Armijo and numerical gates passed"
                trials.append(trial)
                return {
                    "accepted": True,
                    "reason": trial["reason"],
                    "accepted_fraction": fraction,
                    "directional_derivative": directional_derivative,
                    "full_projection": full_projection,
                    "trials": trials,
                }
        except EXPECTED_TRIAL_FAILURES as error:
            trial.update(
                {
                    "accepted": False,
                    "reason": "forward, adjoint, contact, or numerical gate failure",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "seconds": time.perf_counter() - started,
                }
            )
            cost_after = cost_counters()
            trial["forward_solves"] = cost_after[0] - cost_before[0]
            trial["adjoint_solves"] = cost_after[1] - cost_before[1]
        trials.append(trial)
        fraction *= backtrack_factor

    restore_accepted(restore_optimizer=True)
    return {
        "accepted": False,
        "budget_exhausted": False,
        "reason": "outer projected-Adam line search exhausted",
        "directional_derivative": directional_derivative,
        "full_projection": full_projection,
        "trials": trials,
    }


def require_gates(  # noqa: C901, PLR0912, PLR0915 - explicit readiness evidence.
    cfg: Config, output: Path
) -> dict[str, str]:
    expected_solver = forward_solver_contract(cfg)
    paths = [
        GROUP / "data/material-validation/summary.json",
        GROUP / "data/field-validation/summary.json",
        GROUP / "data/coupled-validation/summary.json",
        GROUP / "data/equilibrium-validation/summary.json",
        GROUP / "data/face-gradient-validation/summary.json",
        GROUP / "data/face-gradient-validation-contact/summary.json",
        cfg.contact_validation,
        cfg.rigid_bone_ccd_validation,
    ]
    if cfg.forward_method == "newton_cg":
        paths.extend(
            [cfg.newton_synthetic_validation, cfg.newton_face_gradient_validation]
        )
    missing = []
    evidence = {}
    admitted_shared_basis = None
    for path in paths:
        if not path.exists():
            missing.append(f"missing validation: {path.name} in {path.parent.name}")
            continue
        value = json.loads(path.read_text())
        if value.get("success") is not True:
            missing.append(f"validation did not pass: {path}")
        evidence[str(path)] = sha256(path)
    if not cfg.contact_spec.exists():
        missing.append(f"missing contact specification: {cfg.contact_spec}")
    elif cfg.contact_validation.exists():
        contact_validation = json.loads(cfg.contact_validation.read_text())
        if contact_validation.get("schema") != "joint-contact-validation-v1":
            missing.append("contact validation has the wrong schema")
        if contact_validation.get("contact_spec_sha256") != sha256(cfg.contact_spec):
            missing.append(
                "contact validation does not match the contact specification"
            )
        diagnostics = contact_validation.get("contact_diagnostics", {})
        if diagnostics.get("contact_numerically_valid") is not True:
            missing.append("contact validation lacks valid IPC diagnostics")
    contact_face_path = GROUP / "data/face-gradient-validation-contact/summary.json"
    if contact_face_path.exists() and cfg.contact_spec.exists():
        contact_face = json.loads(contact_face_path.read_text())
        if contact_face.get("contact_enabled") is not True:
            missing.append("full-face contact derivative validation disabled contact")
        if contact_face.get("contact_spec_sha256") != sha256(cfg.contact_spec):
            missing.append(
                "full-face contact derivative validation does not match the contact specification"
            )
        contact_receipt = contact_face.get("last_forward", {}).get("contact", {})
        if contact_receipt.get("contact_numerically_valid") is not True:
            missing.append(
                "full-face contact derivative validation lacks a valid contact receipt"
            )
    if cfg.forward_method == "newton_cg" and cfg.contact_spec.exists():
        if cfg.newton_synthetic_validation.exists():
            synthetic = json.loads(cfg.newton_synthetic_validation.read_text())
            try:
                assert synthetic["schema"] == "joint-contact-validation-v1"
                assert synthetic["success"] is True
                assert synthetic["contact_spec_sha256"] == sha256(cfg.contact_spec)
                validate_forward_solver_contract(
                    synthetic["forward_solver"], expected_solver
                )
                assert (
                    synthetic["contact_diagnostics"]["contact_numerically_valid"]
                    is True
                )
            except (AssertionError, KeyError, TypeError, ValueError) as error:
                missing.append(
                    "Newton synthetic contact/implicit validation does not match "
                    f"the selected solver contract: {error}"
                )
        if cfg.newton_face_gradient_validation.exists():
            full_face = json.loads(cfg.newton_face_gradient_validation.read_text())
            try:
                assert full_face["success"] is True
                assert full_face["contact_enabled"] is True
                assert full_face["contact_spec_sha256"] == sha256(cfg.contact_spec)
                assert len(full_face["checks"]) == 16
                assert float(full_face["maximum_relative_error"]) < 0.02
                validate_forward_solver_contract(
                    full_face["forward_solver"], expected_solver
                )
                tolerances = full_face["forward_tolerances"]
                assert float(tolerances["rtol"]) == cfg.forward_rtol
                assert float(tolerances["atol"]) == cfg.forward_atol
                assert float(tolerances["adjoint_rtol"]) == cfg.adjoint_rtol
                assert int(tolerances["max_steps"]) == cfg.max_forward_steps
                validate_contact_receipt(full_face["last_forward"]["contact"])
            except (AssertionError, KeyError, TypeError, ValueError) as error:
                missing.append(
                    "Newton full-face contact derivative validation does not match "
                    f"the selected solver and production tolerances: {error}"
                )
    if cfg.rigid_bone_ccd_validation.exists():
        rigid_bone_validation = json.loads(cfg.rigid_bone_ccd_validation.read_text())
        try:
            validate_rigid_bone_ccd_validation(
                rigid_bone_validation,
                input_arrays_sha256=sha256(cfg.prepared_dir / "inputs.npz"),
                input_manifest_sha256=sha256(cfg.prepared_dir / "manifest.json"),
            )
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            missing.append(
                "rigid-bone straight-boundary CCD validation does not match "
                f"the current input and source contract: {error}"
            )
    if not cfg.neutral_checkpoint.exists():
        missing.append(f"missing neutral checkpoint: {cfg.neutral_checkpoint}")
    else:
        neutral = torch.load(
            cfg.neutral_checkpoint, map_location="cpu", weights_only=False
        )
        try:
            validate_final_neutral_target(neutral)
            validate_forward_solver_contract(
                neutral["protocol"]["forward_solver"], expected_solver
            )
            if cfg.contact_spec.exists():
                (
                    admitted_shared_basis,
                    _,
                    _,
                    spatial_evidence,
                ) = validate_shared_basis_contract(
                    neutral,
                    contact_spec_sha256=sha256(cfg.contact_spec),
                    expected_solver=expected_solver,
                    expected_tolerances={
                        "rtol": cfg.forward_rtol,
                        "atol": cfg.forward_atol,
                        "adjoint_rtol": cfg.adjoint_rtol,
                        "max_steps": cfg.max_forward_steps,
                    },
                    input_arrays_sha256=sha256(cfg.prepared_dir / "inputs.npz"),
                    input_manifest_sha256=sha256(cfg.prepared_dir / "manifest.json"),
                )
                evidence.update(spatial_evidence)
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            missing.append(
                "neutral checkpoint has not completed the required 1.0 fraction, "
                "80.6 N/m proxy target with the selected solver contract: "
                f"{error}"
            )
        evidence[str(cfg.neutral_checkpoint)] = sha256(cfg.neutral_checkpoint)
    write_json(
        output / "readiness.json",
        {
            "schema": "joint-final-readiness-v1",
            "stage": "readiness",
            "ready": not missing,
            "missing": missing,
            "evidence": evidence,
            "forward_solver": expected_solver,
            "shared_basis": admitted_shared_basis,
        },
    )
    assert not missing, missing
    return evidence


def physics_from_spec(
    cfg: Config, prepared: PreparedInputs, spec: dict, contact_config: dict
) -> JointPhysics:
    materials = spec["materials"]
    skin = materials["skin"]
    return JointPhysics(
        prepared.volume_path,
        prepared.skin_path,
        prepared.arrays,
        bulk_young_mpa={n: materials[n]["young_mpa"] for n in BULK_TISSUES},
        bulk_nu={n: materials[n]["poisson"] for n in BULK_TISSUES},
        skin_young_mpa=skin["reference_map"]["young_mpa"],
        skin_nu=skin["poisson"],
        thickness_m=skin["thickness_m"],
        rtol=cfg.forward_rtol,
        atol=cfg.forward_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        max_steps=cfg.max_forward_steps,
        forward_method=cfg.forward_method,
        newton_linear_rtol=cfg.newton_linear_rtol,
        newton_max_steps=cfg.newton_max_steps,
        contact_config=contact_config,
    )


def validate_contact_receipt(receipt: dict[str, Any]) -> None:
    assert receipt["enabled"] is True, receipt
    assert receipt["contact_numerically_valid"] is True, receipt
    assert math.isfinite(float(receipt["barrier_energy"])), receipt
    assert int(receipt["active_contact_count"]) >= 0, receipt
    distance = receipt["minimum_active_distance_m"]
    assert distance is None or (math.isfinite(float(distance)) and float(distance) > 0)
    for name in ("ccd_boundary_fraction", "ccd_minimum_inner_fraction"):
        value = float(receipt[name])
        assert math.isfinite(value), receipt
        assert 0 < value <= 1, receipt


def validate_rigid_bone_ccd_validation(
    receipt: dict[str, Any],
    *,
    input_arrays_sha256: str,
    input_manifest_sha256: str,
) -> None:
    """Require the frozen straight-boundary-motion CCD implementation evidence."""
    assert receipt["schema"] == "joint-rigid-bone-ccd-validation-v1"
    assert receipt["success"] is True
    assert receipt["input_arrays_sha256"] == input_arrays_sha256
    assert receipt["input_manifest_sha256"] == input_manifest_sha256
    assert receipt["probe_count"] == 13
    assert len(receipt["probes"]) == 13
    assert receipt["probes"][0]["label"] == "zero"
    assert receipt["probes"][0]["numerically_admissible"] is True
    assert receipt["reference"]["numerically_admissible"] is True
    assert receipt["whole_pose_box_validated"] is False
    assert receipt["bone_bone_energy_added"] is False
    crossing = receipt["synthetic_checks"]["crossing_with_disjoint_endpoints"]
    assert crossing["end_intersects"] is False
    assert crossing["numerically_admissible"] is False
    seed_adapter = receipt["seed_adapter_checks"]
    assert seed_adapter["pose_and_displacement_routes_identical"] is True
    assert seed_adapter["moving_cranium_seed_rejected"] is True
    assert seed_adapter["receipt"]["numerically_admissible"] is True
    mapping = receipt["mapping"]
    assert mapping["schema"] == "joint-rigid-bone-map-v1"
    assert mapping["surface_selection"] == ("pure FEM cranium versus pure FEM mandible")
    assert mapping["complete_source_bones_checked"] is False
    assert mapping["anatomical_validation"] is False
    expected_sources = {
        "29-validate-rigid-bone-ccd.py",
        "joint_data.py",
        "joint_rigid_bone_collision.py",
    }
    assert {Path(path).name for path in receipt["sources"]} == expected_sources
    for path, digest in receipt["sources"].items():
        assert sha256(Path(path)) == digest, path


def validate_forward_receipt(
    receipt: dict[str, Any], expected_solver: dict[str, Any]
) -> None:
    assert receipt["success"] is True, receipt
    assert math.isfinite(float(receipt["grad_norm"])), receipt
    selected_method = expected_solver["method"]
    implementation_method = selected_method
    if selected_method == "newton_cg" and receipt["result"] != "initial_equilibrium":
        implementation_method = "inexact_newton_cg"
    assert receipt["method"] == implementation_method, receipt
    validate_forward_solver_contract(receipt["forward_solver"], expected_solver)
    validate_contact_receipt(receipt["contact"])


def validate_neutral_checkpoint(
    initial: dict[str, Any],
    *,
    contact_spec_sha256: str,
    contact_validation_sha256: str,
    expected_solver: dict[str, Any],
) -> None:
    assert initial["schema"] == "joint-inverse-checkpoint-v1"
    assert initial["stage"] == "neutral"
    assert initial["neutral_converged"] is True
    assert initial["optimizer_converged"] is True
    assert initial["neutral_budget_met"] is True
    assert initial["inverse_converged"] is True
    assert initial["contact_validated"] is True
    assert initial["preparation_complete"] is True
    metrics = initial["metrics"]
    assert metrics["preparation_converged"] is True
    assert metrics["preparation_complete"] is True
    assert metrics["optimizer_converged"] is True
    assert metrics["neutral_budget_met"] is True
    convergence = metrics["convergence"]
    assert convergence["stationarity_met"] is True
    assert convergence["objective_stabilized"] is True
    assert convergence["projected_gradient_inf"] <= 1.0e-3
    assert convergence["projected_gradient_inf_tolerance"] == 1.0e-3
    assert convergence["objective_relative_range"] <= 1.0e-5
    assert convergence["objective_range_tolerance"] == 1.0e-5
    assert metrics["qualifying_consecutive"] >= 3
    assert initial["accepted_steps"] >= 5
    protocol = initial["protocol"]
    assert protocol["schema"] == "joint-neutral-convergence-protocol-v1"
    assert protocol["stage"] == "neutral"
    validate_forward_solver_contract(protocol["forward_solver"], expected_solver)
    validate_final_neutral_target(initial)
    assert protocol["contact"]["required_for_final"] is True
    assert protocol["contact"]["enabled"] is True
    assert protocol["contact"]["spec_sha256"] == contact_spec_sha256
    assert protocol["contact"]["validation_sha256"] == contact_validation_sha256
    embedded_validation = protocol["contact"]["validation"]
    assert embedded_validation["schema"] == "joint-contact-validation-v1"
    assert embedded_validation["success"] is True
    assert embedded_validation["contact_spec_sha256"] == contact_spec_sha256
    assert metrics["contact"]["enabled"] is True
    assert metrics["contact"]["contact_numerically_valid"] is True


def validate_shape(metrics: dict, *, neutral: bool) -> None:
    assert metrics["inverted_tetrahedra"] == 0, metrics
    assert metrics["detF_min"] >= 0.25, metrics
    assert metrics["detF_max"] <= 2, metrics
    assert metrics["skin_area_ratio_min"] >= 0.25, metrics
    if neutral:
        assert metrics["surface_motion_rms_mm"] <= 0.25, metrics
        assert metrics["muscle_centroid_motion_rms_mm"] <= 0.5, metrics


def main(cfg: Config):  # noqa: C901, PLR0912, PLR0915
    global COMPLETED  # noqa: PLW0603
    assert cfg.activation_neighbor_rms_budget_dimensionless > 0
    assert cfg.calibration_probe_scale_dimensionless > 0
    assert 0 < cfg.calibration_relative_tightening_factor < 1
    assert 0 <= cfg.calibration_ratio_relative_tolerance < 1
    assert cfg.preparation_max_updates > 0
    assert cfg.preparation_wall_budget_seconds > 0
    assert cfg.final_updates == 100
    assert cfg.final_wall_budget_seconds == 43200
    assert cfg.convergence_window == 5
    assert cfg.convergence_consecutive == 5
    assert 0 < cfg.outer_backtrack_factor < 1
    assert cfg.outer_max_backtracks > 0
    assert 0 < cfg.outer_armijo_coefficient < 1
    assert 0 < cfg.forward_rtol <= 1.0e-6
    assert 0 < cfg.forward_atol <= 1.0e-12
    assert 0 < cfg.adjoint_rtol <= 1.0e-7
    assert cfg.max_forward_steps >= 10000
    assert cfg.forward_method in {"pncg", "newton_cg"}
    if cfg.forward_method == "newton_cg":
        assert cfg.newton_linear_rtol == 1.0e-3
        assert cfg.newton_max_steps == 12
    expected_solver = forward_solver_contract(cfg)
    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    assert prepared.manifest["schema_version"] == 1
    validation = require_gates(cfg, output)
    archive_sources(output)
    configure_cuda()
    contact_config = json.loads(cfg.contact_spec.read_text())
    assert contact_config["schema"] == "joint-bone-contact-v1"
    assert contact_config["enabled"] is True
    assert contact_config["surface_selection"] == "pure-soft-vs-pure-bone"
    assert contact_config["friction"] == "frictionless"
    assert contact_config["attachment_policy"] == "bonded-mixed-faces"
    contact_spec_sha256 = sha256(cfg.contact_spec)
    contact_validation_sha256 = sha256(cfg.contact_validation)
    input_arrays_sha256 = sha256(cfg.prepared_dir / "inputs.npz")
    input_manifest_sha256 = sha256(cfg.prepared_dir / "manifest.json")
    rigid_bone_ccd_validation_sha256 = sha256(cfg.rigid_bone_ccd_validation)
    rigid_bone_ccd_validation = json.loads(cfg.rigid_bone_ccd_validation.read_text())
    validate_rigid_bone_ccd_validation(
        rigid_bone_ccd_validation,
        input_arrays_sha256=input_arrays_sha256,
        input_manifest_sha256=input_manifest_sha256,
    )
    assert prepared.manifest["artifact"]["sha256"] == input_arrays_sha256
    neutral_checkpoint_sha256 = sha256(cfg.neutral_checkpoint)
    initial = torch.load(cfg.neutral_checkpoint, map_location="cpu", weights_only=False)
    validate_neutral_checkpoint(
        initial,
        contact_spec_sha256=contact_spec_sha256,
        contact_validation_sha256=contact_validation_sha256,
        expected_solver=expected_solver,
    )
    assert initial["protocol"]["input_arrays_sha256"] == input_arrays_sha256
    assert initial["protocol"]["input_manifest_sha256"] == input_manifest_sha256
    assert initial["protocol"]["contact"]["spec_sha256"] == contact_spec_sha256
    shared_basis, shared_field, spatial_basis, spatial_evidence = (
        validate_shared_basis_contract(
            initial,
            contact_spec_sha256=contact_spec_sha256,
            expected_solver=expected_solver,
            expected_tolerances={
                "rtol": cfg.forward_rtol,
                "atol": cfg.forward_atol,
                "adjoint_rtol": cfg.adjoint_rtol,
                "max_steps": cfg.max_forward_steps,
            },
            input_arrays_sha256=input_arrays_sha256,
            input_manifest_sha256=input_manifest_sha256,
        )
    )
    assert all(
        validation.get(path) == digest for path, digest in spatial_evidence.items()
    )
    spec = initial["materials"]
    implementation_names = [
        "14-validate-outer-transaction.py",
        "30-joint-pilot.py",
        "31-render-final.py",
        "joint_common.py",
        "joint_data.py",
        "joint_diagnostics.py",
        "joint_equilibrium.py",
        "joint_final_geometry.py",
        "joint_fields.py",
        "joint_materials.py",
        "joint_newton.py",
        "joint_contact.py",
        "joint_physics.py",
        "joint_rigid_bone_collision.py",
    ]
    if shared_basis == "spatial80":
        implementation_names.append("joint_spatial_fields.py")
    implementation_sha256 = {
        name: sha256(GROUP / "src" / name) for name in implementation_names
    }
    physics = physics_from_spec(cfg, prepared, spec, contact_config)
    contact_definition = physics.contact_definition
    assert contact_definition is not None
    assert contact_definition["cranium_triangles"] > 0
    assert contact_definition["mandible_triangles"] > 0
    assert contact_definition["soft_triangles"] > 0
    assert contact_definition["anatomical_validation"] is False
    arrays = prepared.arrays
    rigid_bone_guard = RigidBoneCollision(physics.mesh, arrays["mandible_pivot_m"])
    assert rigid_bone_guard.mapping == rigid_bone_ccd_validation["mapping"]
    assert rigid_bone_guard.reference_receipt == rigid_bone_ccd_validation["reference"]
    if shared_basis == "spatial80":
        assert spatial_basis is not None
        shared: SharedParameters = SpatialSharedFieldParameters(
            Path(spatial_basis["basis_path"]),
            Path(spatial_basis["audit_summary_path"]),
            len(physics.tets),
        )
        assert shared.config == spec
        assert shared.basis_receipt() == spatial_basis
    else:
        assert spatial_basis is None
        shared = SharedFieldParameters(spec)
    with torch.no_grad():
        shared.coefficients.copy_(initial["shared_coefficients"])
    graph = VolumeGraph(
        torch.as_tensor(arrays["graph_i"], dtype=torch.long),
        torch.as_tensor(arrays["graph_j"], dtype=torch.long),
        torch.as_tensor(arrays["graph_conductance_m"]),
        torch.as_tensor(arrays["active_effective_volume_m3"]),
        spec["constraints"]["smooth_length_m"],
    )
    target_indices = arrays["training_target_indices"].tolist()
    count = len(target_indices)
    q = torch.nn.Parameter(torch.zeros((count, len(physics.ids), 6)))
    pose_scale = torch.tensor([10 * math.pi / 180] * 3 + [0.005] * 3)
    pose_origin = torch.zeros(6)
    jaw_lower = torch.full((6,), -1.0)
    jaw_upper = torch.full((6,), 1.0)
    jaw = torch.nn.Parameter(torch.zeros((count, 6)))
    shared_initialization = initial["shared_coefficients"].to(
        device=shared.coefficients.device,
        dtype=shared.coefficients.dtype,
    )
    shared_prior_reference = build_shared_prior_reference(
        shared_initialization, shared_field
    )
    skin_baseline_index = int(shared_field["skin_baseline_index"])
    skin_log_multiplier_index = int(shared_field["skin_log_multiplier_index"])
    assert float(shared_prior_reference[skin_log_multiplier_index]) == 0.0
    assert float(shared_prior_reference[skin_baseline_index]) == float(
        shared_initialization[skin_baseline_index]
    )
    shared_initialization_values = shared_initialization.cpu().tolist()
    shared_prior_reference_values = shared_prior_reference.cpu().tolist()
    shared_initialization_sha256 = canonical_sha256(shared_initialization_values)
    shared_prior_reference_sha256 = canonical_sha256(shared_prior_reference_values)
    seeds = {"neutral": initial["primal"]["neutral"].to(device="cuda")}
    for index in range(count):
        seeds[str(index)] = seeds["neutral"].clone()
    parent_checkpoint_sha256 = None
    parent_checkpoint_update = None
    control_receipt_sha256 = None
    branch = None
    if cfg.stage == "joint_trend":
        assert cfg.initial_checkpoint is not None, (
            "joint trend requires an explicit converged-control parent checkpoint"
        )
        assert cfg.control_receipt is not None, (
            "joint trend requires the converged-control summary receipt"
        )
    if cfg.initial_checkpoint is not None:
        parent_checkpoint_sha256 = sha256(cfg.initial_checkpoint)
        branch = torch.load(
            cfg.initial_checkpoint, map_location="cpu", weights_only=False
        )
        parent_checkpoint_update = branch["update"]
        assert branch["schema"] == "joint-inverse-checkpoint-v1"
        assert branch["stage"] == "control_converge"
        assert branch["preparation_complete"] is True
        assert branch["inverse_converged"] is True
        assert branch["convergence"]["converged"] is True
        assert branch["protocol"]["schema"] == "joint-final-protocol-v1"
        assert branch["protocol"]["stage"] == "control_converge"
        validate_forward_solver_contract(
            branch["protocol"]["forward_solver"], expected_solver
        )
        validate_forward_solver_contract(branch["forward_solver"], expected_solver)
        assert branch["input_arrays_sha256"] == input_arrays_sha256
        assert branch["input_manifest_sha256"] == input_manifest_sha256
        assert branch["neutral_checkpoint_sha256"] == neutral_checkpoint_sha256
        assert branch["protocol"]["input_arrays_sha256"] == input_arrays_sha256
        assert branch["protocol"]["input_manifest_sha256"] == input_manifest_sha256
        assert (
            branch["protocol"]["neutral_checkpoint_sha256"] == neutral_checkpoint_sha256
        )
        assert (
            branch["comparison_fingerprint"]
            == branch["protocol"]["comparison_fingerprint"]
        )
        assert branch["comparison_fingerprint"] == canonical_sha256(
            branch["protocol"]["comparison_contract"]
        )
        assert branch["stop_status"] == "converged numerical preparation"
        assert branch["shared_optimizer_state_present"] is False
        assert branch["contact_validated"] is True
        assert branch["rigid_bone_ccd_validated"] is True
        assert branch["contact_spec_sha256"] == contact_spec_sha256
        assert branch["contact_validation_sha256"] == contact_validation_sha256
        assert branch["rigid_bone_ccd_validation_sha256"] == (
            rigid_bone_ccd_validation_sha256
        )
        assert branch["protocol"]["rigid_bone_ccd"]["validation_sha256"] == (
            rigid_bone_ccd_validation_sha256
        )
        assert branch["materials"] == spec
        assert branch["shared_basis"] == shared_basis
        assert branch["shared_field"] == shared_field
        assert branch["spatial_basis"] == spatial_basis
        assert branch["protocol"]["shared_basis"] == shared_basis
        assert branch["protocol"]["shared_field"] == shared_field
        assert branch["protocol"]["spatial_basis"] == spatial_basis
        assert branch["shared_coefficients"].shape == shared.coefficients.shape
        validate_shared_prior_lineage(
            branch,
            initialization_values=shared_initialization_values,
            prior_reference_values=shared_prior_reference_values,
            initialization_sha256=shared_initialization_sha256,
            prior_reference_sha256=shared_prior_reference_sha256,
        )
        validate_implementation_sha256(
            branch["implementation_sha256"],
            implementation_sha256,
            artifact="control checkpoint",
        )
        validate_implementation_sha256(
            branch["protocol"]["implementation_sha256"],
            implementation_sha256,
            artifact="control protocol",
        )
        expected_state_keys = {"neutral", *(str(index) for index in range(count))}
        assert set(branch["primal"]) == expected_state_keys
        assert set(branch["adjoint"]) == expected_state_keys
        if cfg.stage == "joint_trend":
            assert cfg.control_receipt is not None
            control_receipt_sha256 = sha256(cfg.control_receipt)
            control_receipt = json.loads(cfg.control_receipt.read_text())
            assert control_receipt["schema"] == "joint-final-summary-v1"
            assert control_receipt["success"] is True
            assert control_receipt["stage"] == "control_converge"
            assert control_receipt["preparation_complete"] is True
            assert control_receipt["inverse_converged"] is True
            assert control_receipt["rigid_bone_ccd_validated"] is True
            assert control_receipt["rigid_bone_ccd_validation_sha256"] == (
                rigid_bone_ccd_validation_sha256
            )
            assert control_receipt["shared_basis"] == shared_basis
            assert control_receipt["shared_field"] == shared_field
            assert control_receipt["spatial_basis"] == spatial_basis
            assert control_receipt["shared_initialization_sha256"] == (
                shared_initialization_sha256
            )
            assert control_receipt["shared_prior_reference_sha256"] == (
                shared_prior_reference_sha256
            )
            validate_forward_solver_contract(
                control_receipt["forward_solver"], expected_solver
            )
            assert (
                control_receipt["terminal_checkpoint_sha256"]
                == parent_checkpoint_sha256
            )
        with torch.no_grad():
            shared.coefficients.copy_(branch["shared_coefficients"])
            q.copy_(branch["activation"])
            jaw.copy_(branch["jaw"])
        seeds = {k: v.to(device="cuda") for k, v in branch["primal"].items()}
        physics.runtime.warm_adjoints = {
            k: v.to(device="cuda") for k, v in branch["adjoint"].items()
        }
    optimizer = torch.optim.Adam(
        [
            {"params": [q], "lr": cfg.activation_learning_rate},
            {"params": [jaw], "lr": cfg.jaw_learning_rate},
            {
                "params": [shared.coefficients],
                "lr": (cfg.shared_learning_rate if cfg.stage == "joint_trend" else 0.0),
            },
        ],
        eps=cfg.adam_eps,
    )
    shared_optimizer_state_present_before_release = False
    if branch is not None:
        optimizer.load_state_dict(branch["optimizer"])
        # A branch changes only shared-block release; preserve expression moments.
        for group, lr in zip(
            optimizer.param_groups,
            [
                cfg.activation_learning_rate,
                cfg.jaw_learning_rate,
                cfg.shared_learning_rate if cfg.stage == "joint_trend" else 0.0,
            ],
            strict=True,
        ):
            group["lr"] = lr
    if cfg.stage == "joint_trend":
        shared_optimizer_state_present_before_release = (
            shared.coefficients in optimizer.state
        )
        optimizer.state.pop(shared.coefficients, None)
        assert shared.coefficients not in optimizer.state
    elif cfg.stage == "control_converge":
        assert shared.coefficients not in optimizer.state, (
            "control checkpoint contains forbidden shared Adam moments"
        )
    frozen_control_shared = shared.coefficients.detach().clone()

    def prepare_optimizer_step() -> None:
        if cfg.stage == "control_converge":
            shared.coefficients.grad = None

    def project_parameter_blocks() -> dict[str, Any]:
        activation_projection = project_activation_(
            q, shared.activation_maximum_dimensionless
        )
        if cfg.stage == "joint_trend":
            shared_projection = shared.project_()
        else:
            assert torch.equal(shared.coefficients, frozen_control_shared), (
                "control step changed frozen shared coefficients"
            )
            shared_projection = {
                "applied": False,
                "projection_coordinate_rms": 0.0,
                "reason": "shared block is exactly frozen in control",
            }
        with torch.no_grad():
            jaw.clamp_(min=-1.0, max=1.0)
        return {
            "activation_projection": activation_projection,
            "shared_projection": shared_projection,
        }

    calibration = None
    calibration_sha256 = None
    if cfg.stage != "calibrate":
        calibration_sha256 = sha256(cfg.calibration)
        calibration = json.loads(cfg.calibration.read_text())
        assert calibration["schema"] == "strong-smoothness-calibration-v1"
        assert calibration["stage"] == "calibrate"
        assert calibration["success"] is True
        assert calibration["input_manifest_sha256"] == input_manifest_sha256
        assert calibration["input_arrays_sha256"] == input_arrays_sha256
        assert calibration["neutral_checkpoint_sha256"] == neutral_checkpoint_sha256
        assert calibration["contact_spec_sha256"] == contact_spec_sha256
        assert calibration["contact_validation_sha256"] == contact_validation_sha256
        assert calibration["rigid_bone_ccd_validation_sha256"] == (
            rigid_bone_ccd_validation_sha256
        )
        assert calibration["materials"] == spec
        assert calibration["shared_basis"] == shared_basis
        assert calibration["shared_field"] == shared_field
        assert calibration["spatial_basis"] == spatial_basis
        assert calibration["spatial_validation"] == initial["protocol"].get(
            "spatial_validation"
        )
        validate_shared_prior_lineage(
            calibration,
            initialization_values=shared_initialization_values,
            prior_reference_values=shared_prior_reference_values,
            initialization_sha256=shared_initialization_sha256,
            prior_reference_sha256=shared_prior_reference_sha256,
        )
        validate_forward_solver_contract(calibration["forward_solver"], expected_solver)
        validate_implementation_sha256(
            calibration["implementation_sha256"],
            implementation_sha256,
            artifact="calibration receipt",
        )
        assert (
            calibration["activation_neighbor_rms_budget_dimensionless"]
            == cfg.activation_neighbor_rms_budget_dimensionless
        )
        if branch is not None:
            assert branch["calibration_sha256"] == calibration_sha256
            assert branch["protocol"]["calibration_sha256"] == calibration_sha256

    comparison_contract = {
        "schema": "converged-parent-to-joint-trend-v1",
        "input_arrays_sha256": input_arrays_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "neutral_checkpoint_sha256": neutral_checkpoint_sha256,
        "calibration_sha256": calibration_sha256,
        "parent_control_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_control_checkpoint_update": parent_checkpoint_update,
        "control_receipt_sha256": control_receipt_sha256,
        "implementation_sha256": implementation_sha256,
        "validation_evidence_sha256": validation,
        "target_indices": target_indices,
        "preparation_max_updates": cfg.preparation_max_updates,
        "preparation_wall_budget_seconds": cfg.preparation_wall_budget_seconds,
        "final_updates": cfg.final_updates,
        "final_wall_budget_seconds": cfg.final_wall_budget_seconds,
        "activation_learning_rate": cfg.activation_learning_rate,
        "shared_learning_rate": cfg.shared_learning_rate,
        "jaw_learning_rate": cfg.jaw_learning_rate,
        "adam_eps": cfg.adam_eps,
        "outer_backtrack_factor": cfg.outer_backtrack_factor,
        "outer_max_backtracks": cfg.outer_max_backtracks,
        "outer_armijo_coefficient": cfg.outer_armijo_coefficient,
        "magnitude_weight": cfg.magnitude_weight,
        "shared_prior_weight": cfg.shared_prior_weight,
        "jaw_prior_weight": cfg.jaw_prior_weight,
        "activation_neighbor_rms_budget_dimensionless": (
            cfg.activation_neighbor_rms_budget_dimensionless
        ),
        "forward_rtol": cfg.forward_rtol,
        "forward_atol": cfg.forward_atol,
        "adjoint_rtol": cfg.adjoint_rtol,
        "max_forward_steps": cfg.max_forward_steps,
        "forward_solver": expected_solver,
        "contact_spec_sha256": contact_spec_sha256,
        "contact_validation_sha256": contact_validation_sha256,
        "rigid_bone_ccd_validation_sha256": (rigid_bone_ccd_validation_sha256),
        "rigid_bone_mapping": rigid_bone_guard.mapping,
        "shared_basis": shared_basis,
        "shared_field": shared_field,
        "spatial_basis": spatial_basis,
        "spatial_validation": initial["protocol"].get("spatial_validation"),
        "spatial_smoothness_weight": (
            SPATIAL_SMOOTHNESS_WEIGHT if shared_basis == "spatial80" else 0.0
        ),
        "spatial_smoothness_factor": (
            SPATIAL_SMOOTHNESS_FACTOR if shared_basis == "spatial80" else 0.0
        ),
        "shared_initialization_sha256": shared_initialization_sha256,
        "shared_prior_reference_sha256": shared_prior_reference_sha256,
        "parameter_counts": {
            "activation": count * len(physics.ids) * 6,
            "jaw": count * 6,
            "shared": shared.coefficients.numel(),
            "total": (
                count * len(physics.ids) * 6 + count * 6 + shared.coefficients.numel()
            ),
        },
        "convergence_thresholds": {
            "window": cfg.convergence_window,
            "consecutive": cfg.convergence_consecutive,
            "relative": cfg.projected_gradient_relative_tolerance,
            "relative_requirement": (
                "mandatory current projected-gradient L2 divided by initial L2; "
                "zero initial norm passes only when the current norm is exactly zero"
            ),
            "objective_span": cfg.objective_relative_span_tolerance,
            "activation_step_volume_rms": (cfg.activation_step_volume_rms_tolerance),
            "activation_step_max_tet": cfg.activation_step_max_tet_tolerance,
            "jaw_step_coordinate_rms": cfg.jaw_step_coordinate_rms_tolerance,
            "jaw_step_max_row": cfg.jaw_step_max_row_tolerance,
            "shared_step_coordinate_rms": 1.0e-5,
            "shared_step_maximum_absolute_coordinate": 1.0e-4,
            "absolute_step_source": (
                "exact reversible next projected-Adam proposal with current moments"
            ),
            "activation_mass_normalized_gradient": (
                "interpretive effective-volume Riesz diagnostic only; no absolute gate"
            ),
        },
    }
    comparison_fingerprint = canonical_sha256(comparison_contract)
    protocol = {
        "schema": "joint-final-protocol-v1",
        "stage": cfg.stage,
        "config": cfg.model_dump(mode="json"),
        "materials": spec,
        "input_manifest_sha256": input_manifest_sha256,
        "input_arrays_sha256": input_arrays_sha256,
        "neutral_checkpoint_sha256": neutral_checkpoint_sha256,
        "calibration_sha256": calibration_sha256,
        "parent_control_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_control_checkpoint_update": parent_checkpoint_update,
        "control_receipt_sha256": control_receipt_sha256,
        "comparison_contract": comparison_contract,
        "comparison_fingerprint": comparison_fingerprint,
        "implementation_sha256": implementation_sha256,
        "validation": validation,
        "forward_solver": expected_solver,
        "shared_basis": shared_basis,
        "shared_field": shared_field,
        "spatial_basis": spatial_basis,
        "spatial_validation": initial["protocol"].get("spatial_validation"),
        "jaw_contract": {
            "status": "broad symmetric computational search box; not biological bounds",
            "coordinates": "world rotation vector xyz in radians then translation xyz in metres",
            "normalized_lower": [-1.0] * 6,
            "normalized_upper": [1.0] * 6,
            "rotation_scale_degrees": 10.0,
            "translation_scale_mm": 5.0,
        },
        "contact": {
            "required": True,
            "spec": contact_config,
            "spec_sha256": contact_spec_sha256,
            "validation_sha256": contact_validation_sha256,
            "surface_map": contact_definition,
        },
        "rigid_bone_ccd": {
            "required": True,
            "validation_sha256": rigid_bone_ccd_validation_sha256,
            "validation": rigid_bone_ccd_validation,
            "mapping": rigid_bone_guard.mapping,
            "reference": rigid_bone_guard.reference_receipt,
            "trajectory": "linear boundary-vertex motion from accepted seed",
            "rotation_arc_checked": False,
            "energy_or_derivative_added": False,
            "anatomical_validation": False,
        },
        "anatomical_validation": False,
        "promotion_ready": False,
        "shared_initialization": shared_initialization_values,
        "shared_prior_reference": shared_prior_reference_values,
        "shared_reference": shared_prior_reference_values,
        "shared_initialization_sha256": shared_initialization_sha256,
        "shared_prior_reference_sha256": shared_prior_reference_sha256,
        "shared_prior_contract": {
            "weight": cfg.shared_prior_weight,
            "bulk_and_skin_baseline_center": (
                "admitted converged full-target neutral checkpoint"
            ),
            "skin_log_stiffness_center": 0.0,
            "skin_stiffness_multiplier_center": 1.0,
            "skin_reference_young_mpa": spec["materials"]["skin"]["reference_map"][
                "young_mpa"
            ],
            "bulk_metric": (
                "constant20 Euclidean six-coordinate norm; Spatial80 exact "
                "per-tissue audited M quadratic, preserving constant embedding"
            ),
            "skin_metric": (
                "separate squared baseline-coordinate and log-multiplier departures"
            ),
            "spatial_roughness_included": False,
            "zero_prestress_centered": False,
        },
        "smoothness_contract": {
            "activation": "strong frozen weight, separately per expression",
            "shared_spatial": (
                {
                    "weight": SPATIAL_SMOOTHNESS_WEIGHT,
                    "factor": SPATIAL_SMOOTHNESS_FACTOR,
                    "regularizer": "bulk_spatial_roughness",
                    "objective_term": ("0.5 * 100.0 * mean_t tr(C_t^T G_t C_t)"),
                }
                if shared_basis == "spatial80"
                else {
                    "weight": 0.0,
                    "factor": 0.0,
                    "reason": "constant shared fields have zero spatial roughness",
                }
            ),
        },
        "objective_contract": {
            "common_objective": (
                "neutral_loss + data_loss + weighted_smoothness + "
                "weighted_magnitude + weighted_jaw_prior"
            ),
            "objective": (
                "common_objective + weighted_shared_prior + "
                "weighted_shared_spatial_roughness"
            ),
            "data_loss": "sum of expression losses already divided by expression count",
        },
        "detF_gate": [0.25, 2.0],
        "neutral_budgets_mm": [0.25, 0.5],
        "activation_neighbor_rms_budget_dimensionless": (
            cfg.activation_neighbor_rms_budget_dimensionless
        ),
        "activation_neighbor_rms_budget_MPa": (
            cfg.activation_neighbor_rms_budget_dimensionless
            * shared.activation_reference_mpa
        ),
        "shared_optimizer_state_policy": (
            "shared gradients are cleared before every control step; joint release "
            "starts with no inherited shared Adam state"
        ),
        "shared_optimizer_state_policy_applied": True,
        "shared_optimizer_state_present_before_release": (
            shared_optimizer_state_present_before_release
        ),
        "shared_optimizer_state_present_after_policy": (
            shared.coefficients in optimizer.state
        ),
        "parameter_counts": comparison_contract["parameter_counts"],
        "convergence_thresholds": comparison_contract["convergence_thresholds"],
    }
    write_json(output / "protocol.json", protocol)
    adjoint_counter = {"count": 0}

    def solve_expression(
        index: int,
        physics_instance: JointPhysics,
        seed_map: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        assert torch.all(jaw[index] >= jaw_lower)
        assert torch.all(jaw[index] <= jaw_upper)
        pose = pose_origin + pose_scale * jaw[index]
        active = activation_stresses_mpa(q[index], shared.activation_reference_mpa)
        rigid_bone_ccd = rigid_bone_guard.from_displacement(
            seed_map[str(index)].detach().cpu().numpy(),
            pose.detach().cpu().numpy(),
        )
        assert rigid_bone_ccd["numerically_admissible"] is True, rigid_bone_ccd
        u = physics_instance.solve(
            shared.bulk_stresses_mpa(),
            shared.skin_resultant_n_per_m(),
            shared.skin_stiffness_multiplier(),
            active,
            pose,
            seed_map[str(index)],
            key=str(index),
        )
        validate_forward_receipt(physics_instance.runtime.last_forward, expected_solver)
        raw_geometry_diagnostic = audit_deformed_oral_geometry(
            prepared,
            physics_instance.points + u.detach().cpu().numpy(),
            pose.detach().cpu().numpy(),
        )
        geometry_diagnostic = adapt_final_run_geometry_receipt(
            raw_geometry_diagnostic,
            neutral=False,
            normalized_pose=jaw[index].detach().cpu().numpy(),
            pose_rad_m=pose.detach().cpu().numpy(),
        )
        validate_final_run_geometry_receipt(geometry_diagnostic, neutral=False)
        metrics = physics_instance.metrics(u, target_index=target_indices[index])
        validate_shape(metrics, neutral=False)
        loss = physics_instance.fit_loss(u, target_indices[index]) / count
        loss.backward()
        adjoint_counter["count"] += 1
        seed_map[str(index)] = u.detach().clone()
        return {
            "target": prepared.target_names[target_indices[index]],
            "data": float(loss.detach()),
            "metrics": metrics,
            "geometry_diagnostic": geometry_diagnostic,
            "rigid_bone_ccd": rigid_bone_ccd,
            "contact": physics_instance.runtime.last_forward["contact"],
            "forward": physics_instance.runtime.last_forward,
            "adjoint": physics_instance.runtime.last_adjoint,
        }

    if cfg.stage == "calibrate":
        calibration_started = time.perf_counter()
        generator = torch.Generator(device="cuda").manual_seed(cfg.seed)
        with torch.no_grad():
            random = torch.randn((len(physics.ids), 3, 3), generator=generator)
            probe = cfg.calibration_probe_scale_dimensionless * (
                random @ random.transpose(-1, -2) + torch.eye(3)
            )
            for index in range(count):
                q[index].copy_(symmetric_coordinates(probe))
        probe_regularizers = activation_regularizers(q, graph)
        probe_neighbor_rms = activation_neighbor_rms(q, graph)
        probe_physical = activation_stresses_mpa(
            q[0].detach(), shared.activation_reference_mpa
        )
        probe_eigenvalues = torch.linalg.eigvalsh(probe_physical)
        assert float(probe_eigenvalues.min()) >= -1e-12
        assert float(probe_eigenvalues.max()) <= (
            shared.activation_maximum_dimensionless * shared.activation_reference_mpa
        )
        assert float(probe_neighbor_rms.max()) <= (
            cfg.activation_neighbor_rms_budget_dimensionless
        )
        rows = []
        for index in range(count):
            optimizer.zero_grad(set_to_none=True)
            receipt = solve_expression(index, physics, seeds)
            data_rms = float(q.grad[index].square().mean().sqrt())
            smooth = activation_regularizers(q[index], graph)["smoothness"] / count
            (smooth_gradient,) = torch.autograd.grad(smooth, q)
            smooth_rms = float(smooth_gradient[index].square().mean().sqrt())
            assert data_rms > 0
            assert smooth_rms > 0
            rows.append(
                {
                    "expression": index,
                    "data_gradient_rms": data_rms,
                    "smoothness_gradient_rms": smooth_rms,
                    "reference_weight": data_rms / smooth_rms,
                    "standard_receipt": receipt,
                }
            )
        standard_forward_count = physics.runtime.forward_count
        assert standard_forward_count == count
        del physics
        gc.collect()
        torch.cuda.empty_cache()

        physics = physics_from_spec(cfg, prepared, spec, contact_config)
        repeat_seeds = {"neutral": initial["primal"]["neutral"].to(device="cuda")}
        for index in range(count):
            repeat_seeds[str(index)] = repeat_seeds["neutral"].clone()
        for index, row in enumerate(rows):
            optimizer.zero_grad(set_to_none=True)
            receipt = solve_expression(index, physics, repeat_seeds)
            repeat_data_rms = float(q.grad[index].square().mean().sqrt())
            assert repeat_data_rms > 0
            repeat_weight = repeat_data_rms / row["smoothness_gradient_rms"]
            relative_change = abs(repeat_weight - row["reference_weight"]) / max(
                repeat_weight, row["reference_weight"]
            )
            assert relative_change <= cfg.calibration_ratio_relative_tolerance, (
                "repeat",
                index,
                relative_change,
            )
            row["repeat_data_gradient_rms"] = repeat_data_rms
            row["repeat_reference_weight"] = repeat_weight
            row["repeat_reference_weight_relative_change"] = relative_change
            row["repeat_receipt"] = receipt
        repeat_forward_count = physics.runtime.forward_count
        assert repeat_forward_count == count
        del physics
        gc.collect()
        torch.cuda.empty_cache()

        tight_cfg = cfg.model_copy(
            update={
                "forward_rtol": (
                    cfg.forward_rtol * cfg.calibration_relative_tightening_factor
                ),
                # The validated 1e-12 force floor is retained. Tightening it to
                # 1e-13 previously reached numerical energy resolution rather
                # than providing a more accurate equilibrium.
                "forward_atol": cfg.forward_atol,
                "adjoint_rtol": (
                    cfg.adjoint_rtol * cfg.calibration_relative_tightening_factor
                ),
            }
        )
        assert tight_cfg.forward_rtol < cfg.forward_rtol
        assert tight_cfg.forward_atol == cfg.forward_atol
        assert tight_cfg.adjoint_rtol < cfg.adjoint_rtol
        physics = physics_from_spec(tight_cfg, prepared, spec, contact_config)
        tight_seeds = {"neutral": initial["primal"]["neutral"].to(device="cuda")}
        for index in range(count):
            tight_seeds[str(index)] = tight_seeds["neutral"].clone()
        for index, row in enumerate(rows):
            optimizer.zero_grad(set_to_none=True)
            receipt = solve_expression(index, physics, tight_seeds)
            tight_data_rms = float(q.grad[index].square().mean().sqrt())
            assert tight_data_rms > 0
            tight_weight = tight_data_rms / row["smoothness_gradient_rms"]
            relative_change = abs(tight_weight - row["reference_weight"]) / max(
                tight_weight, row["reference_weight"]
            )
            assert relative_change <= cfg.calibration_ratio_relative_tolerance, (
                "tight",
                index,
                relative_change,
            )
            row["tight_data_gradient_rms"] = tight_data_rms
            row["tight_reference_weight"] = tight_weight
            row["tight_reference_weight_relative_change"] = relative_change
            row["tight_receipt"] = receipt
        assert physics.runtime.forward_count == count
        weight = 3 * max(
            max(
                row["reference_weight"],
                row["repeat_reference_weight"],
                row["tight_reference_weight"],
            )
            for row in rows
        )
        torch.save(
            {
                "schema": "strong-smoothness-probe-v1",
                "stage": "calibrate",
                "activation": q.detach().cpu(),
                "shared": shared.coefficients.detach().cpu(),
                "jaw": jaw.detach().cpu(),
                "shared_basis": shared_basis,
                "shared_field": shared_field,
                "spatial_basis": spatial_basis,
            },
            output / "probe.pt",
        )
        probe_sha256 = sha256(output / "probe.pt")
        write_json(
            output / "calibration.json",
            {
                "schema": "strong-smoothness-calibration-v1",
                "stage": "calibrate",
                "success": True,
                "strong_weight": weight,
                "rows": rows,
                "input_manifest_sha256": input_manifest_sha256,
                "input_arrays_sha256": input_arrays_sha256,
                "neutral_checkpoint_sha256": neutral_checkpoint_sha256,
                "contact_spec_sha256": contact_spec_sha256,
                "contact_validation_sha256": contact_validation_sha256,
                "rigid_bone_ccd_validation_sha256": (rigid_bone_ccd_validation_sha256),
                "forward_solver": expected_solver,
                "implementation_sha256": implementation_sha256,
                "materials": spec,
                "shared_basis": shared_basis,
                "shared_field": shared_field,
                "spatial_basis": spatial_basis,
                "spatial_validation": initial["protocol"].get("spatial_validation"),
                "shared_initialization": shared_initialization_values,
                "shared_prior_reference": shared_prior_reference_values,
                "shared_initialization_sha256": shared_initialization_sha256,
                "shared_prior_reference_sha256": shared_prior_reference_sha256,
                "shared_prior_weight": cfg.shared_prior_weight,
                "probe_sha256": probe_sha256,
                "probe_scale_dimensionless": (
                    cfg.calibration_probe_scale_dimensionless
                ),
                "probe_mass_rms_dimensionless": torch.sqrt(
                    probe_regularizers["magnitude_by_field"]
                )
                .detach()
                .cpu()
                .tolist(),
                "probe_mass_rms_MPa": (
                    shared.activation_reference_mpa
                    * torch.sqrt(probe_regularizers["magnitude_by_field"])
                )
                .detach()
                .cpu()
                .tolist(),
                "probe_neighbor_rms_dimensionless": probe_neighbor_rms.detach()
                .cpu()
                .tolist(),
                "probe_neighbor_rms_MPa": (
                    shared.activation_reference_mpa * probe_neighbor_rms
                )
                .detach()
                .cpu()
                .tolist(),
                "probe_principal_stress_MPa": {
                    "minimum": float(probe_eigenvalues.min()),
                    "maximum": float(probe_eigenvalues.max()),
                },
                "activation_neighbor_rms_budget_dimensionless": (
                    cfg.activation_neighbor_rms_budget_dimensionless
                ),
                "activation_neighbor_rms_budget_MPa": (
                    cfg.activation_neighbor_rms_budget_dimensionless
                    * shared.activation_reference_mpa
                ),
                "standard_tolerances": {
                    "forward_rtol": cfg.forward_rtol,
                    "forward_atol": cfg.forward_atol,
                    "adjoint_rtol": cfg.adjoint_rtol,
                },
                "tight_tolerances": {
                    "forward_rtol": tight_cfg.forward_rtol,
                    "forward_atol": tight_cfg.forward_atol,
                    "adjoint_rtol": tight_cfg.adjoint_rtol,
                },
                "tightening_contract": {
                    "relative_factor": cfg.calibration_relative_tightening_factor,
                    "forward_relative_tolerance_tightened": True,
                    "adjoint_relative_tolerance_tightened": True,
                    "forward_absolute_tolerance_retained": True,
                    "forward_absolute_floor_reason": (
                        "retain validated 1e-12 force floor; 1e-13 previously "
                        "reached numerical energy and Armijo resolution"
                    ),
                },
                "ratio_relative_tolerance": (cfg.calibration_ratio_relative_tolerance),
                "maximum_observed_reference_weight_relative_change": max(
                    max(
                        row["repeat_reference_weight_relative_change"],
                        row["tight_reference_weight_relative_change"],
                    )
                    for row in rows
                ),
                "standard_forward_solves": standard_forward_count,
                "standard_adjoint_solves": count,
                "repeat_forward_solves": repeat_forward_count,
                "repeat_adjoint_solves": count,
                "tight_forward_solves": physics.runtime.forward_count,
                "tight_adjoint_solves": count,
                "total_forward_solves": (
                    standard_forward_count
                    + repeat_forward_count
                    + physics.runtime.forward_count
                ),
                "total_adjoint_solves": 3 * count,
                "elapsed_seconds": time.perf_counter() - calibration_started,
                "convention": (
                    "three times the maximum equal-gradient weight over every "
                    "expression, an independent repeat, and tighter tolerances"
                ),
            },
        )
        COMPLETED = True
        return

    assert calibration is not None
    smoothness_weight = calibration["strong_weight"]
    assert smoothness_weight > 0

    def shared_spatial_regularization() -> tuple[torch.Tensor, torch.Tensor]:
        if shared_basis == "spatial80":
            raw = shared.regularizers()["bulk_spatial_roughness"]
            return raw, SPATIAL_SMOOTHNESS_FACTOR * SPATIAL_SMOOTHNESS_WEIGHT * raw
        raw = shared.coefficients.new_zeros(())
        return raw, raw

    trace = []
    started = time.perf_counter()
    best = float("inf")
    max_updates = (
        cfg.preparation_max_updates
        if cfg.stage == "control_converge"
        else cfg.final_updates
    )
    wall_budget_seconds = (
        cfg.preparation_wall_budget_seconds
        if cfg.stage == "control_converge"
        else cfg.final_wall_budget_seconds
    )
    stop_reason = (
        "preparation update budget"
        if cfg.stage == "control_converge"
        else "final update budget"
    )
    proposal_failure = False
    stationarity_baseline = None
    qualifying_consecutive = 0
    lineage = [
        {
            "role": "converged_neutral",
            "checkpoint_sha256": neutral_checkpoint_sha256,
            "skin_prestress_fraction": initial["protocol"]["skin_prestress_fraction"],
            "skin_target_n_per_m": initial["protocol"]["skin_target_n_per_m"],
        }
    ]
    if branch is not None:
        lineage.extend(branch.get("lineage", []))
        lineage.append(
            {
                "role": "converged_control_parent",
                "checkpoint_sha256": parent_checkpoint_sha256,
                "update": parent_checkpoint_update,
            }
        )

    def save_visualization_snapshot(
        update: int,
        *,
        label: str | None = None,
        saved_checkpoint: dict[str, Any] | None = None,
    ) -> None:
        observation_ids = arrays["observation_node_ids"]
        if saved_checkpoint is None:
            activation_value = q.detach().cpu()
            jaw_value = jaw.detach().cpu()
            shared_value = shared.coefficients.detach().cpu()
            primal = seeds
        else:
            activation_value = saved_checkpoint["activation"]
            jaw_value = saved_checkpoint["jaw"]
            shared_value = saved_checkpoint["shared_coefficients"]
            primal = saved_checkpoint["primal"]
        full_displacement = np.stack(
            [
                primal[str(index)].detach().cpu().to(torch.float64).numpy()
                for index in range(count)
            ]
        )
        assert full_displacement.ndim == 3
        assert full_displacement.shape[0] == count
        assert full_displacement.shape[2] == 3
        assert np.isfinite(full_displacement).all()
        predicted = full_displacement[:, observation_ids]
        suffix = label if label is not None else f"{update:04d}"
        np.savez_compressed(
            output / f"visualization-{suffix}.npz",
            schema=np.asarray("joint-optimization-visualization-snapshot-v2"),
            stage=np.asarray(cfg.stage),
            checkpoint_label=np.asarray(suffix),
            status=np.asarray("accepted_numerically_valid_state"),
            update=np.asarray(update),
            activation=activation_value.to(torch.float32).numpy(),
            activation_reference_mpa=np.asarray(
                shared.activation_reference_mpa, dtype=np.float64
            ),
            activation_cap_mpa=np.asarray(
                shared.activation_reference_mpa
                * shared.activation_maximum_dimensionless,
                dtype=np.float64,
            ),
            symmetric_coordinate_order=np.asarray(
                ["xx", "yy", "zz", "sqrt2_xy", "sqrt2_yz", "sqrt2_xz"]
            ),
            active_cell_ids=arrays["active_cell_ids"],
            active_muscle_ids=arrays["active_muscle_ids"],
            active_effective_volume_m3=arrays["active_effective_volume_m3"],
            jaw_normalized=jaw_value.numpy(),
            jaw_pose_rad_m=(pose_origin + pose_scale * jaw_value).numpy(),
            shared=shared_value.numpy(),
            shared_basis=np.asarray(shared_basis),
            shared_coefficient_count=np.asarray(shared_value.numel(), dtype=np.int64),
            shared_field_json=np.asarray(
                json.dumps(
                    shared_field,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ),
            material_config_json=np.asarray(
                json.dumps(
                    spec,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ),
            spatial_basis_json=np.asarray(
                json.dumps(
                    spatial_basis,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ),
            spatial_smoothness_weight=np.asarray(
                SPATIAL_SMOOTHNESS_WEIGHT if shared_basis == "spatial80" else 0.0,
                dtype=np.float64,
            ),
            spatial_smoothness_factor=np.asarray(
                SPATIAL_SMOOTHNESS_FACTOR if shared_basis == "spatial80" else 0.0,
                dtype=np.float64,
            ),
            observation_node_ids=observation_ids,
            full_displacement_m=full_displacement,
            predicted_observation_displacement_m=predicted,
            target_displacement_m=arrays["target_displacement_m"][target_indices],
            target_indices=np.asarray(target_indices),
        )

    trial_cache_holder: dict[str, Any] = {"value": None}

    def evaluate_proposal_trial() -> dict[str, Any]:
        trial_started = time.perf_counter()
        neighbor_rms = activation_neighbor_rms(q.detach(), graph)
        assert float(neighbor_rms.max()) <= (
            cfg.activation_neighbor_rms_budget_dimensionless
        ), neighbor_rms
        optimizer.zero_grad(set_to_none=True)
        neutral = physics.solve(
            shared.bulk_stresses_mpa(),
            shared.skin_resultant_n_per_m(),
            shared.skin_stiffness_multiplier(),
            None,
            torch.zeros(6),
            seeds["neutral"],
            key="neutral",
        )
        validate_forward_receipt(physics.runtime.last_forward, expected_solver)
        neutral_metrics = physics.metrics(neutral)
        validate_shape(neutral_metrics, neutral=True)
        raw_neutral_geometry_diagnostic = audit_neutral_oral_geometry(
            prepared, physics.points + neutral.detach().cpu().numpy()
        )
        neutral_geometry_diagnostic = adapt_final_run_geometry_receipt(
            raw_neutral_geometry_diagnostic,
            neutral=True,
            normalized_pose=np.zeros(6),
            pose_rad_m=np.zeros(6),
        )
        validate_final_run_geometry_receipt(neutral_geometry_diagnostic, neutral=True)
        neutral_loss = physics.neutral_loss(neutral)
        neutral_loss.backward()
        adjoint_counter["count"] += 1
        seeds["neutral"] = neutral.detach().clone()
        neutral_receipt = {
            "metrics": neutral_metrics,
            "contact": deepcopy(physics.runtime.last_forward["contact"]),
            "forward": deepcopy(physics.runtime.last_forward),
            "adjoint": deepcopy(physics.runtime.last_adjoint),
        }
        expressions = [
            solve_expression(index, physics, seeds) for index in range(count)
        ]
        data_gradient = q.grad.detach().clone()
        regularizers = activation_regularizers(q, graph)
        shared_prior_terms = shared_reference_departure_quadratic(
            shared, shared_prior_reference
        )
        shared_prior = shared_prior_terms["total"]
        _, weighted_shared_spatial_roughness = shared_spatial_regularization()
        jaw_prior = jaw.square().mean()
        penalty = (
            smoothness_weight * regularizers["smoothness"]
            + cfg.magnitude_weight * regularizers["magnitude"]
            + cfg.shared_prior_weight * shared_prior
            + weighted_shared_spatial_roughness
            + cfg.jaw_prior_weight * jaw_prior
        )
        penalty.backward()
        assert all(
            value.grad is not None and torch.isfinite(value.grad).all()
            for value in (q, jaw, shared.coefficients)
        )
        objective = (
            float(neutral_loss.detach())
            + sum(expression["data"] for expression in expressions)
            + float(penalty.detach())
        )
        trial_cache_holder["value"] = {
            "neighbor_rms": neighbor_rms.detach().clone(),
            "neutral_metrics": neutral_metrics,
            "neutral_geometry_diagnostic": neutral_geometry_diagnostic,
            "neutral_loss": float(neutral_loss.detach()),
            "neutral_receipt": neutral_receipt,
            "expressions": expressions,
            "data_gradient": data_gradient,
            "forward_solves": count + 1,
            "adjoint_solves": count + 1,
            "evaluation_seconds": time.perf_counter() - trial_started,
        }
        return {
            "objective": objective,
            "neutral": {
                "metrics": neutral_metrics,
                "contact": neutral_receipt["contact"],
            },
            "expressions": [
                {
                    "target": expression["target"],
                    "metrics": expression["metrics"],
                    "contact": expression["contact"],
                }
                for expression in expressions
            ],
            "activation_neighbor_rms_dimensionless": neighbor_rms.cpu().tolist(),
        }

    torch.cuda.reset_peak_memory_stats()
    accepted_trial_cache = None
    try:
        for update in range(max_updates + 1):
            epoch_started = time.perf_counter()
            cached_evaluation = accepted_trial_cache
            accepted_trial_cache = None
            forward_count_before = physics.runtime.forward_count
            if cached_evaluation is None:
                current_neighbor_rms = activation_neighbor_rms(q.detach(), graph)
                assert float(current_neighbor_rms.max()) <= (
                    cfg.activation_neighbor_rms_budget_dimensionless
                ), current_neighbor_rms
                optimizer.zero_grad(set_to_none=True)
                neutral = physics.solve(
                    shared.bulk_stresses_mpa(),
                    shared.skin_resultant_n_per_m(),
                    shared.skin_stiffness_multiplier(),
                    None,
                    torch.zeros(6),
                    seeds["neutral"],
                    key="neutral",
                )
                validate_forward_receipt(physics.runtime.last_forward, expected_solver)
                neutral_metrics = physics.metrics(neutral)
                validate_shape(neutral_metrics, neutral=True)
                raw_neutral_geometry_diagnostic = audit_neutral_oral_geometry(
                    prepared, physics.points + neutral.detach().cpu().numpy()
                )
                neutral_geometry_diagnostic = adapt_final_run_geometry_receipt(
                    raw_neutral_geometry_diagnostic,
                    neutral=True,
                    normalized_pose=np.zeros(6),
                    pose_rad_m=np.zeros(6),
                )
                validate_final_run_geometry_receipt(
                    neutral_geometry_diagnostic, neutral=True
                )
                neutral_loss = physics.neutral_loss(neutral)
                neutral_loss.backward()
                adjoint_counter["count"] += 1
                seeds["neutral"] = neutral.detach().clone()
                neutral_loss_value = float(neutral_loss.detach())
                neutral_receipt = {
                    "loss": neutral_loss_value,
                    "metrics": neutral_metrics,
                    "geometry_diagnostic": neutral_geometry_diagnostic,
                    "contact": physics.runtime.last_forward["contact"],
                    "forward": physics.runtime.last_forward,
                    "adjoint": physics.runtime.last_adjoint,
                }
                expressions = [
                    solve_expression(index, physics, seeds) for index in range(count)
                ]
                data_gradient = q.grad.detach().clone()
                accepted_forward_solves = count + 1
                accepted_evaluation_seconds = None
            else:
                current_neighbor_rms = cached_evaluation["neighbor_rms"]
                neutral_metrics = cached_evaluation["neutral_metrics"]
                neutral_geometry_diagnostic = cached_evaluation[
                    "neutral_geometry_diagnostic"
                ]
                neutral_loss_value = cached_evaluation["neutral_loss"]
                neutral_receipt = cached_evaluation["neutral_receipt"]
                neutral_receipt["loss"] = neutral_loss_value
                neutral_receipt["geometry_diagnostic"] = neutral_geometry_diagnostic
                expressions = cached_evaluation["expressions"]
                data_gradient = cached_evaluation["data_gradient"]
                accepted_forward_solves = cached_evaluation["forward_solves"]
                accepted_evaluation_seconds = cached_evaluation["evaluation_seconds"]
                assert cached_evaluation["adjoint_solves"] == count + 1
                assert all(
                    value.grad is not None and torch.isfinite(value.grad).all()
                    for value in (q, jaw, shared.coefficients)
                )
            regularizers = activation_regularizers(q, graph)
            shared_prior_terms = shared_reference_departure_quadratic(
                shared, shared_prior_reference
            )
            shared_prior = shared_prior_terms["total"]
            shared_spatial_roughness, weighted_shared_spatial_roughness = (
                shared_spatial_regularization()
            )
            jaw_prior = jaw.square().mean()
            weighted_smoothness = smoothness_weight * regularizers["smoothness"]
            weighted_magnitude = cfg.magnitude_weight * regularizers["magnitude"]
            weighted_shared_prior = cfg.shared_prior_weight * shared_prior
            weighted_jaw_prior = cfg.jaw_prior_weight * jaw_prior
            common_penalty = (
                weighted_smoothness + weighted_magnitude + weighted_jaw_prior
            )
            penalty = (
                common_penalty
                + weighted_shared_prior
                + weighted_shared_spatial_roughness
            )
            if cached_evaluation is None:
                penalty.backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in (q, jaw, shared.coefficients)
            )
            activation_gradient_by_expression = []
            for index in range(count):
                tet_norm = torch.linalg.vector_norm(q.grad[index], dim=-1)
                activation_gradient_by_expression.append(
                    {
                        "l2_norm": float(torch.linalg.vector_norm(q.grad[index])),
                        "coordinate_rms": float(q.grad[index].square().mean().sqrt()),
                        "maximum_tet_frobenius": float(tet_norm.max()),
                    }
                )
            data_loss = sum(expression["data"] for expression in expressions)
            common_objective = (
                neutral_loss_value + float(common_penalty.detach()) + data_loss
            )
            total = (
                common_objective
                + float(weighted_shared_prior.detach())
                + float(weighted_shared_spatial_roughness.detach())
            )
            reconstructed_total = (
                neutral_loss_value
                + data_loss
                + float(weighted_smoothness.detach())
                + float(weighted_magnitude.detach())
                + float(weighted_shared_prior.detach())
                + float(weighted_shared_spatial_roughness.detach())
                + float(weighted_jaw_prior.detach())
            )
            assert math.isclose(total, reconstructed_total, rel_tol=1e-12, abs_tol=0)

            optimizer_preview = reversible_projected_optimizer_preview(
                optimizer=optimizer,
                parameters=(q, jaw, shared.coefficients),
                prepare_step=prepare_optimizer_step,
                project=project_parameter_blocks,
            )
            activation_optimizer_proposal = activation_step_summary(
                optimizer_preview["steps"][0], graph.effective_cell_volume_m3
            )
            jaw_optimizer_proposal = parameter_step_summary(
                optimizer_preview["steps"][1]
            )
            shared_optimizer_proposal = parameter_step_summary(
                optimizer_preview["steps"][2]
            )
            optimizer_proposal_projection = optimizer_preview["projection"]
            optimizer_proposal_directional_derivative = optimizer_preview[
                "directional_derivative"
            ]
            mass_normalized_activation_gradient = (
                mass_normalized_activation_gradient_summary(
                    q.grad, graph.effective_cell_volume_m3
                )
            )
            del optimizer_preview

            activation_mapping = activation_projected_gradient_mapping(
                q,
                q.grad,
                step_size=cfg.activation_learning_rate,
                maximum_dimensionless=shared.activation_maximum_dimensionless,
            )
            activation_stationarity = activation_mapping_summary(
                activation_mapping,
                graph.effective_cell_volume_m3,
                step_size=cfg.activation_learning_rate,
            )
            jaw_mapping = box_projected_gradient_mapping(
                jaw,
                jaw.grad,
                jaw_lower,
                jaw_upper,
                step_size=cfg.jaw_learning_rate,
            )
            jaw_stationarity = parameter_block_mapping_summary(
                jaw_mapping,
                step_size=cfg.jaw_learning_rate,
            )
            shared_mapping = shared_projected_gradient_mapping(
                shared,
                shared.coefficients.grad,
                step_size=cfg.shared_learning_rate,
            )
            shared_stationarity = parameter_block_mapping_summary(
                shared_mapping,
                step_size=cfg.shared_learning_rate,
            )
            activation_spectrum = activation_spectrum_summary(
                q,
                reference_mpa=shared.activation_reference_mpa,
                maximum_dimensionless=shared.activation_maximum_dimensionless,
                effective_cell_volume_m3=graph.effective_cell_volume_m3,
            )
            shared_diagnostics = shared_prior_summary(
                shared,
                reference_coefficients=shared_prior_reference,
            )
            shared_diagnostics["initialization_departure_rms"] = float(
                (shared.coefficients.detach() - shared_initialization)
                .square()
                .mean()
                .sqrt()
            )
            del activation_mapping, jaw_mapping, shared_mapping

            activation_rows = activation_stationarity["expressions"]
            jaw_row_norms = jaw_stationarity["row_norms"]
            if stationarity_baseline is None:
                stationarity_baseline = {
                    "activation_l2": [row["l2_norm"] for row in activation_rows],
                    "jaw_l2": jaw_row_norms,
                    "shared_l2": shared_stationarity["l2_norm"],
                }
            activation_reduction = [
                relative_reduction_summary(
                    row["l2_norm"],
                    reference,
                    tolerance=cfg.projected_gradient_relative_tolerance,
                )
                for row, reference in zip(
                    activation_rows,
                    stationarity_baseline["activation_l2"],
                    strict=True,
                )
            ]
            jaw_reduction = [
                relative_reduction_summary(
                    value,
                    reference,
                    tolerance=cfg.projected_gradient_relative_tolerance,
                )
                for value, reference in zip(
                    jaw_row_norms,
                    stationarity_baseline["jaw_l2"],
                    strict=True,
                )
            ]
            shared_reduction = relative_reduction_summary(
                shared_stationarity["l2_norm"],
                stationarity_baseline["shared_l2"],
                tolerance=cfg.projected_gradient_relative_tolerance,
            )
            activation_relative = [
                row["relative_to_initial"] for row in activation_reduction
            ]
            jaw_relative = [row["relative_to_initial"] for row in jaw_reduction]
            shared_relative = shared_reduction["relative_to_initial"]
            activation_stationary = all(
                reduction["relative_reduction_met"]
                and proposal["effective_volume_tensor_rms"]
                <= cfg.activation_step_volume_rms_tolerance
                and proposal["maximum_tet_frobenius"]
                <= cfg.activation_step_max_tet_tolerance
                for reduction, proposal in zip(
                    activation_reduction,
                    activation_optimizer_proposal["expressions"],
                    strict=True,
                )
            )
            jaw_stationary = all(
                reduction["relative_reduction_met"]
                and row_norm / math.sqrt(6.0) <= cfg.jaw_step_coordinate_rms_tolerance
                and row_norm <= cfg.jaw_step_max_row_tolerance
                for reduction, row_norm in zip(
                    jaw_reduction,
                    jaw_optimizer_proposal["row_norms"],
                    strict=True,
                )
            )
            shared_stationary = bool(
                shared_reduction["relative_reduction_met"]
                and shared_optimizer_proposal["coordinate_rms"] <= 1.0e-5
                and shared_optimizer_proposal["maximum_absolute_coordinate"] <= 1.0e-4
            )
            objective_window = [
                *(row["objective"] for row in trace[-(cfg.convergence_window - 1) :]),
                total,
            ]
            objective_stability = objective_stability_summary(
                objective_window,
                window=cfg.convergence_window,
                relative_span_tolerance=cfg.objective_relative_span_tolerance,
            )
            objective_relative_span = objective_stability["objective_relative_span"]
            objective_stable = bool(objective_stability["objective_stable"])
            stationarity_met = activation_stationary and jaw_stationary
            if cfg.stage == "joint_trend":
                stationarity_met = stationarity_met and shared_stationary
            criteria_met = stationarity_met and objective_stable
            qualifying_consecutive = qualifying_consecutive + 1 if criteria_met else 0
            convergence = {
                "criteria_met": criteria_met,
                "converged": qualifying_consecutive >= cfg.convergence_consecutive,
                "qualifying_consecutive": qualifying_consecutive,
                "required_consecutive": cfg.convergence_consecutive,
                "window": cfg.convergence_window,
                "objective_relative_span": objective_relative_span,
                "objective_relative_span_tolerance": (
                    cfg.objective_relative_span_tolerance
                ),
                "activation_stationary": activation_stationary,
                "activation_relative_l2": activation_relative,
                "activation_relative_reduction_met": [
                    row["relative_reduction_met"] for row in activation_reduction
                ],
                "jaw_stationary": jaw_stationary,
                "jaw_relative_l2": jaw_relative,
                "jaw_relative_reduction_met": [
                    row["relative_reduction_met"] for row in jaw_reduction
                ],
                "shared_stationary": shared_stationary,
                "shared_relative_l2": shared_relative,
                "shared_relative_reduction_met": shared_reduction[
                    "relative_reduction_met"
                ],
                "absolute_step_source": (
                    "exact reversible next projected-Adam proposal"
                ),
            }
            fit_values_mm = [
                expression["metrics"]["area_fit_rms_mm"] for expression in expressions
            ]
            comparison_metrics = {
                "area_fit_rms_mean_mm": sum(fit_values_mm) / count,
                "area_fit_rms_max_mm": max(fit_values_mm),
                "detF_min_all_states": min(
                    neutral_metrics["detF_min"],
                    *(expression["metrics"]["detF_min"] for expression in expressions),
                ),
                "detF_max_all_states": max(
                    neutral_metrics["detF_max"],
                    *(expression["metrics"]["detF_max"] for expression in expressions),
                ),
                "neutral_surface_motion_rms_mm": neutral_metrics[
                    "surface_motion_rms_mm"
                ],
                "neutral_muscle_centroid_motion_rms_mm": neutral_metrics[
                    "muscle_centroid_motion_rms_mm"
                ],
            }
            forward_seconds = float(neutral_receipt["forward"]["seconds"]) + sum(
                float(expression["forward"]["seconds"]) for expression in expressions
            )
            adjoint_seconds = float(neutral_receipt["adjoint"]["seconds"]) + sum(
                float(expression["adjoint"]["seconds"]) for expression in expressions
            )
            new_forward_solves = physics.runtime.forward_count - forward_count_before
            if cached_evaluation is None:
                assert new_forward_solves == count + 1
            else:
                assert new_forward_solves == 0
            forward_solves = accepted_forward_solves
            row = {
                "update": update,
                "elapsed_seconds": time.perf_counter() - started,
                "objective": total,
                "reconstructed_objective": reconstructed_total,
                "comparison_fingerprint": comparison_fingerprint,
                "shared_basis": shared_basis,
                "common_objective": common_objective,
                "data_loss": data_loss,
                "neutral_loss": neutral_loss_value,
                "neutral": neutral_receipt,
                "expressions": expressions,
                "comparison_metrics": comparison_metrics,
                "smoothness": float(regularizers["smoothness"].detach()),
                "smoothness_by_expression": regularizers["smoothness_by_field"]
                .detach()
                .cpu()
                .tolist(),
                "magnitude": float(regularizers["magnitude"].detach()),
                "strong_weight": smoothness_weight,
                "shared_prior": float(shared_prior.detach()),
                "shared_prior_terms": {
                    name: float(value.detach())
                    for name, value in shared_prior_terms.items()
                },
                "jaw_prior": float(jaw_prior.detach()),
                "weighted_smoothness": float(weighted_smoothness.detach()),
                "weighted_magnitude": float(weighted_magnitude.detach()),
                "weighted_shared_prior": float(weighted_shared_prior.detach()),
                "shared_spatial_roughness": float(shared_spatial_roughness.detach()),
                "weighted_shared_spatial_roughness": float(
                    weighted_shared_spatial_roughness.detach()
                ),
                "spatial_smoothness_weight": (
                    SPATIAL_SMOOTHNESS_WEIGHT if shared_basis == "spatial80" else 0.0
                ),
                "spatial_smoothness_factor": (
                    SPATIAL_SMOOTHNESS_FACTOR if shared_basis == "spatial80" else 0.0
                ),
                "weighted_jaw_prior": float(weighted_jaw_prior.detach()),
                "activation_neighbor_rms_dimensionless": current_neighbor_rms.cpu().tolist(),
                "activation_neighbor_rms_MPa": (
                    shared.activation_reference_mpa * current_neighbor_rms
                )
                .cpu()
                .tolist(),
                "activation_data_gradient_rms": float(
                    data_gradient.square().mean().sqrt()
                ),
                "activation_regularizer_gradient_rms": float(
                    (q.grad - data_gradient).square().mean().sqrt()
                ),
                "activation_total_gradient_by_expression": (
                    activation_gradient_by_expression
                ),
                "jaw_gradient": jaw.grad.detach().cpu().tolist(),
                "stationarity": {
                    "activation": activation_stationarity,
                    "jaw": jaw_stationarity,
                    "shared": shared_stationarity,
                    "optimizer_proposal": {
                        "activation": activation_optimizer_proposal,
                        "jaw": jaw_optimizer_proposal,
                        "shared": shared_optimizer_proposal,
                        "projection": optimizer_proposal_projection,
                        "directional_derivative": (
                            optimizer_proposal_directional_derivative
                        ),
                    },
                    "mass_normalized_activation_gradient": (
                        mass_normalized_activation_gradient
                    ),
                },
                "convergence": convergence,
                "activation_spectrum": activation_spectrum,
                "shared_prior_diagnostics": shared_diagnostics,
                "shared_gradient": shared.coefficients.grad.detach().cpu().tolist(),
                "shared": shared.coefficients.detach().cpu().tolist(),
                "jaw_normalized": jaw.detach().cpu().tolist(),
                "jaw_pose_rad_m": (pose_origin + pose_scale * jaw.detach())
                .cpu()
                .tolist(),
                "forward_solves": forward_solves,
                "forward_solves_cumulative": physics.runtime.forward_count,
                "adjoint_solves": count + 1,
                "adjoint_solves_cumulative": adjoint_counter["count"],
                "forward_seconds": forward_seconds,
                "adjoint_seconds": adjoint_seconds,
                "solve_seconds": forward_seconds + adjoint_seconds,
                "evaluation_seconds": (
                    time.perf_counter() - epoch_started
                    if accepted_evaluation_seconds is None
                    else accepted_evaluation_seconds
                ),
                "evaluation_reused_from_outer_trial": cached_evaluation is not None,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            }
            trace.append(row)
            write_json(output / "trace.json", trace)
            checkpoint = {
                "schema": "joint-inverse-checkpoint-v1",
                "stage": cfg.stage,
                "update": update,
                "shared_coefficients": shared.coefficients.detach().cpu(),
                "activation": q.detach().cpu(),
                "jaw": jaw.detach().cpu(),
                "optimizer": optimizer.state_dict(),
                "primal": {k: v.cpu() for k, v in seeds.items()},
                "adjoint": {
                    k: v.cpu() for k, v in physics.runtime.warm_adjoints.items()
                },
                "materials": spec,
                "shared_basis": shared_basis,
                "shared_field": shared_field,
                "spatial_basis": spatial_basis,
                "spatial_validation": initial["protocol"].get("spatial_validation"),
                "shared_initialization_sha256": shared_initialization_sha256,
                "shared_prior_reference_sha256": shared_prior_reference_sha256,
                "input_arrays_sha256": protocol["input_arrays_sha256"],
                "input_manifest_sha256": protocol["input_manifest_sha256"],
                "neutral_checkpoint_sha256": neutral_checkpoint_sha256,
                "calibration_sha256": calibration_sha256,
                "parent_control_checkpoint_sha256": parent_checkpoint_sha256,
                "parent_control_checkpoint_update": parent_checkpoint_update,
                "comparison_fingerprint": comparison_fingerprint,
                "implementation_sha256": implementation_sha256,
                "forward_solver": expected_solver,
                "protocol": protocol,
                "metrics": row,
                "lineage": lineage,
                "stationarity_baseline": stationarity_baseline,
                "convergence": convergence,
                "preparation_complete": (
                    cfg.stage == "control_converge" and convergence["converged"]
                ),
                "inverse_converged": convergence["converged"],
                "contact_validated": True,
                "rigid_bone_ccd_validated": True,
                "contact_spec_sha256": contact_spec_sha256,
                "contact_validation_sha256": contact_validation_sha256,
                "rigid_bone_ccd_validation_sha256": (rigid_bone_ccd_validation_sha256),
                "next_local_evaluation": update,
                "resume_action": "re-evaluate saved state before optimizer step",
                "stop_status": (
                    "converged numerical preparation"
                    if cfg.stage == "control_converge" and convergence["converged"]
                    else "numerically valid evaluated checkpoint; next optimizer step pending"
                ),
                "shared_optimizer_state_present": (
                    shared.coefficients in optimizer.state
                ),
            }
            if cfg.stage == "control_converge":
                assert not checkpoint["shared_optimizer_state_present"]
            torch.save(checkpoint, output / "terminal.pt")
            if total < best:
                best = total
                torch.save(checkpoint, output / "best-numerical.pt")
            snapshot_updates = {0, 1, 2, 5, 10, 20, 50, 100, max_updates}
            if update in snapshot_updates:
                torch.save(checkpoint, output / f"checkpoint-{update:04d}.pt")
                save_visualization_snapshot(update)
            cherries.set_step(update)
            logged_metrics = {
                "objective": total,
                "common_objective": common_objective,
                "data_loss": data_loss,
                "neutral_loss": row["neutral_loss"],
                "weighted_smoothness": row["weighted_smoothness"],
                "weighted_magnitude": row["weighted_magnitude"],
                "weighted_shared_prior": row["weighted_shared_prior"],
                "weighted_shared_spatial_roughness": row[
                    "weighted_shared_spatial_roughness"
                ],
                "weighted_jaw_prior": row["weighted_jaw_prior"],
                "smoothness": row["smoothness"],
                "fit_mean_mm": comparison_metrics["area_fit_rms_mean_mm"],
                "activation_optimizer_proposal_volume_rms_max": max(
                    proposal["effective_volume_tensor_rms"]
                    for proposal in activation_optimizer_proposal["expressions"]
                ),
                "activation_optimizer_proposal_max_tet": max(
                    proposal["maximum_tet_frobenius"]
                    for proposal in activation_optimizer_proposal["expressions"]
                ),
                "jaw_optimizer_proposal_max_row": max(
                    jaw_optimizer_proposal["row_norms"]
                ),
                "forward_seconds": forward_seconds,
                "adjoint_seconds": adjoint_seconds,
            }
            if all(value is not None for value in activation_relative):
                logged_metrics["activation_relative_l2_max"] = max(
                    value for value in activation_relative if value is not None
                )
            if all(value is not None for value in jaw_relative):
                logged_metrics["jaw_relative_l2_max"] = max(
                    value for value in jaw_relative if value is not None
                )
            if shared_relative is not None:
                logged_metrics["shared_relative_l2"] = shared_relative
            if objective_relative_span is not None:
                logged_metrics["objective_relative_span"] = objective_relative_span
            cherries.log_metrics(logged_metrics)
            LOG.info("%s update %d objective %.6g", cfg.stage, update, total)
            if cfg.stage == "control_converge" and convergence["converged"]:
                stop_reason = "projected-gradient and objective convergence"
                break
            if update == max_updates:
                break
            if row["elapsed_seconds"] >= wall_budget_seconds:
                stop_reason = "wall-time budget"
                break
            before = q.detach().clone()
            shared_before = shared.coefficients.detach().clone()
            update_started = time.perf_counter()
            accepted_seeds = {
                key: value.detach().clone() for key, value in seeds.items()
            }
            accepted_warm = {
                key: value.detach().clone()
                for key, value in physics.runtime.warm_adjoints.items()
            }
            accepted_last_forward = deepcopy(physics.runtime.last_forward)
            accepted_last_adjoint = deepcopy(physics.runtime.last_adjoint)
            runtime_forward = physics.runtime.forward
            runtime_model = runtime_forward.model
            accepted_model_materials = optree.tree_map(
                lambda value: (
                    value.detach().clone()
                    if isinstance(value, torch.Tensor)
                    else deepcopy(value)
                ),
                runtime_model.get_materials(),
            )
            accepted_fixed_values = runtime_model.dof_map.fixed_values.detach().clone()
            accepted_runtime_u = runtime_forward.state.u.detach().clone()

            def project_proposal() -> dict[str, Any]:
                proposal_smoothness = float(
                    activation_regularizers(q.detach(), graph)["smoothness"]
                )
                projection = project_parameter_blocks()
                projected_neighbor_rms = activation_neighbor_rms(q.detach(), graph)
                return {
                    "proposal_smoothness": proposal_smoothness,
                    "projected_smoothness": float(
                        activation_regularizers(q.detach(), graph)["smoothness"]
                    ),
                    "projected_neighbor_rms_dimensionless": (
                        projected_neighbor_rms.cpu().tolist()
                    ),
                    "projected_neighbor_rms_MPa": (
                        shared.activation_reference_mpa * projected_neighbor_rms
                    )
                    .cpu()
                    .tolist(),
                    **projection,
                }

            def restore_auxiliary(
                seed_snapshot: dict[str, torch.Tensor] = accepted_seeds,
                warm_snapshot: dict[str, torch.Tensor] = accepted_warm,
                forward_snapshot: dict[str, Any] = accepted_last_forward,
                adjoint_snapshot: dict[str, Any] = accepted_last_adjoint,
                model_materials: dict[str, Any] = accepted_model_materials,
                fixed_values: torch.Tensor = accepted_fixed_values,
                runtime_u: torch.Tensor = accepted_runtime_u,
            ) -> None:
                seeds.clear()
                seeds.update(
                    {
                        key: value.detach().clone()
                        for key, value in seed_snapshot.items()
                    }
                )
                physics.runtime.warm_adjoints = {
                    key: value.detach().clone() for key, value in warm_snapshot.items()
                }
                physics.runtime.last_forward = deepcopy(forward_snapshot)
                physics.runtime.last_adjoint = deepcopy(adjoint_snapshot)
                model = physics.runtime.forward.model
                model.set_materials(
                    optree.tree_map(
                        lambda value: (
                            value.detach().clone()
                            if isinstance(value, torch.Tensor)
                            else deepcopy(value)
                        ),
                        model_materials,
                    )
                )
                model.dof_map.fixed_values = fixed_values.detach().clone()
                physics.runtime.forward.state.u = runtime_u.detach().clone()
                if model.collision is not None:
                    physics.runtime.forward.state.collision = model.collision.state_at(
                        physics.runtime.forward.state.u
                    )
                physics.runtime.forward.last_solution = None

            trial_cache_holder["value"] = None
            proposal_receipt = backtracked_projected_adam_step(
                optimizer=optimizer,
                parameters=(q, jaw, shared.coefficients),
                prepare_step=prepare_optimizer_step,
                project=project_proposal,
                restore_auxiliary=restore_auxiliary,
                evaluate_trial=evaluate_proposal_trial,
                cost_counters=lambda: (
                    physics.runtime.forward_count,
                    adjoint_counter["count"],
                ),
                accepted_objective=total,
                backtrack_factor=cfg.outer_backtrack_factor,
                maximum_trials=cfg.outer_max_backtracks,
                armijo_coefficient=cfg.outer_armijo_coefficient,
                deadline=started + wall_budget_seconds,
            )
            row["outer_proposal"] = proposal_receipt
            row["physical_activation_step_rms_mpa"] = (
                float((q.detach() - before).square().mean().sqrt())
                * shared.activation_reference_mpa
            )
            row["optimizer_projection_seconds"] = time.perf_counter() - update_started
            row["elapsed_seconds_after_update"] = time.perf_counter() - started
            write_json(output / "trace.json", trace)
            if not proposal_receipt["accepted"]:
                accepted_trial_cache = None
                assert torch.equal(q, before)
                assert torch.equal(shared.coefficients, shared_before)
                if proposal_receipt.get("budget_exhausted") is True:
                    stop_reason = "wall-time budget"
                    break
                stop_reason = proposal_receipt["reason"]
                proposal_failure = True
                break
            accepted_trial_cache = trial_cache_holder["value"]
            assert accepted_trial_cache is not None
        inverse_converged = bool(trace[-1]["convergence"]["converged"])
        best_checkpoint = torch.load(
            output / "best-numerical.pt", map_location="cpu", weights_only=False
        )
        save_visualization_snapshot(
            best_checkpoint["update"],
            label="best",
            saved_checkpoint=best_checkpoint,
        )
        save_visualization_snapshot(trace[-1]["update"], label="terminal")
        preparation_complete = cfg.stage == "control_converge" and inverse_converged
        accepted_updates = max(0, len(trace) - 1)
        minimum_joint_updates_met = cfg.stage != "joint_trend" or accepted_updates >= 1
        success = (
            cfg.stage == "joint_trend"
            and not proposal_failure
            and minimum_joint_updates_met
        ) or preparation_complete
        if cfg.stage == "joint_trend" and accepted_updates < 1:
            assert success is False
        terminal_checkpoint_sha256 = sha256(output / "terminal.pt")
        outer_trials = [
            trial
            for trace_row in trace
            for trial in trace_row.get("outer_proposal", {}).get("trials", [])
        ]
        summary = {
            "schema": "joint-final-summary-v1",
            "success": success,
            "preparation_complete": preparation_complete,
            "inverse_converged": inverse_converged,
            "anatomical_validation": False,
            "promotion_ready": False,
            "contact_validated": True,
            "rigid_bone_ccd_validated": True,
            "rigid_bone_ccd_validation_sha256": (rigid_bone_ccd_validation_sha256),
            "stage": cfg.stage,
            "stop_reason": stop_reason,
            "accepted_evaluations": len(trace),
            "accepted_updates": accepted_updates,
            "minimum_joint_updates_met": minimum_joint_updates_met,
            "comparison_fingerprint": comparison_fingerprint,
            "implementation_sha256": implementation_sha256,
            "shared_basis": shared_basis,
            "shared_field": shared_field,
            "spatial_basis": spatial_basis,
            "spatial_validation": initial["protocol"].get("spatial_validation"),
            "shared_initialization_sha256": shared_initialization_sha256,
            "shared_prior_reference_sha256": shared_prior_reference_sha256,
            "parent_control_checkpoint_sha256": parent_checkpoint_sha256,
            "control_receipt_sha256": control_receipt_sha256,
            "terminal_checkpoint_sha256": terminal_checkpoint_sha256,
            "lineage": lineage,
            "parameter_counts": protocol["parameter_counts"],
            "forward_solver": expected_solver,
            "forward_solves": physics.runtime.forward_count,
            "adjoint_solves": adjoint_counter["count"],
            "outer_trial_evaluations": len(outer_trials),
            "outer_rejected_trials": sum(
                trial["accepted"] is False for trial in outer_trials
            ),
            "outer_trial_seconds": sum(trial["seconds"] for trial in outer_trials),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "first": trace[0],
            "terminal": trace[-1],
        }
        write_json(
            output / "summary.json",
            summary,
        )
        COMPLETED = success
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "schema": "joint-final-failure-v1",
                "stage": cfg.stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "numerically_valid_evaluations": len(trace),
                "comparison_fingerprint": comparison_fingerprint,
                "forward_solver": expected_solver,
                "forward_solves": physics.runtime.forward_count,
                "adjoint_solves": adjoint_counter["count"],
                "elapsed_seconds": time.perf_counter() - started,
                "forward": physics.runtime.last_forward,
                "adjoint": physics.runtime.last_adjoint,
            },
        )
        raise


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
    if not COMPLETED:
        raise SystemExit(1)
