"""Converge the full-mesh neutral shared-field preparation.

The isotropic skin resultant is prescribed at a nonzero continuation target.
The remaining shared coordinates use safeguarded projected optimization.  The
approved constant20 basis remains the default; spatial80 is an explicit,
receipt-gated extension with a frozen strong roughness penalty.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Literal

import pydantic_settings as ps
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs, audit_neutral_oral_geometry
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from joint_fields import (
    BULK_TISSUES,
    SharedFieldParameters,
    research_informed_material_config,
    symmetric_matrices,
)
from joint_outer import metric_projected_direction
from joint_physics import JointPhysics
from joint_spatial_fields import (
    ANCHOR_COORDINATE_SLICES,
    ANCHOR_COUNTS,
    EXPECTED_BASIS_SHA256,
    SpatialSharedFieldParameters,
    spatial_field_config,
)
from pydantic import field_validator

from liblaf import cherries

LOG = logging.getLogger(__name__)
COMPLETED = False
PROXY_RESULTANT_N_PER_M = (89.4 + 71.8) / 2
SPATIAL_SMOOTHNESS_WEIGHT = 100.0
SPATIAL_SMOOTHNESS_FACTOR = 0.5
SPATIAL_FIELDS_SOURCE_SHA256 = (
    "fcb62322f0ca29d3d945badef3582e70ce66f46d1e1b3087396986e7f9711543"
)
SharedParameters = SharedFieldParameters | SpatialSharedFieldParameters


def json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def shared_field_receipt(
    shared: SharedParameters,
    shared_basis: Literal["constant20", "spatial80"],
    fixed_coordinate: int,
    free_coordinate_ids: torch.Tensor,
    spatial_smoothness_weight: float,
) -> dict[str, Any]:
    return {
        "schema": shared.config["schema"],
        "basis": shared_basis,
        "coefficient_count": int(shared.coefficients.numel()),
        "bulk_coordinate_slices": (
            {
                name: [value.start, value.stop]
                for name, value in ANCHOR_COORDINATE_SLICES.items()
            }
            if shared_basis == "spatial80"
            else {
                name: [6 * index, 6 * (index + 1)]
                for index, name in enumerate(BULK_TISSUES)
            }
        ),
        "bulk_anchor_counts": (
            copy.deepcopy(ANCHOR_COUNTS)
            if shared_basis == "spatial80"
            else dict.fromkeys(BULK_TISSUES, 1)
        ),
        "skin_baseline_index": fixed_coordinate,
        "skin_log_multiplier_index": int(shared.coefficients.numel() - 1),
        "fixed_coordinate_ids": [fixed_coordinate],
        "free_coordinate_ids": free_coordinate_ids.cpu().tolist(),
        "spatial_smoothness": {
            "weight": spatial_smoothness_weight if shared_basis == "spatial80" else 0.0,
            "factor": SPATIAL_SMOOTHNESS_FACTOR,
            "regularizer": (
                "bulk_spatial_roughness" if shared_basis == "spatial80" else None
            ),
            "objective_term": (
                "0.5 * beta * mean_t tr(C_t^T G_t C_t)"
                if shared_basis == "spatial80"
                else None
            ),
        },
    }


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    prepared_dir: Path = GROUP / "data/prepared"
    initial_checkpoint: Path = GROUP / "data/neutral-prestress-010/best-admissible.pt"
    contact_spec: Path | None = None
    contact_validation: Path = GROUP / "data/contact-validation/summary.json"
    newton_validation: Path = GROUP / "data/contact-validation-newton/summary.json"
    output_dir: Path = GROUP / "data/neutral-convergence-010"
    mode: Literal["smoke", "converge"] = "converge"
    skin_prestress_fraction: float = 0.1
    prior_weight: float = 0.001
    initial_spectral_step: float = 0.003
    minimum_spectral_step: float = 1e-6
    maximum_spectral_step: float = 0.1
    outer_method: Literal["spg", "bfgs"] = "spg"
    bfgs_initial_inverse_scale: float = 0.002
    bfgs_curvature_relative_tolerance: float = 1e-10
    bfgs_subproblem_residual_tolerance: float = 1e-10
    bfgs_subproblem_max_iterations: int = 20000
    armijo_coefficient: float = 1e-4
    backtrack_factor: float = 0.5
    max_line_search_trials: int = 10
    max_accepted_steps: int = 120
    max_forward_evaluations: int = 600
    wall_budget_seconds: float = 7200
    projected_gradient_inf_tolerance: float = 1e-3
    objective_range_tolerance: float = 1e-5
    stabilization_window: int = 5
    convergence_consecutive: int = 3
    minimum_accepted_steps: int = 5
    forward_rtol: float = 1e-6
    forward_atol: float = 1e-12
    adjoint_rtol: float = 1e-7
    max_forward_steps: int = 10000
    forward_method: Literal["pncg", "newton_cg"] = "pncg"
    newton_linear_rtol: float = 1e-3
    newton_max_steps: int = 12
    checkpoint_interval: int = 5
    shared_basis: Literal["constant20", "spatial80"] = "constant20"
    spatial_basis_path: Path | None = None
    spatial_audit_summary: Path | None = None
    spatial_cpu_validation: Path = (
        GROUP / "data/spatial-fields-validation-cpu-v8/summary.json"
    )
    spatial_face_gradient_validation: Path | None = None
    spatial_smoothness_weight: float = SPATIAL_SMOOTHNESS_WEIGHT

    @field_validator("skin_prestress_fraction")
    @classmethod
    def validate_skin_prestress_fraction(cls, value: float) -> float:
        if value not in (0.1, 0.25, 0.5, 1.0):
            message = "skin-prestress fraction must be 0.1, 0.25, 0.5, or 1.0"
            raise ValueError(message)
        return value


def make_physics(
    cfg: Config,
    prepared: PreparedInputs,
    spec: dict,
    contact_config: dict[str, Any] | None,
) -> JointPhysics:
    materials = spec["materials"]
    skin = materials["skin"]
    return JointPhysics(
        prepared.volume_path,
        prepared.skin_path,
        prepared.arrays,
        bulk_young_mpa={name: materials[name]["young_mpa"] for name in BULK_TISSUES},
        bulk_nu={name: materials[name]["poisson"] for name in BULK_TISSUES},
        skin_young_mpa=skin["reference_map"]["young_mpa"],
        skin_nu=skin["poisson"],
        thickness_m=skin["thickness_m"],
        rtol=cfg.forward_rtol,
        atol=cfg.forward_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        max_steps=cfg.max_forward_steps,
        contact_config=contact_config,
        forward_method=cfg.forward_method,
        newton_linear_rtol=cfg.newton_linear_rtol,
        newton_max_steps=cfg.newton_max_steps,
    )


def neutral_terms(
    physics: JointPhysics,
    shared: SharedParameters,
    displacement: torch.Tensor,
    prior_weight: float,
    *,
    shared_basis: Literal["constant20", "spatial80"],
    spatial_smoothness_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    surface = (
        (
            physics.weights_t[:, None] * displacement[physics.observation_t].square()
        ).sum()
        * 1e6
        / 0.25**2
    )
    centers = displacement[physics.muscle_tets_t].mean(dim=1)
    muscle = (physics.muscle_mass_t[:, None] * centers.square()).sum() * 1e6 / 0.5**2
    regularizers = shared.regularizers()
    weighted_prior = prior_weight * regularizers["prior_total"]
    spatial_roughness = (
        regularizers["bulk_spatial_roughness"]
        if shared_basis == "spatial80"
        else displacement.new_zeros(())
    )
    weighted_spatial_roughness = (
        SPATIAL_SMOOTHNESS_FACTOR * spatial_smoothness_weight * spatial_roughness
    )
    terms = {
        "surface_loss": surface,
        "muscle_loss": muscle,
        "prior_total": regularizers["prior_total"],
        "weighted_prior": weighted_prior,
        "bulk_spatial_roughness": spatial_roughness,
        "weighted_spatial_roughness": weighted_spatial_roughness,
        **{
            name: value
            for name, value in regularizers.items()
            if name.endswith(("_prior", "_magnitude", "_spatial_roughness"))
        },
    }
    return surface + muscle + weighted_prior + weighted_spatial_roughness, terms


def hard_shape_valid(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["inverted_tetrahedra"] == 0
        and metrics["detF_min"] >= 0.25
        and metrics["detF_max"] <= 2.0
        and metrics["skin_area_ratio_min"] >= 0.25
    )


def neutral_budget_met(metrics: dict[str, Any]) -> bool:
    return bool(
        hard_shape_valid(metrics)
        and metrics["surface_motion_rms_mm"] <= 0.25
        and metrics["muscle_centroid_motion_rms_mm"] <= 0.5
    )


def oral_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": value["mode"],
        "neutral_invariants_ok": value["neutral_invariants_ok"],
        "admissible": value["admissible"],
        "gate": value["gate"],
        "new_lip_pairs": value["fem_lip"]["new_contact_pairs"],
        "worsened_lip_pairs": value["fem_lip"]["worsened_inherited_pairs"],
        "new_mandible_upper_pairs": value["fem_mandible_oral"]["upper_oral"][
            "new_contact_pairs"
        ],
        "new_mandible_lower_pairs": value["fem_mandible_oral"]["lower_oral"][
            "new_contact_pairs"
        ],
        "mandible_support_max_error_m": value["mandible_pose_consistency"][
            "max_error_m"
        ],
        "anatomical_admissibility_claimed": False,
    }


def main(cfg: Config) -> None:  # noqa: C901, PLR0912, PLR0915
    global COMPLETED  # noqa: PLW0603
    assert cfg.prior_weight > 0
    assert cfg.skin_prestress_fraction in (0.1, 0.25, 0.5, 1.0)
    assert 0 < cfg.minimum_spectral_step <= cfg.initial_spectral_step
    assert cfg.initial_spectral_step <= cfg.maximum_spectral_step
    assert cfg.outer_method in {"spg", "bfgs"}
    assert cfg.bfgs_initial_inverse_scale == 0.002
    assert cfg.bfgs_curvature_relative_tolerance == 1e-10
    assert cfg.bfgs_subproblem_residual_tolerance == 1e-10
    assert cfg.bfgs_subproblem_max_iterations == 20000
    assert 0 < cfg.backtrack_factor < 1
    assert 0 < cfg.armijo_coefficient < 1
    assert cfg.max_line_search_trials > 0
    assert cfg.projected_gradient_inf_tolerance == 1e-3
    assert cfg.objective_range_tolerance == 1e-5
    assert cfg.stabilization_window >= 2
    assert cfg.convergence_consecutive >= 1
    assert cfg.minimum_accepted_steps >= cfg.stabilization_window
    assert cfg.forward_method in {"pncg", "newton_cg"}
    assert cfg.shared_basis in {"constant20", "spatial80"}
    if cfg.shared_basis == "spatial80":
        assert cfg.spatial_basis_path is not None
        assert cfg.spatial_audit_summary is not None
        assert cfg.spatial_face_gradient_validation is not None
        assert cfg.spatial_smoothness_weight == SPATIAL_SMOOTHNESS_WEIGHT
    if cfg.forward_method == "newton_cg":
        assert cfg.newton_linear_rtol == 1e-3
        assert cfg.newton_max_steps == 12
        assert cfg.contact_spec is not None
    if cfg.mode == "smoke":
        assert cfg.max_accepted_steps <= 10
    else:
        assert cfg.contact_spec is not None, (
            "production convergence requires an explicit contact spec"
        )

    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)
    archive_sources(output)
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz",
        cfg.prepared_dir / "manifest.json",
        verify_sources=True,
    )
    configure_cuda()
    initial = torch.load(cfg.initial_checkpoint, map_location="cpu", weights_only=False)
    assert initial["schema"] == "joint-inverse-checkpoint-v1"
    assert initial["stage"] == "neutral"
    assert initial["protocol"]["input_arrays_sha256"] == sha256(
        cfg.prepared_dir / "inputs.npz"
    )
    current_manifest_sha256 = sha256(cfg.prepared_dir / "manifest.json")
    initial_manifest_sha256 = initial["protocol"]["input_manifest_sha256"]
    initial_fraction = float(initial["protocol"]["skin_prestress_fraction"])
    fraction_tolerance = 1e-14
    if cfg.skin_prestress_fraction < initial_fraction - fraction_tolerance:
        message = "skin-prestress continuation cannot decrease its target"
        raise ValueError(message)
    same_target = math.isclose(
        cfg.skin_prestress_fraction,
        initial_fraction,
        rel_tol=0,
        abs_tol=fraction_tolerance,
    )
    target_increase = cfg.skin_prestress_fraction > (
        initial_fraction + fraction_tolerance
    )
    assert same_target or target_increase
    manifest_matches = initial_manifest_sha256 == current_manifest_sha256
    if not manifest_matches:
        assert cfg.skin_prestress_fraction == 0.1, (
            "a stale-manifest checkpoint may seed only the 10% revalidation stage"
        )
    contact_config = (
        json.loads(cfg.contact_spec.read_text())
        if cfg.contact_spec is not None
        else None
    )
    if contact_config is not None:
        assert contact_config["schema"] == "joint-bone-contact-v1"
        assert contact_config["enabled"] is True
        contact_validation = json.loads(cfg.contact_validation.read_text())
        assert contact_validation["schema"] == "joint-contact-validation-v1"
        assert contact_validation["success"] is True
        assert contact_validation["contact_spec_sha256"] == sha256(cfg.contact_spec)
    else:
        contact_validation = None
    if cfg.forward_method == "newton_cg":
        newton_validation = json.loads(cfg.newton_validation.read_text())
        assert newton_validation["schema"] == "joint-contact-validation-v1"
        assert newton_validation["success"] is True
        assert newton_validation["forward_solver"]["method"] == "newton_cg"
        assert newton_validation["contact_spec_sha256"] == sha256(cfg.contact_spec)
    else:
        newton_validation = None
    initial_shared_basis = initial["protocol"].get("shared_basis", "constant20")
    if initial_shared_basis not in {"constant20", "spatial80"}:
        message = f"unknown incoming shared basis: {initial_shared_basis}"
        raise ValueError(message)
    incoming_spec = initial["materials"]
    if initial_shared_basis == "spatial80":
        assert incoming_spec == spatial_field_config()
        constitutive_spec = research_informed_material_config()
    else:
        constitutive_spec = incoming_spec
    physics = make_physics(cfg, prepared, constitutive_spec, contact_config)
    if cfg.shared_basis == "constant20":
        if initial_shared_basis != "constant20":
            message = "spatial80 checkpoints cannot be reduced to constant20"
            raise ValueError(message)
        shared: SharedParameters = SharedFieldParameters(constitutive_spec)
        spatial_basis_receipt = None
        spatial_cpu_validation = None
        spatial_face_gradient_validation = None
    else:
        assert cfg.spatial_basis_path is not None
        assert cfg.spatial_audit_summary is not None
        assert cfg.spatial_face_gradient_validation is not None
        shared = SpatialSharedFieldParameters(
            cfg.spatial_basis_path,
            cfg.spatial_audit_summary,
            len(physics.tets),
            material_config=constitutive_spec,
            device="cuda",
        )
        spatial_basis_receipt = shared.basis_receipt()
        spatial_cpu_validation = json.loads(cfg.spatial_cpu_validation.read_text())
        assert spatial_cpu_validation["schema"] == "joint-spatial-field-validation-v1"
        assert spatial_cpu_validation["success"] is True
        assert (
            spatial_cpu_validation["status"]
            == "passed_cpu_inactive_spatial_field_validation"
        )
        assert spatial_cpu_validation["basis"] == spatial_basis_receipt
        assert spatial_basis_receipt["basis_sha256"] == EXPECTED_BASIS_SHA256
        assert (
            spatial_cpu_validation["sources"][
                str((GROUP / "src/joint_spatial_fields.py").resolve())
            ]
            == SPATIAL_FIELDS_SOURCE_SHA256
        )
        assert (
            spatial_cpu_validation["strong_roughness_contract"][
                "included_in_prior_total"
            ]
            is False
        )
        spatial_face_gradient_validation = json.loads(
            cfg.spatial_face_gradient_validation.read_text()
        )
        assert (
            spatial_face_gradient_validation["schema"]
            == "joint-spatial-face-gradient-validation-v1"
        )
        assert spatial_face_gradient_validation["success"] is True
        assert spatial_face_gradient_validation["basis"] == spatial_basis_receipt
        assert spatial_face_gradient_validation["contact_enabled"] is True
        assert spatial_face_gradient_validation["spatial_smoothness_weight"] == (
            SPATIAL_SMOOTHNESS_WEIGHT
        )
        assert spatial_face_gradient_validation["spatial_smoothness_factor"] == (
            SPATIAL_SMOOTHNESS_FACTOR
        )
    target_resultant = cfg.skin_prestress_fraction * PROXY_RESULTANT_N_PER_M
    target_coordinate = target_resultant / shared.skin_resultant_scale_n_per_m
    with torch.no_grad():
        if cfg.shared_basis == "spatial80" and initial_shared_basis == "constant20":
            assert isinstance(shared, SpatialSharedFieldParameters)
            shared.load_constant20_(initial["shared_coefficients"])
        else:
            shared.coefficients.copy_(initial["shared_coefficients"])
        fixed_coordinate = (
            shared.skin_baseline_index
            if isinstance(shared, SpatialSharedFieldParameters)
            else 18
        )
        shared.coefficients[fixed_coordinate] = target_coordinate
        before_initial_projection = shared.coefficients.detach().clone()
        initial_projection = shared.project_()
        initial_projection_max_change = float(
            (shared.coefficients - before_initial_projection).abs().max()
        )
        assert initial_projection_max_change <= 1e-12
        assert math.isclose(
            float(shared.coefficients[fixed_coordinate]),
            target_coordinate,
            rel_tol=0,
            abs_tol=1e-14,
        )
    spec = copy.deepcopy(shared.config)
    initial_contact_sha256 = initial["protocol"].get("contact", {}).get("spec_sha256")
    current_contact_sha256 = (
        sha256(cfg.contact_spec) if cfg.contact_spec is not None else None
    )
    if spatial_face_gradient_validation is not None:
        assert (
            spatial_face_gradient_validation["contact_spec_sha256"]
            == current_contact_sha256
        )
        assert spatial_face_gradient_validation["input_arrays_sha256"] == sha256(
            cfg.prepared_dir / "inputs.npz"
        )
        assert (
            spatial_face_gradient_validation["input_manifest_sha256"]
            == current_manifest_sha256
        )
    if target_increase:
        assert initial.get("neutral_converged") is True, (
            "higher prestress continuation requires a converged prior stage"
        )
        assert manifest_matches
        assert initial_contact_sha256 == current_contact_sha256
    elif initial_contact_sha256 is not None:
        assert initial_contact_sha256 == current_contact_sha256

    pose = torch.zeros(6)
    accepted_u = initial["primal"]["neutral"].to(device="cuda")
    physics.runtime.warm_adjoints = {
        key: value.to(device="cuda")
        for key, value in initial.get("adjoint", {}).items()
    }
    free_coordinate_ids = (
        shared.free_coordinate_indices
        if isinstance(shared, SpatialSharedFieldParameters)
        else torch.tensor(
            [*range(18), 19], dtype=torch.int64, device=shared.coefficients.device
        )
    )
    same_basis = initial_shared_basis == cfg.shared_basis
    if not same_basis:
        assert same_target, "basis transitions must hold the prestress target fixed"
        assert initial_shared_basis == "constant20"
        assert cfg.shared_basis == "spatial80"
    if cfg.shared_basis == "spatial80" and initial_shared_basis == "spatial80":
        assert initial["protocol"]["spatial_basis"] == spatial_basis_receipt
    shared_field = shared_field_receipt(
        shared,
        cfg.shared_basis,
        fixed_coordinate,
        free_coordinate_ids,
        cfg.spatial_smoothness_weight,
    )
    objective_fingerprint_payload = {
        "schema": "joint-neutral-objective-fingerprint-v1",
        "shared_field": shared_field,
        "spatial_basis": spatial_basis_receipt,
        "skin_prestress_fraction": cfg.skin_prestress_fraction,
        "skin_target_n_per_m": target_resultant,
        "prior_weight": cfg.prior_weight,
        "spatial_smoothness_weight": (
            cfg.spatial_smoothness_weight if cfg.shared_basis == "spatial80" else 0.0
        ),
        "spatial_smoothness_factor": SPATIAL_SMOOTHNESS_FACTOR,
        "input_arrays_sha256": sha256(cfg.prepared_dir / "inputs.npz"),
        "input_manifest_sha256": current_manifest_sha256,
        "contact_spec_sha256": current_contact_sha256,
        "forward_solver": physics.runtime.forward_solver,
        "forward_tolerances": physics.runtime.tolerances,
        "loss": "neutral surface + muscle + weighted prior + weighted spatial roughness",
    }
    objective_fingerprint = {
        "sha256": json_sha256(objective_fingerprint_payload),
        "payload": objective_fingerprint_payload,
    }
    incoming_fingerprint = initial["protocol"].get("objective_fingerprint")
    history_eligible = bool(
        same_target
        and same_basis
        and incoming_fingerprint == objective_fingerprint
        and initial_contact_sha256 == current_contact_sha256
        and manifest_matches
    )
    saved_optimizer = initial.get("neutral_optimizer", {})
    saved_method_matches = saved_optimizer.get("outer_method") == cfg.outer_method
    saved_state_available = bool(
        (
            cfg.outer_method == "bfgs"
            and isinstance(saved_optimizer.get("inverse_hessian"), torch.Tensor)
        )
        or (
            cfg.outer_method == "spg"
            and saved_optimizer.get("spectral_step") is not None
        )
    )
    history_reused = history_eligible and saved_method_matches and saved_state_available
    spectral_step = cfg.initial_spectral_step
    if cfg.outer_method == "spg" and history_reused:
        spectral_step = float(
            saved_optimizer.get("spectral_step", cfg.initial_spectral_step)
        )
    spectral_step = min(
        cfg.maximum_spectral_step, max(cfg.minimum_spectral_step, spectral_step)
    )
    bfgs_inverse = (
        torch.eye(
            len(free_coordinate_ids),
            dtype=shared.coefficients.dtype,
            device=shared.coefficients.device,
        )
        * cfg.bfgs_initial_inverse_scale
    )
    if cfg.outer_method == "bfgs" and history_reused:
        bfgs_inverse = saved_optimizer["inverse_hessian"].to(
            device=shared.coefficients.device, dtype=shared.coefficients.dtype
        )
    assert bfgs_inverse.shape == (
        len(free_coordinate_ids),
        len(free_coordinate_ids),
    )
    assert torch.isfinite(bfgs_inverse).all()
    assert float(torch.linalg.eigvalsh(bfgs_inverse).min()) > 0
    history_action = "reused" if history_reused else "reset"
    history_reason = (
        "same target, basis, objective fingerprint, contact, inputs, and method"
        if history_action == "reused"
        else "target, basis, objective fingerprint, contact, inputs, or method changed"
    )
    # Every invocation is a new continuation segment. The incoming checkpoint
    # supplies a physical seed only; convergence windows and budgets start fresh.
    accepted_steps = 0
    evaluation_count = 0
    adjoint_count = 0
    trace: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    started = time.perf_counter()
    forward_seconds_total = 0.0
    adjoint_seconds_total = 0.0
    best_objective = math.inf
    qualifying_consecutive = 0

    constraints = spec["constraints"]
    basis_transition = (
        {
            "schema": "joint-constant20-to-spatial80-transition-v1",
            "source_basis": initial_shared_basis,
            "target_basis": cfg.shared_basis,
            "source_coefficient_shape": list(initial["shared_coefficients"].shape),
            "source_coefficients_sha256": tensor_sha256(initial["shared_coefficients"]),
            "mapping": (
                "repeat each tissue's six constant coordinates at all 4/4/5 "
                "anchors; map skin indices 18/19 to 78/79"
            ),
            "target_coordinate_set_after_embedding": fixed_coordinate,
            "initial_projection": initial_projection,
            "initial_projection_max_change": initial_projection_max_change,
            "inverse_hessian_action": "reset",
        }
        if not same_basis
        else None
    )
    protocol = {
        "schema": "joint-neutral-convergence-protocol-v1",
        "stage": "neutral",
        "mode": cfg.mode,
        "config": cfg.model_dump(mode="json"),
        "input_manifest_sha256": current_manifest_sha256,
        "input_arrays_sha256": sha256(cfg.prepared_dir / "inputs.npz"),
        "initial_checkpoint_sha256": sha256(cfg.initial_checkpoint),
        "initial_checkpoint_manifest_sha256": initial_manifest_sha256,
        "initial_checkpoint_manifest_matches_current": manifest_matches,
        "initial_checkpoint_revalidation": (
            "not required; manifest hashes match"
            if manifest_matches
            else (
                "seed only: current inputs passed source verification and the initial "
                "state receives a fresh strict equilibrium, shape, contact, and oral audit"
            )
        ),
        "initial_history_policy": (
            "seed only; accepted-step, evaluation, stabilization, and qualification "
            "history reset for this continuation segment"
        ),
        "continuation_relation": (
            "constant20_to_spatial80_same_target"
            if same_target and not same_basis
            else ("same_target_resume" if same_target else "target_increase")
        ),
        "initial_skin_prestress_fraction": initial_fraction,
        "target_increase_requires_prior_convergence": target_increase,
        "bfgs_inverse_history_policy": (
            "reuse only for same-target, same-basis BFGS continuation; reset to "
            "declared H0 unless the full objective fingerprint also matches"
        ),
        "objective_fingerprint": objective_fingerprint,
        "continuation_eligibility": {
            "incoming_objective_fingerprint": incoming_fingerprint,
            "history_eligible": history_eligible,
            "optimizer_history_action": history_action,
            "reason": history_reason,
        },
        "basis_transition": basis_transition,
        "comet_git_metadata": (
            "disabled experiment-locally; the 164k-file dirty checkout made SDK "
            "git patch discovery block startup and does not affect numerical evidence"
        ),
        "materials": spec,
        "shared_basis": cfg.shared_basis,
        "spatial_basis": spatial_basis_receipt,
        "shared_field": shared_field,
        "spatial_validation": {
            "cpu": (
                {
                    "path": str(cfg.spatial_cpu_validation.resolve()),
                    "sha256": sha256(cfg.spatial_cpu_validation),
                    "receipt": spatial_cpu_validation,
                }
                if spatial_cpu_validation is not None
                else None
            ),
            "full_face_directional": (
                {
                    "path": str(cfg.spatial_face_gradient_validation.resolve()),
                    "sha256": sha256(cfg.spatial_face_gradient_validation),
                    "receipt": spatial_face_gradient_validation,
                }
                if spatial_face_gradient_validation is not None
                else None
            ),
        },
        "free_coordinates": len(free_coordinate_ids),
        "fixed_coordinate": fixed_coordinate,
        "skin_prestress_fraction": cfg.skin_prestress_fraction,
        "skin_target_n_per_m": target_resultant,
        "skin_target_status": (
            "prescribed continuation proxy; not a measured prestress map"
        ),
        "optimizer": (
            "spectral projected gradient with monotone Armijo backtracking and BB1 step updates"
            if cfg.outer_method == "spg"
            else (
                f"full-memory projected BFGS on {len(free_coordinate_ids)} free "
                "coordinates with monotone "
                "Armijo backtracking"
            )
        ),
        "outer_optimizer": {
            "method": cfg.outer_method,
            "bfgs_initial_inverse_scale": cfg.bfgs_initial_inverse_scale,
            "metric_projection": {
                "objective": "g.d + 0.5*d.T*inverse(H)*d",
                "residual_tolerance": cfg.bfgs_subproblem_residual_tolerance,
                "max_iterations": cfg.bfgs_subproblem_max_iterations,
                "direct_policy": (
                    "use the unconstrained BFGS direction only when it is already "
                    "feasible; otherwise solve the convex metric-projected quadratic"
                ),
            },
            "bfgs_curvature_policy": (
                "apply the inverse-BFGS update only when sTy exceeds "
                "1e-10*||s||*||y||; otherwise retain the prior positive-definite "
                "inverse Hessian and record the skipped update; projected direction "
                "must be strict descent"
            ),
            "fallback": None,
        },
        "forward_solver": physics.runtime.forward_solver,
        "forward_solver_validation": {
            "path": (
                str(cfg.newton_validation.resolve())
                if newton_validation is not None
                else None
            ),
            "sha256": (
                sha256(cfg.newton_validation) if newton_validation is not None else None
            ),
            "receipt": newton_validation,
        },
        "projected_gradient_definition": (
            f"x - projection(x - gradient) on the {len(free_coordinate_ids)} free "
            "dimensionless coordinates; "
            "infinity norm reported"
        ),
        "convergence_requires": (
            "optimizer stationarity and objective stabilization separately from "
            "neutral shape budgets"
        ),
        "neutral_budgets": {
            "surface_motion_rms_mm": 0.25,
            "muscle_centroid_motion_rms_mm": 0.5,
            "detF": [0.25, 2.0],
            "skin_area_ratio_min": 0.25,
        },
        "oral_status": (
            "diagnostic only for neutral preparation; no anatomical admissibility claim"
        ),
        "contact": {
            "required_for_final": True,
            "enabled": contact_config is not None,
            "spec_path": (
                str(cfg.contact_spec.resolve())
                if cfg.contact_spec is not None
                else None
            ),
            "spec_sha256": (
                sha256(cfg.contact_spec) if cfg.contact_spec is not None else None
            ),
            "validation_sha256": (
                sha256(cfg.contact_validation) if contact_config is not None else None
            ),
            "validation": contact_validation,
            "config": contact_config,
            "surface_map": physics.contact_definition,
            "status": (
                "explicit soft-tissue-vs-bone IPC barrier"
                if contact_config is not None
                else "optimizer-functionality evidence only; contact not configured"
            ),
        },
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "material-config.json", spec)
    write_json(output / "shared-field-config.json", shared.config)

    def project(candidate: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        current = shared.coefficients.detach().clone()
        with torch.no_grad():
            shared.coefficients.copy_(candidate)
            shared.coefficients[fixed_coordinate] = target_coordinate
            receipt = shared.project_()
            projected = shared.coefficients.detach().clone()
            shared.coefficients.copy_(current)
        assert math.isclose(
            float(projected[fixed_coordinate]),
            target_coordinate,
            rel_tol=0,
            abs_tol=1e-14,
        )
        return projected, receipt

    def project_free(candidate: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        assert candidate.shape == free_coordinate_ids.shape
        full = template.detach().clone()
        full[free_coordinate_ids] = candidate
        projected, _ = project(full)
        return projected[free_coordinate_ids]

    def material_receipt() -> dict[str, Any]:
        epsilon = float(constraints["baseline_epsilon"])
        upper = float(constraints["baseline_upper_mu_multiple"])
        receipt: dict[str, Any] = {
            "shared_basis": cfg.shared_basis,
            "bulk_eigenvalue_bounds_mpa": {
                name: [-(1 - epsilon) * mu, upper * mu]
                for name, mu in zip(BULK_TISSUES, shared.bulk_mu_mpa, strict=True)
            },
            "skin_resultant_n_per_m": float(
                shared.skin_resultant_n_per_m()[0, 0].detach()
            ),
            "skin_target_n_per_m": target_resultant,
            "skin_stiffness_multiplier": float(
                shared.skin_stiffness_multiplier().detach()
            ),
            "skin_stiffness_multiplier_bounds": constraints["skin_multiplier_bounds"],
        }
        if isinstance(shared, SpatialSharedFieldParameters):
            receipt["bulk_anchor_eigenvalue_ranges_dimensionless"] = {}
            receipt["bulk_anchor_eigenvalue_ranges_mpa"] = {}
            for name, mu_mpa in zip(BULK_TISSUES, shared.bulk_mu_mpa, strict=True):
                anchor_values = torch.linalg.eigvalsh(
                    symmetric_matrices(shared.anchor_coordinates(name).detach())
                )
                dimensionless_range = [
                    float(anchor_values.min()),
                    float(anchor_values.max()),
                ]
                receipt["bulk_anchor_eigenvalue_ranges_dimensionless"][name] = (
                    dimensionless_range
                )
                receipt["bulk_anchor_eigenvalue_ranges_mpa"][name] = [
                    mu_mpa * dimensionless_range[0],
                    mu_mpa * dimensionless_range[1],
                ]
            receipt["spatial_basis"] = spatial_basis_receipt
        else:
            eigenvalues = torch.linalg.eigvalsh(shared.bulk_stresses_mpa().detach())
            receipt["bulk_eigenvalues_mpa"] = {
                name: eigenvalues[index].cpu().tolist()
                for index, name in enumerate(BULK_TISSUES)
            }
        return receipt

    def restore_runtime(
        x: torch.Tensor,
        u: torch.Tensor,
        warm: dict[str, torch.Tensor],
        forward_receipt: dict[str, Any],
        adjoint_receipt: dict[str, Any],
    ) -> None:
        with torch.no_grad():
            shared.coefficients.copy_(x)
            materials = physics.materials(
                shared.bulk_stresses_mpa(),
                shared.skin_resultant_n_per_m(),
                shared.skin_stiffness_multiplier(),
                None,
            )
            physics.runtime.forward.model.set_materials(materials)
            physics.runtime.forward.model.dof_map.fixed_values = physics.boundary(
                pose
            ).detach()
            physics.runtime.forward.state.u = u.detach().clone()
            if physics.runtime.forward.model.collision is not None:
                physics.runtime.forward.state.collision = (
                    physics.runtime.forward.model.collision.state_at(
                        physics.runtime.forward.state.u
                    )
                )
        physics.runtime.warm_adjoints = {
            key: value.detach().clone() for key, value in warm.items()
        }
        physics.runtime.last_forward = copy.deepcopy(forward_receipt)
        physics.runtime.last_adjoint = copy.deepcopy(adjoint_receipt)

    def evaluate(
        seed: torch.Tensor, *, backward: bool
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, Any],
        torch.Tensor | None,
    ]:
        nonlocal evaluation_count, adjoint_count
        shared.coefficients.grad = None
        evaluation_count += 1
        u = physics.solve(
            shared.bulk_stresses_mpa(),
            shared.skin_resultant_n_per_m(),
            shared.skin_stiffness_multiplier(),
            None,
            pose,
            seed,
            key="neutral",
        )
        objective, terms = neutral_terms(
            physics,
            shared,
            u,
            cfg.prior_weight,
            shared_basis=cfg.shared_basis,
            spatial_smoothness_weight=cfg.spatial_smoothness_weight,
        )
        metrics = physics.metrics(u)
        gradient = None
        if backward:
            objective.backward()
            adjoint_count += 1
            assert shared.coefficients.grad is not None
            gradient = shared.coefficients.grad.detach().clone()
            gradient[fixed_coordinate] = 0
            assert torch.isfinite(gradient).all()
        return u, objective, terms, metrics, gradient

    def convergence_values(
        x: torch.Tensor, gradient: torch.Tensor, objectives: list[float]
    ) -> dict[str, Any]:
        mapped, _ = project(x - gradient)
        projected_gradient = x - mapped
        projected_gradient[fixed_coordinate] = 0
        window = objectives[-cfg.stabilization_window :]
        stabilized = len(window) == cfg.stabilization_window
        objective_range = (
            (max(window) - min(window)) / max(1.0, abs(window[-1]))
            if stabilized
            else None
        )
        stationarity = float(projected_gradient.abs().max())
        return {
            "projected_gradient_inf": stationarity,
            "projected_gradient_l2": float(
                torch.linalg.vector_norm(projected_gradient)
            ),
            "raw_gradient_inf": float(gradient.abs().max()),
            "projected_gradient_free": projected_gradient[free_coordinate_ids]
            .cpu()
            .tolist(),
            "raw_gradient_free": gradient[free_coordinate_ids].cpu().tolist(),
            "objective_relative_range": objective_range,
            "projected_gradient_inf_tolerance": (cfg.projected_gradient_inf_tolerance),
            "objective_range_tolerance": cfg.objective_range_tolerance,
            "stabilization_window": cfg.stabilization_window,
            "stationarity_met": (stationarity <= cfg.projected_gradient_inf_tolerance),
            "objective_stabilized": (
                stabilized
                and objective_range is not None
                and objective_range <= cfg.objective_range_tolerance
            ),
        }

    def outer_optimizer_receipt(
        curvature_update: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        receipt: dict[str, Any] = {"method": cfg.outer_method}
        if cfg.outer_method == "spg":
            receipt["spectral_step"] = spectral_step
        else:
            eigenvalues = torch.linalg.eigvalsh(bfgs_inverse)
            receipt.update(
                {
                    "inverse_hessian_min_eigenvalue": float(eigenvalues.min()),
                    "inverse_hessian_max_eigenvalue": float(eigenvalues.max()),
                    "curvature_update": curvature_update,
                }
            )
        return receipt

    def checkpoint(row: dict[str, Any], u: torch.Tensor) -> dict[str, Any]:
        optimizer_converged = row["optimizer_converged"]
        budget_met = row["neutral_budget_met"]
        contact_validated = row["contact_validated"]
        preparation_complete = optimizer_converged and budget_met and contact_validated
        return {
            "schema": "joint-inverse-checkpoint-v1",
            "stage": "neutral",
            "update": row["update"],
            "accepted_steps": accepted_steps,
            "shared_coefficients": shared.coefficients.detach().cpu(),
            "shared_field": shared_field,
            "optimizer": {
                "type": cfg.outer_method,
                "spectral_step": spectral_step if cfg.outer_method == "spg" else None,
            },
            "neutral_optimizer": {
                "spectral_step": spectral_step,
                "outer_method": cfg.outer_method,
                "inverse_hessian": (
                    bfgs_inverse.detach().cpu() if cfg.outer_method == "bfgs" else None
                ),
                "evaluation_count": evaluation_count,
                "adjoint_count": adjoint_count,
            },
            "next_update": row["update"] + 1,
            "primal": {"neutral": u.detach().cpu()},
            "adjoint": {
                key: value.detach().cpu()
                for key, value in physics.runtime.warm_adjoints.items()
            },
            "protocol": protocol,
            "materials": spec,
            "metrics": row,
            "neutral_budget_met": budget_met,
            "optimizer_converged": optimizer_converged,
            "contact_validated": contact_validated,
            "preparation_complete": preparation_complete,
            "neutral_converged": preparation_complete,
            "inverse_converged": preparation_complete,
        }

    def contact_receipt() -> dict[str, Any]:
        receipt = physics.runtime.last_forward.get("contact")
        if receipt is None:
            return {
                "enabled": False,
                "contact_numerically_valid": False,
                "status": "not configured",
            }
        assert receipt["enabled"] is True
        return copy.deepcopy(receipt)

    try:
        accepted_x = shared.coefficients.detach().clone()
        u, objective, terms, metrics, gradient = evaluate(accepted_u, backward=True)
        assert gradient is not None
        assert hard_shape_valid(metrics), metrics
        accepted_u = u.detach().clone()
        accepted_x = shared.coefficients.detach().clone()
        forward_seconds_total += float(physics.runtime.last_forward["seconds"])
        adjoint_seconds_total += float(physics.runtime.last_adjoint["seconds"])
        objectives = [float(objective.detach())]
        convergence = convergence_values(accepted_x, gradient, objectives)
        contact = contact_receipt()
        contact_validated = bool(
            contact["enabled"] and contact["contact_numerically_valid"]
        )
        if contact_config is not None:
            assert contact_validated, contact
        oral = oral_summary(
            audit_neutral_oral_geometry(
                prepared, physics.points + accepted_u.cpu().numpy()
            )
        )
        assert oral["neutral_invariants_ok"], oral
        row: dict[str, Any] = {
            "update": 0,
            "accepted_steps": accepted_steps,
            "evaluation_count": evaluation_count,
            "elapsed_seconds": time.perf_counter() - started,
            "objective": objectives[-1],
            "terms": {name: float(value.detach()) for name, value in terms.items()},
            **{
                key: convergence[key]
                for key in (
                    "projected_gradient_inf",
                    "projected_gradient_l2",
                    "raw_gradient_inf",
                )
            },
            "convergence": convergence,
            "qualifying_consecutive": 0,
            "optimizer_converged": False,
            "neutral_budget_met": neutral_budget_met(metrics),
            "contact": contact,
            "contact_validated": contact_validated,
            "preparation_converged": False,
            "preparation_complete": False,
            "acceptance": {"kind": "initial_evaluation", "trials": 0},
            "parameter_step_inf": 0.0,
            "spectral_step": spectral_step if cfg.outer_method == "spg" else None,
            "outer_optimizer": outer_optimizer_receipt(),
            "shared": accepted_x.cpu().tolist(),
            "material": material_receipt(),
            "metrics": metrics,
            "oral_diagnostic": oral,
            "forward": copy.deepcopy(physics.runtime.last_forward),
            "adjoint": copy.deepcopy(physics.runtime.last_adjoint),
            "solve_counts": {
                "forward": evaluation_count,
                "adjoint": adjoint_count,
            },
            "solve_seconds": {
                "forward": forward_seconds_total,
                "adjoint": adjoint_seconds_total,
            },
        }
        trace.append(row)
        write_json(output / "trace.json", trace)
        state = checkpoint(row, accepted_u)
        atomic_torch_save(state, output / "terminal.pt")
        if row["neutral_budget_met"]:
            best_objective = row["objective"]
            atomic_torch_save(state, output / "best-admissible.pt")
        restore_runtime(
            accepted_x,
            accepted_u,
            physics.runtime.warm_adjoints,
            row["forward"],
            row["adjoint"],
        )

        stop_reason = "accepted-step budget"
        while accepted_steps < cfg.max_accepted_steps:
            if evaluation_count >= cfg.max_forward_evaluations:
                stop_reason = "forward-evaluation budget"
                break
            if time.perf_counter() - started >= cfg.wall_budget_seconds:
                stop_reason = "wall-time budget"
                break
            if cfg.outer_method == "spg":
                proposal = accepted_x - spectral_step * gradient
                direction_state = {
                    "method": "spg",
                    "spectral_step": spectral_step,
                }
            else:
                direction_free, subproblem = metric_projected_direction(
                    accepted_x[free_coordinate_ids],
                    gradient[free_coordinate_ids],
                    bfgs_inverse,
                    lambda value, template=accepted_x: project_free(value, template),
                    residual_tolerance=cfg.bfgs_subproblem_residual_tolerance,
                    max_iterations=cfg.bfgs_subproblem_max_iterations,
                )
                proposal = accepted_x.detach().clone()
                proposal[free_coordinate_ids] += direction_free
                direction_state = {
                    **outer_optimizer_receipt(),
                    "metric_subproblem": subproblem,
                }
            projected, projection = project(proposal)
            if (
                cfg.outer_method == "bfgs"
                and direction_state["metric_subproblem"]["stationary"]
            ):
                stop_reason = "metric subproblem stationary before convergence contract"
                break
            if cfg.outer_method == "bfgs":
                assert projection["projection_coordinate_rms"] <= 1e-10
            direction = projected - accepted_x
            direction[fixed_coordinate] = 0
            directional_derivative = float(torch.dot(gradient, direction))
            if not directional_derivative < 0:
                stop_reason = "no projected descent direction"
                break

            accepted_snapshot = {
                "x": accepted_x.detach().clone(),
                "u": accepted_u.detach().clone(),
                "warm": {
                    key: value.detach().clone()
                    for key, value in physics.runtime.warm_adjoints.items()
                },
                "forward": copy.deepcopy(physics.runtime.last_forward),
                "adjoint": copy.deepcopy(physics.runtime.last_adjoint),
            }
            fraction = 1.0
            rejected: list[dict[str, Any]] = []
            proposal_accepted = False
            previous_objective = objectives[-1]
            previous_x = accepted_x.detach().clone()
            previous_gradient = gradient.detach().clone()
            trial_budget_stop = None
            for trial_index in range(cfg.max_line_search_trials):
                if evaluation_count >= cfg.max_forward_evaluations:
                    trial_budget_stop = "forward-evaluation budget"
                    break
                if time.perf_counter() - started >= cfg.wall_budget_seconds:
                    trial_budget_stop = "wall-time budget"
                    break
                candidate = accepted_x + fraction * direction
                with torch.no_grad():
                    shared.coefficients.copy_(candidate)
                    shared.coefficients[fixed_coordinate] = target_coordinate
                trial_started = time.perf_counter()
                trial: dict[str, Any] = {
                    "update": len(trace),
                    "trial": trial_index,
                    "evaluation_count": evaluation_count + 1,
                    "fraction": fraction,
                    "outer_method": cfg.outer_method,
                    "spectral_step": (
                        spectral_step if cfg.outer_method == "spg" else None
                    ),
                    "direction_state": direction_state,
                }
                trial_u = None
                trial_objective = None
                trial_terms = None
                try:
                    trial_u, trial_objective, trial_terms, trial_metrics, _ = evaluate(
                        accepted_u, backward=False
                    )
                    forward_seconds_total += float(
                        physics.runtime.last_forward["seconds"]
                    )
                    value = float(trial_objective.detach())
                    armijo_rhs = previous_objective + (
                        cfg.armijo_coefficient * fraction * directional_derivative
                    )
                    trial.update(
                        {
                            "objective": value,
                            "armijo_rhs": armijo_rhs,
                            "metrics": trial_metrics,
                            "forward": copy.deepcopy(physics.runtime.last_forward),
                            "seconds": time.perf_counter() - trial_started,
                        }
                    )
                    if not hard_shape_valid(trial_metrics):
                        trial["accepted"] = False
                        trial["reason"] = "hard shape gate"
                    elif value > armijo_rhs:
                        trial["accepted"] = False
                        trial["reason"] = "Armijo decrease"
                    else:
                        trial_objective.backward()
                        adjoint_count += 1
                        adjoint_seconds_total += float(
                            physics.runtime.last_adjoint["seconds"]
                        )
                        assert shared.coefficients.grad is not None
                        candidate_gradient = shared.coefficients.grad.detach().clone()
                        candidate_gradient[fixed_coordinate] = 0
                        assert torch.isfinite(candidate_gradient).all()
                        trial["accepted"] = True
                        trial["reason"] = "Armijo and hard shape gates passed"
                        trial["adjoint"] = copy.deepcopy(physics.runtime.last_adjoint)
                        trials.append(trial)
                        accepted_steps += 1
                        accepted_x = shared.coefficients.detach().clone()
                        accepted_u = trial_u.detach().clone()
                        gradient = candidate_gradient
                        objectives.append(value)
                        metrics = trial_metrics
                        terms = trial_terms
                        proposal_accepted = True
                        break
                    trials.append(trial)
                except (
                    ForwardConvergenceError,
                    FloatingPointError,
                ) as error:
                    trial.update(
                        {
                            "accepted": False,
                            "reason": "forward or adjoint failure",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "forward": copy.deepcopy(physics.runtime.last_forward),
                            "seconds": time.perf_counter() - trial_started,
                        }
                    )
                    trials.append(trial)
                if not trials[-1]["accepted"]:
                    rejected.append(copy.deepcopy(trials[-1]))
                    trial_u = None
                    trial_objective = None
                    trial_terms = None
                    restore_runtime(
                        accepted_snapshot["x"],
                        accepted_snapshot["u"],
                        accepted_snapshot["warm"],
                        accepted_snapshot["forward"],
                        accepted_snapshot["adjoint"],
                    )
                    accepted_x = accepted_snapshot["x"].detach().clone()
                    accepted_u = accepted_snapshot["u"].detach().clone()
                    fraction *= cfg.backtrack_factor
            write_json(output / "trials.json", trials)
            if not proposal_accepted:
                stop_reason = trial_budget_stop or "line search exhausted"
                break

            step_vector = accepted_x - previous_x
            gradient_change = gradient - previous_gradient
            curvature_update: dict[str, Any] | None = None
            if cfg.outer_method == "spg":
                curvature = float(torch.dot(step_vector, gradient_change))
                if curvature > 1e-16:
                    spectral_step = float(
                        torch.dot(step_vector, step_vector) / curvature
                    )
                else:
                    spectral_step *= cfg.backtrack_factor
                spectral_step = min(
                    cfg.maximum_spectral_step,
                    max(cfg.minimum_spectral_step, spectral_step),
                )
            else:
                step_free = step_vector[free_coordinate_ids]
                change_free = gradient_change[free_coordinate_ids]
                curvature = float(torch.dot(step_free, change_free))
                step_norm = float(torch.linalg.vector_norm(step_free))
                change_norm = float(torch.linalg.vector_norm(change_free))
                curvature_threshold = (
                    cfg.bfgs_curvature_relative_tolerance * step_norm * change_norm
                )
                applied = curvature > curvature_threshold
                if applied:
                    rho = 1.0 / curvature
                    identity = torch.eye(
                        len(free_coordinate_ids),
                        dtype=bfgs_inverse.dtype,
                        device=bfgs_inverse.device,
                    )
                    left = identity - rho * torch.outer(step_free, change_free)
                    bfgs_inverse = left @ bfgs_inverse @ left.T + rho * torch.outer(
                        step_free, step_free
                    )
                    bfgs_inverse = 0.5 * (bfgs_inverse + bfgs_inverse.T)
                    assert torch.isfinite(bfgs_inverse).all()
                    assert float(torch.linalg.eigvalsh(bfgs_inverse).min()) > 0
                curvature_update = {
                    "applied": applied,
                    "sTy": curvature,
                    "threshold": curvature_threshold,
                    "step_norm": step_norm,
                    "gradient_change_norm": change_norm,
                    "policy": (
                        "retain prior inverse Hessian when positive-curvature "
                        "threshold is not met"
                    ),
                }
            convergence = convergence_values(accepted_x, gradient, objectives)
            qualifies = bool(
                accepted_steps >= cfg.minimum_accepted_steps
                and convergence["stationarity_met"]
                and convergence["objective_stabilized"]
            )
            qualifying_consecutive = qualifying_consecutive + 1 if qualifies else 0
            optimizer_converged = qualifying_consecutive >= cfg.convergence_consecutive
            budget_met = neutral_budget_met(metrics)
            contact = contact_receipt()
            contact_validated = bool(
                contact["enabled"] and contact["contact_numerically_valid"]
            )
            if contact_config is not None:
                assert contact_validated, contact
            oral = oral_summary(
                audit_neutral_oral_geometry(
                    prepared, physics.points + accepted_u.cpu().numpy()
                )
            )
            assert oral["neutral_invariants_ok"], oral
            row = {
                "update": len(trace),
                "accepted_steps": accepted_steps,
                "evaluation_count": evaluation_count,
                "elapsed_seconds": time.perf_counter() - started,
                "objective": objectives[-1],
                "terms": {name: float(value.detach()) for name, value in terms.items()},
                **{
                    key: convergence[key]
                    for key in (
                        "projected_gradient_inf",
                        "projected_gradient_l2",
                        "raw_gradient_inf",
                    )
                },
                "convergence": convergence,
                "qualifying_consecutive": qualifying_consecutive,
                "optimizer_converged": optimizer_converged,
                "neutral_budget_met": budget_met,
                "contact": contact,
                "contact_validated": contact_validated,
                "preparation_converged": (
                    optimizer_converged and budget_met and contact_validated
                ),
                "preparation_complete": (
                    optimizer_converged and budget_met and contact_validated
                ),
                "acceptance": {
                    "kind": "projected_armijo",
                    "accepted": True,
                    "trials": len(rejected) + 1,
                    "rejected_trials": rejected,
                    "fraction": trials[-1]["fraction"],
                    "spectral_step_used": trials[-1]["spectral_step"],
                    "outer_method": cfg.outer_method,
                    "direction_state": direction_state,
                    "directional_derivative": directional_derivative,
                    "armijo_rhs": trials[-1]["armijo_rhs"],
                    "projection": projection,
                },
                "parameter_step_inf": float(step_vector.abs().max()),
                "spectral_step": (spectral_step if cfg.outer_method == "spg" else None),
                "outer_optimizer": outer_optimizer_receipt(curvature_update),
                "shared": accepted_x.cpu().tolist(),
                "material": material_receipt(),
                "metrics": metrics,
                "oral_diagnostic": oral,
                "forward": copy.deepcopy(physics.runtime.last_forward),
                "adjoint": copy.deepcopy(physics.runtime.last_adjoint),
                "solve_counts": {
                    "forward": evaluation_count,
                    "adjoint": adjoint_count,
                },
                "solve_seconds": {
                    "forward": forward_seconds_total,
                    "adjoint": adjoint_seconds_total,
                },
            }
            trace.append(row)
            write_json(output / "trace.json", trace)
            state = checkpoint(row, accepted_u)
            atomic_torch_save(state, output / "terminal.pt")
            if budget_met and row["objective"] < best_objective:
                best_objective = row["objective"]
                atomic_torch_save(state, output / "best-admissible.pt")
            if accepted_steps % cfg.checkpoint_interval == 0:
                atomic_torch_save(state, output / f"checkpoint-{accepted_steps:04d}.pt")
            cherries.set_step(accepted_steps)
            logged_metrics = {
                "neutral/objective": row["objective"],
                "neutral/surface_loss": row["terms"]["surface_loss"],
                "neutral/muscle_loss": row["terms"]["muscle_loss"],
                "neutral/projected_gradient_inf": row["projected_gradient_inf"],
                "neutral/surface_motion_rms_mm": metrics["surface_motion_rms_mm"],
                "neutral/muscle_centroid_motion_rms_mm": metrics[
                    "muscle_centroid_motion_rms_mm"
                ],
                "neutral/detF_min": metrics["detF_min"],
            }
            if row["convergence"]["objective_relative_range"] is not None:
                logged_metrics["neutral/objective_relative_range"] = row["convergence"][
                    "objective_relative_range"
                ]
            cherries.log_metrics(logged_metrics)
            LOG.info(
                "accepted %d objective %.8g projected-gradient-inf %.4g",
                accepted_steps,
                row["objective"],
                row["projected_gradient_inf"],
            )
            restore_runtime(
                accepted_x,
                accepted_u,
                physics.runtime.warm_adjoints,
                row["forward"],
                row["adjoint"],
            )
            if optimizer_converged:
                stop_reason = (
                    "optimizer converged within neutral budgets"
                    if budget_met
                    else "optimizer converged outside neutral budgets"
                )
                if row["preparation_complete"]:
                    atomic_torch_save(state, output / "converged.pt")
                break

        terminal = trace[-1]
        optimizer_converged = bool(terminal["optimizer_converged"])
        budget_met = bool(terminal["neutral_budget_met"])
        contact_validated = bool(terminal["contact_validated"])
        numerical_preparation_converged = optimizer_converged and budget_met
        preparation_complete = numerical_preparation_converged and contact_validated
        status = (
            "smoke_completed"
            if cfg.mode == "smoke"
            else (
                "converged"
                if preparation_complete
                else (
                    "stationary_outside_neutral_budget"
                    if optimizer_converged and not budget_met
                    else (
                        "contact_not_validated"
                        if numerical_preparation_converged
                        else "not_converged"
                    )
                )
            )
        )
        summary = {
            "schema": "joint-neutral-convergence-summary-v1",
            "success": preparation_complete,
            "status": status,
            "mode": cfg.mode,
            "optimizer_converged": optimizer_converged,
            "neutral_budget_met": budget_met,
            "numerical_preparation_converged": numerical_preparation_converged,
            "preparation_converged": preparation_complete,
            "contact_validated": contact_validated,
            "preparation_complete": preparation_complete,
            "stop_reason": stop_reason,
            "accepted_steps": accepted_steps,
            "accepted_evaluations": len(trace),
            "forward_evaluations": evaluation_count,
            "adjoint_evaluations": adjoint_count,
            "elapsed_seconds": time.perf_counter() - started,
            "solve_seconds": {
                "forward": forward_seconds_total,
                "adjoint": adjoint_seconds_total,
            },
            "skin_prestress_fraction": cfg.skin_prestress_fraction,
            "skin_target_n_per_m": target_resultant,
            "first": trace[0],
            "terminal": terminal,
            "best_admissible_objective": (
                best_objective if math.isfinite(best_objective) else None
            ),
            "oral_claim": "diagnostic only; no anatomical admissibility claim",
        }
        write_json(output / "summary.json", summary)
        cherries.log_output(output)
        if cfg.mode == "smoke":
            COMPLETED = True
            return
        assert preparation_complete, summary
        COMPLETED = True
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "schema": "joint-neutral-convergence-failure-v1",
                "error_type": type(error).__name__,
                "error": str(error),
                "accepted_steps": accepted_steps,
                "forward_evaluations": evaluation_count,
                "adjoint_evaluations": adjoint_count,
                "elapsed_seconds": time.perf_counter() - started,
                "forward": physics.runtime.last_forward,
                "adjoint": physics.runtime.last_adjoint,
            },
        )
        raise


if __name__ == "__main__":
    # Comet's documented config keys avoid a minutes-long scan of this very large
    # dirty checkout. Cherries still records hashes and our archived source tree.
    os.environ.setdefault("COMET_AUTO_LOG_GIT_METADATA", "false")
    os.environ.setdefault("COMET_AUTO_LOG_GIT_PATCH", "false")
    cherries.main(main, profile=ProfileJoint)
    if not COMPLETED:
        raise SystemExit(1)
