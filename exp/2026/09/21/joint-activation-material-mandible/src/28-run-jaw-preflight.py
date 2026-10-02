"""No-optimization full-face jaw proposal-domain preflight."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
import pydantic_settings as ps
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import (
    PreparedInputs,
    audit_deformed_oral_geometry,
    audit_neutral_oral_geometry,
)
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from joint_fields import SharedFieldParameters, research_informed_material_config
from joint_final_geometry import adapt_final_run_geometry_receipt
from joint_rigid_bone_collision import RigidBoneCollision
from joint_spatial_fields import SpatialSharedFieldParameters, spatial_field_config

from liblaf import cherries

LOG = logging.getLogger(__name__)
COMPLETED = False
GEOMETRY_REPRODUCIBILITY_BUDGET_M = 1e-6
PREVIOUS_DIAGNOSTIC = {
    "path": "data/jaw-preflight-spatial25-diagnostic-002/summary.json",
    "sha256": "120e2c1287376e08fcc6a3a305bddd737fe523bb8dbe5bba7659a6d73282d481",
    "status": "failed_required_zero_or_nonzero_test",
    "zero_maximum_coordinate_difference_m": 1.9021703167654294e-7,
    "legacy_one_degree_status": "soft_bone_boundary_ccd_rejection",
}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    prepared_dir: Path = GROUP / "data/prepared"
    neutral_checkpoint: Path
    contact_spec: Path = GROUP / "data/contact/config.json"
    contact_validation: Path
    newton_synthetic_validation: Path = (
        GROUP / "data/contact-validation-newton/summary.json"
    )
    rigid_bone_ccd_validation: Path = (
        GROUP / "data/rigid-bone-ccd-validation-003/summary.json"
    )
    admission_mode: Literal["diagnostic_only", "final_launch_ready"] = (
        "final_launch_ready"
    )
    output_dir: Path = cherries.output("jaw-preflight", mkdir=True)
    forward_rtol: float = 1e-6
    forward_atol: float = 1e-12
    adjoint_rtol: float = 1e-7
    max_forward_steps: int = 10000
    forward_method: str = "newton_cg"
    newton_linear_rtol: float = 1e-3
    newton_max_steps: int = 12
    geometry_reproducibility_budget_m: float = GEOMETRY_REPRODUCIBILITY_BUDGET_M


def _load_pilot() -> ModuleType:
    path = Path(__file__).with_name("30-joint-pilot.py")
    spec = importlib.util.spec_from_file_location("joint_pilot_jaw_preflight", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tensor_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().to(torch.float64).numpy())
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _geometry_reproducibility_metrics(
    displacement_m: np.ndarray,
    reference_m: np.ndarray,
    surface_node_ids: np.ndarray,
) -> dict[str, object]:
    displacement_m = np.asarray(displacement_m, dtype=np.float64)
    reference_m = np.asarray(reference_m, dtype=np.float64)
    surface_node_ids = np.asarray(surface_node_ids, dtype=np.int64)
    assert displacement_m.shape == reference_m.shape
    assert displacement_m.ndim == 2
    assert displacement_m.shape[1] == 3
    assert surface_node_ids.ndim == 1
    assert surface_node_ids.size > 0
    assert np.all((surface_node_ids >= 0) & (surface_node_ids < len(displacement_m)))
    assert np.unique(surface_node_ids).size == surface_node_ids.size
    difference_norm_m = np.linalg.norm(displacement_m - reference_m, axis=1)
    assert np.isfinite(difference_norm_m).all()
    return {
        "reference": "frozen_neutral_seed_displacement",
        "candidate": "resolved_zero_pose_displacement",
        "coordinate_frame": "world_m",
        "maximum_euclidean_nodal_displacement_difference_m": float(
            difference_norm_m.max()
        ),
        "surface_node_rms_euclidean_displacement_difference_m": float(
            np.sqrt(np.mean(np.square(difference_norm_m[surface_node_ids])))
        ),
        "volume_node_rms_euclidean_displacement_difference_m": float(
            np.sqrt(np.mean(np.square(difference_norm_m)))
        ),
        "surface_node_count": int(surface_node_ids.size),
        "volume_node_count": int(displacement_m.shape[0]),
    }


def _shape_checks(metrics: dict, *, neutral: bool) -> dict[str, bool]:
    checks = {
        "no_inverted_tetrahedra": metrics["inverted_tetrahedra"] == 0,
        "detF_min": metrics["detF_min"] >= 0.25,
        "detF_max": metrics["detF_max"] <= 2.0,
        "skin_area_ratio_min": metrics["skin_area_ratio_min"] >= 0.25,
    }
    if neutral:
        checks["surface_motion_rms"] = metrics["surface_motion_rms_mm"] <= 0.25
        checks["muscle_centroid_motion_rms"] = (
            metrics["muscle_centroid_motion_rms_mm"] <= 0.5
        )
    return checks


def _failure_kind(receipt: dict | None) -> str:
    if receipt is not None and receipt.get("failure") == (
        "Dirichlet boundary proposal crosses a contact surface"
    ):
        return "soft_bone_boundary_ccd_rejection"
    return "nonlinear_equilibrium_failure"


def _validate_diagnostic_checkpoint(
    initial: dict,
    *,
    pilot: ModuleType,
    expected_solver: dict,
    input_arrays_sha256: str,
    input_manifest_sha256: str,
    contact_spec_sha256: str,
    contact_validation_sha256: str,
) -> None:
    """Admit a frozen neutral state for diagnosis without a convergence claim."""
    assert initial["schema"] == "joint-inverse-checkpoint-v1"
    assert initial["stage"] == "neutral"
    protocol = initial["protocol"]
    assert protocol["schema"] == "joint-neutral-convergence-protocol-v1"
    assert protocol["stage"] == "neutral"
    assert protocol["input_arrays_sha256"] == input_arrays_sha256
    assert protocol["input_manifest_sha256"] == input_manifest_sha256
    assert protocol["contact"]["enabled"] is True
    assert protocol["contact"]["spec_sha256"] == contact_spec_sha256
    assert protocol["contact"]["validation_sha256"] == contact_validation_sha256
    assert initial["metrics"]["contact"]["enabled"] is True
    assert initial["metrics"]["contact"]["contact_numerically_valid"] is True
    pilot.validate_forward_solver_contract(protocol["forward_solver"], expected_solver)


def main(cfg: Config) -> None:  # noqa: C901, PLR0912, PLR0915
    global COMPLETED  # noqa: PLW0603
    assert cfg.forward_method == "newton_cg"
    assert cfg.newton_linear_rtol == 1e-3
    assert cfg.newton_max_steps == 12
    assert cfg.forward_rtol == 1e-6
    assert cfg.forward_atol == 1e-12
    assert cfg.adjoint_rtol == 1e-7
    assert cfg.max_forward_steps == 10000
    assert cfg.geometry_reproducibility_budget_m == GEOMETRY_REPRODUCIBILITY_BUDGET_M
    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=False)
    archive_sources(output)
    pilot = _load_pilot()
    expected_solver = pilot.forward_solver_contract(cfg)
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz",
        cfg.prepared_dir / "manifest.json",
        verify_sources=True,
    )
    input_arrays_sha256 = sha256(cfg.prepared_dir / "inputs.npz")
    input_manifest_sha256 = sha256(cfg.prepared_dir / "manifest.json")
    contact_spec_sha256 = sha256(cfg.contact_spec)
    contact_validation_sha256 = sha256(cfg.contact_validation)
    rigid_validation_sha256 = sha256(cfg.rigid_bone_ccd_validation)
    contact_config = json.loads(cfg.contact_spec.read_text())
    assert contact_config["schema"] == "joint-bone-contact-v1"
    assert contact_config["enabled"] is True
    contact_validation = json.loads(cfg.contact_validation.read_text())
    assert contact_validation["schema"] == "joint-contact-validation-v1"
    assert contact_validation["success"] is True
    assert contact_validation["contact_spec_sha256"] == contact_spec_sha256
    newton_validation = json.loads(cfg.newton_synthetic_validation.read_text())
    assert newton_validation["schema"] == "joint-contact-validation-v1"
    assert newton_validation["success"] is True
    assert newton_validation["contact_spec_sha256"] == contact_spec_sha256
    pilot.validate_forward_solver_contract(
        newton_validation["forward_solver"], expected_solver
    )
    rigid_validation = json.loads(cfg.rigid_bone_ccd_validation.read_text())
    pilot.validate_rigid_bone_ccd_validation(
        rigid_validation,
        input_arrays_sha256=input_arrays_sha256,
        input_manifest_sha256=input_manifest_sha256,
    )
    initial = torch.load(cfg.neutral_checkpoint, map_location="cpu", weights_only=False)
    if cfg.admission_mode == "final_launch_ready":
        pilot.validate_neutral_checkpoint(
            initial,
            contact_spec_sha256=contact_spec_sha256,
            contact_validation_sha256=contact_validation_sha256,
            expected_solver=expected_solver,
        )
    else:
        _validate_diagnostic_checkpoint(
            initial,
            pilot=pilot,
            expected_solver=expected_solver,
            input_arrays_sha256=input_arrays_sha256,
            input_manifest_sha256=input_manifest_sha256,
            contact_spec_sha256=contact_spec_sha256,
            contact_validation_sha256=contact_validation_sha256,
        )
    shared_basis, shared_field, spatial_basis, _ = pilot.validate_shared_basis_contract(
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

    incoming_spec = initial["materials"]
    if shared_basis == "spatial80":
        assert incoming_spec == spatial_field_config()
        constitutive_spec = research_informed_material_config()
    else:
        assert shared_basis == "constant20"
        constitutive_spec = incoming_spec

    configure_cuda()
    physics = pilot.physics_from_spec(cfg, prepared, constitutive_spec, contact_config)
    if shared_basis == "spatial80":
        assert spatial_basis is not None
        shared = SpatialSharedFieldParameters(
            Path(spatial_basis["basis_path"]),
            Path(spatial_basis["audit_summary_path"]),
            len(physics.tets),
        )
        assert shared.config == incoming_spec
        assert shared.basis_receipt() == spatial_basis
    else:
        assert spatial_basis is None
        shared = SharedFieldParameters(constitutive_spec)
    assert shared_field["coefficient_count"] == shared.coefficients.numel()
    with torch.no_grad():
        shared.coefficients.copy_(initial["shared_coefficients"])

    guard = RigidBoneCollision(physics.mesh, prepared.arrays["mandible_pivot_m"])
    assert guard.mapping == rigid_validation["mapping"]
    assert guard.reference_receipt == rigid_validation["reference"]
    seed = initial["primal"]["neutral"].to(device="cuda", dtype=torch.float64)
    seed_sha256 = _tensor_sha256(seed)
    seed_np = seed.detach().cpu().numpy()
    scale = np.asarray([*np.deg2rad([10.0] * 3), *([0.005] * 3)])
    cases = [
        {
            "label": "zero",
            "role": "required_zero_geometry_reproducibility",
            "normalized_pose": np.zeros(6),
        }
    ]
    for coordinate in range(6):
        for sign in (-1, 1):
            normalized = np.zeros(6)
            normalized[coordinate] = sign
            cases.append(
                {
                    "label": f"coordinate_{coordinate}_{sign:+d}",
                    "role": "diagnostic_broad_axis_endpoint",
                    "normalized_pose": normalized,
                }
            )
    legacy_pose = np.asarray(
        prepared.manifest["joint_pilot_contract"]["initial_pose_rad_m"],
        dtype=np.float64,
    )
    cases.append(
        {
            "label": "legacy_one_degree_center",
            "role": "diagnostic_legacy_one_degree_center",
            "normalized_pose": legacy_pose / scale,
        }
    )
    small_world_x_pose = np.zeros(6)
    small_world_x_pose[0] = np.deg2rad(0.01)
    cases.append(
        {
            "label": "world_x_positive_0p01_degree",
            "role": "required_nonzero_world_x_0p01_degree",
            "normalized_pose": small_world_x_pose / scale,
        }
    )
    assert len(cases) == 15

    rows = []
    started = time.perf_counter()
    for case in cases:
        normalized = case["normalized_pose"]
        pose_np = scale * normalized
        trial_seed = seed.detach().clone()
        rigid_receipt = guard.from_displacement(
            trial_seed.detach().cpu().numpy(), pose_np
        )
        row = {
            "label": case["label"],
            "role": case["role"],
            "normalized_pose": normalized.tolist(),
            "pose_rad_m": pose_np.tolist(),
            "seed_sha256": _tensor_sha256(trial_seed),
            "seed_matches_neutral": _tensor_sha256(trial_seed) == seed_sha256,
            "rigid_bone_ccd": rigid_receipt,
            "forward_attempted": False,
            "numerically_admissible": False,
        }
        if not rigid_receipt["numerically_admissible"]:
            row["status"] = "rigid_bone_boundary_ccd_rejection"
            rows.append(row)
            write_json(output / "probes.json", rows)
            LOG.info(
                "%s rejected by rigid-bone CCD at fraction %.8g",
                case["label"],
                rigid_receipt["collision_free_fraction"],
            )
            continue
        pose = torch.as_tensor(pose_np, device=seed.device, dtype=seed.dtype)
        row["forward_attempted"] = True
        try:
            displacement = physics.solve(
                shared.bulk_stresses_mpa(),
                shared.skin_resultant_n_per_m(),
                shared.skin_stiffness_multiplier(),
                None,
                pose,
                trial_seed,
                key=case["label"],
            )
        except ForwardConvergenceError as error:
            row["status"] = _failure_kind(error.receipt)
            row["forward"] = deepcopy(error.receipt)
            rows.append(row)
            write_json(output / "probes.json", rows)
            LOG.info("%s failed: %s", case["label"], row["status"])
            continue

        forward = deepcopy(physics.runtime.last_forward)
        pilot.validate_forward_receipt(forward, expected_solver)
        metrics = physics.metrics(displacement)
        neutral = case["label"] == "zero"
        shape_checks = _shape_checks(metrics, neutral=neutral)
        deformed = physics.points + displacement.detach().cpu().numpy()
        if neutral:
            raw_geometry = audit_neutral_oral_geometry(prepared, deformed)
        else:
            raw_geometry = audit_deformed_oral_geometry(prepared, deformed, pose_np)
        geometry = adapt_final_run_geometry_receipt(
            raw_geometry,
            neutral=neutral,
            normalized_pose=normalized,
            pose_rad_m=pose_np,
        )
        displacement_np = displacement.detach().cpu().to(torch.float64).numpy()
        reproducibility = (
            _geometry_reproducibility_metrics(
                displacement_np,
                seed_np,
                physics.skin_ids,
            )
            if neutral
            else None
        )
        if reproducibility is not None:
            reproducibility["maximum_euclidean_nodal_difference_budget_m"] = (
                cfg.geometry_reproducibility_budget_m
            )
            reproducibility["passed"] = bool(
                reproducibility["maximum_euclidean_nodal_displacement_difference_m"]
                <= cfg.geometry_reproducibility_budget_m
            )
        row.update(
            {
                "status": "accepted"
                if (
                    all(shape_checks.values())
                    and geometry["numerical_geometry_admissible"]
                )
                else "post_solve_numerical_rejection",
                "forward": forward,
                "metrics": metrics,
                "shape_checks": shape_checks,
                "geometry": geometry,
                "contact": forward["contact"],
                "geometry_reproducibility": reproducibility,
            }
        )
        row["numerically_admissible"] = row["status"] == "accepted"
        if row["numerically_admissible"]:
            state_path = output / f"state-{case['label']}.npz"
            assert displacement_np.shape == physics.points.shape
            assert np.isfinite(displacement_np).all()
            np.savez_compressed(
                state_path,
                schema=np.asarray("joint-jaw-preflight-state-v1"),
                label=np.asarray(case["label"]),
                role=np.asarray(case["role"]),
                normalized_pose=np.asarray(normalized, dtype=np.float64),
                pose_rad_m=np.asarray(pose_np, dtype=np.float64),
                full_displacement_m=displacement_np,
                neutral_checkpoint_sha256=np.asarray(sha256(cfg.neutral_checkpoint)),
                input_arrays_sha256=np.asarray(input_arrays_sha256),
                input_manifest_sha256=np.asarray(input_manifest_sha256),
                seed_sha256=np.asarray(seed_sha256),
            )
            row["state"] = {
                "schema": "joint-jaw-preflight-state-v1",
                "path": str(state_path.resolve()),
                "sha256": sha256(state_path),
                "displacement_shape": list(displacement_np.shape),
                "displacement_dtype": displacement_np.dtype.str,
            }
        rows.append(row)
        write_json(output / "probes.json", rows)
        LOG.info("%s: %s", case["label"], row["status"])

    zero = next(row for row in rows if row["label"] == "zero")
    legacy = next(row for row in rows if row["label"] == "legacy_one_degree_center")
    nonzero = next(
        row for row in rows if row["label"] == "world_x_positive_0p01_degree"
    )
    endpoint_rows = [
        row for row in rows if row["role"] == "diagnostic_broad_axis_endpoint"
    ]
    assert len(endpoint_rows) == 12
    zero_geometry_reproducibility_met = bool(
        zero["numerically_admissible"] and zero["geometry_reproducibility"]["passed"]
    )
    required_nonzero_world_x_0p01_degree_met = bool(nonzero["numerically_admissible"])
    success = (
        zero_geometry_reproducibility_met and required_nonzero_world_x_0p01_degree_met
    )
    final_launch_ready = bool(success and cfg.admission_mode == "final_launch_ready")
    if not success:
        status = "failed_required_geometry_or_nonzero_candidate"
    elif final_launch_ready:
        status = "passed_final_launch_preflight"
    else:
        status = "passed_diagnostic_only"
    summary = {
        "schema": "joint-jaw-preflight-v2",
        "success": success,
        "status": status,
        "admission_mode": cfg.admission_mode,
        "diagnostic_only": cfg.admission_mode == "diagnostic_only",
        "final_launch_ready": final_launch_ready,
        "neutral_checkpoint": str(cfg.neutral_checkpoint.resolve()),
        "neutral_checkpoint_sha256": sha256(cfg.neutral_checkpoint),
        "input_arrays_sha256": input_arrays_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "contact_spec_sha256": contact_spec_sha256,
        "contact_validation_sha256": contact_validation_sha256,
        "rigid_bone_ccd_validation_sha256": rigid_validation_sha256,
        "forward_solver": expected_solver,
        "forward_tolerances": {
            "rtol": cfg.forward_rtol,
            "atol": cfg.forward_atol,
            "adjoint_rtol": cfg.adjoint_rtol,
            "max_steps": cfg.max_forward_steps,
        },
        "shared_basis": shared_basis,
        "shared_field": shared_field,
        "spatial_basis": spatial_basis,
        "seed_sha256": seed_sha256,
        "probe_count": len(rows),
        "broad_axis_endpoint_count": len(endpoint_rows),
        "broad_axis_endpoint_accepted_count": sum(
            row["numerically_admissible"] for row in endpoint_rows
        ),
        "geometry_reproducibility": {
            "budget_m": cfg.geometry_reproducibility_budget_m,
            "budget_status": (
                "model QA choice: 1 micrometre, one two-hundred-fiftieth of "
                "the 0.25 mm neutral surface-motion RMS budget"
            ),
            "comparison": "resolved zero-pose equilibrium versus frozen neutral primal",
            "metrics": zero.get("geometry_reproducibility"),
            "met": zero_geometry_reproducibility_met,
        },
        "zero_geometry_reproducibility_met": zero_geometry_reproducibility_met,
        "required_nonzero_world_x_0p01_degree_met": (
            required_nonzero_world_x_0p01_degree_met
        ),
        "legacy_center_diagnostic_admissible": bool(legacy["numerically_admissible"]),
        "required_labels": ["zero", "world_x_positive_0p01_degree"],
        "diagnostic_labels": [
            *[f"coordinate_{i}_{sign}" for i in range(6) for sign in (-1, 1)],
            "legacy_one_degree_center",
        ],
        "protocol_revision": {
            "reason": (
                "v1 incorrectly reused the 1e-12 force absolute tolerance as a "
                "displacement identity tolerance; v2 declares an independent "
                "geometry reproducibility budget"
            ),
            "previous_diagnostic": PREVIOUS_DIAGNOSTIC,
            "candidate_policy": (
                "the 0.01 degree world-x candidate is predeclared in source; it "
                "does not replace a failed probe adaptively, and the failed one-degree "
                "candidate remains diagnostic evidence"
            ),
        },
        "whole_pose_box_validated": False,
        "anatomical_validation": False,
        "bone_bone_energy_added": False,
        "rotation_arc_checked": False,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "sources": {
            str(path.resolve()): sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("30-joint-pilot.py"),
                Path(__file__).with_name("joint_final_geometry.py"),
                Path(__file__).with_name("joint_rigid_bone_collision.py"),
            )
        },
    }
    write_json(output / "summary.json", summary)
    cherries.log_metrics(
        {
            "jaw_preflight/success": float(success),
            "jaw_preflight/endpoint_accepted": summary[
                "broad_axis_endpoint_accepted_count"
            ],
            "jaw_preflight/zero_geometry_reproducibility_met": float(
                zero_geometry_reproducibility_met
            ),
            "jaw_preflight/required_nonzero_world_x_0p01_degree_met": float(
                required_nonzero_world_x_0p01_degree_met
            ),
        }
    )
    COMPLETED = success


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
    if not COMPLETED:
        raise SystemExit(1)
