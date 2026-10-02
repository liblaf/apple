"""CPU validation for complete-source-bone contact and pose pullbacks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_contact import OwnedContact
from joint_full_skull_contact import (
    ADMISSION_SCHEMA,
    CONTACT_SCHEMA,
    FullSkullContactAdapter,
    FullSkullGeometry,
    build_full_skull_contact,
    extend_dof_map,
    load_admitted_initialization,
    load_full_skull_geometry,
    validate_full_skull_admission,
)

from liblaf import cherries
from liblaf.apple.forward.dof_map import DofMap


class Config(cherries.BaseConfig):
    geometry: Path = GROUP / "data/full-skull-initialization-audit-001/geometry.npz"
    geometry_audit: Path = (
        GROUP / "data/full-skull-initialization-audit-001/summary.json"
    )
    output_dir: Path = GROUP / "data/full-skull-contact-adapter-validation-002"
    initialization_admission: Path = (
        GROUP / "data/full-skull-initialization-candidate-002/admission.json"
    )


def synthetic_geometry() -> FullSkullGeometry:
    fem = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    cranium = np.asarray(
        [[0.15, 0.18, 0.035], [0.62, 0.11, 0.055], [0.08, 0.67, 0.047]],
        dtype=np.float64,
    )
    mandible = np.asarray(
        [[0.22, 0.16, -0.038], [0.73, 0.21, -0.052], [0.31, 0.64, -0.061]],
        dtype=np.float64,
    )
    audit = {
        "schema": "joint-full-skull-initialization-audit-v1",
        "input_arrays_sha256": "synthetic-arrays",
        "input_manifest_sha256": "synthetic-manifest",
        "units": "metres",
        "frame": "unchanged registered source and FEM world frame",
    }
    triangle = np.asarray([[0, 1, 2]], dtype=np.int32)
    return FullSkullGeometry(
        fem_reference_points_m=fem,
        soft_global_ids=np.asarray([0, 1, 2]),
        soft_faces=triangle,
        fixed_global_ids=np.asarray([3]),
        mandible_pivot_m=np.zeros(3, dtype=np.float64),
        cranium_points_m=cranium,
        cranium_faces=triangle,
        cranium_source_vertex_ids=np.arange(3),
        cranium_source_triangle_ids=np.arange(1),
        mandible_points_m=mandible,
        mandible_faces=triangle,
        mandible_source_vertex_ids=np.arange(3),
        mandible_source_triangle_ids=np.arange(1),
        geometry_path=Path("synthetic-geometry.npz"),
        geometry_sha256="synthetic-geometry",
        audit_path=Path("synthetic-summary.json"),
        audit_sha256="synthetic-audit",
        audit=audit,
    )


def contact_config() -> dict:
    return {
        "schema": CONTACT_SCHEMA,
        "enabled": True,
        "surface_selection": "pure-soft-vs-complete-source-bones",
        "attachment_policy": "no-source-triangle-exclusions",
        "friction": "frictionless",
        "dhat_m": 0.1,
        "stiffness_mpa": 0.01,
    }


def energy_and_gradient(
    contact: OwnedContact, u: torch.Tensor
) -> tuple[float, torch.Tensor]:
    state = contact.state_at(u)
    gradient = torch.zeros_like(u)
    contact.grad(state, u, gradient)
    return float(contact.fun(state, u)), gradient


def derivative_checks(adapter: FullSkullContactAdapter) -> dict:
    contact = adapter.collision
    u = torch.zeros((adapter.geometry.full_node_count, 3), dtype=torch.float64)
    direction = torch.as_tensor(
        [
            [0.1, -0.2, 0.3],
            [-0.2, 0.1, 0.2],
            [0.1, 0.1, -0.1],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.05, 0.02, -0.03],
            [-0.03, 0.04, -0.02],
            [-0.02, -0.06, 0.05],
            [0.02, -0.01, 0.03],
        ],
        dtype=torch.float64,
    )
    direction /= torch.linalg.vector_norm(direction)
    state = contact.state_at(u)
    energy = float(contact.fun(state, u))
    gradient = torch.zeros_like(u)
    contact.grad(state, u, gradient)
    energy_exact = float(torch.sum(gradient * direction))
    hp = torch.zeros_like(u)
    contact.hess_prod(state, u, direction, hp)
    rows = []
    for step in (3.0e-4, 1.0e-4):
        plus = float(
            contact.fun(contact.state_at(u + step * direction), u + step * direction)
        )
        minus = float(
            contact.fun(contact.state_at(u - step * direction), u - step * direction)
        )
        energy_fd = (plus - minus) / (2 * step)
        energy_error = abs(energy_fd - energy_exact) / max(
            abs(energy_fd), abs(energy_exact), 1.0e-12
        )
        _, gradient_plus = energy_and_gradient(contact, u + step * direction)
        _, gradient_minus = energy_and_gradient(contact, u - step * direction)
        hp_fd = (gradient_plus - gradient_minus) / (2 * step)
        hp_error = float(torch.linalg.vector_norm(hp - hp_fd)) / max(
            float(torch.linalg.vector_norm(hp)),
            float(torch.linalg.vector_norm(hp_fd)),
            1.0e-12,
        )
        assert energy_error < 2.0e-3, (energy_fd, energy_exact, energy_error)
        assert hp_error < 2.0e-5, hp_error
        rows.append(
            {
                "step": step,
                "energy_directional_finite_difference": energy_fd,
                "energy_directional_exact": energy_exact,
                "energy_directional_relative_error": energy_error,
                "hessian_product_relative_error": hp_error,
            }
        )
    force_balance = torch.linalg.vector_norm(gradient.sum(dim=0))
    force_scale = torch.linalg.vector_norm(gradient)
    force_balance_relative = float(force_balance / force_scale)
    assert force_balance_relative < 1.0e-12
    return {
        "active_contact_count": len(state.collisions),
        "energy": energy,
        "steps": rows,
        "maximum_energy_directional_relative_error": max(
            row["energy_directional_relative_error"] for row in rows
        ),
        "maximum_hessian_product_relative_error": max(
            row["hessian_product_relative_error"] for row in rows
        ),
        "hessian_product": "exact IPC potential Hessian; distinct from the inherited Gauss-Newton hess_quad used by PNCG",
        "force_balance_relative": force_balance_relative,
    }


def pose_pullback_check(adapter: FullSkullContactAdapter) -> dict:
    geometry = adapter.geometry
    contact = adapter.collision
    original_points = torch.as_tensor(geometry.fem_reference_points_m)
    original_jaw = torch.as_tensor([3], dtype=torch.long)
    pose = torch.as_tensor(
        [0.002, -0.001, 0.0015, 0.0003, -0.0002, 0.0004],
        dtype=torch.float64,
    )
    u = adapter.full_boundary_displacement(original_points, original_jaw, pose)
    p = torch.zeros_like(u)
    p[:3] = torch.as_tensor(
        [[0.2, -0.1, 0.3], [-0.1, 0.2, 0.1], [0.05, -0.1, -0.2]],
        dtype=torch.float64,
    )
    state = contact.state_at(u)
    hp = torch.zeros_like(u)
    contact.hess_prod(state, u, p, hp)
    fixed_ids = torch.as_tensor(
        [
            *geometry.fixed_global_ids,
            *geometry.cranium_global_ids,
            *geometry.mandible_global_ids,
        ],
        dtype=torch.long,
    )
    pose_variable = pose.detach().clone().requires_grad_()
    boundary = adapter.full_boundary_displacement(
        original_points, original_jaw, pose_variable
    )
    predicted = torch.autograd.grad(
        (boundary[fixed_ids] * hp[fixed_ids]).sum(), pose_variable
    )[0]

    direction = torch.as_tensor(
        [0.4, -0.2, 0.1, 0.05, -0.03, 0.02], dtype=torch.float64
    )
    direction /= torch.linalg.vector_norm(direction)
    step = 1.0e-6

    def scalar(at_pose: torch.Tensor) -> float:
        at_u = adapter.full_boundary_displacement(
            original_points, original_jaw, at_pose
        )
        _, gradient = energy_and_gradient(contact, at_u)
        return float(torch.sum(gradient * p))

    finite_difference = (
        scalar(pose + step * direction) - scalar(pose - step * direction)
    ) / (2 * step)
    exact = float(torch.dot(predicted, direction))
    relative_error = abs(finite_difference - exact) / max(
        abs(finite_difference), abs(exact), 1.0e-12
    )
    assert relative_error < 3.0e-4, (finite_difference, exact, relative_error)
    mandible_pullback_norm = float(
        torch.linalg.vector_norm(hp[torch.as_tensor(geometry.mandible_global_ids)])
    )
    assert mandible_pullback_norm > 0
    return {
        "directional_finite_difference": finite_difference,
        "directional_exact_fixed_hvp_pullback": exact,
        "relative_error": relative_error,
        "appended_mandible_hvp_norm": mandible_pullback_norm,
    }


def ccd_and_filter_checks(adapter: FullSkullContactAdapter) -> dict:
    geometry = adapter.geometry
    contact = adapter.collision
    soft = 0
    cranium = len(geometry.soft_global_ids)
    mandible = cranium + geometry.cranium_node_count
    assert contact.collision_mesh.can_collide(soft, cranium)
    assert contact.collision_mesh.can_collide(soft, mandible)
    assert not contact.collision_mesh.can_collide(soft, soft + 1)
    assert not contact.collision_mesh.can_collide(cranium, mandible)

    u0 = torch.zeros((geometry.full_node_count, 3), dtype=torch.float64)
    delta = torch.zeros_like(u0)
    delta[torch.as_tensor(geometry.mandible_global_ids), 2] = 0.08
    state = contact.state_at(u0)
    fraction = float(contact.max_step_size(state, u0, delta))
    assert 0 < fraction < 1, fraction
    return {
        "soft_cranium_enabled": True,
        "soft_mandible_enabled": True,
        "soft_soft_disabled": True,
        "bone_bone_and_self_disabled": True,
        "prescribed_mandible_crossing_ccd_fraction": fraction,
    }


def dof_checks(adapter: FullSkullContactAdapter) -> dict:
    geometry = adapter.geometry
    original = DofMap(
        n_points=geometry.fem_node_count,
        dim=3,
        fixed_indices=torch.as_tensor([9, 10, 11]),
        fixed_values=torch.zeros(3, dtype=torch.float64),
        free_indices=torch.arange(9),
    )
    extended = extend_dof_map(original, geometry)
    assert extended.n_points == geometry.full_node_count
    assert torch.equal(extended.free_indices, original.free_indices)
    assert torch.equal(extended.fixed_indices[:3], original.fixed_indices)
    assert len(extended.fixed_indices) == 3 + 3 * (
        geometry.cranium_node_count + geometry.mandible_node_count
    )
    return {
        "original_free_indices_preserved": True,
        "appended_coordinates_all_fixed": True,
    }


def admission_checks(geometry: FullSkullGeometry, path: Path) -> dict:
    admission = json.loads(path.read_text())
    assert admission["schema"] == ADMISSION_SCHEMA
    displacement = load_admitted_initialization(admission, geometry)
    assert displacement.shape == (geometry.fem_node_count, 3)
    assert not np.any(displacement[geometry.fixed_global_ids])
    malformed = copy.deepcopy(admission)
    malformed["complete_source_triangles_retained"] = "true"
    rejected = False
    try:
        validate_full_skull_admission(malformed, geometry)
    except ValueError:
        rejected = True
    assert rejected
    rejected = False
    malformed = copy.deepcopy(admission)
    malformed["excluded_source_triangles"] = 1
    try:
        validate_full_skull_admission(malformed, geometry)
    except ValueError:
        rejected = True
    assert rejected
    malformed = copy.deepcopy(admission)
    malformed["initialization_displacement_sha256"] = "0" * 64
    rejected = False
    try:
        validate_full_skull_admission(malformed, geometry)
    except ValueError:
        rejected = True
    assert rejected
    malformed = copy.deepcopy(admission)
    malformed["final_launch_ready"] = True
    rejected = False
    try:
        validate_full_skull_admission(malformed, geometry)
    except ValueError:
        rejected = True
    assert rejected
    return {
        "strict_boolean_schema_rejection": True,
        "source_triangle_exclusion_rejection": True,
        "initialization_displacement_hash_rejection": True,
        "final_launch_claim_rejection": True,
        "admitted_initialization_shape": list(displacement.shape),
    }


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    audited = load_full_skull_geometry(cfg.geometry, cfg.geometry_audit)
    assert audited.audit["full_skull_contact_admitted"] is False
    assert audited.audit["initialization_repaired"] is False
    synthetic = synthetic_geometry()
    adapter = build_full_skull_contact(synthetic, contact_config())
    summary = {
        "schema": "joint-full-skull-contact-adapter-validation-v1",
        "success": True,
        "status": "passed_cpu_adapter_and_initialization_binding_validation",
        "scope": "complete-source node append, exact IPC derivatives/HVP/fixed-pose pullback, prescribed-bone CCD, collision filter, and hash-bound soft-bone initialization; bone-bone contact is excluded and no FEM equilibrium or final-launch admission is claimed",
        "audited_geometry": audited.binding_receipt(),
        "audited_geometry_contact_admitted": False,
        "soft_bone_initialization_admitted": True,
        "bone_bone_contact_validated": False,
        "final_launch_ready": False,
        "initialization_admission": {
            "path": str(cfg.initialization_admission.resolve()),
            "sha256": sha256(cfg.initialization_admission),
        },
        "synthetic_contact": adapter.contact_definition,
        "checks": {
            "derivatives": derivative_checks(adapter),
            "pose_pullback": pose_pullback_check(adapter),
            "ccd_and_filter": ccd_and_filter_checks(adapter),
            "dof": dof_checks(adapter),
            "admission": admission_checks(audited, cfg.initialization_admission),
        },
        "implementation_sha256": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(
                Path(__file__).with_name("joint_full_skull_contact.py").resolve()
            ): sha256(
                Path(__file__).with_name("joint_full_skull_contact.py").resolve()
            ),
        },
    }
    write_json(cfg.output_dir / "summary.json", summary)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
