"""CPU admission checks for fixed source-eye IPC obstacles.

This validates the collision law independently of the large CUDA FEM solve, then
hash-binds the real frozen material tensors after eye-inclusive construction.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_full_skull_contact import FullSkullGeometry
from joint_rigid_eye_contact import (
    RigidEyeGeometry,
    _build_contact,
    _extend_dof_map,
    build_eye_collision_physics,
)

from liblaf import cherries
from liblaf.apple.forward.dof_map import DofMap


class Config(cherries.BaseConfig):
    eyes_dir: Path = GROUP / "data/rigid-eyes-001"
    output_dir: Path = GROUP / "data/rigid-eye-contact-validation-001"
    run_real_material_identity: bool = True


def synthetic_geometry() -> FullSkullGeometry:
    triangle = np.asarray([[0, 1, 2]], dtype=np.int32)
    fem = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    audit = {
        "input_arrays_sha256": "synthetic",
        "input_manifest_sha256": "synthetic",
        "units": "metres",
        "frame": "unchanged registered source and FEM world frame",
    }
    return FullSkullGeometry(
        fem_reference_points_m=fem,
        soft_global_ids=np.asarray([0, 1, 2]),
        soft_faces=triangle,
        fixed_global_ids=np.asarray([3]),
        mandible_pivot_m=np.zeros(3),
        cranium_points_m=np.asarray(
            [[0.1, 0.1, 0.08], [0.8, 0.1, 0.08], [0.1, 0.8, 0.08]]
        ),
        cranium_faces=triangle,
        cranium_source_vertex_ids=np.arange(3),
        cranium_source_triangle_ids=np.arange(1),
        mandible_points_m=np.asarray(
            [[0.1, 0.1, -0.08], [0.8, 0.1, -0.08], [0.1, 0.8, -0.08]]
        ),
        mandible_faces=triangle,
        mandible_source_vertex_ids=np.arange(3),
        mandible_source_triangle_ids=np.arange(1),
        geometry_path=Path("synthetic.npz"),
        geometry_sha256="synthetic",
        audit_path=Path("synthetic.json"),
        audit_sha256="synthetic",
        audit=audit,
    )


def synthetic_eyes() -> RigidEyeGeometry:
    points = np.asarray(
        [[0.15, 0.15, 0.04], [0.65, 0.15, 0.04], [0.15, 0.65, 0.04]], dtype=np.float64
    )
    triangles = np.asarray([[0, 1, 2]], dtype=np.int64)
    manifest = {
        "artifacts": {
            "eyes.npz": {"path": "synthetic-eyes.npz", "sha256": "synthetic"}
        },
        "source": {"path": "synthetic-source", "sha256": "synthetic"},
        "units": "metres",
        "frame": "unchanged registered source and FEM world frame",
    }
    return RigidEyeGeometry(
        points,
        triangles,
        np.arange(3),
        np.arange(1),
        np.zeros(3, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        Path(),
        manifest,
        "synthetic",
    )


def synthetic_contact() -> tuple[Any, FullSkullGeometry, RigidEyeGeometry]:
    geometry, eyes = synthetic_geometry(), synthetic_eyes()
    config = {
        "dhat_m": 0.1,
        "stiffness_mpa": 0.01,
        "collision_set_type": "IPC",
        "ccd_max_iterations": 1000,
        "ccd_min_distance_m": 0.0,
    }
    base = SimpleNamespace(
        full_skull=SimpleNamespace(geometry=geometry),
        contact_definition={"config": config},
        full_skull_receipt=lambda: {"synthetic": True},
    )
    contact, _ = _build_contact(base, eyes)
    return contact, geometry, eyes


def derivative_and_filter_checks() -> dict[str, Any]:
    contact, geometry, eyes = synthetic_contact()
    n = geometry.full_node_count + eyes.node_count
    u = torch.zeros((n, 3), dtype=torch.float64)
    state = contact.state_at(u)
    energy = float(contact.fun(state, u))
    assert energy > 0
    assert len(state.collisions) > 0
    gradient = torch.zeros_like(u)
    contact.grad(state, u, gradient)
    direction = torch.Generator(device="cpu").manual_seed(85)
    p = torch.randn(u.shape, generator=direction, dtype=torch.float64)
    p /= torch.linalg.vector_norm(p)
    hp = torch.zeros_like(u)
    contact.hess_prod(state, u, p, hp)
    h = 1.0e-5
    ep = float(contact.fun(state, u + h * p))
    em = float(contact.fun(state, u - h * p))
    gp = torch.zeros_like(u)
    gm = torch.zeros_like(u)
    contact.grad(state, u + h * p, gp)
    contact.grad(state, u - h * p, gm)
    slope_error = abs((ep - em) / (2 * h) - float(torch.sum(gradient * p))) / max(
        abs((ep - em) / (2 * h)), abs(float(torch.sum(gradient * p))), 1e-30
    )
    hp_error = float(
        torch.linalg.vector_norm((gp - gm) / (2 * h) - hp)
        / torch.linalg.vector_norm(hp)
    )
    assert slope_error < 2e-5
    assert hp_error < 2e-5
    # All rigid surfaces share one patch. Moving only eyes cannot create a rigid-rigid pair.
    rigid_only = u.clone()
    rigid_only[-eyes.node_count :, 2] = -0.02
    rigid_state = contact.state_at(rigid_only)
    assert len(rigid_state.collisions) == len(state.collisions)
    crossing = torch.zeros_like(u)
    crossing[:3, 2] = 0.12
    fraction = float(contact.max_step_size(state, u, crossing))
    assert 0 < fraction < 1
    original = DofMap(
        n_points=geometry.full_node_count,
        dim=3,
        fixed_indices=torch.tensor([9, 10, 11]),
        fixed_values=torch.zeros(3),
        free_indices=torch.arange(9),
    )
    extended = _extend_dof_map(original, eyes.node_count)
    assert torch.equal(extended.free_indices, original.free_indices)
    assert len(extended.fixed_indices) == 12
    return {
        "energy": energy,
        "active_pairs": len(state.collisions),
        "gradient_relative_error": slope_error,
        "hvp_relative_error": hp_error,
        "rigid_rigid_pairs_excluded": True,
        "ccd_eye_crossing_fraction": fraction,
        "eye_dofs_fixed": True,
    }


def real_identity(eyes_dir: Path) -> dict[str, Any]:
    from joint_frozen_neutral import FrozenNeutral, load_script

    # The frozen model is a CUDA Warp model; the synthetic checks above are CPU.
    load_script("68-run-simple-skin-forward.py").configure_cuda()
    neutral = FrozenNeutral.load()
    physics, baseline = build_eye_collision_physics(neutral, eyes_dir)
    now = physics.runtime.forward.model.get_materials()
    identical = all(
        torch.equal(now[name][key], value)
        for name, fields in baseline.items()
        for key, value in fields.items()
    )
    assert identical
    pose = torch.as_tensor(
        [0.002, -0.001, 0.0015, 0.0003, -0.0002, 0.0004],
        device=physics.points_t.device,
        dtype=physics.points_t.dtype,
    )
    boundary = physics.full_skull.full_boundary_displacement(
        physics.points_t, physics.jaw_t, pose
    )
    eye_boundary = boundary[physics.full_skull.eye_global_ids]
    assert torch.count_nonzero(eye_boundary) == 0
    return {
        "material_identity": True,
        "eyes_fixed_under_nonzero_mandible_pose": True,
        "eye_vertices": physics.eyes.node_count,
        "eye_triangles": physics.eyes.triangle_count,
        "full_nodes": physics.full_skull.full_node_count,
        "receipt": physics.full_skull_receipt(),
    }


def main(cfg: Config) -> None:
    torch.set_default_dtype(torch.float64)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    synthetic = derivative_and_filter_checks()
    real = (
        real_identity(cfg.eyes_dir)
        if cfg.run_real_material_identity
        else {"skipped": True}
    )
    result = {
        "schema": "joint-rigid-eye-contact-validation-v1",
        "success": True,
        "synthetic_cpu": synthetic,
        "real_model": real,
        "implementation_sha256": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(
                Path(__file__).with_name("joint_rigid_eye_contact.py").resolve()
            ): sha256(Path(__file__).with_name("joint_rigid_eye_contact.py").resolve()),
        },
    }
    write_json(cfg.output_dir / "summary.json", result)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
