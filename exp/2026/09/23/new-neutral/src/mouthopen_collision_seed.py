"""Screen a harmonic jaw carry while retaining collision in every equilibrium."""

# ruff: noqa: EM101, TRY003, TRY301

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from joint_common import write_json
from joint_coupled_predictor import audit_coupled_motion, rotation_sagitta
from joint_equilibrium import ForwardConvergenceError
from mouthopen_harmonic_carry import carry_near_mandible
from scipy.spatial.transform import Rotation


@torch.no_grad()
def fixed_pose_geometry(physics: Any, pose: torch.Tensor) -> dict:
    """Check the original all-fixed FEM cells before expensive seed construction."""
    model = physics.runtime.forward.model
    reference = np.asarray(physics.points)
    fixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    tets = np.asarray(physics.tets)
    selected = tets[fixed[tets].all(axis=1)]
    u = torch.zeros_like(physics.runtime.forward.state.u)
    u.flatten()[model.dof_map.fixed_indices] = physics.boundary(pose)
    moved = reference + u[: len(reference)].cpu().numpy()
    rest_edges = reference[selected[:, 1:]] - reference[selected[:, :1]]
    edges = moved[selected[:, 1:]] - moved[selected[:, :1]]
    detf = np.linalg.det(edges) / np.linalg.det(rest_edges)
    assert np.isfinite(detf).all()
    return {
        "all_fixed_tetrahedra": len(selected),
        "detF_min": float(detf.min()),
        "inverted_tetrahedra": int(np.count_nonzero(detf <= 0)),
    }


@torch.no_grad()
def prepare_collision_seed(
    physics: Any,
    materials: Any,
    old_q: torch.Tensor,
    new_q: torch.Tensor,
    old_pose_rad_m: torch.Tensor,
    new_pose_rad_m: torch.Tensor,
    seed: torch.Tensor,
    output_dir: Path,
    *,
    deadline: float | None = None,
    **_unused: Any,
) -> tuple[torch.Tensor, dict]:
    """Return a CCD-admitted seed; the caller performs the collision-on solve."""
    del materials, old_q, new_q
    started = time.perf_counter()
    output_dir.mkdir(parents=True)
    receipt: dict[str, Any] = {
        "method": "harmonic-jaw-carry-with-CCD-admission",
        "collision_disabled": False,
        "equilibrium_claimed": False,
        "final_strict_equilibrium_required": True,
        "success": False,
    }
    try:
        if deadline is not None and time.perf_counter() >= deadline:
            raise ForwardConvergenceError("declared seed wall budget exhausted")
        fixed = fixed_pose_geometry(physics, new_pose_rad_m)
        receipt["fixed_pose_geometry"] = fixed
        if fixed["detF_min"] < 0.1:
            raise ForwardConvergenceError(
                "jaw proposal violates fully prescribed tetrahedron quality floor",
                receipt=fixed,
            )
        if torch.equal(old_pose_rad_m, new_pose_rad_m):
            carried, carry = seed.detach().clone(), {"method": "unchanged-pose"}
        else:
            carried, carry = carry_near_mandible(
                physics, seed, old_pose_rad_m, new_pose_rad_m, lambda value: value
            )
        receipt["carry"] = carry
        geometry = physics.metrics(carried[: len(physics.points)])
        receipt["geometry"] = geometry
        if geometry["inverted_tetrahedra"]:
            raise ForwardConvergenceError(
                "harmonic carry inverts tetrahedra", receipt=geometry
            )
        old = old_pose_rad_m.cpu().numpy()
        new = new_pose_rad_m.cpu().numpy()
        angle = float(
            (
                Rotation.from_rotvec(new[:3]) * Rotation.from_rotvec(old[:3]).inv()
            ).magnitude()
        )
        collision = physics.runtime.forward.model.collision
        pivot = torch.as_tensor(physics.pivot_t, device=seed.device, dtype=seed.dtype)
        radius = float(
            torch.linalg.vector_norm(collision.vertices - pivot, dim=1).max()
        )
        contact = audit_coupled_motion(
            collision, seed, carried, rotation_margin_m=rotation_sagitta(radius, angle)
        )
        receipt["contact"] = contact
        if not contact["admitted"]:
            raise ForwardConvergenceError(
                "harmonic carry failed collision CCD", receipt=contact
            )
        receipt["success"] = True
    except ForwardConvergenceError as error:
        receipt["failure"] = {"message": str(error), "receipt": error.receipt}
        raise
    finally:
        receipt["seconds"] = time.perf_counter() - started
        write_json(output_dir / "summary.json", receipt)
    return carried, receipt
