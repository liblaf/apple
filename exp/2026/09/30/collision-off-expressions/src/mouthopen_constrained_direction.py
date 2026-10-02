# Copyright (c) 2026 liblaf
"""Project a proposed jaw update using coupled tissue determinant sensitivities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from collision_off_seed import _damped_equilibrium_tangent
from joint_common import write_json
from joint_coupled_predictor import clone_materials


@torch.no_grad()
def project_coupled_pose(
    physics: Any,
    materials: Any,
    physical_pose: Any,
    q: torch.Tensor,
    next_q: torch.Tensor,
    pose: torch.Tensor,
    requested_dp: torch.Tensor,
    displacement: torch.Tensor,
    output: Path,
    *,
    activation_threshold: float,
    margin: float,
    epsilon: float,
    deadline: float | None,
) -> tuple[torch.Tensor, dict]:
    """Return a linearized feasible direction; the caller must fully validate it.

    Probe states serve only to differentiate determinants. They are never
    accepted as equilibria. Actual trial states still pass collision-off
    nonlinear equilibrium and the retained-tet allowance.
    """
    from mouthopen_pose_projection import project_pose_direction

    assert 0 < margin < activation_threshold
    assert epsilon > 0
    output.mkdir(parents=True)
    runtime = physics.runtime
    model = runtime.forward.model
    base = physics.base
    source_ids = base._mouthopen_retained_tetrahedron_ids  # noqa: SLF001
    tets = np.asarray(base.tets)[source_ids]
    reference = np.asarray(base.points)
    rest_edges = np.transpose(
        reference[tets[:, 1:]] - reference[tets[:, :1]], (0, 2, 1)
    )
    rest_det = np.linalg.det(rest_edges)
    assert np.all(rest_det > 0)

    def determinants(u: torch.Tensor) -> np.ndarray:
        points = reference + u[: len(reference)].cpu().numpy()
        edges = np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
        result = np.linalg.det(edges) / rest_det
        assert np.isfinite(result).all()
        return result

    original_materials = clone_materials(model.get_materials())
    original_fixed = model.dof_map.fixed_values.clone()
    old_materials = materials(q)
    receipts = []
    j_old = determinants(displacement)
    active = (j_old > 0) & (j_old <= activation_threshold)
    assert active.any()
    j0 = j_old[active]
    try:

        def tangent(new_materials: dict, fixed: torch.Tensor, *, changed: bool):
            seed, receipt = _damped_equilibrium_tangent(
                physics,
                old_materials,
                new_materials,
                displacement,
                fixed,
                deadline=deadline,
                predictor_relative_shift=runtime.adjoint_relative_shift,
                predictor_rtol=runtime.tolerances["adjoint_rtol"],
                material_changed=changed,
            )
            receipts.append(receipt)
            return seed

        fixed = physics.boundary(physical_pose(pose))
        strain_seed = tangent(materials(next_q), fixed, changed=True)
        jq = determinants(strain_seed)[active]
        jacobian = np.empty((len(j0), 6))
        for coordinate in range(6):
            probe_pose = pose.clone()
            probe_pose[coordinate] += epsilon
            probe_seed = tangent(
                old_materials,
                physics.boundary(physical_pose(probe_pose)),
                changed=False,
            )
            jacobian[:, coordinate] = (determinants(probe_seed)[active] - j0) / epsilon
        # Preserve every currently positive near-boundary tet. Existing
        # inverted cells remain governed by the original count/volume gates.
        required = margin - jq
        projected, qp = project_pose_direction(
            requested_dp.cpu().numpy(),
            jacobian,
            required,
        )
        np.savez_compressed(
            output / "linear-constraints.npz",
            original_cell_ids=source_ids[active],
            j_old=j0,
            j_strain_seed=jq,
            jacobian=jacobian,
            required=required,
            requested=requested_dp.cpu().numpy(),
            projected=projected,
        )
        receipt = {
            "method": "coupled-tangent-linearized-pose-projection",
            "accepted_equilibrium": False,
            "final_collision_off_forward_and_inversion_checks_required": True,
            "active_cell_count": int(active.sum()),
            "activation_threshold": activation_threshold,
            "determinant_margin": margin,
            "normalized_pose_probe_epsilon": epsilon,
            "pose_probe_method": "forward differences of damped coupled tissue tangents",
            "q_seed_method": "finite Raw6 force-change coupled tangent",
            "minimum_old_positive_active_J": float(j0.min()),
            "minimum_strain_seed_active_J": float(jq.min()),
            "minimum_linearized_projected_J": float((jq + jacobian @ projected).min()),
            "projection": qp,
            "tangent_solves": copy.deepcopy(receipts),
        }
        write_json(output / "summary.json", receipt)
        return torch.as_tensor(projected, device=pose.device, dtype=pose.dtype), receipt
    finally:
        model.set_materials(original_materials)
        model.dof_map.fixed_values = original_fixed
