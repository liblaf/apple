"""Project a proposed jaw update using coupled tissue determinant sensitivities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from joint_common import write_json
from joint_coupled_predictor import clone_materials
from mouthopen_coupled_seed import _damped_equilibrium_tangent


@torch.no_grad()
def project_coupled_pose(  # noqa: C901, PLR0915
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
    pose_gradient: torch.Tensor | None = None,
    strain_directional: float | None = None,
    descent_target: float | None = None,
    analytic_determinants: bool = False,
    feasible_witness: bool = False,
    witness_policy: str = "zero_pose",
) -> tuple[torch.Tensor, dict]:
    """Return a linearized feasible direction; the caller must fully validate it.

    Probe states serve only to differentiate determinants. They are never
    accepted as equilibria. Actual trial states still pass collision CCD,
    nonlinear equilibrium, and the original retained-tet allowance.
    """
    from mouthopen_pose_projection import (
        determinant_directional_derivative,
        project_pose_descent_direction,
        project_pose_direction,
    )

    assert 0 < margin < activation_threshold
    assert epsilon > 0
    assert not feasible_witness or descent_target is not None
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
        analytic_arrays = {}
        if analytic_determinants:
            old_u_np = displacement.cpu().numpy()
            active_tets = tets[active]

            def determinant_derivative(seed: torch.Tensor) -> np.ndarray:
                return determinant_directional_derivative(
                    reference,
                    active_tets,
                    old_u_np,
                    (seed.cpu().numpy() - old_u_np) / epsilon,
                )

            strain_seed = tangent(
                materials(q + epsilon * (next_q - q)), fixed, changed=True
            )
            strain_delta_j = determinant_derivative(strain_seed)
            jq = j0 + strain_delta_j
            analytic_arrays = {
                "q_analytic_delta_j": strain_delta_j,
                "q_probe_actual_det_f": determinants(strain_seed)[active],
            }
        else:
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
            if analytic_determinants:
                jacobian[:, coordinate] = determinant_derivative(probe_seed)
            else:
                jacobian[:, coordinate] = (
                    determinants(probe_seed)[active] - j0
                ) / epsilon
        # Preserve every currently positive near-boundary tet. Existing
        # inverted cells remain governed by the original count/volume gates.
        required = margin - jq
        descent_arrays = {}
        if descent_target is not None:
            assert pose_gradient is not None
            assert strain_directional is not None
            descent_arrays = {
                "pose_gradient": pose_gradient.cpu().numpy(),
                "strain_directional": np.asarray(strain_directional),
                "descent_target": np.asarray(descent_target),
            }
        # Persist the complete problem before the solver can fail.
        np.savez_compressed(
            output / "projection-inputs.npz",
            original_cell_ids=source_ids[active],
            j_old=j0,
            j_strain_seed=jq,
            jacobian=jacobian,
            required=required,
            requested=requested_dp.cpu().numpy(),
            **descent_arrays,
            **analytic_arrays,
        )
        write_json(
            output / "input-receipt.json",
            {
                "stage": "before_projection_solve",
                "accepted_equilibrium": False,
                "analytic_determinants": analytic_determinants,
                "feasible_witness": feasible_witness,
                "witness_policy": witness_policy,
                "determinant_margin": margin,
                "normalized_probe_epsilon": epsilon,
                "tangent_solves": copy.deepcopy(receipts),
            },
        )
        if descent_target is None:
            assert pose_gradient is None
            assert strain_directional is None
            projected, qp = project_pose_direction(
                requested_dp.cpu().numpy(), jacobian, required
            )
        else:
            assert pose_gradient is not None
            assert strain_directional is not None
            witness_receipt = {}
            try:
                projected, qp = project_pose_descent_direction(
                    requested_dp.cpu().numpy(),
                    jacobian,
                    required,
                    pose_gradient=pose_gradient.cpu().numpy(),
                    strain_directional=strain_directional,
                    descent_target=descent_target,
                    feasible_witness=feasible_witness,
                    witness_policy=witness_policy,
                    witness_receipt=witness_receipt,
                )
            finally:
                if feasible_witness:
                    write_json(output / "witness-receipt.json", witness_receipt)
        if feasible_witness:
            descent_arrays["original_descent_target"] = descent_arrays["descent_target"]
            descent_arrays["descent_target"] = np.asarray(
                qp["descent"]["joint_directional_upper_bound"]
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
            **descent_arrays,
            **analytic_arrays,
        )
        receipt = {
            "method": "coupled-tangent-linearized-pose-projection",
            "accepted_equilibrium": False,
            "final_ccd_forward_and_inversion_checks_required": True,
            "active_cell_count": int(active.sum()),
            "activation_threshold": activation_threshold,
            "determinant_margin": margin,
            "normalized_pose_probe_epsilon": epsilon,
            "analytic_determinants": analytic_determinants,
            "pose_probe_method": (
                "cofactor determinant derivative along small-pose damped coupled tissue tangents"
                if analytic_determinants
                else "forward differences of damped coupled tissue tangents"
            ),
            "q_seed_method": (
                "cofactor determinant derivative along small Raw6 force-change coupled tangent"
                if analytic_determinants
                else "finite Raw6 force-change coupled tangent"
            ),
            "strain_J_semantics": "old J plus analytic directional derivative"
            if analytic_determinants
            else "finite strain seed determinant",
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
