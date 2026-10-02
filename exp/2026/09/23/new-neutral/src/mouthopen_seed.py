# ruff: noqa: EM101, PLR0915, PT018, TRY003, TRY301
"""Material-agnostic CCD-admitted predictor seeds for the MouthOpen inverse."""

from __future__ import annotations

import copy
import logging
import math
import time
from typing import Any

import torch
from joint_coupled_predictor import (
    audit_coupled_motion,
    equilibrium_predictor,
    rotation_sagitta,
)
from joint_equilibrium import ForwardConvergenceError

LOG = logging.getLogger(__name__)


def _axis_radius(
    vertices: torch.Tensor, axis: torch.Tensor, pivot: torch.Tensor
) -> float:
    relative = vertices - pivot
    perpendicular = relative - (relative @ axis).unsqueeze(-1) * axis
    return float(torch.linalg.vector_norm(perpendicular, dim=-1).max())


@torch.no_grad()
def prepare_seed(
    physics: Any,
    materials: Any,
    pose: Any,
    old_q: torch.Tensor,
    new_q: torch.Tensor,
    old_jaw: torch.Tensor,
    new_jaw: torch.Tensor,
    seed: torch.Tensor,
    *,
    linear_rtol: float = 1e-7,
    max_steps: int = 256,
    max_attempts: int = 512,
    callback: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Predict and CCD-admit a full runtime seed for a Raw6/jaw change.

    The intermediate correctors exist only to make the next tangent prediction
    reliable.  The caller owns the final strict equilibrium at ``new_q`` and
    ``new_jaw``.
    """
    assert 0 < linear_rtol <= 1e-7
    assert max_steps > 0 and max_attempts >= max_steps
    assert old_q.shape == new_q.shape and old_jaw.shape == new_jaw.shape
    assert bool(
        torch.isfinite(old_q).all()
        and torch.isfinite(new_q).all()
        and torch.isfinite(old_jaw).all()
        and torch.isfinite(new_jaw).all()
        and torch.isfinite(seed).all()
    )
    runtime = physics.runtime
    model = runtime.forward.model
    collision = model.collision
    assert collision is not None
    assert seed.shape == runtime.forward.state.u.shape
    started = time.perf_counter()
    current_seed = seed.detach().clone()
    current_q, current_jaw = old_q.detach(), old_jaw.detach()
    target_q, target_jaw = new_q.detach(), new_jaw.detach()
    axis = (pose(torch.ones_like(old_jaw)) - pose(torch.zeros_like(old_jaw)))[:3]
    axis = axis / torch.linalg.vector_norm(axis)
    assert abs(float(torch.linalg.vector_norm(axis)) - 1.0) < 1e-12
    pivot = torch.as_tensor(
        physics.full_skull.geometry.mandible_pivot_m,
        device=seed.device,
        dtype=seed.dtype,
    )
    radius = _axis_radius(collision.vertices, axis, pivot)
    total_angle = float(
        torch.linalg.vector_norm(pose(target_jaw)[:3] - pose(current_jaw)[:3])
    )
    max_initial_angle = math.radians(0.25)
    max_angle = math.radians(0.5)
    angular_step = min(max_initial_angle, total_angle)
    progress = 0.0
    fraction = 1.0 if total_angle == 0 else angular_step / total_angle
    accepted = 0
    attempts: list[dict[str, Any]] = []
    for attempt in range(max_attempts):
        if accepted >= max_steps or fraction < 1e-8:
            break
        next_progress = 1.0 if fraction == 1.0 else progress + fraction * (1 - progress)
        candidate_q = (
            target_q
            if next_progress == 1.0
            else torch.lerp(old_q, new_q, next_progress)
        )
        candidate_jaw = (
            target_jaw
            if next_progress == 1.0
            else torch.lerp(old_jaw, new_jaw, next_progress)
        )
        row: dict[str, Any] = {
            "attempt": attempt + 1,
            "progress_before": progress,
            "progress_proposed": next_progress,
            "remaining_fraction": fraction,
        }
        try:
            fixed_target = physics.boundary(pose(candidate_jaw))
            delta_angle = float(
                torch.linalg.vector_norm(
                    pose(candidate_jaw)[:3] - pose(current_jaw)[:3]
                )
            )
            margin = rotation_sagitta(radius, delta_angle)
            current_contact = collision.diagnostics(
                collision.state_at(current_seed), current_seed
            )
            current_gap = current_contact["minimum_active_distance_m"]
            if current_gap is not None and margin >= current_gap:
                raise ForwardConvergenceError(
                    "hinge sagitta exceeds current active contact clearance",
                    receipt={
                        "geometry": {
                            "admitted": False,
                            "precheck": "rotation sagitta below current active gap",
                            "current_minimum_active_distance_m": current_gap,
                            "rotation_sagitta_bound_m": margin,
                        }
                    },
                )
            predicted, predictor = equilibrium_predictor(
                model=model,
                solver=runtime.solver,
                displacement=current_seed,
                old_materials=materials(current_q),
                new_materials=materials(candidate_q),
                fixed_target=fixed_target,
                linear_rtol=linear_rtol,
                residual_scale=fraction,
            )
            geometry = audit_coupled_motion(
                collision, current_seed, predicted, rotation_margin_m=margin
            )
            row.update(
                predictor=predictor,
                geometry=geometry,
                maximum_collision_vertex_axis_radius_m=radius,
                delta_hinge_angle_rad=delta_angle,
                rotation_sagitta_bound_m=margin,
            )
            if not geometry["admitted"]:
                raise ForwardConvergenceError(
                    "coupled predictor CCD admission failed", receipt=row
                )
            # Correct each admitted nonfinal predictor before asking it to seed
            # another tangent calculation.  The final strict solve is caller-owned.
            if next_progress < 1.0:
                correction_started = time.perf_counter()
                predicted = runtime.primal(
                    materials(candidate_q), fixed_target, predicted
                )
                row["internal_corrector"] = copy.deepcopy(runtime.last_forward)
                row["internal_corrector_wall_seconds"] = (
                    time.perf_counter() - correction_started
                )
        except ForwardConvergenceError as error:
            # ``row`` can itself be the error receipt from the CCD gate; copy
            # before adding it to this row so the JSON receipt stays acyclic.
            receipt = copy.deepcopy(error.receipt)
            row.update(admitted=False, failure=str(error), receipt=receipt)
            attempts.append(row)
            ccd = error.receipt.get("geometry", {}).get("ccd_fraction")
            shrink = 0.5 if ccd is None else min(0.5, max(0.1, 0.8 * float(ccd)))
            fraction *= shrink
            LOG.info(
                "MouthOpen seed rejected attempt %d at progress %.6f; next fraction %.6g",
                attempt + 1,
                next_progress,
                fraction,
            )
            continue
        row["admitted"] = True
        attempts.append(row)
        current_seed = predicted.detach().clone()
        current_q, current_jaw = candidate_q, candidate_jaw
        progress = next_progress
        accepted += 1
        if callback is not None:
            callback(
                current_q.detach().clone(),
                current_jaw.detach().clone(),
                current_seed,
                copy.deepcopy(row),
            )
        LOG.info(
            "MouthOpen seed accepted progress %.6f after %d admitted substeps",
            progress,
            accepted,
        )
        if progress == 1.0:
            return current_seed, {
                "success": True,
                "method": "adaptive-coupled-raw6-jaw-predictor-continuation",
                "seed_only": True,
                "physical_model_changed": False,
                "final_strict_equilibrium_required": True,
                "accepted_substeps": accepted,
                "progress": progress,
                "seconds": time.perf_counter() - started,
                "attempts": attempts,
            }
        # Continue from the local accepted angular scale.  Repeating the whole
        # remaining jaw rotation would create futile CCD broad-phase warnings.
        angular_step = min(max_angle, delta_angle * 1.5)
        remaining_angle = total_angle * (1.0 - progress)
        fraction = (
            1.0 if remaining_angle == 0 else min(1.0, angular_step / remaining_angle)
        )
    raise ForwardConvergenceError(
        "coupled Raw6/jaw seed continuation exhausted its budget before target",
        receipt={
            "success": False,
            "method": "adaptive-coupled-raw6-jaw-predictor-continuation",
            "seed_only": True,
            "physical_model_changed": False,
            "final_strict_equilibrium_required": True,
            "accepted_substeps": accepted,
            "progress": progress,
            "seconds": time.perf_counter() - started,
            "attempts": attempts,
        },
    )


__all__ = ["prepare_seed"]
