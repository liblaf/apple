# ruff: noqa: EM101, TRY003
"""Construct a feasible target seed without shrinking the outer inverse step."""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

import torch
from joint_coupled_seed import predict_expression_seed
from joint_equilibrium import ForwardConvergenceError

LOG = logging.getLogger(__name__)


@torch.no_grad()
def prepare_expression_seed(
    physics: Any,
    *,
    old_stress: torch.Tensor,
    new_stress: torch.Tensor,
    old_pose: torch.Tensor,
    new_pose: torch.Tensor,
    seed: torch.Tensor,
    axis: torch.Tensor,
    pivot: torch.Tensor,
    linear_rtol: float,
    max_steps: int = 32,
    max_attempts: int = 96,
    equilibrate_substeps: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Follow CCD-admitted coupled predictors to the exact requested parameters.

    Internal states are never outer iterates. By default each nonfinal admitted
    predictor is relaxed by the original strict PNCG before predicting again.
    The optional unrelaxed variant is retained for the recorded ablation only.
    The caller must run the original strict equilibrium solver at the final
    parameters and perform its normal adjoint and objective acceptance checks.
    A bounded failure returns no partial result. Only private detached seeds
    are retained, and the underlying predictor restores model mutations.
    """
    assert max_steps > 0
    assert max_attempts >= max_steps
    started = time.perf_counter()
    current_seed = seed.detach().clone()
    original_stress, target_stress = old_stress.detach(), new_stress.detach()
    original_pose, target_pose = old_pose.detach(), new_pose.detach()
    current_stress = original_stress
    current_pose = original_pose
    progress = 0.0
    fraction = 1.0
    accepted = 0
    attempts = []
    receipt = {
        "method": "adaptive-coupled-predictor-continuation",
        "success": False,
        "seed_only": True,
        "physical_model_changed": False,
        "intermediate_equilibrium_required": equilibrate_substeps,
        "final_strict_equilibrium_required": True,
        "attempts": attempts,
    }
    for attempt in range(max_attempts):
        if accepted >= max_steps or fraction < 1e-8:
            break
        next_progress = 1.0 if fraction == 1 else progress + fraction * (1 - progress)
        # Snap the final trial to the exact caller tensors, avoiding interpolation roundoff.
        candidate_pose = (
            target_pose
            if next_progress == 1
            else torch.lerp(original_pose, target_pose, next_progress)
        )
        candidate_stress = (
            target_stress
            if next_progress == 1
            else torch.lerp(original_stress, target_stress, next_progress)
        )
        trial = {
            "attempt": attempt + 1,
            "progress_before": progress,
            "progress_proposed": next_progress,
            "remaining_fraction": fraction,
        }
        try:
            candidate_seed, details = predict_expression_seed(
                physics,
                old_stress=current_stress,
                new_stress=candidate_stress,
                old_pose=current_pose,
                new_pose=candidate_pose,
                seed=current_seed,
                axis=axis,
                pivot=pivot,
                linear_rtol=linear_rtol,
                residual_scale=fraction,
            )
            if equilibrate_substeps and next_progress < 1:
                correction_started = time.perf_counter()
                candidate_seed = (
                    physics.solve(
                        skin_multiplier=torch.ones(
                            (), device=seed.device, dtype=seed.dtype
                        ),
                        active_stress=candidate_stress,
                        pose=candidate_pose,
                        seed=candidate_seed,
                        seed_pose=candidate_pose,
                        key="coupled-internal-corrector",
                    )
                    .detach()
                    .clone()
                )
                details["internal_corrector"] = copy.deepcopy(
                    physics.runtime.last_forward
                )
                details["internal_corrector_wall_seconds"] = (
                    time.perf_counter() - correction_started
                )
        except ForwardConvergenceError as error:
            trial.update(admitted=False, failure=str(error), details=error.receipt)
            attempts.append(trial)
            # A new parameter step is always recomputed and re-audited. The
            # previous CCD fraction is only a proposal-size heuristic.
            ccd = error.receipt.get("geometry", {}).get("ccd_fraction")
            shrink = 0.5 if ccd is None else min(0.5, max(0.1, 0.8 * float(ccd)))
            fraction *= shrink
            LOG.info(
                "Coupled seed attempt %d rejected at %.4f progress; next local fraction %.5g",
                attempt + 1,
                next_progress,
                fraction,
            )
            continue
        trial.update(admitted=True, details=details)
        attempts.append(trial)
        current_seed = candidate_seed
        current_pose, current_stress = candidate_pose, candidate_stress
        progress = next_progress
        accepted += 1
        LOG.info(
            "Coupled seed progress %.6f in %d admitted substeps", progress, accepted
        )
        if progress == 1:
            receipt.update(
                success=True,
                progress=1.0,
                accepted_substeps=accepted,
                seconds=time.perf_counter() - started,
            )
            return current_seed.detach().clone(), receipt
        # Try the entire remaining target from the updated contact geometry.
        # If necessary, the next trial is again reduced and fully recomputed.
        fraction = 1.0
    receipt.update(
        progress=progress,
        accepted_substeps=accepted,
        seconds=time.perf_counter() - started,
    )
    raise ForwardConvergenceError(
        "coupled seed continuation exhausted its budget before the target",
        receipt=receipt,
    )
