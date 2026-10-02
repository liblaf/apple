"""CPU-only, alpha-specific pose increments with separate residual predictions.

The cache holds coefficients at one accepted physical state. A trial solves for
an actual pose increment, not a direction to multiply by alpha again. Residual
response is a model intercept only and is never added to the production seed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mouthopen_pose_projection import project_pose_descent_direction


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _immutable(value: np.ndarray, dtype: Any = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AffineDirectionCache:
    """Owned, read-only arrays bound to one persisted coefficient file."""

    original_ids: np.ndarray
    old_j: np.ndarray
    q_delta_j: np.ndarray
    pose_jacobian: np.ndarray
    residual_delta_j: np.ndarray
    pose_gradient: np.ndarray
    requested_dp: np.ndarray
    strain_slope: float
    original_target: float
    source_path: Path
    source_sha256: str


def save_affine_cache(
    directory: Path,
    *,
    original_ids: np.ndarray,
    old_j: np.ndarray,
    q_delta_j: np.ndarray,
    pose_jacobian: np.ndarray,
    residual_delta_j: np.ndarray,
    pose_gradient: np.ndarray,
    requested_dp: np.ndarray,
    strain_slope: float,
    original_target: float,
) -> AffineDirectionCache:
    """Copy and persist all retained-cell coefficients once per outer iteration."""
    arrays = {
        "original_ids": _immutable(original_ids, np.int64),
        "old_j": _immutable(old_j),
        "q_delta_j": _immutable(q_delta_j),
        "pose_jacobian": _immutable(pose_jacobian),
        "residual_delta_j": _immutable(residual_delta_j),
        "pose_gradient": _immutable(pose_gradient),
        "requested_dp": _immutable(requested_dp),
    }
    n = len(arrays["original_ids"])
    assert arrays["original_ids"].shape == (n,)
    assert np.unique(arrays["original_ids"]).size == n
    assert np.all(arrays["original_ids"] >= 0)
    for name in ("old_j", "q_delta_j", "residual_delta_j"):
        assert arrays[name].shape == (n,)
    assert arrays["pose_jacobian"].shape == (n, 6)
    assert arrays["pose_gradient"].shape == (6,)
    assert arrays["requested_dp"].shape == (6,)
    assert all(np.isfinite(value).all() for value in arrays.values())
    assert np.any(arrays["old_j"] > 0)
    assert np.isfinite(strain_slope)
    assert np.isfinite(original_target)
    assert original_target < 0
    assert np.linalg.norm(arrays["pose_gradient"]) > 0
    directory.mkdir(parents=True, exist_ok=False)
    path = directory.resolve() / "coefficients.npz"
    np.savez_compressed(
        path,
        **arrays,
        strain_slope=np.asarray(strain_slope),
        original_target=np.asarray(original_target),
    )
    cache = AffineDirectionCache(
        **arrays,
        strain_slope=float(strain_slope),
        original_target=float(original_target),
        source_path=path,
        source_sha256=_sha256(path),
    )
    _write_json(
        directory / "manifest.json",
        {
            "schema": "mouthopen-affine-direction-cache-v1",
            "coefficients": {"path": str(path), "sha256": cache.source_sha256},
            "retained_cell_count": n,
            "positive_retained_cell_count": int(np.sum(cache.old_j > 0)),
            "residual_response_is_separate_intercept": True,
            "accepted_equilibrium": False,
        },
    )
    return cache


def project_affine_increment(
    cache: AffineDirectionCache,
    alpha: float,
    output: Path,
    *,
    margin: float = 1e-6,
    activation_threshold: float = 0.05,
    max_constraint_passes: int = 8,
) -> tuple[np.ndarray, dict]:
    """Certify one actual increment and check both models on every positive cell.

    The seed model is ``Jold + alpha*q_dJ + A*increment``. The affine model
    adds the unscaled residual response. Existing nonpositive cells remain
    governed by the caller's original physical inversion allowance. No result
    certifies a physical candidate; CCD, correction and acceptance are required.
    """
    output.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "status": "running",
        "alpha": float(alpha),
        "margin": margin,
        "activation_threshold": activation_threshold,
        "max_constraint_passes": max_constraint_passes,
        "cache": {"path": str(cache.source_path), "sha256": cache.source_sha256},
        "actual_increment_already_scaled": True,
        "residual_response_used_in_seed": False,
        "accepted_equilibrium": False,
        "final_ccd_forward_inversion_and_armijo_required": True,
        "constraint_passes": [],
    }
    _write_json(output / "input-receipt.json", receipt)
    try:
        assert 0 < alpha <= 1
        assert 0 < margin < activation_threshold
        assert max_constraint_passes >= 1
        assert _sha256(cache.source_path) == cache.source_sha256
        positive = cache.old_j > 0
        seed_intercept = cache.old_j + alpha * cache.q_delta_j
        affine_intercept = seed_intercept + cache.residual_delta_j
        lower = (
            margin
            - cache.old_j
            - np.minimum(cache.residual_delta_j, 0)
            - alpha * cache.q_delta_j
        )
        active = positive & (
            (cache.old_j <= activation_threshold)
            | (cache.old_j + cache.residual_delta_j <= activation_threshold)
            | (seed_intercept <= activation_threshold)
            | (affine_intercept <= activation_threshold)
        )
        # Always seed the active set with its most restrictive intercept. This
        # also keeps the existing certified geometry-LP interface nonempty.
        positive_indices = np.flatnonzero(positive)
        most_restrictive = positive_indices[np.argmax(lower[positive])]
        active[most_restrictive] = True
        requested = alpha * cache.requested_dp
        strain_slope = alpha * cache.strain_slope
        original_target = alpha * cache.original_target
        receipt.update(
            original_increment_target=original_target,
            strain_increment_slope=strain_slope,
            requested_pose_increment=requested.tolist(),
            positive_retained_cells=int(positive.sum()),
            existing_nonpositive_cells=int((~positive).sum()),
            initial_active_ids=cache.original_ids[active].tolist(),
        )
        for attempt in range(max_constraint_passes):
            pass_receipt: dict[str, Any] = {
                "attempt": attempt,
                "active_original_ids": cache.original_ids[active].tolist(),
                "witness": {},
            }
            receipt["constraint_passes"].append(pass_receipt)
            np.savez_compressed(
                output / f"constraints-{attempt:02d}.npz",
                original_cell_ids=cache.original_ids[active],
                matrix=cache.pose_jacobian[active],
                lower=lower[active],
                requested_increment=requested,
                pose_gradient=cache.pose_gradient,
                strain_increment_slope=np.asarray(strain_slope),
                original_increment_target=np.asarray(original_target),
            )
            increment, projection = project_pose_descent_direction(
                requested,
                cache.pose_jacobian[active],
                lower[active],
                pose_gradient=cache.pose_gradient,
                strain_directional=strain_slope,
                descent_target=original_target,
                feasible_witness=True,
                witness_policy="attainable_optimum",
                witness_receipt=pass_receipt["witness"],
            )
            pass_receipt["projection"] = projection
            predicted_seed = seed_intercept + cache.pose_jacobian @ increment
            predicted_affine = predicted_seed + cache.residual_delta_j
            slack = np.minimum(predicted_seed, predicted_affine) - margin
            violating = positive & (slack < -1e-10)
            pass_receipt.update(
                minimum_full_positive_slack=float(slack[positive].min()),
                violating_original_ids=cache.original_ids[violating].tolist(),
            )
            if violating.any():
                assert np.any(violating & ~active), (
                    "Active QP violated a supplied constraint"
                )
                active |= violating
                _write_json(output / "summary.json", receipt)
                continue
            actual_slope = float(strain_slope + cache.pose_gradient @ increment)
            chosen_target = projection["descent"]["joint_directional_upper_bound"]
            assert actual_slope < 0
            assert actual_slope <= chosen_target + 1e-11
            limiting = positive_indices[np.argsort(slack[positive])[:8]]
            receipt.update(
                status="certified_linear_increment",
                projection=projection,
                actual_pose_increment=increment.tolist(),
                actual_joint_increment_slope=actual_slope,
                chosen_increment_target=chosen_target,
                all_positive_cells_checked=True,
                minimum_seed_positive_J=float(predicted_seed[positive].min()),
                minimum_affine_positive_J=float(predicted_affine[positive].min()),
                minimum_full_positive_slack=float(slack[positive].min()),
                limiting_original_ids=cache.original_ids[limiting].tolist(),
            )
            np.savez_compressed(
                output / "predictions.npz",
                original_retained_ids=cache.original_ids,
                old_det_f=cache.old_j,
                predicted_seed_det_f=predicted_seed,
                predicted_affine_det_f=predicted_affine,
                actual_pose_increment=increment,
                residual_delta_j=cache.residual_delta_j,
            )
            _write_json(output / "summary.json", receipt)
            return increment, receipt
        message = (
            "Full positive-cell constraint closure exceeded its declared pass limit"
        )
        raise RuntimeError(message)  # noqa: TRY301 - persist failure before propagating
    except Exception as error:
        receipt.update(
            status="projection_failed",
            failure={
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        _write_json(output / "summary.json", receipt)
        raise


__all__ = ["AffineDirectionCache", "project_affine_increment", "save_affine_cache"]
