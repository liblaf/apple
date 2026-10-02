"""Scale-aware diagnostics for the final joint inverse experiment."""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import torch
from joint_fields import (
    BULK_TISSUES,
    DEFAULT_EIGEN_BATCH_SIZE,
    SharedFieldParameters,
    project_activation_,
    symmetric_matrices,
)
from joint_spatial_fields import SpatialSharedFieldParameters

SharedParameters = SharedFieldParameters | SpatialSharedFieldParameters


def shared_reference_departure_quadratic(
    shared: SharedParameters,
    reference_coefficients: torch.Tensor,
    *,
    coefficients: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return a basis-invariant shared-reference departure quadratic.

    Spatial bulk fields use the audited tissue-normalized ``M`` forms. A
    repeated constant anchor field therefore has exactly the same value and
    pulled-back gradient as its constant20 six-vector. Skin coordinates remain
    separate scalar squared departures. The strong spatial ``G`` term is not
    included here.
    """
    values = shared.coefficients if coefficients is None else coefficients
    if values.shape != shared.coefficients.shape:
        msg = "shared values must match the admitted coefficient vector"
        raise ValueError(msg)
    if reference_coefficients.shape != values.shape:
        msg = "shared reference must match the admitted coefficient vector"
        raise ValueError(msg)
    reference = reference_coefficients.to(device=values.device, dtype=values.dtype)
    result: dict[str, torch.Tensor] = {}
    bulk_terms = []
    for tissue_index, tissue in enumerate(BULK_TISSUES):
        if isinstance(shared, SpatialSharedFieldParameters):
            coordinate_slice = shared.config["parameterization"][
                "bulk_anchor_coordinate_slices"
            ][tissue]
            departure = (
                values[coordinate_slice[0] : coordinate_slice[1]]
                - reference[coordinate_slice[0] : coordinate_slice[1]]
            ).reshape(-1, 6)
            matrix_m = getattr(shared, f"_{tissue}_M")
            term = torch.sum(departure * (matrix_m @ departure))
        else:
            start = 6 * tissue_index
            departure = values[start : start + 6] - reference[start : start + 6]
            term = departure.square().sum()
        result[f"{tissue}_reference_departure"] = term
        bulk_terms.append(term)
    parameterization = shared.config["parameterization"]
    skin_baseline_index = int(parameterization["skin_isotropic_resultant_coordinate"])
    skin_log_multiplier_index = int(
        parameterization["skin_log_stiffness_multiplier_coordinate"]
    )
    skin_baseline = (
        values[skin_baseline_index] - reference[skin_baseline_index]
    ).square()
    skin_log_multiplier = (
        values[skin_log_multiplier_index] - reference[skin_log_multiplier_index]
    ).square()
    result["bulk_reference_departure"] = torch.stack(bulk_terms).sum()
    result["skin_baseline_reference_departure"] = skin_baseline
    result["skin_log_multiplier_reference_departure"] = skin_log_multiplier
    result["total"] = (
        result["bulk_reference_departure"] + skin_baseline + skin_log_multiplier
    )
    return result


@torch.no_grad()
def projected_gradient_mapping(
    parameters: torch.Tensor,
    gradient: torch.Tensor,
    *,
    step_size: float,
    project: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Return ``(x - projection(x - eta * grad)) / eta``.

    The mapping is zero at a first-order constrained stationary point. The
    caller must keep ``step_size`` fixed across a reported convergence trace.
    """
    if parameters.shape != gradient.shape:
        msg = "parameters and gradient must have identical shapes"
        raise ValueError(msg)
    if not math.isfinite(step_size) or step_size <= 0.0:
        msg = "projected-gradient step size must be finite and positive"
        raise ValueError(msg)
    if not bool(torch.isfinite(parameters).all()):
        msg = "parameters must be finite"
        raise ValueError(msg)
    if not bool(torch.isfinite(gradient).all()):
        msg = "gradient must be finite"
        raise ValueError(msg)
    proposal = parameters.detach() - step_size * gradient.detach()
    projected = project(proposal.clone())
    if projected.shape != parameters.shape:
        msg = "projection changed the parameter shape"
        raise ValueError(msg)
    if not bool(torch.isfinite(projected).all()):
        msg = "projection returned nonfinite values"
        raise ValueError(msg)
    return (parameters.detach() - projected) / step_size


def objective_stability_summary(
    values: list[float],
    *,
    window: int,
    relative_span_tolerance: float,
) -> dict[str, float | bool | None]:
    """Report objective stabilization without a nonfinite incomplete-window sentinel."""
    if window <= 0:
        msg = "objective stabilization window must be positive"
        raise ValueError(msg)
    if not math.isfinite(relative_span_tolerance) or relative_span_tolerance < 0:
        msg = "objective relative-span tolerance must be finite and nonnegative"
        raise ValueError(msg)
    if not values or len(values) > window:
        msg = "objective window must contain between one and window values"
        raise ValueError(msg)
    if not all(math.isfinite(value) for value in values):
        msg = "objective window values must be finite"
        raise ValueError(msg)
    relative_span = None
    if len(values) == window:
        relative_span = (max(values) - min(values)) / max(
            1.0e-30, *(abs(value) for value in values)
        )
    return {
        "objective_relative_span": relative_span,
        "objective_stable": (
            relative_span is not None and relative_span <= relative_span_tolerance
        ),
    }


def relative_reduction_summary(
    current: float, initial: float, *, tolerance: float
) -> dict[str, float | bool | None]:
    """Compare a norm with its initial value without an epsilon denominator."""
    if not math.isfinite(current) or current < 0.0:
        msg = "current norm must be finite and nonnegative"
        raise ValueError(msg)
    if not math.isfinite(initial) or initial < 0.0:
        msg = "initial norm must be finite and nonnegative"
        raise ValueError(msg)
    if not math.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
        msg = "relative-reduction tolerance must be finite and in [0, 1)"
        raise ValueError(msg)
    if initial == 0.0:
        return {
            "relative_to_initial": 0.0 if current == 0.0 else None,
            "relative_reduction_met": current == 0.0,
        }
    relative = current / initial
    return {
        "relative_to_initial": relative,
        "relative_reduction_met": relative <= tolerance,
    }


def reversible_projected_optimizer_preview(
    *,
    optimizer: torch.optim.Optimizer,
    parameters: tuple[torch.nn.Parameter, ...],
    prepare_step: Callable[[], None],
    project: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Preview one exact optimizer step and projection, then restore all owned state."""
    accepted_parameters = tuple(value.detach().clone() for value in parameters)
    accepted_gradients = tuple(
        None if value.grad is None else value.grad.detach().clone()
        for value in parameters
    )
    optimizer_state = deepcopy(optimizer.state_dict())
    try:
        prepare_step()
        optimizer.step()
        projection = project()
        steps = tuple(
            value.detach().clone() - accepted
            for value, accepted in zip(parameters, accepted_parameters, strict=True)
        )
        directional_derivative = sum(
            float(torch.sum(gradient * step))
            for gradient, step in zip(accepted_gradients, steps, strict=True)
            if gradient is not None
        )
    finally:
        with torch.no_grad():
            for parameter, accepted in zip(
                parameters, accepted_parameters, strict=True
            ):
                parameter.copy_(accepted)
        for parameter, gradient in zip(parameters, accepted_gradients, strict=True):
            parameter.grad = None if gradient is None else gradient.detach().clone()
        optimizer.load_state_dict(optimizer_state)
    return {
        "steps": steps,
        "projection": projection,
        "directional_derivative": directional_derivative,
    }


@torch.no_grad()
def activation_projected_gradient_mapping(
    coordinates: torch.Tensor,
    gradient: torch.Tensor,
    *,
    step_size: float,
    maximum_dimensionless: float,
    batch_size: int = DEFAULT_EIGEN_BATCH_SIZE,
) -> torch.Tensor:
    """Projected-gradient mapping for dense PSD, capped activation tensors."""

    def project(proposal: torch.Tensor) -> torch.Tensor:
        project_activation_(
            proposal,
            maximum_dimensionless,
            batch_size=batch_size,
        )
        return proposal

    return projected_gradient_mapping(
        coordinates,
        gradient,
        step_size=step_size,
        project=project,
    )


@torch.no_grad()
def box_projected_gradient_mapping(
    parameters: torch.Tensor,
    gradient: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    step_size: float,
) -> torch.Tensor:
    """Projected-gradient mapping for broadcastable box constraints."""
    if not bool(torch.all(upper > lower)):
        msg = "every upper box limit must exceed its lower limit"
        raise ValueError(msg)

    def project(proposal: torch.Tensor) -> torch.Tensor:
        return torch.minimum(torch.maximum(proposal, lower), upper)

    return projected_gradient_mapping(
        parameters,
        gradient,
        step_size=step_size,
        project=project,
    )


@torch.no_grad()
def shared_projected_gradient_mapping(
    shared: SharedParameters,
    gradient: torch.Tensor,
    *,
    step_size: float,
) -> torch.Tensor:
    """Projected-gradient mapping for either admitted shared-field basis."""
    accepted = shared.coefficients.detach().clone()

    def project(proposal: torch.Tensor) -> torch.Tensor:
        try:
            shared.coefficients.copy_(proposal)
            shared.project_()
            return shared.coefficients.detach().clone()
        finally:
            shared.coefficients.copy_(accepted)

    return projected_gradient_mapping(
        shared.coefficients,
        gradient,
        step_size=step_size,
        project=project,
    )


def _rms(value: torch.Tensor) -> float:
    return float(value.square().mean().sqrt())


@torch.no_grad()
def activation_step_summary(
    step: torch.Tensor, effective_cell_volume_m3: torch.Tensor
) -> dict[str, Any]:
    """Summarize an actual dense activation parameter change."""
    if step.ndim < 2 or step.shape[-1] != 6:
        msg = "activation step must have shape (..., cells, 6)"
        raise ValueError(msg)
    if effective_cell_volume_m3.shape != (step.shape[-2],):
        msg = "effective cell volumes must match the activation cell count"
        raise ValueError(msg)
    if bool((effective_cell_volume_m3 < 0.0).any()):
        msg = "effective cell volumes cannot be negative"
        raise ValueError(msg)
    volume = effective_cell_volume_m3.sum()
    if not float(volume) > 0.0:
        msg = "effective tissue volume must be positive"
        raise ValueError(msg)
    weights = effective_cell_volume_m3 / volume
    rows = []
    for expression, field in enumerate(step.reshape(-1, step.shape[-2], 6)):
        tet_norm = torch.linalg.vector_norm(field, dim=-1)
        rows.append(
            {
                "expression": expression,
                "coordinate_rms": _rms(field),
                "effective_volume_tensor_rms": float(
                    torch.sqrt((weights * tet_norm.square()).sum())
                ),
                "maximum_tet_frobenius": float(tet_norm.max()),
            }
        )
    return {
        "schema": "joint-activation-optimizer-proposal-v1",
        "field_shape": list(step.shape),
        "expressions": rows,
    }


@torch.no_grad()
def parameter_step_summary(step: torch.Tensor) -> dict[str, Any]:
    """Summarize an actual small-block optimizer parameter change."""
    if step.ndim == 0:
        msg = "parameter step must have at least one dimension"
        raise ValueError(msg)
    rows = step.reshape(-1, step.shape[-1])
    row_norms = torch.linalg.vector_norm(rows, dim=-1)
    return {
        "schema": "joint-parameter-block-optimizer-proposal-v1",
        "shape": list(step.shape),
        "coordinate_rms": _rms(step),
        "maximum_row_norm": float(row_norms.max()),
        "maximum_absolute_coordinate": float(step.abs().max()),
        "row_norms": row_norms.cpu().tolist(),
    }


@torch.no_grad()
def mass_normalized_activation_gradient_summary(
    gradient: torch.Tensor, effective_cell_volume_m3: torch.Tensor
) -> dict[str, Any]:
    """Report the effective-volume Riesz gradient for interpretation only."""
    if gradient.ndim < 2 or gradient.shape[-1] != 6:
        msg = "activation gradient must have shape (..., cells, 6)"
        raise ValueError(msg)
    if effective_cell_volume_m3.shape != (gradient.shape[-2],):
        msg = "effective cell volumes must match the activation cell count"
        raise ValueError(msg)
    if bool((effective_cell_volume_m3 <= 0.0).any()):
        msg = "mass-normalized diagnostics require positive effective cell volumes"
        raise ValueError(msg)
    weights = effective_cell_volume_m3 / effective_cell_volume_m3.sum()
    rows = []
    for expression, field in enumerate(gradient.reshape(-1, gradient.shape[-2], 6)):
        riesz = field / effective_cell_volume_m3[:, None]
        tensor_norm = torch.linalg.vector_norm(riesz, dim=-1)
        rows.append(
            {
                "expression": expression,
                "effective_volume_riesz_tensor_rms": float(
                    torch.sqrt((weights * tensor_norm.square()).sum())
                ),
                "maximum_tet_riesz_frobenius": float(tensor_norm.max()),
            }
        )
    return {
        "schema": "joint-activation-mass-normalized-gradient-v1",
        "field_shape": list(gradient.shape),
        "normalization": "g_t divided by effective cell volume V_t; interpretive only",
        "used_for_convergence": False,
        "expressions": rows,
    }


@torch.no_grad()
def activation_mapping_summary(
    mapping: torch.Tensor,
    effective_cell_volume_m3: torch.Tensor,
    *,
    step_size: float,
) -> dict[str, Any]:
    """Summarize dense projected gradients without coordinate-count dilution."""
    if mapping.ndim < 2 or mapping.shape[-1] != 6:
        msg = "activation mapping must have shape (..., cells, 6)"
        raise ValueError(msg)
    if effective_cell_volume_m3.shape != (mapping.shape[-2],):
        msg = "effective cell volumes must match the activation cell count"
        raise ValueError(msg)
    if bool((effective_cell_volume_m3 < 0.0).any()):
        msg = "effective cell volumes cannot be negative"
        raise ValueError(msg)
    volume = effective_cell_volume_m3.sum()
    if not float(volume) > 0.0:
        msg = "effective tissue volume must be positive"
        raise ValueError(msg)
    fields = mapping.reshape(-1, mapping.shape[-2], 6)
    weights = effective_cell_volume_m3 / volume
    rows = []
    for expression, field in enumerate(fields):
        tet_norm = torch.linalg.vector_norm(field, dim=-1)
        volume_rms = torch.sqrt((weights * tet_norm.square()).sum())
        rows.append(
            {
                "expression": expression,
                "l2_norm": float(torch.linalg.vector_norm(field)),
                "coordinate_rms": _rms(field),
                "effective_volume_tensor_rms": float(volume_rms),
                "maximum_tet_frobenius": float(tet_norm.max()),
                "projected_step_effective_volume_tensor_rms": float(
                    step_size * volume_rms
                ),
                "projected_step_maximum_tet_frobenius": float(
                    step_size * tet_norm.max()
                ),
            }
        )
    return {
        "schema": "joint-activation-projected-gradient-v1",
        "field_shape": list(mapping.shape),
        "step_size": step_size,
        "expressions": rows,
    }


@torch.no_grad()
def parameter_block_mapping_summary(
    mapping: torch.Tensor,
    *,
    step_size: float,
) -> dict[str, Any]:
    """Summarize a small projected-gradient block by leading row."""
    if mapping.ndim == 0:
        msg = "parameter block mapping must have at least one dimension"
        raise ValueError(msg)
    rows = mapping.reshape(-1, mapping.shape[-1])
    norms = torch.linalg.vector_norm(rows, dim=-1)
    return {
        "schema": "joint-parameter-block-projected-gradient-v1",
        "shape": list(mapping.shape),
        "step_size": step_size,
        "l2_norm": float(torch.linalg.vector_norm(mapping)),
        "coordinate_rms": _rms(mapping),
        "maximum_row_norm": float(norms.max()),
        "projected_step_coordinate_rms": step_size * _rms(mapping),
        "projected_step_maximum_row_norm": float(step_size * norms.max()),
        "row_norms": norms.cpu().tolist(),
    }


@torch.no_grad()
def activation_spectrum_summary(  # noqa: C901 - validates and streams dense fields.
    coordinates: torch.Tensor,
    *,
    reference_mpa: float,
    maximum_dimensionless: float,
    effective_cell_volume_m3: torch.Tensor | None = None,
    batch_size: int = DEFAULT_EIGEN_BATCH_SIZE,
    occupancy_relative_tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Summarize dense activation spectra without materializing 3x3 fields.

    Quantiles are unweighted over principal stresses. Occupancy is reported both
    by eigenvalue count and, when cell volumes are supplied, by effective tissue
    volume over tetrahedra touching the lower or upper spectral bound.
    """
    if coordinates.ndim < 2 or coordinates.shape[-1] != 6:
        msg = "activation coordinates must have shape (..., cells, 6)"
        raise ValueError(msg)
    if reference_mpa <= 0.0 or maximum_dimensionless <= 0.0:
        msg = "activation reference and maximum must be positive"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch size must be positive"
        raise ValueError(msg)
    cells = coordinates.shape[-2]
    if effective_cell_volume_m3 is not None:
        if effective_cell_volume_m3.shape != (cells,):
            msg = "effective cell volumes must match the activation cell count"
            raise ValueError(msg)
        if bool((effective_cell_volume_m3 < 0.0).any()):
            msg = "effective cell volumes cannot be negative"
            raise ValueError(msg)
        if not float(effective_cell_volume_m3.sum()) > 0.0:
            msg = "effective tissue volume must be positive"
            raise ValueError(msg)

    fields = coordinates.reshape(-1, cells, 6)
    cap_mpa = reference_mpa * maximum_dimensionless
    bound_tolerance_mpa = max(
        cap_mpa * occupancy_relative_tolerance,
        torch.finfo(coordinates.dtype).eps * cap_mpa * 32.0,
    )
    rows = []
    for expression, field in enumerate(fields):
        eigenvalue_chunks = []
        lower_tet_chunks = []
        upper_tet_chunks = []
        for start in range(0, cells, batch_size):
            stop = min(start + batch_size, cells)
            eigenvalues = reference_mpa * torch.linalg.eigvalsh(
                symmetric_matrices(field[start:stop])
            )
            eigenvalue_chunks.append(eigenvalues.detach().cpu())
            lower_tet_chunks.append(
                (eigenvalues[:, 0] <= bound_tolerance_mpa).detach().cpu()
            )
            upper_tet_chunks.append(
                (eigenvalues[:, -1] >= cap_mpa - bound_tolerance_mpa).detach().cpu()
            )
        eigenvalues = torch.cat(eigenvalue_chunks)
        lower_tets = torch.cat(lower_tet_chunks)
        upper_tets = torch.cat(upper_tet_chunks)
        flat = eigenvalues.flatten()
        quantiles = torch.quantile(
            flat,
            flat.new_tensor([0.0, 0.5, 0.9, 0.99, 1.0]),
        )
        row: dict[str, Any] = {
            "expression": expression,
            "cells": cells,
            "coordinate_rms_dimensionless": _rms(field),
            "tensor_frobenius_rms_mpa": (
                reference_mpa * float(field.square().sum(dim=-1).mean().sqrt())
            ),
            "principal_stress_quantiles_mpa": {
                name: float(value)
                for name, value in zip(
                    ("minimum", "median", "p90", "p99", "maximum"),
                    quantiles,
                    strict=True,
                )
            },
            "lower_bound_eigenvalue_fraction": float(
                (flat <= bound_tolerance_mpa).to(torch.float64).mean()
            ),
            "upper_cap_eigenvalue_fraction": float(
                (flat >= cap_mpa - bound_tolerance_mpa).to(torch.float64).mean()
            ),
            "lower_bound_tet_fraction": float(lower_tets.to(torch.float64).mean()),
            "upper_cap_tet_fraction": float(upper_tets.to(torch.float64).mean()),
        }
        if effective_cell_volume_m3 is not None:
            weights = effective_cell_volume_m3.detach().cpu().to(torch.float64)
            denominator = weights.sum()
            row["lower_bound_effective_volume_fraction"] = float(
                weights[lower_tets].sum() / denominator
            )
            row["upper_cap_effective_volume_fraction"] = float(
                weights[upper_tets].sum() / denominator
            )
        rows.append(row)
    return {
        "schema": "joint-activation-spectrum-v1",
        "field_shape": list(coordinates.shape),
        "reference_mpa": reference_mpa,
        "maximum_dimensionless": maximum_dimensionless,
        "cap_mpa": cap_mpa,
        "occupancy_relative_tolerance": occupancy_relative_tolerance,
        "expressions": rows,
    }


@torch.no_grad()
def shared_prior_summary(
    shared: SharedParameters,
    *,
    reference_coefficients: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Report interpretable departures for either admitted shared-field basis."""
    coefficients = shared.coefficients.detach()
    if reference_coefficients is None:
        reference_coefficients = torch.zeros_like(coefficients)
    if reference_coefficients.shape != coefficients.shape:
        msg = "shared reference must match the coefficient vector"
        raise ValueError(msg)
    reference_coefficients = reference_coefficients.to(
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    constraints = shared.config["constraints"]
    epsilon = float(constraints["baseline_epsilon"])
    upper_multiple = float(constraints["baseline_upper_mu_multiple"])
    bulk = {}
    if isinstance(shared, SpatialSharedFieldParameters):
        regularizers = shared.regularizers()
        for index, name in enumerate(BULK_TISSUES):
            mu_mpa = shared.bulk_mu_mpa[index]
            block = shared.anchor_coordinates(name).detach()
            coordinate_slice = shared.config["parameterization"][
                "bulk_anchor_coordinate_slices"
            ][name]
            reference = reference_coefficients[
                coordinate_slice[0] : coordinate_slice[1]
            ].reshape_as(block)
            matrix_m = getattr(shared, f"_{name}_M")
            departure = block - reference
            normalized_frobenius = torch.sqrt(torch.sum(block * (matrix_m @ block)))
            reference_departure = torch.sqrt(
                torch.sum(departure * (matrix_m @ departure))
            )
            anchor_ratios = torch.linalg.eigvalsh(symmetric_matrices(block))
            bulk[name] = {
                "mu_mpa": mu_mpa,
                "anchors": block.shape[0],
                "normalized_frobenius": float(normalized_frobenius),
                "reference_departure_frobenius": float(reference_departure),
                "anchor_principal_stress_mpa": (mu_mpa * anchor_ratios).cpu().tolist(),
                "anchor_principal_stress_over_mu": anchor_ratios.cpu().tolist(),
                "lower_bound_margin_over_mu": float(
                    anchor_ratios.min() + 1.0 - epsilon
                ),
                "upper_bound_margin_over_mu": float(
                    upper_multiple - anchor_ratios.max()
                ),
                "spatial_roughness": float(regularizers[f"{name}_spatial_roughness"]),
                "volume_normalized_magnitude": float(regularizers[f"{name}_magnitude"]),
            }
    else:
        stresses = shared.bulk_stresses_mpa()
        for index, name in enumerate(BULK_TISSUES):
            mu_mpa = shared.bulk_mu_mpa[index]
            principal = torch.linalg.eigvalsh(stresses[index])
            ratios = principal / mu_mpa
            block = shared.bulk_coordinates[index]
            reference = reference_coefficients[6 * index : 6 * (index + 1)]
            bulk[name] = {
                "mu_mpa": mu_mpa,
                "anchors": 1,
                "normalized_frobenius": float(torch.linalg.vector_norm(block)),
                "reference_departure_frobenius": float(
                    torch.linalg.vector_norm(block - reference)
                ),
                "principal_stress_mpa": principal.cpu().tolist(),
                "principal_stress_over_mu": ratios.cpu().tolist(),
                "lower_bound_margin_over_mu": float(ratios.min() + 1.0 - epsilon),
                "upper_bound_margin_over_mu": float(upper_multiple - ratios.max()),
            }
    skin = shared.config["materials"]["skin"]
    proxy_n_per_m = float(skin["continuation"]["source_proxy_mean_n_per_m"])
    resultant_n_per_m = float(shared.skin_resultant_n_per_m()[0, 0])
    multiplier = float(shared.skin_stiffness_multiplier())
    regularizers = shared.regularizers()
    parameterization = shared.config["parameterization"]
    skin_baseline_index = int(parameterization["skin_isotropic_resultant_coordinate"])
    skin_log_multiplier_index = int(
        parameterization["skin_log_stiffness_multiplier_coordinate"]
    )
    return {
        "schema": "joint-shared-prior-summary-v1",
        "coefficient_count": coefficients.numel(),
        "coefficient_rms": _rms(coefficients),
        "reference_departure_rms": _rms(coefficients - reference_coefficients),
        "bulk": bulk,
        "skin": {
            "baseline_coordinate": float(shared.skin_baseline_coordinate),
            "resultant_n_per_m": resultant_n_per_m,
            "literature_proxy_n_per_m": proxy_n_per_m,
            "fraction_of_literature_proxy": resultant_n_per_m / proxy_n_per_m,
            "study_proxy_departure_n_per_m": resultant_n_per_m - proxy_n_per_m,
            "stiffness_multiplier": multiplier,
            "log_stiffness_multiplier": float(shared.skin_log_multiplier),
            "study_reference_stiffness_multiplier": 1.0,
            "study_reference_log_stiffness_multiplier": 0.0,
            "study_reference_log_stiffness_departure": float(
                shared.skin_log_multiplier
            ),
            "reference_departure_baseline_coordinate": float(
                shared.skin_baseline_coordinate
                - reference_coefficients[skin_baseline_index]
            ),
            "reference_departure_log_multiplier": float(
                shared.skin_log_multiplier
                - reference_coefficients[skin_log_multiplier_index]
            ),
        },
        "prior_blocks": {
            name: float(value)
            for name, value in regularizers.items()
            if name.endswith("_prior") or name == "prior_total"
        },
        "spatial_regularizers": {
            name: float(value)
            for name, value in regularizers.items()
            if value.ndim == 0
            and ("spatial_roughness" in name or "spatial_magnitude" in name)
        },
        "basis": (
            "spatial80"
            if isinstance(shared, SpatialSharedFieldParameters)
            else "constant20"
        ),
        "basis_limit": (
            "4/4/5 coarse bulk anchors, one uniform isotropic skin resultant, "
            "and one global skin stiffness multiplier"
            if isinstance(shared, SpatialSharedFieldParameters)
            else (
                "three spatially constant bulk tensors, one spatially constant "
                "isotropic skin resultant, and one global skin stiffness multiplier"
            )
        ),
        "spatial_basis": (
            shared.basis_receipt()
            if isinstance(shared, SpatialSharedFieldParameters)
            else None
        ),
    }


__all__ = [
    "activation_mapping_summary",
    "activation_projected_gradient_mapping",
    "activation_spectrum_summary",
    "activation_step_summary",
    "box_projected_gradient_mapping",
    "mass_normalized_activation_gradient_summary",
    "parameter_block_mapping_summary",
    "parameter_step_summary",
    "projected_gradient_mapping",
    "relative_reduction_summary",
    "reversible_projected_optimizer_preview",
    "shared_prior_summary",
    "shared_projected_gradient_mapping",
    "shared_reference_departure_quadratic",
]
