"""CPU validation for joint baseline-stress and activation field helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import torch
from joint_common import ProfileJoint
from joint_fields import (
    APPROVED_FIELD_CONFIG,
    BULK_TISSUES,
    N_SHARED_COEFFICIENTS,
    SharedFieldParameters,
    VolumeGraph,
    activation_regularizers,
    activation_stresses_mpa,
    project_activation_,
    research_informed_material_config,
    symmetric_coordinates,
    symmetric_matrices,
)

from liblaf import cherries

logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("field-validation", mkdir=True)
    seed: int = 20260921
    dense_cells: int = 288_235
    dense_edges: int = 501_409


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(axis: torch.Tensor, angle: float) -> torch.Tensor:
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis
    skew = torch.stack(
        (
            torch.stack((x.new_zeros(()), -z, y)),
            torch.stack((z, x.new_zeros(()), -x)),
            torch.stack((-y, x, x.new_zeros(()))),
        )
    )
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return identity + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(actual - expected)
    denominator = torch.maximum(
        torch.linalg.vector_norm(expected), expected.new_tensor(1.0e-14)
    )
    return float(numerator / denominator)


def check_metric(
    checks: list[dict[str, Any]], name: str, value: float, limit: float
) -> None:
    passed = math.isfinite(value) and value <= limit
    checks.append({"name": name, "value": value, "limit": limit, "passed": passed})
    if not passed:
        msg = f"{name}: {value} exceeds {limit}"
        raise AssertionError(msg)


def small_checks(  # noqa: C901, PLR0915 - one receipt records all field contracts.
    checks: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = torch.float64

    q = torch.randn((11, 6), generator=generator, dtype=dtype)
    matrices = symmetric_matrices(q)
    check_metric(
        checks,
        "coordinate_round_trip",
        float((symmetric_coordinates(matrices) - q).abs().max()),
        1.0e-14,
    )
    check_metric(
        checks,
        "frobenius_orthonormality",
        float((matrices.square().sum((-2, -1)) - q.square().sum(-1)).abs().max()),
        1.0e-13,
    )

    transform = rotation(torch.tensor((1.0, -2.0, 0.5), dtype=dtype), 0.71)
    rotated = transform @ matrices @ transform.T
    check_metric(
        checks,
        "rotation_preserves_coordinate_norm",
        float(
            (symmetric_coordinates(rotated).square().sum(-1) - q.square().sum(-1))
            .abs()
            .max()
        ),
        2.0e-13,
    )

    maximum = 2.5
    projected = q.clone()
    project_activation_(projected, maximum, batch_size=4)
    eigenvalues = torch.linalg.eigvalsh(symmetric_matrices(projected))
    check_metric(
        checks,
        "activation_projection_lower",
        float((-eigenvalues).clamp_min(0).max()),
        2.0e-14,
    )
    check_metric(
        checks,
        "activation_projection_upper",
        float((eigenvalues - maximum).clamp_min(0).max()),
        2.0e-14,
    )
    idempotent = projected.clone()
    project_activation_(idempotent, maximum, batch_size=3)
    check_metric(
        checks,
        "activation_projection_idempotence",
        float((idempotent - projected).abs().max()),
        2.0e-14,
    )
    rotated_before = symmetric_coordinates(transform @ matrices @ transform.T)
    project_activation_(rotated_before, maximum, batch_size=5)
    rotated_after = symmetric_coordinates(
        transform @ symmetric_matrices(projected) @ transform.T
    )
    check_metric(
        checks,
        "activation_projection_rotation_covariance",
        float((rotated_before - rotated_after).abs().max()),
        2.0e-13,
    )

    graph = VolumeGraph(
        i=torch.tensor((0, 1, 2, 3, 4, 5), dtype=torch.long),
        j=torch.tensor((1, 2, 3, 4, 5, 6), dtype=torch.long),
        conductance_m=torch.tensor((0.2, 0.3, 0.4, 0.5, 0.6, 0.7), dtype=dtype),
        effective_cell_volume_m3=torch.tensor(
            (0.4, 0.7, 0.5, 0.9, 0.8, 0.6, 0.3), dtype=dtype
        ),
        smooth_length_m=0.005,
    )
    fields = torch.randn((2, 7, 6), generator=generator, dtype=dtype)
    result = activation_regularizers(fields, graph)
    delta = fields[..., graph.i, :] - fields[..., graph.j, :]
    manual_smooth = (
        graph.smooth_length_m**2
        / graph.tissue_volume_m3
        * (graph.conductance_m * delta.square().sum(-1)).sum(-1)
    )
    manual_magnitude = (
        graph.effective_cell_volume_m3
        / graph.tissue_volume_m3
        * fields.square().sum(-1)
    ).sum(-1)
    check_metric(
        checks,
        "smoothness_formula",
        float((result["smoothness_by_field"] - manual_smooth).abs().max()),
        1.0e-14,
    )
    check_metric(
        checks,
        "magnitude_formula",
        float((result["magnitude_by_field"] - manual_magnitude).abs().max()),
        1.0e-14,
    )

    rotated_fields = symmetric_coordinates(
        transform @ symmetric_matrices(fields) @ transform.T
    )
    rotated_regularizers = activation_regularizers(rotated_fields, graph)
    check_metric(
        checks,
        "smoothness_rotation_invariance",
        float((rotated_regularizers["smoothness"] - result["smoothness"]).abs()),
        1.0e-13,
    )
    check_metric(
        checks,
        "magnitude_rotation_invariance",
        float((rotated_regularizers["magnitude"] - result["magnitude"]).abs()),
        1.0e-13,
    )

    differentiable = fields[0].clone().requires_grad_()
    objective_terms = activation_regularizers(differentiable, graph)
    objective = objective_terms["smoothness"] + 0.3 * objective_terms["magnitude"]
    gradient = torch.autograd.grad(objective, differentiable)[0]
    direction = torch.randn(differentiable.shape, generator=generator, dtype=dtype)
    direction /= torch.linalg.vector_norm(direction)
    step = 1.0e-6
    with torch.no_grad():
        plus_terms = activation_regularizers(differentiable + step * direction, graph)
        minus_terms = activation_regularizers(differentiable - step * direction, graph)
        finite_difference = (
            plus_terms["smoothness"]
            + 0.3 * plus_terms["magnitude"]
            - minus_terms["smoothness"]
            - 0.3 * minus_terms["magnitude"]
        ) / (2.0 * step)
    analytic = (gradient * direction).sum()
    check_metric(
        checks,
        "regularizer_directional_derivative",
        relative_error(analytic, finite_difference),
        2.0e-9,
    )

    shared = SharedFieldParameters(dtype=dtype)
    if shared.coefficients.numel() != N_SHARED_COEFFICIENTS:
        msg = "shared parameter vector does not have 20 coefficients"
        raise AssertionError(msg)
    with torch.no_grad():
        shared.bulk_coordinates.copy_(
            torch.tensor(
                (
                    (-5.0, 12.0, 0.4, 2.0, -1.0, 0.5),
                    (11.0, -4.0, 0.2, 0.7, -0.9, 1.1),
                    (-2.0, -3.0, 15.0, 1.0, 0.2, -0.5),
                ),
                dtype=dtype,
            )
        )
        shared.skin_baseline_coordinate.copy_(torch.tensor(-9.0, dtype=dtype))
        shared.skin_log_multiplier.copy_(torch.tensor(4.0, dtype=dtype))
    shared.project_()
    constraints = APPROVED_FIELD_CONFIG["constraints"]
    epsilon = float(constraints["baseline_epsilon"])
    upper_multiple = float(constraints["baseline_upper_mu_multiple"])
    for index, name in enumerate(BULK_TISSUES):
        mu = shared.bulk_mu_mpa[index]
        values = torch.linalg.eigvalsh(shared.bulk_stresses_mpa()[index].detach())
        check_metric(
            checks,
            f"{name}_signed_projection_lower",
            float((-(1.0 - epsilon) * mu - values).clamp_min(0).max()),
            2.0e-13,
        )
        check_metric(
            checks,
            f"{name}_signed_projection_upper",
            float((values - upper_multiple * mu).clamp_min(0).max()),
            2.0e-13,
        )
    multiplier = float(shared.skin_stiffness_multiplier().detach())
    skin_coordinate = float(shared.skin_baseline_coordinate.detach())
    if not (
        -(1.0 - epsilon) * multiplier <= skin_coordinate <= upper_multiple * multiplier
    ):
        msg = "skin baseline projection violated its stiffness-scaled bound"
        raise AssertionError(msg)
    regularizers = shared.regularizers()
    if float(regularizers["bulk_spatial_smoothness"]) != 0.0:
        msg = "constant bulk field reported nonzero smoothness"
        raise AssertionError(msg)
    if regularizers["bulk_spatial_smoothness"].requires_grad:
        msg = "constant bulk smoothness manufactured a coordinate gradient"
        raise AssertionError(msg)
    if float(regularizers["skin_spatial_smoothness"]) != 0.0:
        msg = "constant skin field reported nonzero smoothness"
        raise AssertionError(msg)

    zero_shared = SharedFieldParameters(dtype=dtype)
    multiplier_value = zero_shared.skin_stiffness_multiplier()
    multiplier_gradient = torch.autograd.grad(
        multiplier_value, zero_shared.coefficients, retain_graph=True
    )[0][19]
    check_metric(
        checks,
        "skin_positive_multiplier_derivative",
        float((multiplier_gradient - multiplier_value).detach().abs()),
        1.0e-14,
    )

    reference_mpa = zero_shared.activation_reference_mpa
    physical = activation_stresses_mpa(fields, reference_mpa)
    check_metric(
        checks,
        "activation_physical_normalization",
        float(
            (
                physical.square().sum((-2, -1))
                - reference_mpa**2 * fields.square().sum(-1)
            )
            .abs()
            .max()
        ),
        1.0e-15,
    )

    frozen = research_informed_material_config()
    if frozen is APPROVED_FIELD_CONFIG:
        msg = "configuration helper did not return an owned copy"
        raise AssertionError(msg)
    for name in (*BULK_TISSUES, "skin"):
        material = frozen["materials"][name]
        if not material.get("source") or not any("status" in key for key in material):
            msg = f"{name} configuration lacks source/status labels"
            raise AssertionError(msg)
    if "no measured" not in frozen["materials"]["aponeurosis"]["baseline_stress_prior"]:
        msg = "aponeurosis baseline stress was not labeled as unmeasured"
        raise AssertionError(msg)
    constraints = frozen["constraints"]
    check_metric(
        checks,
        "activation_cap_configuration",
        abs(
            float(constraints["activation_cap_mpa"])
            - float(constraints["activation_reference_mpa"])
            * float(constraints["activation_maximum_dimensionless"])
        ),
        1.0e-15,
    )
    check_metric(
        checks,
        "skin_resultant_unit_conversion",
        float(
            (
                zero_shared.skin_resultant_mpa_m()
                - zero_shared.skin_resultant_n_per_m() * 1.0e-6
            )
            .detach()
            .abs()
            .max()
        ),
        1.0e-15,
    )

    return {
        "activation_reference_mpa": reference_mpa,
        "activation_cap_mpa": zero_shared.activation_cap_mpa,
        "skin_resultant_scale_n_per_m": zero_shared.skin_resultant_scale_n_per_m,
        "skin_resultant_scale_mpa_m": zero_shared.skin_resultant_scale_mpa_m,
        "shared_coefficient_count": shared.coefficients.numel(),
        "small_graph_edges": graph.i.numel(),
    }


def dense_check(checks: list[dict[str, Any]], cells: int, edges: int) -> dict[str, Any]:
    dtype = torch.float64
    cell_id = torch.arange(cells, dtype=dtype)
    basis = torch.tensor((1.0, -0.7, 0.4, 0.3, -0.2, 0.1), dtype=dtype)
    coordinates = (1.0e-3 * torch.sin(0.001 * cell_id))[:, None] * basis[None, :]
    coordinates.requires_grad_()
    edge_id = torch.arange(edges, dtype=torch.long)
    i = edge_id.remainder(cells)
    j = (i + 1 + torch.div(edge_id, cells, rounding_mode="floor")).remainder(cells)
    conductance = torch.full((edges,), 1.0e-3, dtype=dtype)
    tissue_volume = 1.0e-3
    effective_volume = torch.full((cells,), tissue_volume / cells, dtype=dtype)
    graph = VolumeGraph(
        i=i,
        j=j,
        conductance_m=conductance,
        effective_cell_volume_m3=effective_volume,
        smooth_length_m=0.005,
    )
    started = time.perf_counter()
    values = activation_regularizers(coordinates, graph)
    objective = values["smoothness"] + values["magnitude"]
    objective.backward()
    elapsed = time.perf_counter() - started
    if coordinates.grad is None or not bool(torch.isfinite(coordinates.grad).all()):
        msg = "dense activation regularizer returned no finite gradient"
        raise AssertionError(msg)
    if coordinates.grad.shape != coordinates.shape:
        msg = "dense activation gradient shape changed"
        raise AssertionError(msg)
    checks.append(
        {
            "name": "dense_cpu_regularizer_finite_gradient",
            "value": 0.0,
            "limit": 0.0,
            "passed": True,
        }
    )
    return {
        "cells": cells,
        "edges": edges,
        "coordinate_count": coordinates.numel(),
        "elapsed_seconds": elapsed,
        "smoothness": float(values["smoothness"].detach()),
        "magnitude": float(values["magnitude"].detach()),
        "gradient_rms": float(coordinates.grad.square().mean().sqrt()),
        "coordinate_bytes": coordinates.numel() * coordinates.element_size(),
    }


def main(cfg: Config) -> None:
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    small = small_checks(checks, cfg.seed)
    dense = dense_check(checks, cfg.dense_cells, cfg.dense_edges)
    if not all(bool(row["passed"]) for row in checks):
        msg = "one or more field validation checks failed"
        raise AssertionError(msg)

    config_path = cfg.output_dir / "research-informed-material-config.json"
    checks_path = cfg.output_dir / "checks.csv"
    summary_path = cfg.output_dir / "summary.json"
    write_json(config_path, research_informed_material_config())
    with checks_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("name", "value", "limit", "passed"))
        writer.writeheader()
        writer.writerows(checks)
    source_path = Path(__file__).with_name("joint_fields.py")
    summary = {
        "success": True,
        "status": "passed_cpu_field_validation",
        "scope": (
            "field coordinates, signed/PSD projections, priors, physical graph "
            "regularization, rotation covariance, and dense CPU gradients; no FEM solve"
        ),
        "seed": cfg.seed,
        "checks": {row["name"]: row for row in checks},
        "small": small,
        "dense": dense,
        "sources": {
            str(source_path): sha256(source_path),
            str(Path(__file__)): sha256(Path(__file__)),
        },
        "outputs": {
            "checks": str(checks_path),
            "configuration": str(config_path),
        },
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
    }
    write_json(summary_path, summary)
    cherries.log_metrics(
        {
            "field_validation/check_count": len(checks),
            "field_validation/max_normalized_error": max(
                row["value"] / row["limit"] for row in checks if row["limit"] > 0.0
            ),
            "field_validation/dense_elapsed_seconds": dense["elapsed_seconds"],
            "field_validation/dense_gradient_rms": dense["gradient_rms"],
        }
    )
    logger.info("Wrote passed CPU field validation to %s", summary_path)


if __name__ == "__main__":
    profile = "debug" if os.getenv("DEBUG") else ProfileJoint
    cherries.main(main, profile=profile)
