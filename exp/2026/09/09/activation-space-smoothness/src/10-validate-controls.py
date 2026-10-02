"""CPU validation for matched activation fields and smoothness."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Callable
from pathlib import Path

import torch
from activation_controls import (
    EIGEN_BATCH_SIZE,
    baseline_z,
    common_initial_controls,
    control_c,
    eigenvalues,
    learned_axis_b,
    learned_axis_c,
    learned_axis_raw6,
    learned_axis_tensile,
    learned_axis_z,
    project_psd_,
    raw6_c,
    raw6_matrices,
    smoothness,
    symmetric_coordinates,
    symmetric_matrices,
    tensile_z,
)

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    output_dir: Path = cherries.output("11-learned-axis-validation", mkdir=True)
    seed: int = 20260909
    finite_difference_step: float = 1.0e-6
    smooth_length_m: float = 0.005


def maximum_absolute(value: torch.Tensor) -> float:
    return float(value.detach().abs().max())


def scalar(value: torch.Tensor) -> float:
    return float(value.detach())


def directional_check(
    q: torch.Tensor,
    direction: torch.Tensor,
    objective: Callable[[torch.Tensor], torch.Tensor],
    step: float,
) -> dict[str, float]:
    q = q.detach().clone().requires_grad_()
    value = objective(q)
    (gradient,) = torch.autograd.grad(value, q)
    autodiff = (gradient * direction).sum()
    finite_difference = (
        objective(q.detach() + step * direction)
        - objective(q.detach() - step * direction)
    ) / (2.0 * step)
    absolute_error = (autodiff - finite_difference).abs()
    scale = torch.maximum(
        torch.stack((autodiff.abs(), finite_difference.abs())).max(),
        q.new_tensor(1.0),
    )
    return {
        "autodiff": scalar(autodiff),
        "finite_difference": scalar(finite_difference),
        "absolute_error": scalar(absolute_error),
        "scaled_error": scalar(absolute_error / scale),
    }


def map_directional_check(
    q: torch.Tensor,
    direction: torch.Tensor,
    mapping: Callable[[torch.Tensor], torch.Tensor],
    step: float,
) -> dict[str, float]:
    _, tangent = torch.autograd.functional.jvp(mapping, q, direction)
    finite_difference = (
        mapping(q + step * direction) - mapping(q - step * direction)
    ) / (2.0 * step)
    error = tangent - finite_difference
    return {
        "maximum_absolute_error": maximum_absolute(error),
        "tangent_norm": scalar(torch.linalg.vector_norm(tangent)),
        "finite_difference_norm": scalar(torch.linalg.vector_norm(finite_difference)),
    }


def field_checks(cfg: Config, generator: torch.Generator) -> dict[str, object]:
    count = 5
    direction = torch.randn((count, 6), generator=generator)
    direction /= torch.linalg.vector_norm(direction)
    probe = torch.randn((count, 3, 3), generator=generator)
    probe = 0.5 * (probe + probe.transpose(-1, -2))
    i = torch.tensor((0, 1, 2, 3), dtype=torch.long)
    j = torch.tensor((1, 2, 3, 4), dtype=torch.long)
    edge_weight = torch.tensor((0.7, 1.1, 0.9, 1.4))
    active_volume = 2.3e-6

    def objective(
        mapping: Callable[[torch.Tensor], torch.Tensor], q: torch.Tensor
    ) -> torch.Tensor:
        z = mapping(q)
        return (probe * z).sum() + 0.4 * smoothness(
            z,
            i,
            j,
            edge_weight,
            smooth_length_m=cfg.smooth_length_m,
            active_volume=active_volume,
        )

    states = {
        "zero": torch.zeros((count, 6)),
        "nonzero": 0.08 * torch.randn((count, 6), generator=generator),
    }
    result: dict[str, object] = {}
    for name, q in states.items():
        result[name] = {}
        for model, mapping in (("baseline", baseline_z), ("tensile", tensile_z)):
            result[name][model] = {
                "map": map_directional_check(
                    q, direction, mapping, cfg.finite_difference_step
                ),
                "objective": directional_check(
                    q,
                    direction,
                    lambda value, mapping=mapping: objective(mapping, value),
                    cfg.finite_difference_step,
                ),
            }

    zero_z = torch.zeros((count, 3, 3), requires_grad=True)
    zero_smoothness = smoothness(
        zero_z,
        i,
        j,
        edge_weight,
        smooth_length_m=cfg.smooth_length_m,
        active_volume=active_volume,
    )
    (zero_gradient,) = torch.autograd.grad(zero_smoothness, zero_z)
    nonzero_z = tensile_z(states["nonzero"].requires_grad_())
    nonzero_smoothness = smoothness(
        nonzero_z,
        i,
        j,
        edge_weight,
        smooth_length_m=cfg.smooth_length_m,
        active_volume=active_volume,
    )
    (nonzero_gradient,) = torch.autograd.grad(nonzero_smoothness, states["nonzero"])
    result["smoothness"] = {
        "zero_value": scalar(zero_smoothness),
        "zero_gradient_max_abs": maximum_absolute(zero_gradient),
        "nonzero_value": scalar(nonzero_smoothness),
        "nonzero_gradient_norm": scalar(torch.linalg.vector_norm(nonzero_gradient)),
    }
    return result


def coordinate_and_projection_checks(
    generator: torch.Generator,
) -> dict[str, object]:
    q = torch.randn((EIGEN_BATCH_SIZE + 3, 6), generator=generator)
    coordinate_error = symmetric_coordinates(symmetric_matrices(q)) - q
    coordinate_norm_error = symmetric_matrices(q).square().sum(
        dim=(-2, -1)
    ) - q.square().sum(-1)
    values, vectors = torch.linalg.eigh(symmetric_matrices(q))
    expected_chunked = symmetric_coordinates(
        (vectors * values.clamp_min(0.0)[:, None, :]) @ vectors.transpose(-1, -2)
    )
    chunked = q.clone()
    chunked_stats = project_psd_(chunked)

    matrices = torch.stack(
        (
            torch.diag(torch.tensor((-1.0, 2.0, 7.0))),
            torch.tensor(((0.4, -0.8, 0.0), (-0.8, 0.4, 0.0), (0.0, 0.0, 5.0))),
        )
    )
    before = torch.linalg.eigvalsh(matrices)
    projected = symmetric_coordinates(matrices)
    explicit_stats = project_psd_(projected)
    after = eigenvalues(tensile_z(projected))
    idempotent = projected.clone()
    idempotent_stats = project_psd_(idempotent)
    expected = before.clamp_min(0.0)
    return {
        "coordinate_roundtrip_max_abs": maximum_absolute(coordinate_error),
        "coordinate_frobenius_isometry_max_abs": maximum_absolute(
            coordinate_norm_error
        ),
        "chunked_count": len(q),
        "chunked_projection_max_abs": maximum_absolute(chunked - expected_chunked),
        "chunked_projection": chunked_stats,
        "input_eigenvalues": before.tolist(),
        "output_eigenvalues": after.tolist(),
        "negative_only_projection_max_abs": maximum_absolute(after - expected),
        "largest_positive_eigenvalue_preserved": scalar(after.max()),
        "projection": explicit_stats,
        "idempotence_max_abs": maximum_absolute(idempotent - projected),
        "idempotent_projection_rms": idempotent_stats["projection_rms"],
    }


def learned_axis_checks(cfg: Config, generator: torch.Generator) -> dict[str, object]:
    count = 5
    direction = torch.randn((count, 3), generator=generator)
    direction /= torch.linalg.vector_norm(direction)
    probe = torch.randn((count, 3, 3), generator=generator)
    probe = 0.5 * (probe + probe.transpose(-1, -2))
    i = torch.tensor((0, 1, 2, 3), dtype=torch.long)
    j = torch.tensor((1, 2, 3, 4), dtype=torch.long)
    edge_weight = torch.tensor((0.7, 1.1, 0.9, 1.4))
    active_volume = 2.3e-6

    def objective(v: torch.Tensor) -> torch.Tensor:
        z = learned_axis_z(v)
        return (probe * z).sum() + 0.4 * smoothness(
            z,
            i,
            j,
            edge_weight,
            smooth_length_m=cfg.smooth_length_m,
            active_volume=active_volume,
        )

    states = {
        "zero": torch.zeros((count, 3)),
        "nonzero": 0.08 * torch.randn((count, 3), generator=generator),
    }
    derivatives: dict[str, object] = {}
    for name, v in states.items():
        derivatives[name] = {
            "z_map": map_directional_check(
                v, direction, learned_axis_z, cfg.finite_difference_step
            ),
            "raw6_map": map_directional_check(
                v, direction, learned_axis_raw6, cfg.finite_difference_step
            ),
            "tensile_map": map_directional_check(
                v, direction, learned_axis_tensile, cfg.finite_difference_step
            ),
            "objective": directional_check(
                v, direction, objective, cfg.finite_difference_step
            ),
        }

    v = states["nonzero"]
    b = learned_axis_b(v)
    z = learned_axis_z(v)
    raw_b = raw6_matrices(learned_axis_raw6(v))
    raw_z = baseline_z(learned_axis_raw6(v))
    tensile = tensile_z(learned_axis_tensile(v))
    squared_norm = v.square().sum(-1)
    b_spectrum = torch.linalg.eigvalsh(b)
    z_spectrum = torch.linalg.eigvalsh(z)
    contraction_spectrum = torch.linalg.eigvalsh(torch.linalg.inv(b))
    expected_b = torch.stack(
        (
            torch.ones_like(squared_norm),
            torch.ones_like(squared_norm),
            1 + squared_norm,
        ),
        dim=-1,
    )
    expected_z = torch.stack(
        (
            torch.zeros_like(squared_norm),
            torch.zeros_like(squared_norm),
            (2 + squared_norm) * squared_norm,
        ),
        dim=-1,
    )
    expected_contraction = torch.stack(
        (
            1 / (1 + squared_norm),
            torch.ones_like(squared_norm),
            torch.ones_like(squared_norm),
        ),
        dim=-1,
    )
    return {
        "derivatives": derivatives,
        "conversion": {
            "raw6_B_max_abs": maximum_absolute(raw_b - b),
            "raw6_Z_max_abs": maximum_absolute(raw_z - z),
            "tensile_Z_max_abs": maximum_absolute(tensile - z),
        },
        "rank_one_spectrum": {
            "B_max_abs": maximum_absolute(b_spectrum - expected_b),
            "Z_max_abs": maximum_absolute(z_spectrum - expected_z),
            "active_contraction_max_abs": maximum_absolute(
                contraction_spectrum - expected_contraction
            ),
            "minimum_Z_eigenvalue": scalar(z_spectrum.min()),
            "positive_Z_eigenvalues_per_cell": (z_spectrum > 1.0e-12).sum(-1).tolist(),
        },
    }


def raw6_coordinates(matrix: torch.Tensor) -> torch.Tensor:
    """Pack a symmetric ``C = B - I`` in historical unscaled coordinates."""
    assert matrix.ndim == 3
    assert matrix.shape[1:] == (3, 3)
    assert torch.equal(matrix, matrix.transpose(-1, -2))
    return torch.stack(
        (
            matrix[:, 0, 0],
            matrix[:, 1, 1],
            matrix[:, 2, 2],
            matrix[:, 0, 1],
            matrix[:, 1, 2],
            matrix[:, 0, 2],
        ),
        dim=-1,
    )


def control_c_checks(  # noqa: PLR0915
    cfg: Config, generator: torch.Generator
) -> dict[str, object]:
    """Validate the optimized C field for learned-axis and Raw6 controls."""
    count = 5
    i = torch.tensor((0, 1, 2, 3), dtype=torch.long)
    j = torch.tensor((1, 2, 3, 4), dtype=torch.long)
    edge_weight = torch.tensor((0.7, 1.1, 0.9, 1.4))
    active_volume = 2.3e-6
    raw = 0.08 * torch.randn((count, 6), generator=generator)
    learned = 0.08 * torch.randn((count, 3), generator=generator)
    raw_direction = torch.randn(raw.shape, generator=generator)
    learned_direction = torch.randn(learned.shape, generator=generator)
    raw_direction /= torch.linalg.vector_norm(raw_direction)
    learned_direction /= torch.linalg.vector_norm(learned_direction)

    def penalty(value: torch.Tensor, model: str) -> torch.Tensor:
        return smoothness(
            control_c(value, model),
            i,
            j,
            edge_weight,
            cfg.smooth_length_m,
            active_volume,
        )

    derivatives = {
        "raw6": {
            "map": map_directional_check(
                raw,
                raw_direction,
                lambda value: control_c(value, "raw6"),
                cfg.finite_difference_step,
            ),
            "smoothness": directional_check(
                raw,
                raw_direction,
                lambda value: penalty(value, "raw6"),
                cfg.finite_difference_step,
            ),
        },
        "learned-axis": {
            "map": map_directional_check(
                learned,
                learned_direction,
                lambda value: control_c(value, "learned-axis"),
                cfg.finite_difference_step,
            ),
            "smoothness": directional_check(
                learned,
                learned_direction,
                lambda value: penalty(value, "learned-axis"),
                cfg.finite_difference_step,
            ),
        },
    }

    raw_field = raw6_c(raw)
    raw_identity_error = baseline_z(raw) - (2.0 * raw_field + raw_field @ raw_field)
    learned_field = learned_axis_c(learned)
    learned_identity_error = (
        learned_axis_z(learned)
        - (2.0 + learned_field.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None])
        * learned_field
    )

    learned_for_gradient = learned.detach().clone().requires_grad_()
    c = learned_axis_c(learned_for_gradient)
    value = smoothness(
        c,
        i,
        j,
        edge_weight,
        cfg.smooth_length_m,
        active_volume,
    )
    (autodiff_gradient,) = torch.autograd.grad(value, learned_for_gradient)
    delta = c[i] - c[j]
    factor = 4.0 * cfg.smooth_length_m**2 / active_volume
    analytic_gradient = torch.zeros_like(learned_for_gradient)
    analytic_gradient.index_add_(
        0,
        i,
        factor
        * edge_weight[:, None]
        * (delta @ learned_for_gradient[i, :, None]).squeeze(-1),
    )
    analytic_gradient.index_add_(
        0,
        j,
        factor
        * edge_weight[:, None]
        * ((-delta) @ learned_for_gradient[j, :, None]).squeeze(-1),
    )

    direct_field = torch.zeros((3, 3, 3))
    direct_field[1, 0, 0] = 1.0
    direct_field[2, 0, 0] = 1.0
    direct_field[2, 1, 1] = 2.0
    direct_i = torch.tensor((0, 1), dtype=torch.long)
    direct_j = torch.tensor((1, 2), dtype=torch.long)
    direct_weight = torch.tensor((0.7, 1.1))
    direct_value = smoothness(
        direct_field,
        direct_i,
        direct_j,
        direct_weight,
        cfg.smooth_length_m,
        active_volume,
    )
    direct_expected = cfg.smooth_length_m**2 / active_volume * (0.7 * 1.0 + 1.1 * 4.0)

    labels = torch.tensor((0, 0, 1, 1, 2), dtype=torch.int64)
    constant_v = common_initial_controls(
        labels, "learned-axis", seed=cfg.seed, strength=0.001
    ).requires_grad_()
    constant_i = torch.tensor((0, 2), dtype=torch.long)
    constant_j = torch.tensor((1, 3), dtype=torch.long)
    constant_value = smoothness(
        learned_axis_c(constant_v),
        constant_i,
        constant_j,
        torch.tensor((0.8, 1.2)),
        cfg.smooth_length_m,
        active_volume,
    )
    (constant_gradient,) = torch.autograd.grad(constant_value, constant_v)

    branch_b = torch.stack(
        (
            torch.diag(torch.tensor((1.2, 0.9, 1.1))),
            torch.diag(torch.tensor((1.1, 0.8, 1.05))),
        )
    )
    branch_q = raw6_coordinates(branch_b - torch.eye(3))
    flipped_b = branch_b.clone()
    flipped_b[0] *= -1.0
    flipped_q = raw6_coordinates(flipped_b - torch.eye(3))
    branch_i = torch.tensor((0,), dtype=torch.long)
    branch_j = torch.tensor((1,), dtype=torch.long)
    branch_weight = torch.ones(1)
    branch_smoothness = smoothness(
        raw6_c(branch_q),
        branch_i,
        branch_j,
        branch_weight,
        cfg.smooth_length_m,
        active_volume,
    )
    flipped_smoothness = smoothness(
        raw6_c(flipped_q),
        branch_i,
        branch_j,
        branch_weight,
        cfg.smooth_length_m,
        active_volume,
    )

    return {
        "derivatives": derivatives,
        "identities": {
            "raw6_Z_equals_2C_plus_C2_max_abs": maximum_absolute(raw_identity_error),
            "learned_axis_Z_equals_2_plus_traceC_times_C_max_abs": (
                maximum_absolute(learned_identity_error)
            ),
            "learned_axis_B_equals_I_plus_C_max_abs": maximum_absolute(
                learned_axis_b(learned) - torch.eye(3) - learned_field
            ),
        },
        "analytic_learned_axis_gradient_max_abs": maximum_absolute(
            autodiff_gradient - analytic_gradient
        ),
        "direct_normalization": {
            "value": scalar(direct_value),
            "expected": direct_expected,
            "absolute_error": abs(scalar(direct_value) - direct_expected),
        },
        "constant_per_muscle_start": {
            "value": scalar(constant_value),
            "gradient_max_abs": maximum_absolute(constant_gradient),
            "nonzero_C": scalar(learned_axis_c(constant_v).abs().max()),
        },
        "learned_axis_sign_invariance": {
            "C_max_abs": maximum_absolute(
                learned_axis_c(learned) - learned_axis_c(-learned)
            ),
            "Z_max_abs": maximum_absolute(
                learned_axis_z(learned) - learned_axis_z(-learned)
            ),
            "smoothness_abs": scalar(
                (
                    penalty(learned, "learned-axis") - penalty(-learned, "learned-axis")
                ).abs()
            ),
        },
        "raw6_sign_branch": {
            "Z_max_abs": maximum_absolute(baseline_z(branch_q) - baseline_z(flipped_q)),
            "C_max_abs": maximum_absolute(raw6_c(branch_q) - raw6_c(flipped_q)),
            "smoothness_original": scalar(branch_smoothness),
            "smoothness_flipped": scalar(flipped_smoothness),
            "smoothness_abs_difference": scalar(
                (branch_smoothness - flipped_smoothness).abs()
            ),
        },
    }


def common_initialization_checks(cfg: Config) -> dict[str, object]:
    labels = torch.tensor((0, 0, 1, 2, 2, 1), dtype=torch.int64)
    strength = 0.001
    v = common_initial_controls(
        labels, "learned-axis", seed=cfg.seed, strength=strength
    )
    raw = common_initial_controls(labels, "raw6", seed=cfg.seed, strength=strength)
    tensile = common_initial_controls(
        labels, "tensor", seed=cfg.seed, strength=strength
    )
    repeated = common_initial_controls(
        labels, "learned-axis", seed=cfg.seed, strength=strength
    )
    alternate = common_initial_controls(
        labels, "learned-axis", seed=20260910, strength=strength
    )
    learned_z = learned_axis_z(v)
    raw_z = baseline_z(raw)
    tensile_field = tensile_z(tensile)
    axes = v / math.sqrt(strength)
    unique_axes = axes[torch.tensor((0, 2, 3))]
    same_label_error = torch.stack(
        (axes[0] - axes[1], axes[2] - axes[5], axes[3] - axes[4])
    )
    return {
        "seed": cfg.seed,
        "alternate_seed": 20260910,
        "strength_s_equals_norm_v_squared": strength,
        "expected_Z_positive_eigenvalue": (2.0 + strength) * strength,
        "axis_by_label": unique_axes.tolist(),
        "axis_sha256": hashlib.sha256(unique_axes.numpy().tobytes()).hexdigest(),
        "axis_norm_max_abs_error": maximum_absolute(
            torch.linalg.vector_norm(unique_axes, dim=-1) - 1.0
        ),
        "same_label_axis_max_abs_error": maximum_absolute(same_label_error),
        "repeat_bitwise_equal": torch.equal(v, repeated),
        "alternate_seed_max_abs_difference": maximum_absolute(v - alternate),
        "raw6_B_max_abs": maximum_absolute(raw6_matrices(raw) - learned_axis_b(v)),
        "raw6_Z_max_abs": maximum_absolute(raw_z - learned_z),
        "tensile_Z_max_abs": maximum_absolute(tensile_field - learned_z),
        "learned_strength_max_abs_error": maximum_absolute(
            v.square().sum(-1) - strength
        ),
        "anatomical_fiber_used": False,
    }


def default_cuda_initialization_check(cfg: Config) -> dict[str, object]:
    """Exercise CPU-seeded initialization while Torch defaults to CUDA."""
    available = torch.cuda.is_available()
    if not available:
        return {"cuda_available": False, "exercised": False}
    labels = torch.tensor((0, 0, 1, 2, 2, 1), dtype=torch.int64, device="cpu")
    expected = common_initial_controls(
        labels,
        "learned-axis",
        seed=cfg.seed,
        strength=0.001,
        device="cpu",
    )
    original = torch.get_default_device()
    try:
        torch.set_default_device("cuda")
        actual = common_initial_controls(
            labels,
            "learned-axis",
            seed=cfg.seed,
            strength=0.001,
        )
    finally:
        torch.set_default_device(original)
    return {
        "cuda_available": True,
        "exercised": True,
        "output_device": str(actual.device),
        "bitwise_equal_to_cpu_default": torch.equal(actual, expected),
        "maximum_absolute_error": maximum_absolute(actual - expected),
    }


def material_equivalence_check(generator: torch.Generator) -> dict[str, float]:
    """Check Q=mu(BB^T-I) with a physical determinant bulk term."""
    mu = 0.013
    lambda_code = 0.18
    q = 0.1 * torch.randn((1, 6), generator=generator)
    b = raw6_matrices(q)[0]
    z = baseline_z(q)[0]
    physical_q = mu * z
    f = (
        torch.eye(3) + 0.08 * torch.randn((3, 3), generator=generator)
    ).requires_grad_()

    def bulk(value: torch.Tensor) -> torch.Tensor:
        determinant = torch.linalg.det(value)
        return -mu * (determinant - 1.0) + 0.5 * lambda_code * (determinant - 1.0) ** 2

    def baseline_energy(value: torch.Tensor) -> torch.Tensor:
        return 0.5 * mu * ((value @ b).square().sum() - 3.0) + bulk(value)

    def tensile_energy(value: torch.Tensor) -> torch.Tensor:
        c = value.transpose(-1, -2) @ value
        identity = torch.eye(3, dtype=value.dtype, device=value.device)
        return (
            0.5 * mu * (value.square().sum() - 3.0)
            + bulk(value)
            + 0.5 * (physical_q * (c - identity)).sum()
        )

    baseline_value = baseline_energy(f)
    tensile_value = tensile_energy(f)
    (baseline_force,) = torch.autograd.grad(baseline_value, f, create_graph=True)
    (tensile_force,) = torch.autograd.grad(tensile_value, f, create_graph=True)
    baseline_hessian = torch.autograd.functional.hessian(baseline_energy, f)
    tensile_hessian = torch.autograd.functional.hessian(tensile_energy, f)
    expected_offset = 0.5 * torch.trace(physical_q)
    return {
        "energy_offset_abs_error": scalar(
            (baseline_value - tensile_value - expected_offset).abs()
        ),
        "force_max_abs_error": maximum_absolute(baseline_force - tensile_force),
        "hessian_max_abs_error": maximum_absolute(baseline_hessian - tensile_hessian),
        "energy_offset": scalar(expected_offset),
        "det_f": scalar(torch.linalg.det(f)),
        "physical_active_stress_eigenvalue_min": scalar(
            (mu * eigenvalues(z[None]))[0, 0]
        ),
        "physical_active_stress_eigenvalue_max": scalar(
            (mu * eigenvalues(z[None]))[0, -1]
        ),
    }


def learned_axis_material_equivalence_check(
    generator: torch.Generator,
) -> dict[str, float]:
    """Check the learned-axis Raw6 law against its equivalent tensile stress."""
    mu = 0.013
    lambda_code = 0.18
    v = 0.1 * torch.randn((1, 3), generator=generator)
    b = learned_axis_b(v)[0]
    raw_b = raw6_matrices(learned_axis_raw6(v))[0]
    z = learned_axis_z(v)[0]
    tensile_field = tensile_z(learned_axis_tensile(v))[0]
    physical_q = mu * z
    f = (
        torch.eye(3) + 0.08 * torch.randn((3, 3), generator=generator)
    ).requires_grad_()

    def bulk(value: torch.Tensor) -> torch.Tensor:
        determinant = torch.linalg.det(value)
        return -mu * (determinant - 1.0) + 0.5 * lambda_code * (determinant - 1.0) ** 2

    def learned_energy(value: torch.Tensor) -> torch.Tensor:
        return 0.5 * mu * ((value @ b).square().sum() - 3.0) + bulk(value)

    def raw6_energy(value: torch.Tensor) -> torch.Tensor:
        return 0.5 * mu * ((value @ raw_b).square().sum() - 3.0) + bulk(value)

    def tensile_energy(value: torch.Tensor) -> torch.Tensor:
        c = value.transpose(-1, -2) @ value
        identity = torch.eye(3, dtype=value.dtype, device=value.device)
        return (
            0.5 * mu * (value.square().sum() - 3.0)
            + bulk(value)
            + 0.5 * (physical_q * (c - identity)).sum()
        )

    learned_value = learned_energy(f)
    raw6_value = raw6_energy(f)
    tensile_value = tensile_energy(f)
    (learned_force,) = torch.autograd.grad(learned_value, f, create_graph=True)
    (raw6_force,) = torch.autograd.grad(raw6_value, f, create_graph=True)
    (tensile_force,) = torch.autograd.grad(tensile_value, f, create_graph=True)
    learned_hessian = torch.autograd.functional.hessian(learned_energy, f)
    raw6_hessian = torch.autograd.functional.hessian(raw6_energy, f)
    tensile_hessian = torch.autograd.functional.hessian(tensile_energy, f)
    expected_offset = 0.5 * torch.trace(physical_q)
    return {
        "B_conversion_max_abs_error": maximum_absolute(raw_b - b),
        "Z_conversion_max_abs_error": maximum_absolute(tensile_field - z),
        "raw6_energy_max_abs_error": scalar((learned_value - raw6_value).abs()),
        "tensile_energy_offset_abs_error": scalar(
            (learned_value - tensile_value - expected_offset).abs()
        ),
        "raw6_force_max_abs_error": maximum_absolute(learned_force - raw6_force),
        "tensile_force_max_abs_error": maximum_absolute(learned_force - tensile_force),
        "raw6_hessian_max_abs_error": maximum_absolute(learned_hessian - raw6_hessian),
        "tensile_hessian_max_abs_error": maximum_absolute(
            learned_hessian - tensile_hessian
        ),
        "det_f": scalar(torch.linalg.det(f)),
        "s": scalar(v.square().sum()),
    }


def validate(result: dict[str, object]) -> None:  # noqa: PLR0915
    projection = result["coordinates_and_projection"]
    assert projection["coordinate_roundtrip_max_abs"] < 1.0e-14
    assert projection["coordinate_frobenius_isometry_max_abs"] < 1.0e-12
    assert projection["chunked_count"] > EIGEN_BATCH_SIZE
    assert projection["chunked_projection_max_abs"] < 1.0e-13
    assert projection["negative_only_projection_max_abs"] < 1.0e-14
    assert projection["largest_positive_eigenvalue_preserved"] > 6.99
    assert projection["idempotence_max_abs"] < 1.0e-14

    fields = result["fields"]
    for state in ("zero", "nonzero"):
        for model in ("baseline", "tensile"):
            assert fields[state][model]["map"]["maximum_absolute_error"] < 1.0e-9
            assert fields[state][model]["objective"]["scaled_error"] < 1.0e-8
    assert fields["zero"]["tensile"]["map"]["tangent_norm"] > 0.0
    assert fields["smoothness"]["zero_value"] == 0.0
    assert fields["smoothness"]["zero_gradient_max_abs"] == 0.0
    assert fields["smoothness"]["nonzero_value"] > 0.0
    assert fields["smoothness"]["nonzero_gradient_norm"] > 0.0

    material = result["material_equivalence"]
    assert material["energy_offset_abs_error"] < 1.0e-14
    assert material["force_max_abs_error"] < 1.0e-14
    assert material["hessian_max_abs_error"] < 1.0e-13

    learned = result["learned_axis"]
    for state in ("zero", "nonzero"):
        for mapping in ("z_map", "raw6_map", "tensile_map"):
            assert (
                learned["derivatives"][state][mapping]["maximum_absolute_error"]
                < 1.0e-9
            )
        assert learned["derivatives"][state]["objective"]["scaled_error"] < 1.0e-8
    assert learned["derivatives"]["zero"]["z_map"]["tangent_norm"] == 0.0
    assert learned["derivatives"]["nonzero"]["z_map"]["tangent_norm"] > 0.0
    assert learned["conversion"]["raw6_B_max_abs"] < 1.0e-14
    assert learned["conversion"]["raw6_Z_max_abs"] < 1.0e-14
    assert learned["conversion"]["tensile_Z_max_abs"] < 1.0e-14
    spectrum = learned["rank_one_spectrum"]
    assert spectrum["B_max_abs"] < 1.0e-14
    assert spectrum["Z_max_abs"] < 1.0e-14
    assert spectrum["active_contraction_max_abs"] < 1.0e-14
    assert spectrum["minimum_Z_eigenvalue"] > -1.0e-14
    assert spectrum["positive_Z_eigenvalues_per_cell"] == [1] * 5

    initial = result["common_initialization"]
    assert initial["axis_norm_max_abs_error"] < 1.0e-14
    assert initial["same_label_axis_max_abs_error"] == 0.0
    assert initial["repeat_bitwise_equal"] is True
    assert initial["alternate_seed_max_abs_difference"] > 0.01
    assert initial["raw6_B_max_abs"] < 1.0e-14
    assert initial["raw6_Z_max_abs"] < 1.0e-14
    assert initial["tensile_Z_max_abs"] < 1.0e-14
    assert initial["learned_strength_max_abs_error"] < 1.0e-14
    assert initial["anatomical_fiber_used"] is False

    cuda_default = result["default_cuda_initialization"]
    if cuda_default["cuda_available"]:
        assert cuda_default["exercised"] is True
        assert cuda_default["output_device"] == "cpu"
        assert cuda_default["bitwise_equal_to_cpu_default"] is True
        assert cuda_default["maximum_absolute_error"] == 0.0

    learned_material = result["learned_axis_material_equivalence"]
    assert learned_material["B_conversion_max_abs_error"] < 1.0e-14
    assert learned_material["Z_conversion_max_abs_error"] < 1.0e-14
    assert learned_material["raw6_energy_max_abs_error"] < 1.0e-14
    assert learned_material["tensile_energy_offset_abs_error"] < 1.0e-14
    assert learned_material["raw6_force_max_abs_error"] < 1.0e-14
    assert learned_material["tensile_force_max_abs_error"] < 1.0e-14
    assert learned_material["raw6_hessian_max_abs_error"] < 1.0e-13
    assert learned_material["tensile_hessian_max_abs_error"] < 1.0e-13

    c_fields = result["control_C"]
    for model in ("raw6", "learned-axis"):
        assert c_fields["derivatives"][model]["map"]["maximum_absolute_error"] < 1.0e-9
        assert c_fields["derivatives"][model]["smoothness"]["scaled_error"] < 1.0e-8
    identities = c_fields["identities"]
    assert identities["raw6_Z_equals_2C_plus_C2_max_abs"] < 1.0e-14
    assert identities["learned_axis_Z_equals_2_plus_traceC_times_C_max_abs"] < 1.0e-14
    assert identities["learned_axis_B_equals_I_plus_C_max_abs"] < 1.0e-14
    assert c_fields["analytic_learned_axis_gradient_max_abs"] < 1.0e-12
    assert c_fields["direct_normalization"]["absolute_error"] < 1.0e-14
    constant = c_fields["constant_per_muscle_start"]
    assert constant["nonzero_C"] > 0.0
    assert constant["value"] == 0.0
    assert constant["gradient_max_abs"] == 0.0
    sign = c_fields["learned_axis_sign_invariance"]
    assert sign["C_max_abs"] == 0.0
    assert sign["Z_max_abs"] == 0.0
    assert sign["smoothness_abs"] == 0.0
    branch = c_fields["raw6_sign_branch"]
    assert branch["Z_max_abs"] < 1.0e-14
    assert branch["C_max_abs"] > 1.0
    assert branch["smoothness_abs_difference"] > 1.0


def main(cfg: Config) -> None:
    assert cfg.finite_difference_step > 0.0
    assert cfg.smooth_length_m == 0.005
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    result = {
        "status": "passed",
        "device": "cpu",
        "dtype": str(torch.get_default_dtype()),
        "finite_difference_step": cfg.finite_difference_step,
        "smooth_length_m": cfg.smooth_length_m,
        "coordinates_and_projection": coordinate_and_projection_checks(generator),
        "fields": field_checks(cfg, generator),
        "material_equivalence": material_equivalence_check(generator),
        "learned_axis": learned_axis_checks(cfg, generator),
        "common_initialization": common_initialization_checks(cfg),
        "default_cuda_initialization": default_cuda_initialization_check(cfg),
        "learned_axis_material_equivalence": (
            learned_axis_material_equivalence_check(generator)
        ),
        "control_C": control_c_checks(cfg, generator),
    }
    validate(result)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    output = cfg.output_dir / "summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    cherries.log_metrics(
        {
            "validation/field_fd_max": max(
                result["fields"][state][model]["objective"]["scaled_error"]
                for state in ("zero", "nonzero")
                for model in ("baseline", "tensile")
            ),
            "validation/force_error": result["material_equivalence"][
                "force_max_abs_error"
            ],
            "validation/hessian_error": result["material_equivalence"][
                "hessian_max_abs_error"
            ],
            "validation/learned_axis_field_fd_max": max(
                result["learned_axis"]["derivatives"][state]["objective"][
                    "scaled_error"
                ]
                for state in ("zero", "nonzero")
            ),
            "validation/learned_axis_force_error": result[
                "learned_axis_material_equivalence"
            ]["tensile_force_max_abs_error"],
            "validation/learned_axis_hessian_error": result[
                "learned_axis_material_equivalence"
            ]["tensile_hessian_max_abs_error"],
            "validation/control_C_fd_max": max(
                result["control_C"]["derivatives"][model]["smoothness"]["scaled_error"]
                for model in ("raw6", "learned-axis")
            ),
            "validation/control_C_gradient_error": result["control_C"][
                "analytic_learned_axis_gradient_max_abs"
            ],
        }
    )
    LOG.info("Wrote passing CPU validation to %s", output)


if __name__ == "__main__":
    cherries.main(main)
