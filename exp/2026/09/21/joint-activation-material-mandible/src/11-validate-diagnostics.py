"""CPU validation for scale-aware joint-optimization diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pydantic_settings as ps
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, write_json
from joint_diagnostics import (
    activation_mapping_summary,
    activation_projected_gradient_mapping,
    activation_spectrum_summary,
    activation_step_summary,
    box_projected_gradient_mapping,
    mass_normalized_activation_gradient_summary,
    objective_stability_summary,
    parameter_block_mapping_summary,
    parameter_step_summary,
    projected_gradient_mapping,
    relative_reduction_summary,
    shared_prior_summary,
    shared_projected_gradient_mapping,
    shared_reference_departure_quadratic,
)
from joint_fields import SharedFieldParameters
from joint_spatial_fields import (
    EXPECTED_MESH_CELL_COUNT,
    SpatialSharedFieldParameters,
    constant20_to_spatial80,
)

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("diagnostics-validation", mkdir=True)
    spatial_basis_path: Path = (
        GROUP / "data/spatial-baseline-audit-005/basis-and-normal-equations.npz"
    )
    spatial_audit_summary_path: Path = (
        GROUP / "data/spatial-baseline-audit-005/summary.json"
    )


def main(  # noqa: PLR0915 - one receipt validates every diagnostics family.
    cfg: Config,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64
    checks: dict[str, float] = {}

    parameters = torch.tensor([-1.0, 0.5, 2.0], dtype=dtype)
    gradient = torch.tensor([0.3, -0.2, 0.7], dtype=dtype)
    unconstrained = projected_gradient_mapping(
        parameters,
        gradient,
        step_size=0.25,
        project=lambda value: value,
    )
    checks["unconstrained_mapping_error"] = float(
        (unconstrained - gradient).abs().max()
    )

    lower = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
    upper = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
    box_parameters = torch.tensor([0.0, 0.5, 1.0], dtype=dtype)
    box_gradient = torch.tensor([2.0, -0.2, -3.0], dtype=dtype)
    box = box_projected_gradient_mapping(
        box_parameters,
        box_gradient,
        lower,
        upper,
        step_size=0.1,
    )
    checks["box_kkt_mapping_error"] = float(
        (box - torch.tensor([0.0, -0.2, 0.0], dtype=dtype)).abs().max()
    )

    activation = torch.zeros((2, 4097, 6), dtype=dtype)
    activation[..., 0] = 2.0
    activation[..., 1] = 4.0
    activation[..., 2] = 10.0
    activation_gradient = torch.zeros_like(activation)
    activation_gradient[..., 0] = -0.5
    activation_gradient[..., 2] = -2.0
    activation_mapping = activation_projected_gradient_mapping(
        activation,
        activation_gradient,
        step_size=0.1,
        maximum_dimensionless=10.0,
        batch_size=257,
    )
    checks["activation_cap_mapping_error"] = float(
        activation_mapping[..., 2].abs().max()
    )
    checks["activation_interior_mapping_error"] = float(
        (activation_mapping[..., 0] + 0.5).abs().max()
    )
    volume = torch.linspace(1.0, 2.0, 4097, dtype=dtype)
    mapping_summary = activation_mapping_summary(
        activation_mapping,
        volume,
        step_size=0.1,
    )
    expected_tet_norm = 0.5
    checks["mapping_max_tet_error"] = abs(
        mapping_summary["expressions"][0]["maximum_tet_frobenius"] - expected_tet_norm
    )
    activation_step = 0.1 * activation_mapping
    activation_step_receipt = activation_step_summary(activation_step, volume)
    checks["activation_step_max_tet_error"] = abs(
        activation_step_receipt["expressions"][0]["maximum_tet_frobenius"]
        - 0.1 * expected_tet_norm
    )

    coarse_volume = torch.tensor([2.0, 1.0], dtype=dtype)
    coarse_density = torch.tensor(
        [
            [[1.0, -2.0, 0.5, 0.25, -0.5, 0.75], [0.5] * 6],
            [[-0.25, 0.5, 1.0, -1.5, 0.25, 2.0], [1.5] * 6],
        ],
        dtype=dtype,
    )
    coarse_gradient = coarse_density * coarse_volume[None, :, None]
    coarse_mass = mass_normalized_activation_gradient_summary(
        coarse_gradient, coarse_volume
    )
    refinement = 8
    refined_volume = coarse_volume.repeat_interleave(refinement) / refinement
    refined_gradient = coarse_gradient.repeat_interleave(refinement, dim=1) / refinement
    refined_mass = mass_normalized_activation_gradient_summary(
        refined_gradient, refined_volume
    )
    for coarse_row, refined_row in zip(
        coarse_mass["expressions"], refined_mass["expressions"], strict=True
    ):
        checks["mass_refinement_rms_error"] = max(
            checks.get("mass_refinement_rms_error", 0.0),
            abs(
                coarse_row["effective_volume_riesz_tensor_rms"]
                - refined_row["effective_volume_riesz_tensor_rms"]
            ),
        )
        checks["mass_refinement_max_error"] = max(
            checks.get("mass_refinement_max_error", 0.0),
            abs(
                coarse_row["maximum_tet_riesz_frobenius"]
                - refined_row["maximum_tet_riesz_frobenius"]
            ),
        )
    checks["raw_refinement_coordinate_rms_ratio_error"] = abs(
        float(refined_gradient.square().mean().sqrt())
        / float(coarse_gradient.square().mean().sqrt())
        - 1.0 / refinement
    )
    checks["raw_refinement_l2_ratio_error"] = abs(
        float(torch.linalg.vector_norm(refined_gradient))
        / float(torch.linalg.vector_norm(coarse_gradient))
        - 1.0 / math.sqrt(refinement)
    )

    spectrum = activation_spectrum_summary(
        activation,
        reference_mpa=0.012,
        maximum_dimensionless=10.0,
        effective_cell_volume_m3=volume,
        batch_size=257,
    )
    checks["spectrum_max_error_mpa"] = abs(
        spectrum["expressions"][0]["principal_stress_quantiles_mpa"]["maximum"] - 0.12
    )
    checks["spectrum_cap_occupancy_error"] = abs(
        spectrum["expressions"][0]["upper_cap_eigenvalue_fraction"] - 1.0 / 3.0
    )
    checks["spectrum_cap_tet_occupancy_error"] = abs(
        spectrum["expressions"][0]["upper_cap_tet_fraction"] - 1.0
    )

    jaw_mapping = torch.tensor([[3.0, 4.0], [0.0, 5.0]], dtype=dtype)
    jaw_summary = parameter_block_mapping_summary(jaw_mapping, step_size=0.01)
    checks["small_block_max_row_error"] = abs(jaw_summary["maximum_row_norm"] - 5.0)
    jaw_step_summary = parameter_step_summary(0.01 * jaw_mapping)
    checks["small_block_step_max_row_error"] = abs(
        jaw_step_summary["maximum_row_norm"] - 0.05
    )

    shared = SharedFieldParameters(dtype=dtype)
    with torch.no_grad():
        shared.coefficients[0] = 0.5
        shared.coefficients[18] = 0.1
        shared.coefficients[19] = math.log(1.25)
    shared_summary = shared_prior_summary(shared)
    checks["shared_fat_norm_error"] = abs(
        shared_summary["bulk"]["fat"]["normalized_frobenius"] - 0.5
    )
    checks["shared_skin_resultant_error_n_per_m"] = abs(
        shared_summary["skin"]["resultant_n_per_m"]
        - 0.1 * shared.skin_resultant_scale_n_per_m
    )
    checks["shared_skin_multiplier_error"] = abs(
        shared_summary["skin"]["stiffness_multiplier"] - 1.25
    )
    shared_gradient = torch.zeros_like(shared.coefficients)
    shared_gradient[0] = -0.25
    shared_mapping = shared_projected_gradient_mapping(
        shared,
        shared_gradient,
        step_size=0.1,
    )
    checks["shared_mapping_error"] = float((shared_mapping[0] + 0.25).abs())

    spatial = SpatialSharedFieldParameters(
        cfg.spatial_basis_path,
        cfg.spatial_audit_summary_path,
        EXPECTED_MESH_CELL_COUNT,
        device="cpu",
        dtype=dtype,
    )
    spatial.load_constant20_(shared.coefficients)
    spatial_before = spatial.coefficients.detach().clone()
    spatial_summary = shared_prior_summary(spatial)
    assert spatial_summary["basis"] == "spatial80"
    assert spatial_summary["coefficient_count"] == 80
    assert spatial_summary["bulk"]["fat"]["anchors"] == 4
    checks["spatial_constant_fat_norm_error"] = abs(
        spatial_summary["bulk"]["fat"]["normalized_frobenius"] - 0.5
    )
    checks["spatial_constant_roughness_error"] = abs(
        spatial_summary["spatial_regularizers"]["bulk_spatial_roughness"]
    )
    spatial_gradient = torch.zeros_like(spatial.coefficients)
    spatial_gradient[0] = -0.25
    spatial_mapping = shared_projected_gradient_mapping(
        spatial,
        spatial_gradient,
        step_size=0.1,
    )
    checks["spatial_shared_mapping_error"] = float((spatial_mapping[0] + 0.25).abs())
    checks["spatial_mapping_restore_error"] = float(
        (spatial.coefficients.detach() - spatial_before).abs().max()
    )
    constant_values = torch.linspace(-0.3, 0.4, 20, dtype=dtype).requires_grad_()
    constant_reference = torch.linspace(0.2, -0.1, 20, dtype=dtype)
    constant_departure = shared_reference_departure_quadratic(
        shared,
        constant_reference,
        coefficients=constant_values,
    )["total"]
    spatial_values = constant20_to_spatial80(constant_values)
    spatial_reference = constant20_to_spatial80(constant_reference)
    spatial_departure = shared_reference_departure_quadratic(
        spatial,
        spatial_reference,
        coefficients=spatial_values,
    )["total"]
    constant_gradient = torch.autograd.grad(
        constant_departure, constant_values, retain_graph=True
    )[0]
    spatial_pulled_back_gradient = torch.autograd.grad(
        spatial_departure, constant_values
    )[0]
    checks["spatial_prior_constant_embedding_value_error"] = abs(
        float(spatial_departure.detach() - constant_departure.detach())
    )
    checks["spatial_prior_constant_embedding_gradient_error"] = float(
        (spatial_pulled_back_gradient - constant_gradient).abs().max()
    )

    incomplete_stability = objective_stability_summary(
        [1.0], window=5, relative_span_tolerance=1.0e-4
    )
    assert incomplete_stability == {
        "objective_relative_span": None,
        "objective_stable": False,
    }
    json.dumps(incomplete_stability, allow_nan=False)
    complete_stability = objective_stability_summary(
        [1.0, 0.99999, 0.99998, 0.99997, 0.99996],
        window=5,
        relative_span_tolerance=1.0e-4,
    )
    assert complete_stability["objective_stable"] is True
    assert complete_stability["objective_relative_span"] is not None
    checks["objective_full_window_span_error"] = abs(
        complete_stability["objective_relative_span"] - 4.0e-5
    )
    json.dumps(complete_stability, allow_nan=False)
    zero_reduction = relative_reduction_summary(0.0, 0.0, tolerance=0.01)
    assert zero_reduction == {
        "relative_to_initial": 0.0,
        "relative_reduction_met": True,
    }
    nonzero_from_zero = relative_reduction_summary(1.0e-30, 0.0, tolerance=0.01)
    assert nonzero_from_zero == {
        "relative_to_initial": None,
        "relative_reduction_met": False,
    }
    json.dumps(nonzero_from_zero, allow_nan=False)

    tolerance = 1.0e-12
    assert max(checks.values()) <= tolerance, checks
    receipt = {
        "success": True,
        "scope": "CPU algebra and chunking checks; no FEM or optimization solve",
        "checks": checks,
        "tolerance": tolerance,
        "activation_shape": list(activation.shape),
        "activation_spectrum": spectrum,
        "activation_mapping": mapping_summary,
        "activation_optimizer_step": activation_step_receipt,
        "mass_normalized_gradient": {
            "coarse": coarse_mass,
            "refined": refined_mass,
        },
        "relative_reduction": {
            "zero_from_zero": zero_reduction,
            "nonzero_from_zero": nonzero_from_zero,
        },
        "shared_prior": shared_summary,
        "spatial_shared_prior": spatial_summary,
        "objective_stability": {
            "incomplete_window": incomplete_stability,
            "complete_window": complete_stability,
        },
    }
    archive_sources(cfg.output_dir)
    write_json(cfg.output_dir / "summary.json", receipt)
    cherries.log_metrics({"diagnostics/maximum_error": max(checks.values())})
    print(receipt)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
