"""CPU validation for the inactive 80-coordinate spatial field prototype."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import pydantic_settings as ps
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_fields import BULK_TISSUES, symmetric_matrices
from joint_spatial_fields import (
    ANCHOR_COUNTS,
    EXPECTED_BASIS_SHA256,
    EXPECTED_MESH_CELL_COUNT,
    N_SPATIAL_SHARED_COEFFICIENTS,
    SKIN_BASELINE_INDEX,
    SKIN_LOG_MULTIPLIER_INDEX,
    SpatialSharedFieldParameters,
    constant20_to_spatial80,
)

from liblaf import cherries

LOG = logging.getLogger(__name__)
COMPLETED = False


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    basis_path: Path = (
        GROUP / "data/spatial-baseline-audit-005/basis-and-normal-equations.npz"
    )
    audit_summary_path: Path = GROUP / "data/spatial-baseline-audit-005/summary.json"
    output_dir: Path
    seed: int = 20260921


def check(checks: list[dict[str, Any]], name: str, value: float, limit: float) -> None:
    passed = math.isfinite(value) and value <= limit
    checks.append({"name": name, "value": value, "limit": limit, "passed": passed})
    if not passed:
        msg = f"{name}: {value} exceeds {limit}"
        raise AssertionError(msg)


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1.0e-12)


def reconstruction_scalar(
    fields: SpatialSharedFieldParameters,
    probes: dict[str, torch.Tensor],
    rows: dict[str, torch.Tensor],
) -> torch.Tensor:
    value = fields.coefficients.new_zeros(())
    for tissue in BULK_TISSUES:
        coordinates = fields.bulk_dimensionless_support_coordinates(tissue)
        value = value + torch.sum(coordinates[rows[tissue]] * probes[tissue])
    return value


def validate_implicit_cuda_device(
    cfg: Config,
    constant: torch.Tensor,
    checks: list[dict[str, Any]],
) -> str:
    """Exercise the production implicit-device constructor on CUDA."""
    if not torch.cuda.is_available():
        msg = "CUDA is required for the implicit-default-device readiness check"
        raise RuntimeError(msg)
    original_device = torch.get_default_device()
    cuda_fields: SpatialSharedFieldParameters | None = None
    try:
        torch.set_default_device("cuda")
        cuda_fields = SpatialSharedFieldParameters(
            cfg.basis_path,
            cfg.audit_summary_path,
            EXPECTED_MESH_CELL_COUNT,
            dtype=torch.float64,
        )
        devices = {
            cuda_fields.coefficients.device.type,
            *(value.device.type for _, value in cuda_fields.named_buffers()),
        }
        check(
            checks,
            "implicit_default_cuda_device_consistency",
            float(devices != {"cuda"}),
            0.0,
        )
        cuda_fields.load_constant20_(constant)
        reconstructed = cuda_fields.bulk_dimensionless_support_coordinates("fat")
        expected = constant[:6].to(device="cuda", dtype=torch.float64)
        reconstruction_error = float(
            (reconstructed - expected).abs().max().detach().cpu()
        )
        check(
            checks,
            "implicit_default_cuda_reconstruction",
            reconstruction_error,
            2.0e-15,
        )
        assert cuda_fields.activation_reference_mpa > 0.0
        assert cuda_fields.activation_maximum_dimensionless > 0.0
        torch.cuda.synchronize()
        return str(cuda_fields.coefficients.device)
    finally:
        del cuda_fields
        torch.set_default_device(original_device)
        torch.cuda.empty_cache()


def main(cfg: Config) -> None:  # noqa: PLR0915 - one receipt validates one API.
    global COMPLETED  # noqa: PLW0603
    started = time.perf_counter()
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    checks: list[dict[str, Any]] = []
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    fields = SpatialSharedFieldParameters(
        cfg.basis_path,
        cfg.audit_summary_path,
        EXPECTED_MESH_CELL_COUNT,
        device="cpu",
        dtype=torch.float64,
    )
    receipt = fields.basis_receipt()
    assert receipt["basis_sha256"] == EXPECTED_BASIS_SHA256
    assert receipt["anchor_counts"] == ANCHOR_COUNTS
    assert receipt["mesh_cell_count"] == EXPECTED_MESH_CELL_COUNT
    assert list(fields.state_dict()) == ["coefficients"]

    constant = 0.2 * torch.randn(20, generator=generator)
    constant[18] = 0.37
    constant[19] = -0.08
    embedded = constant20_to_spatial80(constant)
    assert embedded.shape == (N_SPATIAL_SHARED_COEFFICIENTS,)
    fields.load_constant20_(constant)
    max_coordinate_error = 0.0
    for tissue_index, tissue in enumerate(BULK_TISSUES):
        expected = constant[6 * tissue_index : 6 * (tissue_index + 1)]
        reconstructed = fields.bulk_dimensionless_support_coordinates(tissue)
        max_coordinate_error = max(
            max_coordinate_error,
            float((reconstructed - expected).abs().max().detach()),
        )
    check(checks, "constant20_coordinate_embedding", max_coordinate_error, 2.0e-15)
    regularizers = fields.regularizers()
    check(
        checks,
        "constant20_zero_spatial_roughness",
        abs(float(regularizers["bulk_spatial_roughness"].detach())),
        2.0e-15,
    )
    expected_prior = sum(
        float(constant[6 * index : 6 * (index + 1)].square().sum())
        for index in range(3)
    ) + float(constant[18].square() + constant[19].square())
    check(
        checks,
        "constant20_prior_preservation",
        relative_error(float(regularizers["prior_total"].detach()), expected_prior),
        2.0e-14,
    )
    implicit_default_device = validate_implicit_cuda_device(cfg, constant, checks)

    full = fields.bulk_stresses_mpa()
    assert full.shape == (3, EXPECTED_MESH_CELL_COUNT, 3, 3)
    full_error = 0.0
    off_support_maximum = 0.0
    for tissue_index, tissue in enumerate(BULK_TISSUES):
        expected = fields.bulk_mu_mpa[tissue_index] * symmetric_matrices(
            constant[6 * tissue_index : 6 * (tissue_index + 1)]
        )
        ids = fields.cell_ids(tissue)
        full_error = max(
            full_error,
            float((full[tissue_index, ids] - expected).abs().max().detach()),
        )
        mask = torch.zeros(EXPECTED_MESH_CELL_COUNT, dtype=torch.bool)
        mask[ids] = True
        outside = torch.nonzero(~mask, as_tuple=False)[0, 0]
        off_support_maximum = max(
            off_support_maximum,
            float(full[tissue_index, outside].abs().max().detach()),
        )
    check(checks, "constant20_full_stress_embedding_mpa", full_error, 5.0e-16)
    check(checks, "off_support_stress_is_zero", off_support_maximum, 0.0)
    del full

    base = 0.15 * torch.randn(80, generator=generator)
    fields.coefficients.data.copy_(base)
    rows = {
        tissue: torch.arange(min(31, len(fields.cell_ids(tissue))))
        for tissue in BULK_TISSUES
    }
    probes = {
        tissue: torch.randn((len(rows[tissue]), 6), generator=generator)
        for tissue in BULK_TISSUES
    }
    direction = torch.randn(80, generator=generator)
    direction /= torch.linalg.vector_norm(direction)
    loss = reconstruction_scalar(fields, probes, rows)
    loss.backward()
    analytic = float(fields.coefficients.grad @ direction)
    fields.coefficients.grad = None
    step = 1.0e-4
    with torch.no_grad():
        fields.coefficients.copy_(base + step * direction)
        plus = float(reconstruction_scalar(fields, probes, rows))
        fields.coefficients.copy_(base - step * direction)
        minus = float(reconstruction_scalar(fields, probes, rows))
        fields.coefficients.copy_(base)
    finite_difference = (plus - minus) / (2 * step)
    check(
        checks,
        "reconstruction_directional_derivative",
        relative_error(finite_difference, analytic),
        2.0e-9,
    )

    quadratic = fields.regularizers()
    quadratic_loss = (
        quadratic["bulk_spatial_roughness"] + 0.37 * quadratic["bulk_spatial_magnitude"]
    )
    quadratic_loss.backward()
    analytic = float(fields.coefficients.grad @ direction)
    fields.coefficients.grad = None
    with torch.no_grad():
        fields.coefficients.copy_(base + step * direction)
        plus_values = fields.regularizers()
        plus = float(
            plus_values["bulk_spatial_roughness"]
            + 0.37 * plus_values["bulk_spatial_magnitude"]
        )
        fields.coefficients.copy_(base - step * direction)
        minus_values = fields.regularizers()
        minus = float(
            minus_values["bulk_spatial_roughness"]
            + 0.37 * minus_values["bulk_spatial_magnitude"]
        )
        fields.coefficients.copy_(base)
    finite_difference = (plus - minus) / (2 * step)
    check(
        checks,
        "quadratic_directional_derivative",
        relative_error(finite_difference, analytic),
        2.0e-9,
    )

    with torch.no_grad():
        fields.coefficients.copy_(8.0 * torch.randn(80, generator=generator))
    projection = fields.project_()
    lower, upper = -0.9, 10.0
    cell_lower_violation = 0.0
    cell_upper_violation = 0.0
    for tissue in BULK_TISSUES:
        anchor_eigenvalues = torch.linalg.eigvalsh(
            symmetric_matrices(fields.anchor_coordinates(tissue))
        )
        check(
            checks,
            f"{tissue}_anchor_lower_bound",
            float((lower - anchor_eigenvalues).clamp_min(0).max().detach()),
            2.0e-13,
        )
        check(
            checks,
            f"{tissue}_anchor_upper_bound",
            float((anchor_eigenvalues - upper).clamp_min(0).max().detach()),
            2.0e-13,
        )
        phi = fields.basis_weights(tissue)
        anchors = fields.anchor_coordinates(tissue)
        for block in phi.split(65_536):
            eigenvalues = torch.linalg.eigvalsh(symmetric_matrices(block @ anchors))
            cell_lower_violation = max(
                cell_lower_violation,
                float((lower - eigenvalues).clamp_min(0).max().detach()),
            )
            cell_upper_violation = max(
                cell_upper_violation,
                float((eigenvalues - upper).clamp_min(0).max().detach()),
            )
    check(checks, "all_cell_lower_bound", cell_lower_violation, 3.0e-13)
    check(checks, "all_cell_upper_bound", cell_upper_violation, 3.0e-13)
    assert projection["minimum_bulk_anchor_eigenvalue"] >= lower - 2.0e-13
    assert projection["maximum_bulk_anchor_eigenvalue"] <= upper + 2.0e-13

    fields.set_skin_resultant_target_(80.6)
    expected_skin = 80.6 * torch.eye(2)
    check(
        checks,
        "uniform_skin_target_n_per_m",
        float((fields.skin_resultant_n_per_m() - expected_skin).abs().max().detach()),
        2.0e-14,
    )
    assert fields.skin_baseline_index == SKIN_BASELINE_INDEX
    assert fields.skin_log_multiplier_index == SKIN_LOG_MULTIPLIER_INDEX
    assert SKIN_BASELINE_INDEX not in fields.free_coordinate_ids
    assert SKIN_LOG_MULTIPLIER_INDEX in fields.free_coordinate_ids
    assert len(fields.free_coordinate_ids) == 79

    default_model_sources = (
        Path(__file__).with_name("joint_fields.py"),
        Path(__file__).with_name("joint_physics.py"),
    )
    default_mentions = sum(
        "joint_spatial_fields" in path.read_text() for path in default_model_sources
    )
    check(
        checks,
        "constant_field_and_physics_defaults_unchanged",
        float(default_mentions),
        0.0,
    )

    bad_metadata = json.loads(cfg.audit_summary_path.read_text())
    bad_metadata["hashes"]["basis_arrays"] = "0" * 64
    bad_path = cfg.output_dir / "invalid-metadata.json"
    write_json(bad_path, bad_metadata)
    try:
        SpatialSharedFieldParameters(
            cfg.basis_path,
            bad_path,
            EXPECTED_MESH_CELL_COUNT,
            device="cpu",
            dtype=torch.float64,
        )
    except ValueError as error:
        if "does not identify" not in str(error):
            msg = "metadata rejection did not identify the basis mismatch"
            raise AssertionError(msg) from error
    else:
        msg = "mismatched audit metadata was accepted"
        raise AssertionError(msg)
    bad_path.unlink()

    assert all(row["passed"] for row in checks)
    result = {
        "schema": "joint-spatial-field-validation-v1",
        "success": True,
        "status": "passed_cpu_inactive_spatial_field_validation",
        "scope": (
            "immutable basis binding, constant embedding, differentiable spatial "
            "reconstruction and G/M forms, spectral projection, cell-bound "
            "preservation, fixed uniform skin target, and unchanged default field "
            "and physics modules; no FEM "
            "equilibrium or inverse optimization"
        ),
        "device": "cpu",
        "implicit_default_device_probe": implicit_default_device,
        "seed": cfg.seed,
        "basis": receipt,
        "projection": projection,
        "strong_roughness_contract": {
            "regularizer_key": "bulk_spatial_roughness",
            "definition": "mean_t tr(C_t^T G_t C_t)",
            "objective_term_for_beta": "0.5 * beta * bulk_spatial_roughness",
            "included_in_prior_total": False,
            "activation_requirement": "freeze and record beta > 0 before activation",
        },
        "checks": {row["name"]: row for row in checks},
        "sources": {
            str(Path(__file__).with_name("joint_spatial_fields.py")): sha256(
                Path(__file__).with_name("joint_spatial_fields.py")
            ),
            str(Path(__file__)): sha256(Path(__file__)),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(cfg.output_dir / "summary.json", result)
    cherries.log_metrics(
        {
            "spatial_fields/check_count": len(checks),
            "spatial_fields/max_cell_lower_violation": cell_lower_violation,
            "spatial_fields/max_cell_upper_violation": cell_upper_violation,
            "spatial_fields/elapsed_seconds": result["elapsed_seconds"],
        }
    )
    LOG.info("Wrote passed inactive spatial-field validation to %s", cfg.output_dir)
    COMPLETED = True


if __name__ == "__main__":
    profile = "debug" if os.getenv("DEBUG") else ProfileJoint
    cherries.main(main, profile=profile)
    if not COMPLETED:
        raise SystemExit(1)
