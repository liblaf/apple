"""Full-face directional derivatives for the opt-in spatial shared basis."""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path

import pydantic_settings as ps
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs
from joint_equilibrium import configure_cuda
from joint_fields import BULK_TISSUES
from joint_physics import JointPhysics
from joint_spatial_fields import (
    ANCHOR_COORDINATE_SLICES,
    ANCHOR_COUNTS,
    SpatialSharedFieldParameters,
)

from liblaf import cherries

LOG = logging.getLogger(__name__)
COMPLETED = False
SPATIAL_SMOOTHNESS_WEIGHT = 100.0
SPATIAL_SMOOTHNESS_FACTOR = 0.5


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    prepared_dir: Path = GROUP / "data/prepared"
    initial_checkpoint: Path = (
        GROUP / "data/neutral-continuation-metric-bfgs-002/neutral-025/terminal.pt"
    )
    basis_path: Path = (
        GROUP / "data/spatial-baseline-audit-005/basis-and-normal-equations.npz"
    )
    basis_audit_summary: Path = GROUP / "data/spatial-baseline-audit-005/summary.json"
    cpu_validation: Path = GROUP / "data/spatial-fields-validation-cpu-v7/summary.json"
    contact_spec: Path = GROUP / "data/contact/config.json"
    contact_validation: Path = GROUP / "data/contact-validation/summary.json"
    newton_validation: Path = GROUP / "data/contact-validation-newton/summary.json"
    output_dir: Path = GROUP / "data/spatial-face-gradient-validation-001"
    prior_weight: float = 0.001
    spatial_smoothness_weight: float = SPATIAL_SMOOTHNESS_WEIGHT
    base_spatial_offset: float = 0.02
    forward_rtol: float = 1e-6
    forward_atol: float = 1e-12
    adjoint_rtol: float = 1e-7
    max_forward_steps: int = 10000
    newton_linear_rtol: float = 1e-3
    newton_max_steps: int = 12


def neutral_objective(
    physics: JointPhysics,
    shared: SpatialSharedFieldParameters,
    displacement: torch.Tensor,
    *,
    prior_weight: float,
    spatial_smoothness_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    surface = (
        (
            physics.weights_t[:, None] * displacement[physics.observation_t].square()
        ).sum()
        * 1e6
        / 0.25**2
    )
    centers = displacement[physics.muscle_tets_t].mean(dim=1)
    muscle = (physics.muscle_mass_t[:, None] * centers.square()).sum() * 1e6 / 0.5**2
    regularizers = shared.regularizers()
    weighted_prior = prior_weight * regularizers["prior_total"]
    weighted_roughness = (
        SPATIAL_SMOOTHNESS_FACTOR
        * spatial_smoothness_weight
        * regularizers["bulk_spatial_roughness"]
    )
    terms = {
        "surface_loss": surface,
        "muscle_loss": muscle,
        "prior_total": regularizers["prior_total"],
        "weighted_prior": weighted_prior,
        "bulk_spatial_roughness": regularizers["bulk_spatial_roughness"],
        "weighted_spatial_roughness": weighted_roughness,
    }
    return surface + muscle + weighted_prior + weighted_roughness, terms


def main(cfg: Config) -> None:  # noqa: PLR0915
    global COMPLETED  # noqa: PLW0603
    assert cfg.spatial_smoothness_weight == SPATIAL_SMOOTHNESS_WEIGHT
    assert cfg.prior_weight == 0.001
    assert cfg.base_spatial_offset == 0.02
    assert cfg.forward_rtol == 1e-6
    assert cfg.forward_atol == 1e-12
    assert cfg.adjoint_rtol == 1e-7
    assert cfg.newton_linear_rtol == 1e-3
    assert cfg.newton_max_steps == 12
    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)
    archive_sources(output)
    source_names = (
        Path(__file__).name,
        "joint_common.py",
        "joint_data.py",
        "joint_fields.py",
        "joint_spatial_fields.py",
        "joint_physics.py",
        "joint_equilibrium.py",
        "joint_contact.py",
        "joint_newton.py",
        "joint_materials.py",
    )
    source_hashes = {
        str(GROUP / "src" / name): sha256(GROUP / "src" / name) for name in source_names
    }
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz",
        cfg.prepared_dir / "manifest.json",
        verify_sources=True,
    )
    configure_cuda()
    initial = torch.load(cfg.initial_checkpoint, map_location="cpu", weights_only=False)
    assert initial["schema"] == "joint-inverse-checkpoint-v1"
    assert initial["stage"] == "neutral"
    assert initial["protocol"].get("shared_basis", "constant20") == "constant20"
    assert initial["protocol"]["skin_prestress_fraction"] == 0.25
    assert initial["optimizer_converged"] is True
    assert initial["neutral_budget_met"] is False
    assert initial["protocol"]["input_arrays_sha256"] == sha256(
        cfg.prepared_dir / "inputs.npz"
    )
    assert initial["protocol"]["input_manifest_sha256"] == sha256(
        cfg.prepared_dir / "manifest.json"
    )

    contact_config = json.loads(cfg.contact_spec.read_text())
    assert contact_config["schema"] == "joint-bone-contact-v1"
    assert contact_config["enabled"] is True
    contact_validation = json.loads(cfg.contact_validation.read_text())
    assert contact_validation["schema"] == "joint-contact-validation-v1"
    assert contact_validation["success"] is True
    assert contact_validation["contact_spec_sha256"] == sha256(cfg.contact_spec)
    newton_validation = json.loads(cfg.newton_validation.read_text())
    assert newton_validation["schema"] == "joint-contact-validation-v1"
    assert newton_validation["success"] is True
    assert newton_validation["forward_solver"]["method"] == "newton_cg"
    assert newton_validation["contact_spec_sha256"] == sha256(cfg.contact_spec)

    spec = initial["materials"]
    materials = spec["materials"]
    skin = materials["skin"]
    physics = JointPhysics(
        prepared.volume_path,
        prepared.skin_path,
        prepared.arrays,
        bulk_young_mpa={name: materials[name]["young_mpa"] for name in BULK_TISSUES},
        bulk_nu={name: materials[name]["poisson"] for name in BULK_TISSUES},
        skin_young_mpa=skin["reference_map"]["young_mpa"],
        skin_nu=skin["poisson"],
        thickness_m=skin["thickness_m"],
        rtol=cfg.forward_rtol,
        atol=cfg.forward_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        max_steps=cfg.max_forward_steps,
        forward_method="newton_cg",
        newton_linear_rtol=cfg.newton_linear_rtol,
        newton_max_steps=cfg.newton_max_steps,
        contact_config=contact_config,
    )
    shared = SpatialSharedFieldParameters(
        cfg.basis_path,
        cfg.basis_audit_summary,
        len(physics.tets),
        material_config=spec,
    )
    cpu_validation = json.loads(cfg.cpu_validation.read_text())
    assert cpu_validation["schema"] == "joint-spatial-field-validation-v1"
    assert cpu_validation["success"] is True
    assert cpu_validation["status"] == "passed_cpu_inactive_spatial_field_validation"
    assert cpu_validation["basis"] == shared.basis_receipt()
    assert (
        cpu_validation["strong_roughness_contract"]["included_in_prior_total"] is False
    )
    for source, digest in cpu_validation["sources"].items():
        assert sha256(Path(source)) == digest, source

    with torch.no_grad():
        shared.load_constant20_(initial["shared_coefficients"])
        tensor = shared.coefficients.new_tensor([0.4, -0.3, 0.2, 0.1, -0.2, 0.3])
        tensor /= torch.linalg.vector_norm(tensor)
        for tissue in BULK_TISSUES:
            count = ANCHOR_COUNTS[tissue]
            pattern = torch.linspace(
                -1.0,
                1.0,
                count,
                dtype=shared.coefficients.dtype,
                device=shared.coefficients.device,
            )
            pattern -= pattern.mean()
            pattern /= torch.linalg.vector_norm(pattern)
            shared.anchor_coordinates(tissue).add_(
                cfg.base_spatial_offset * pattern[:, None] * tensor[None, :]
            )
        # Keep both finite-difference stiffness sides inside the model bounds.
        shared.coefficients[shared.skin_log_multiplier_index] -= 0.01
        before_projection = shared.coefficients.detach().clone()
        projection = shared.project_()
        assert float((shared.coefficients - before_projection).abs().max()) <= 1e-12

    pose = torch.zeros(6)
    seed = initial["primal"]["neutral"].to(device="cuda")

    def evaluate() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        u = physics.solve(
            shared.bulk_stresses_mpa(),
            shared.skin_resultant_n_per_m(),
            shared.skin_stiffness_multiplier(),
            None,
            pose,
            seed,
            key="spatial-face-gradient-check",
        )
        return neutral_objective(
            physics,
            shared,
            u,
            prior_weight=cfg.prior_weight,
            spatial_smoothness_weight=cfg.spatial_smoothness_weight,
        )

    value, terms = evaluate()
    base_forward = copy.deepcopy(physics.runtime.last_forward)
    assert base_forward["contact"]["contact_numerically_valid"] is True
    seed = physics.runtime.forward.state.u.detach().clone()
    torch.save(seed.cpu(), output / "base-equilibrium.pt")
    mechanical = terms["surface_loss"] + terms["muscle_loss"]
    regularization = terms["weighted_prior"] + terms["weighted_spatial_roughness"]
    mechanical_gradient = torch.autograd.grad(
        mechanical, shared.coefficients, retain_graph=True
    )[0].detach()
    regularizer_gradient = torch.autograd.grad(regularization, shared.coefficients)[
        0
    ].detach()
    gradient = mechanical_gradient + regularizer_gradient
    assert torch.isfinite(gradient).all()

    directions: list[tuple[str, torch.Tensor]] = []
    for tissue in BULK_TISSUES:
        section = ANCHOR_COORDINATE_SLICES[tissue]
        direction = torch.zeros_like(shared.coefficients)
        direction[section.start : section.start + 6] = tensor
        directions.append((f"{tissue}_anchor_0", direction))
        anchor_gradient = mechanical_gradient[section].reshape(ANCHOR_COUNTS[tissue], 6)
        contrast = anchor_gradient - anchor_gradient.mean(dim=0, keepdim=True)
        norm = torch.linalg.vector_norm(contrast)
        assert float(norm) > 1e-10, (tissue, float(norm))
        direction = torch.zeros_like(shared.coefficients)
        direction[section] = (contrast / norm).reshape(-1)
        directions.append((f"{tissue}_spatial_contrast", direction))
    for name, index in (
        ("skin_baseline", shared.skin_baseline_index),
        ("skin_stiffness", shared.skin_log_multiplier_index),
    ):
        direction = torch.zeros_like(shared.coefficients)
        direction[index] = 1.0
        directions.append((name, direction))

    original = shared.coefficients.detach().clone()
    rows: list[dict[str, object]] = []
    for name, direction in directions:
        analytic = float(torch.dot(gradient, direction))
        assert abs(analytic) > 1e-10, (name, analytic)
        mechanical_analytic = float(torch.dot(mechanical_gradient, direction))
        assert abs(mechanical_analytic) > 1e-10, (name, mechanical_analytic)
        for step in (0.003, 0.001):
            LOG.info("Checking spatial full-face %s h=%.3g", name, step)
            with torch.no_grad():
                shared.coefficients.copy_(original + step * direction)
                plus, plus_terms = evaluate()
                plus_value = float(plus.detach())
                plus_forward = copy.deepcopy(physics.runtime.last_forward)
                shared.coefficients.copy_(original - step * direction)
                minus, minus_terms = evaluate()
                minus_value = float(minus.detach())
                minus_forward = copy.deepcopy(physics.runtime.last_forward)
                shared.coefficients.copy_(original)
            finite = (plus_value - minus_value) / (2 * step)
            relative = abs(finite - analytic) / max(abs(finite), abs(analytic), 1e-14)
            mechanical_finite = float(
                (
                    plus_terms["surface_loss"]
                    + plus_terms["muscle_loss"]
                    - minus_terms["surface_loss"]
                    - minus_terms["muscle_loss"]
                )
                / (2 * step)
            )
            mechanical_relative = abs(mechanical_finite - mechanical_analytic) / max(
                abs(mechanical_finite), abs(mechanical_analytic)
            )
            assert plus_forward["contact"]["contact_numerically_valid"] is True
            assert minus_forward["contact"]["contact_numerically_valid"] is True
            row = {
                "name": name,
                "step": step,
                "analytic": analytic,
                "finite_difference": finite,
                "relative_error": relative,
                "mechanical_analytic": mechanical_analytic,
                "mechanical_finite_difference": mechanical_finite,
                "mechanical_relative_error": mechanical_relative,
                "direction": direction.detach().cpu().tolist(),
                "plus_forward": plus_forward,
                "minus_forward": minus_forward,
            }
            rows.append(row)
            write_json(output / "checks.json", rows)
            LOG.info("Spatial %s h=%.3g relative error %.6g", name, step, relative)
            assert relative < 0.02, row
            assert mechanical_relative < 0.02, row

    for source, digest in source_hashes.items():
        assert sha256(Path(source)) == digest, (
            f"Source changed during validation: {source}"
        )
    summary = {
        "schema": "joint-spatial-face-gradient-validation-v1",
        "success": True,
        "scope": (
            "full-mesh contact-enabled neutral-objective directional derivatives "
            "for three anchors, three spatial contrast modes, skin baseline and stiffness; mechanical and total objectives checked separately"
        ),
        "basis": shared.basis_receipt(),
        "shared_field_schema": shared.config["schema"],
        "initial_checkpoint": str(cfg.initial_checkpoint.resolve()),
        "initial_checkpoint_sha256": sha256(cfg.initial_checkpoint),
        "input_arrays_sha256": sha256(cfg.prepared_dir / "inputs.npz"),
        "input_manifest_sha256": sha256(cfg.prepared_dir / "manifest.json"),
        "cpu_validation_path": str(cfg.cpu_validation.resolve()),
        "cpu_validation_sha256": sha256(cfg.cpu_validation),
        "contact_enabled": True,
        "contact_spec_sha256": sha256(cfg.contact_spec),
        "newton_validation_sha256": sha256(cfg.newton_validation),
        "spatial_smoothness_weight": cfg.spatial_smoothness_weight,
        "spatial_smoothness_factor": SPATIAL_SMOOTHNESS_FACTOR,
        "base_spatial_offset": cfg.base_spatial_offset,
        "base_projection": projection,
        "base_objective": float(value.detach()),
        "base_terms": {name: float(item.detach()) for name, item in terms.items()},
        "base_forward": base_forward,
        "checks": rows,
        "maximum_relative_error": max(float(row["relative_error"]) for row in rows),
        "maximum_mechanical_relative_error": max(
            float(row["mechanical_relative_error"]) for row in rows
        ),
        "implementation_sha256": source_hashes,
        "finite_difference_seed_policy": "same frozen converged base equilibrium for each independent plus/minus solve",
        "base_coefficients": original.detach().cpu().tolist(),
        "forward_tolerances": physics.runtime.tolerances,
        "forward_solver": physics.runtime.forward_solver,
        "forward_count": physics.runtime.forward_count,
        "adjoint": physics.runtime.last_adjoint,
    }
    write_json(output / "summary.json", summary)
    cherries.log_output(output)
    COMPLETED = True


if __name__ == "__main__":
    os.environ.setdefault("COMET_AUTO_LOG_GIT_METADATA", "false")
    os.environ.setdefault("COMET_AUTO_LOG_GIT_PATCH", "false")
    cherries.main(main, profile=ProfileJoint)
    if not COMPLETED:
        raise SystemExit(1)
