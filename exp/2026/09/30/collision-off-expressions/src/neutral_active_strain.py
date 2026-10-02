"""Convert the neutral load calibration into explicit multiplicative strain."""

from __future__ import annotations

from typing import Any

import torch
from active_strain_materials import StableNeoHookeanActiveMembrane

from liblaf.apple.warp.fem import StableNeoHookeanActive


def install_active_strain(model: Any) -> tuple[dict, dict, dict]:
    """Replace every material and return strain arrays plus calibration evidence.

    The skin uses B B^T = I + T/(h mu) in its orthonormal rest tangent frame.
    This preserves the existing physical-volume energy's forces and tangents.
    Bulk neutral activation is exactly identity. No additive stress field is
    retained by any installed potential.
    """
    registry = model.warp_model.__wrapped__.potentials
    previous = dict(registry)
    old = model.get_materials()
    values: dict[str, dict[str, torch.Tensor]] = {}
    for name in ("fat", "muscle", "aponeurosis"):
        assert type(previous[name]).__name__ == "StableNeoHookeanStress"
        fields = dict(old[name])
        stress = fields.pop("active_stress")
        assert not bool(torch.count_nonzero(stress))
        # The library packs B as I + symmetric(activation_inv), so zero means I.
        fields["activation_inv"] = torch.zeros(
            (len(stress), 6), device=stress.device, dtype=stress.dtype
        )
        potential = StableNeoHookeanActive(cells=previous[name].cells, name=name)
        potential.materials = potential.material_struct()
        assert set(fields) == set(potential.MATERIAL_FIELDS)
        potential.set_materials(fields)
        registry[name] = potential
        values[name] = fields

    fields = dict(old["skin"])
    tension = fields.pop("baseline_stress")
    mu_h = fields["mu"] * fields["thickness"]
    metric = (
        torch.eye(2, device=tension.device, dtype=tension.dtype)
        + tension / mu_h[:, None, None]
    )
    eigenvalues, directions = torch.linalg.eigh(metric)
    assert bool(torch.isfinite(eigenvalues).all())
    assert bool((eigenvalues > 0).all()), "Skin tension has no SPD strain mapping"
    stretch = (
        directions * torch.sqrt(eigenvalues).unsqueeze(-2)
    ) @ directions.transpose(-1, -2)
    reconstructed = mu_h[:, None, None] * (
        stretch @ stretch.transpose(-1, -2)
        - torch.eye(2, device=tension.device, dtype=tension.dtype)
    )
    mapping_error = float(
        torch.linalg.vector_norm(reconstructed - tension)
        / torch.linalg.vector_norm(tension)
    )
    assert mapping_error < 1e-12
    fields["activation_inv"] = stretch.contiguous()
    skin = StableNeoHookeanActiveMembrane(
        cells=previous["skin"].cells,
        name="skin",
        thickness=previous["skin"].thickness,
    )
    skin.materials = skin.material_struct()
    assert set(fields) == set(skin.MATERIAL_FIELDS)
    skin.set_materials(fields)
    registry["skin"] = skin
    values["skin"] = fields
    assert all(
        "active_stress" not in item and "baseline_stress" not in item
        for item in values.values()
    )
    arrays = {
        "skin_activation_inverse": stretch.detach().cpu().numpy(),
        "skin_activation": torch.linalg.inv(stretch).detach().cpu().numpy(),
        "source_skin_tension_mpa_m": tension.detach().cpu().numpy(),
        "skin_mu_mpa": fields["mu"].detach().cpu().numpy(),
        "skin_thickness_m": fields["thickness"].detach().cpu().numpy(),
    }
    principal = torch.sqrt(eigenvalues)
    reference_area = 0.5 * fields["fraction"] * fields["rest_metric_sqrt_det"]
    # W_strain - W_stress is constant over physical displacement.
    offset = float(
        (0.5 * reference_area * tension.diagonal(dim1=-2, dim2=-1).sum(-1)).sum()
    )
    receipt = {
        "formulation": "multiplicative active strain with physical-volume regularization",
        "bulk": "StableNeoHookeanActive; B = I; stored symmetric increment = 0",
        "skin": "StableNeoHookeanActiveMembrane; B = sqrt(I + T/(h*mu))",
        "volume_convention": "J = det(F) in bulk; J = physical surface area ratio * relaxed normal stretch in skin",
        "skin_normal_activation": 1.0,
        "additive_stress_fields_in_installed_model": False,
        "calibration": "Equivalent prestretch derived from previous prescribed skin tension; not an independently measured prestrain",
        "skin_B_principal_min": float(principal.min()),
        "skin_B_principal_max": float(principal.max()),
        "skin_A_principal_min": float(1.0 / principal.max()),
        "skin_A_principal_max": float(1.0 / principal.min()),
        "tension_mapping_relative_error": mapping_error,
        "constant_energy_offset_mpa_m3": offset,
        "installed_potentials": {
            name: type(value).__name__ for name, value in registry.items()
        },
        "physical_force_and_tangent_equivalence_expected": True,
    }
    return values, receipt, {"previous_potentials": previous, "arrays": arrays}


def verify_equivalence(model: Any, state: Any, context: dict, receipt: dict) -> None:
    """Audit both installed formulations at the same full face state."""
    registry = model.warp_model.__wrapped__.potentials
    active = dict(registry)
    generator = torch.Generator(device=state.u.device).manual_seed(20260923)
    direction = torch.randn(
        state.u.shape, device=state.u.device, dtype=state.u.dtype, generator=generator
    )
    energy = model.fun(state)
    gradient = model.grad(state)
    product = model.hess_prod(state, direction)
    try:
        registry.update(context["previous_potentials"])
        old_energy = model.fun(state)
        old_gradient = model.grad(state)
        old_product = model.hess_prod(state, direction)
    finally:
        registry.update(active)
    errors = {
        "force_relative_error": float(
            torch.linalg.vector_norm(gradient - old_gradient)
            / torch.linalg.vector_norm(old_gradient)
        ),
        "hvp_relative_error": float(
            torch.linalg.vector_norm(product - old_product)
            / torch.linalg.vector_norm(old_product)
        ),
        "energy_offset_observed": float(energy - old_energy),
        "energy_offset_error": float(energy - old_energy)
        - receipt["constant_energy_offset_mpa_m3"],
    }
    assert errors["force_relative_error"] < 1e-10, errors
    assert errors["hvp_relative_error"] < 1e-10, errors
    assert abs(errors["energy_offset_error"]) < 1e-12 * max(
        abs(float(energy)), abs(float(old_energy)), 1e-12
    ), errors
    receipt["full_face_seed_equivalence"] = errors
