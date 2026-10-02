"""Repeatable CPU checks for full-boundary IPC transition contact."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch

from liblaf import cherries

sys.path.insert(0, str(Path(__file__).parent))
GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402
from transition_contact import (  # noqa: E402
    OwnedContact,
    audit_contact,
    build_self_contact,
)


class Config(cherries.BaseConfig):
    output: Path = cherries.output("10-contact-checks/summary.json", mkdir=True)


def _two_tetrahedra() -> pv.UnstructuredGrid:
    """Two disconnected tetrahedra with opposing, nonadjacent boundary faces."""
    points = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, -1.0, 0.30],
            [1.0, -1.0, 0.30],
            [0.0, 1.0, 0.30],
            [0.0, 0.0, 1.30],
        ],
        dtype=np.float64,
    )
    cells = np.array([4, 0, 1, 2, 3, 4, 4, 5, 6, 7], dtype=np.int64)
    volume = pv.UnstructuredGrid(cells, np.full(2, pv.CellType.TETRA), points)
    volume.point_data["GlobalPointId"] = np.arange(volume.n_points, dtype=np.int64)
    return volume


def _energy_gradient(
    contact: OwnedContact, u: torch.Tensor
) -> tuple[float, torch.Tensor]:
    state = contact.state_at(u)
    energy = float(contact.fun(state, u))
    gradient = torch.zeros_like(u)
    contact.grad(state, u, gradient)
    return energy, gradient


def _fixed_feature_gradient(
    contact: OwnedContact, state: Any, u: torch.Tensor
) -> torch.Tensor:
    """Differentiate a collision set held fixed across the HVP stencil."""
    positions = (contact.vertices + u[contact.indices]).numpy(force=True)
    local_gradient = torch.as_tensor(
        contact.potential.gradient(state.collisions, contact.collision_mesh, positions)
    ).reshape(-1, 3)
    gradient = torch.zeros_like(u)
    gradient.index_add_(0, contact.indices, local_gradient)
    return gradient


def main(cfg: Config) -> None:
    torch.set_default_dtype(torch.float64)
    volume = _two_tetrahedra()
    contact, receipt = build_self_contact(volume)
    u = torch.zeros((volume.n_points, 3), dtype=torch.float64)
    energy, gradient = _energy_gradient(contact, u)
    assert energy > 0
    assert torch.linalg.vector_norm(gradient) > 0

    approach = torch.zeros_like(u)
    approach[4:, 2] = -0.40
    state = contact.state_at(u)
    ccd_fraction = float(contact.max_step_size(state, u, approach))
    assert 0 < ccd_fraction < 1
    unit_approach = approach / torch.linalg.vector_norm(approach)
    epsilon = 1e-6
    energy_plus, _ = _energy_gradient(contact, u + epsilon * unit_approach)
    energy_minus, _ = _energy_gradient(contact, u - epsilon * unit_approach)
    directional_fd = (energy_plus - energy_minus) / (2 * epsilon)
    directional_gradient = float(torch.sum(gradient * unit_approach))
    directional_relative_error = abs(directional_gradient - directional_fd) / abs(
        directional_gradient
    )
    assert directional_gradient > 0
    assert directional_relative_error < 2e-7
    force = -gradient
    assert torch.all(force[:3, 2] < 0)
    assert torch.all(force[4:7, 2] > 0)

    # A generic perturbation and one held-fixed active set checks the exact IPC
    # Hessian without allowing a broad-phase feature transition in the stencil.
    generator = torch.Generator().manual_seed(37)
    hessian_u = u + 3e-3 * torch.randn(u.shape, generator=generator)
    direction = torch.randn(u.shape, generator=generator)
    direction /= torch.linalg.vector_norm(direction)
    state = contact.state_at(hessian_u)
    hp_exact = torch.zeros_like(u)
    contact.hess_prod(state, hessian_u, direction, hp_exact)
    gradient_plus = _fixed_feature_gradient(
        contact, state, hessian_u + epsilon * direction
    )
    gradient_minus = _fixed_feature_gradient(
        contact, state, hessian_u - epsilon * direction
    )
    hp_fd = (gradient_plus - gradient_minus) / (2 * epsilon)
    hessian_relative_error = float(
        torch.linalg.vector_norm(hp_exact - hp_fd) / torch.linalg.vector_norm(hp_exact)
    )
    assert hessian_relative_error < 2e-5

    audit = audit_contact(contact, u)
    assert audit["active_contact_count"] > 0
    assert audit["complete_boundary_no_intersections"]
    summary = {
        "schema": "full-fem-boundary-self-contact-check-v1",
        "device": "cpu",
        "receipt": receipt,
        "energy": energy,
        "gradient_l2": float(torch.linalg.vector_norm(gradient)),
        "directional_energy_fd": directional_fd,
        "directional_gradient": directional_gradient,
        "directional_relative_error": directional_relative_error,
        "repulsive_force_sign": {
            "lower_facing_triangle_z": "negative",
            "upper_base_z": "positive",
        },
        "ccd_approach_fraction": ccd_fraction,
        "hessian_hvp_relative_error": hessian_relative_error,
        "audit": audit,
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    cherries.log_metrics(
        {
            "contact/energy": energy,
            "contact/gradient_l2": summary["gradient_l2"],
            "contact/directional_relative_error": directional_relative_error,
            "contact/ccd_approach_fraction": ccd_fraction,
            "contact/hessian_hvp_relative_error": hessian_relative_error,
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
