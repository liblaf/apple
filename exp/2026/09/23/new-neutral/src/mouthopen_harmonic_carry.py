"""Harmonically extend the exact MouthOpen mandible carry outside its band."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from mouthopen_pose_jump import _full_with_boundary, _harmonic_increment
from mouthopen_pose_jump import carry_near_mandible as _exact_carry_near_mandible


@torch.no_grad()
def carry_near_mandible(
    physics: Any,
    u: torch.Tensor,
    old_jaw: torch.Tensor,
    new_jaw: torch.Tensor,
    pose: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Keep the exact d-hat band and harmonically distribute its increment.

    The selected band is derived from the immutable exact-carry result rather
    than from a second closest-feature implementation.  All selected free
    vertices retain the exact rigid carry; every other free FEM vertex receives
    only the graph-harmonic extension.  This is a geometry seed, never an
    equilibrium or contact-valid state.
    """
    started = time.perf_counter()
    model = physics.runtime.forward.model
    fem_count = len(physics.points)
    assert u.shape[0] >= fem_count
    assert u.shape[1] == 3
    exact, exact_receipt = _exact_carry_near_mandible(
        physics, u, old_jaw, new_jaw, pose
    )
    old_fixed = physics.boundary(pose(old_jaw))
    new_fixed = physics.boundary(pose(new_jaw))
    current = _full_with_boundary(model, u, old_fixed)
    delta = exact[:fem_count] - current[:fem_count]

    fixed_ids = np.unique(model.dof_map.fixed_indices.detach().cpu().numpy() // 3)
    fixed_fem = fixed_ids[fixed_ids < fem_count]
    free = np.ones(fem_count, dtype=bool)
    free[fixed_fem] = False
    moved = torch.linalg.vector_norm(delta, dim=1).detach().cpu().numpy() > 0
    carried_ids = np.flatnonzero(free & moved)
    assert len(carried_ids) == exact_receipt["carried_fem_vertices"], {
        "derived": len(carried_ids),
        "exact_receipt": exact_receipt["carried_fem_vertices"],
    }
    assert not np.intersect1d(carried_ids, fixed_fem).size
    boundary = np.union1d(carried_ids, fixed_fem)
    assert boundary.size

    reference = np.asarray(physics.points, dtype=np.float64)
    increments = delta.detach().cpu().numpy()
    harmonic = _harmonic_increment(
        reference,
        np.asarray(physics.tets, dtype=np.int64),
        boundary,
        np.empty(0, dtype=np.int64),
        increments[boundary],
    )
    candidate = exact.detach().clone()
    candidate[:fem_count] = current[:fem_count] + torch.as_tensor(
        harmonic, device=u.device, dtype=u.dtype
    )
    # Assignment restores bitwise exact band values after the add/subtract
    # round trip.  The full map then restores all appended and original fixed
    # coordinates from the new jaw boundary.
    carried_t = torch.as_tensor(carried_ids, device=u.device, dtype=torch.long)
    candidate[carried_t] = exact[carried_t]
    candidate = _full_with_boundary(model, candidate, new_fixed)
    torch.testing.assert_close(candidate[carried_t], exact[carried_t], rtol=0, atol=0)
    torch.testing.assert_close(
        candidate.flatten()[model.dof_map.fixed_indices], new_fixed, rtol=0, atol=0
    )

    geometry = physics.metrics(candidate[:fem_count])
    return candidate, {
        "method": "exact-dhat-mandible-carry-with-harmonic-free-fem-increment",
        "seed_only": True,
        "equilibrium_claimed": False,
        "contact_validity_claimed": False,
        "exact_carry": exact_receipt,
        "carried_fem_vertices": len(carried_ids),
        "fixed_fem_vertices": len(fixed_fem),
        "harmonic_boundary_vertices": len(boundary),
        "selected_free_vertices_exact": True,
        "new_fixed_values_exact": True,
        "geometry": geometry,
        "seconds": time.perf_counter() - started,
    }


__all__ = ["carry_near_mandible"]
