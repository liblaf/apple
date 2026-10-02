# Copyright (c) 2026 liblaf
"""Constitutive policy for fully prescribed original FEM tetrahedra.

The policy removes only bulk constitutive quadrature. It leaves the original
node numbering, IsFixed DofMap, and skin potential unchanged.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
import warp as wp

from liblaf.apple.warp.fem import StableNeoHookeanActive

BULK_NAMES = ("fat", "aponeurosis", "muscle")
_RETAINED_IDS = "_mouthopen_retained_tetrahedron_ids"
_EXCLUDED_IDS = "_mouthopen_excluded_tetrahedron_ids"


def _base(physics: Any) -> Any:
    """Return the owner of the original FEM arrays through eye wrappers."""
    return physics.base if hasattr(physics, "base") else physics


def _sha256_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _filtered_potential(
    potential: StableNeoHookeanActive,
    fields: dict[str, torch.Tensor],
    retained: torch.Tensor,
) -> StableNeoHookeanActive:
    assert type(potential) is StableNeoHookeanActive
    cells = wp.to_torch(potential.cells)
    assert cells.ndim == 2
    assert cells.shape[1] == 4
    assert retained.device == cells.device
    assert all(value.shape[0] == len(cells) for value in fields.values())
    selected_cells = cells.index_select(0, retained).contiguous()
    selected_fields = {
        name: value.index_select(0, retained).contiguous()
        for name, value in fields.items()
    }
    assert all(
        value.shape[0] == len(selected_cells) for value in selected_fields.values()
    )
    result = StableNeoHookeanActive(
        cells=wp.from_torch(selected_cells, dtype=wp.vec4i), name=potential.name
    )
    result.materials = result.material_struct()
    result.set_materials(selected_fields)
    return result


def exclude_fully_fixed_tetrahedra(physics: Any) -> dict[str, Any]:
    """Remove all-original-IsFixed cells from the three bulk potentials.

    ``base.ids`` remains the historical original mesh-cell map.  New local
    material indices for the retained active cells are exposed through
    ``base.active_t``; their original mesh-cell IDs are stored in
    ``base.retained_active_cell_ids`` for endpoint provenance.
    """
    base = _base(physics)
    model = physics.runtime.forward.model
    registry = model.warp_model.__wrapped__.potentials
    assert set(BULK_NAMES) <= set(registry)
    assert all(type(registry[name]) is StableNeoHookeanActive for name in BULK_NAMES)
    tets = np.asarray(base.tets, dtype=np.int64)
    is_fixed = np.asarray(base.mesh.point_data["IsFixed"], dtype=bool)
    assert tets.ndim == 2
    assert tets.shape[1] == 4
    assert is_fixed.shape == (len(base.points),)
    excluded_mask = is_fixed[tets].all(axis=1)
    retained_ids = np.flatnonzero(~excluded_mask).astype(np.int64)
    excluded_ids = np.flatnonzero(excluded_mask).astype(np.int64)
    assert retained_ids.size
    device = model.dof_map.fixed_values.device
    retained = torch.as_tensor(retained_ids, dtype=torch.long, device=device)
    materials = model.get_materials()
    for name in BULK_NAMES:
        potential = registry[name]
        fields = {
            field_name: (
                value if torch.is_tensor(value) else wp.to_torch(value)
            ).contiguous()
            for field_name, value in materials[name].items()
        }
        assert potential.cells.shape[0] == len(tets)
        registry[name] = _filtered_potential(potential, fields, retained)

    # ``ids`` is intentionally not rewritten: it is the source mesh-cell map
    # used by historical diagnostics.  The material arrays are now indexed by
    # retained-cell-local position, so MouthOpen packing uses ``active_t``.
    original_active = np.asarray(base.ids, dtype=np.int64)
    assert original_active.ndim == 1
    assert np.all((original_active >= 0) & (original_active < len(tets)))
    retained_active = original_active[~excluded_mask[original_active]]
    assert retained_active.size
    local = np.full(len(tets), -1, dtype=np.int64)
    local[retained_ids] = np.arange(len(retained_ids), dtype=np.int64)
    local_active = local[retained_active]
    assert np.all(local_active >= 0)
    base.retained_active_cell_ids = retained_active.copy()
    base.active_t = torch.as_tensor(local_active, dtype=torch.long, device=device)
    base.active_tetrahedron_ids = retained_active.copy()
    setattr(base, _RETAINED_IDS, retained_ids.copy())
    setattr(base, _EXCLUDED_IDS, excluded_ids.copy())

    return {
        "schema": "mouthopen-fully-fixed-tetrahedron-exclusion-v1",
        "policy": "bulk constitutive cells with four original IsFixed vertices are excluded; nodes, DofMap, and skin are retained",
        "original_tetrahedra": len(tets),
        "retained_tetrahedra": len(retained_ids),
        "excluded_tetrahedra": len(excluded_ids),
        "retained_tetrahedron_ids_sha256": _sha256_ids(retained_ids),
        "excluded_tetrahedron_ids_sha256": _sha256_ids(excluded_ids),
        "original_active_tetrahedra": len(original_active),
        "retained_active_tetrahedra": len(retained_active),
        "retained_active_cell_ids_sha256": _sha256_ids(retained_active),
        "bulk_material_cell_counts": {
            name: int(registry[name].cells.shape[0]) for name in BULK_NAMES
        },
        "skin_cells_unchanged": int(registry["skin"].cells.shape[0]),
        "dof_map_points_unchanged": int(model.dof_map.n_points),
    }


def geometry_metrics(physics: Any, u: torch.Tensor) -> dict[str, float | int]:
    """Return physical determinant diagnostics over retained bulk tetrahedra."""
    base = _base(physics)
    retained_ids = getattr(base, _RETAINED_IDS)
    excluded_ids = getattr(base, _EXCLUDED_IDS)
    tets = np.asarray(base.tets, dtype=np.int64)[retained_ids]
    reference = np.asarray(base.points, dtype=np.float64)
    displacement = u[: len(reference)].detach().cpu().numpy()
    assert displacement.shape == reference.shape
    deformed = reference + displacement
    reference_edges = np.transpose(
        reference[tets[:, 1:]] - reference[tets[:, :1]], (0, 2, 1)
    )
    deformed_edges = np.transpose(
        deformed[tets[:, 1:]] - deformed[tets[:, :1]], (0, 2, 1)
    )
    rest_det = np.linalg.det(reference_edges)
    det_f = np.linalg.det(deformed_edges) / rest_det
    assert np.isfinite(rest_det).all()
    assert np.all(rest_det > 0)
    assert np.isfinite(det_f).all()
    inverted = det_f <= 0
    rest_volume = rest_det / 6.0
    return {
        "retained_tetrahedra": len(tets),
        "excluded_tetrahedra": len(excluded_ids),
        "inverted_tetrahedra": int(inverted.sum()),
        "inverted_fraction": float(inverted.mean()),
        "inverted_rest_volume_fraction": float(
            rest_volume[inverted].sum() / rest_volume.sum()
        ),
        "detF_min": float(det_f.min()),
        "detF_p001": float(np.quantile(det_f, 0.001)),
        "detF_max": float(det_f.max()),
    }


__all__ = ["exclude_fully_fixed_tetrahedra", "geometry_metrics"]
