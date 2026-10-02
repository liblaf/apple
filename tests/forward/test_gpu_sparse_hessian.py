# ruff: noqa: SLF001
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from liblaf.apple.forward.hessian._assembled_fem import _topology
from liblaf.apple.forward.hessian._gpu_sparse import (
    GpuFreeSparseHessian,
    _assemble_numpy,
    _CachedGpuPlan,
    _digest,
    _make_plan,
    _pattern_hash,
    _tensor_token,
)


def _reference(
    fem: np.ndarray,
    contact: np.ndarray,
    local_to_full: np.ndarray,
    free: np.ndarray,
    shift: float,
) -> np.ndarray:
    full = fem.copy()
    for row, col in zip(*np.nonzero(contact), strict=True):
        full[local_to_full[row], local_to_full[col]] += contact[row, col]
    restricted = full[np.ix_(free, free)]
    return 0.5 * (restricted + restricted.T) + shift * np.eye(len(free))


def _planned(
    fem: np.ndarray,
    contact: np.ndarray,
    local_to_full: np.ndarray,
    free: np.ndarray,
    shift: float,
) -> np.ndarray:
    inverse = np.full(len(fem), -1, dtype=np.int64)
    inverse[free] = np.arange(len(free))
    fem_rows, fem_cols = np.nonzero(fem)
    keep = (inverse[fem_rows] >= 0) & (inverse[fem_cols] >= 0)
    fem_keys = inverse[fem_rows[keep]] * len(free) + inverse[fem_cols[keep]]
    contact_rows, contact_cols = np.nonzero(contact)
    full_rows = local_to_full[contact_rows]
    full_cols = local_to_full[contact_cols]
    retained = (inverse[full_rows] >= 0) & (inverse[full_cols] >= 0)
    contact_keys = (
        inverse[full_rows[retained]] * len(free) + inverse[full_cols[retained]]
    )
    plan = _make_plan(
        n_free=len(free),
        fem_keys=fem_keys,
        contact_keys=contact_keys,
        contact_pattern_hash="test",
        shifted=bool(shift),
    )
    return _assemble_numpy(
        plan,
        fem_values=fem[fem_rows[keep], fem_cols[keep]],
        contact_values=contact[contact_rows[retained], contact_cols[retained]],
        shift=shift,
    ).toarray()


@pytest.mark.parametrize("shift", [0.0, 0.75])
def test_symbolic_plan_matches_dense_reference(shift: float) -> None:
    fem = np.diag(np.arange(1.0, 13.0))
    fem[2, 8] = fem[8, 2] = 0.25
    local_to_full = np.array([9, 10, 11, 0, 1, 2], dtype=np.int64)
    contact = np.zeros((6, 6))
    contact[0, 5] = contact[5, 0] = 3.5
    contact[1, 1] = 2.0
    contact[2, 4] = contact[4, 2] = 1.25
    free = np.array([10, 2, 9, 0, 8], dtype=np.int64)

    np.testing.assert_allclose(
        _planned(fem, contact, local_to_full, free, shift),
        _reference(fem, contact, local_to_full, free, shift),
        rtol=0,
        atol=1e-14,
    )


def test_contact_cache_reuses_only_entries_in_union() -> None:
    keys = np.array([0, 1, 3, 4, 8], dtype=np.int64)
    assembly = object.__new__(GpuFreeSparseHessian)
    assembly._device = torch.device("cpu")
    assembly._plan = _CachedGpuPlan(
        contact_pattern_hash="old",
        shifted=True,
        keys=keys,
        pattern_hash=_pattern_hash(keys, 3),
        lower_nnz=4,
    )
    assembly._gpu = {}

    assert assembly._reuse_existing_union(np.array([1, 4]), "inside")
    torch.testing.assert_close(assembly._gpu["contact_direct"], torch.tensor([1, 3]))
    assert assembly._plan.contact_pattern_hash == "inside"
    assert not assembly._reuse_existing_union(np.array([2]), "outside")


def test_free_mapping_mutation_fails_fast() -> None:
    free = torch.tensor([0, 2, 4], dtype=torch.int64)
    assembly = object.__new__(GpuFreeSparseHessian)
    assembly._model = SimpleNamespace(dof_map=SimpleNamespace(free_indices=free))
    assembly._free = free.numpy().copy()
    assembly._free_digest = _digest(assembly._free)
    assembly._free_token = _tensor_token(free)

    assembly._check_free_mapping()
    assembly._model.dof_map.free_indices = free.clone()
    assembly._check_free_mapping()
    assembly._model.dof_map.free_indices[1] = 3
    with pytest.raises(ValueError, match="free-DOF mapping changed"):
        assembly._check_free_mapping()


def test_new_fem_operator_does_not_reuse_stale_topology() -> None:
    class Model:
        n_points = 4

    model = Model()
    potential = SimpleNamespace(cells=torch.tensor([[0, 1, 2]]))

    first, cache_hit = _topology(model, {"surface": potential})
    assert not cache_hit
    same, cache_hit = _topology(model, {"surface": potential})
    assert cache_hit
    assert same is first

    potential.cells[0, 2] = 3
    changed, cache_hit = _topology(model, {"surface": potential})
    assert not cache_hit
    assert changed is not first
    assert not np.array_equal(changed.keys, first.keys)
