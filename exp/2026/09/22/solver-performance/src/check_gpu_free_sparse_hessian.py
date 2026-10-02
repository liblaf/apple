# ruff: noqa: C901, PLR0915, SLF001
"""Independent dense-reference checks for CPU maps and cached CUDA assembly."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import scipy.sparse

source = Path(__file__).with_name("gpu_free_sparse_hessian.py")
spec = importlib.util.spec_from_file_location("gpu_free_sparse_hessian", source)
assert spec is not None
assert spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def reference(
    fem: np.ndarray,
    local: np.ndarray,
    local_to_full: np.ndarray,
    free: np.ndarray,
    shift: float,
) -> np.ndarray:
    full = fem.copy()
    for row in range(local.shape[0]):
        for col in range(local.shape[1]):
            full[local_to_full[row], local_to_full[col]] += local[row, col]
    restricted = full[np.ix_(free, free)]
    return 0.5 * (restricted + restricted.T) + shift * np.eye(len(free))


def plan_result(
    fem: np.ndarray,
    local: np.ndarray,
    local_to_full: np.ndarray,
    free: np.ndarray,
    shift: float,
) -> np.ndarray:
    inverse = np.full(fem.shape[0], -1, dtype=np.int64)
    inverse[free] = np.arange(len(free))
    fem_rows, fem_cols = np.nonzero(fem)
    fem_keys = inverse[fem_rows] * len(free) + inverse[fem_cols]
    fem_keep = (inverse[fem_rows] >= 0) & (inverse[fem_cols] >= 0)
    fem_keys = fem_keys[fem_keep]
    fem_values = fem[fem_rows[fem_keep], fem_cols[fem_keep]]
    local_rows, local_cols = np.nonzero(local)
    contact_rows, contact_cols = local_to_full[local_rows], local_to_full[local_cols]
    contact_keep = (inverse[contact_rows] >= 0) & (inverse[contact_cols] >= 0)
    contact_keys = (
        inverse[contact_rows[contact_keep]] * len(free)
        + inverse[contact_cols[contact_keep]]
    )
    contact_values = local[local_rows[contact_keep], local_cols[contact_keep]]
    plan = module._make_plan(
        n_free=len(free),
        fem_keys=fem_keys.astype(np.int64),
        contact_keys=contact_keys.astype(np.int64),
        contact_pattern_hash="test",
        shifted=bool(shift),
    )
    return module._assemble_numpy(
        plan, fem_values=fem_values, contact_values=contact_values, shift=shift
    ).toarray()


def main() -> None:
    # Arbitrary free ordering and partial constraints: local contact maps both
    # retained and eliminated DOFs.  The (0, 5) edge is outside FEM sparsity.
    fem = np.diag(np.arange(1.0, 13.0))
    fem[2, 8] = fem[8, 2] = 0.25
    local_to_full = np.array([9, 10, 11, 0, 1, 2], dtype=np.int64)
    local = np.zeros((6, 6))
    local[0, 5] = local[5, 0] = 3.5  # contact-only edge, outside FEM pattern
    local[1, 1] = 2.0
    local[2, 4] = 1.25  # col 1 is constrained below and must disappear
    local[4, 2] = 1.25
    free = np.array([10, 2, 9, 0, 8], dtype=np.int64)
    for shift in (0.0, 0.75):
        np.testing.assert_allclose(
            plan_result(fem, local, local_to_full, free, shift),
            reference(fem, local, local_to_full, free, shift),
            rtol=0,
            atol=1e-14,
        )
    # Same contact pattern, changed numeric entries: cached symbolic slots must
    # produce the new exact matrix rather than retaining the first values.
    changed = local.copy()
    changed[0, 5] = changed[5, 0] = -1.25
    changed[1, 1] = 7.0
    np.testing.assert_allclose(
        plan_result(fem, changed, local_to_full, free, 0.25),
        reference(fem, changed, local_to_full, free, 0.25),
        rtol=0,
        atol=1e-14,
    )
    # A new off-FEM contact edge changes the union pattern and remains present.
    new_pattern = changed.copy()
    new_pattern[0, 3] = new_pattern[3, 0] = 4.0
    actual = plan_result(fem, new_pattern, local_to_full, free, 0.0)
    expected = reference(fem, new_pattern, local_to_full, free, 0.0)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    assert actual[2, 3] == actual[3, 2] == 4.0
    print("gpu_free_sparse_hessian CPU symbolic checks passed")


def cuda_class_check() -> None:
    """Exercise the real GPU class cache and scatter path with tiny mocks."""
    if not module.torch.cuda.is_available():
        print("CUDA unavailable; CPU symbolic checks completed")
        return

    class DofMap:
        def __init__(self, free: np.ndarray) -> None:
            self.free_indices = module.torch.as_tensor(
                free, device="cuda", dtype=module.torch.int64
            )

    class Collision:
        def __init__(self) -> None:
            self.indices = module.torch.tensor(
                [3, 0], device="cuda", dtype=module.torch.int64
            )

    class Model:
        n_points = 4
        dim = 3

        def __init__(self, free: np.ndarray) -> None:
            self.dof_map = DofMap(free)
            self.collision = Collision()

    class CollisionState:
        def __init__(self, hess: np.ndarray) -> None:
            self.hess = scipy.sparse.csr_matrix(hess)

    class State:
        def __init__(self, hess: np.ndarray) -> None:
            self.u = module.torch.zeros(
                (4, 3), device="cuda", dtype=module.torch.float64
            )
            self.collision = CollisionState(hess)

    class Fem:
        def __init__(self, dense: np.ndarray) -> None:
            self.dense = dense
            self.matrix = self._matrix()

        def _matrix(self):
            return module.torch.as_tensor(
                self.dense, device="cuda", dtype=module.torch.float64
            ).to_sparse_bsr((3, 3))

        def setup(self, _state: State) -> None:
            self.matrix = self._matrix()

    free = np.array([10, 1, 9, 2, 8], dtype=np.int64)
    local_to_full = np.array([9, 10, 11, 0, 1, 2], dtype=np.int64)
    fem_dense = np.diag(np.arange(1.0, 13.0))
    fem_dense[2, 8] = fem_dense[8, 2] = 0.25
    fem = Fem(fem_dense)

    def local(*entries: tuple[int, int, float]) -> np.ndarray:
        value = np.zeros((6, 6))
        for row, col, number in entries:
            value[row, col] = value[col, row] = number
        return value

    original = local((0, 5, 3.5), (1, 1, 2.0))  # includes contact-only edge
    state = State(original)
    assembled = module.GpuFreeSparseHessian(Model(free), state, fem)

    def check(
        expected_local: np.ndarray,
        shift: float,
        *,
        cache: bool | None = None,
        rebuilt: bool | None = None,
    ) -> None:
        expected = reference(fem_dense, expected_local, local_to_full, free, shift)
        module.torch.cuda.synchronize()
        np.testing.assert_allclose(
            assembled.matrix.to_dense().cpu().numpy(), expected, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            assembled.lower.to_dense().cpu().numpy(),
            np.tril(expected),
            rtol=0,
            atol=1e-12,
        )
        if cache is not None:
            assert assembled.metadata["symbolic_cache_hit"] is cache
        if rebuilt is not None:
            assert assembled.metadata["symbolic_rebuilt"] is rebuilt

    check(original, 0.0, rebuilt=True)
    changed_values = local((0, 5, -1.25), (1, 1, 7.0))
    state.collision = CollisionState(changed_values)
    assembled.setup(state, fem, 0.0)
    check(changed_values, 0.0, cache=True)
    # Different contact pattern whose diagonal falls in the cached FEM union.
    inside_union = local((1, 1, 5.0))
    state.collision = CollisionState(inside_union)
    assembled.setup(state, fem, 0.0)
    check(inside_union, 0.0, cache=True)
    # A new retained contact edge is outside both FEM and current IPC union.
    off_union = local((0, 4, 4.0))
    state.collision = CollisionState(off_union)
    assembled.setup(state, fem, 0.0)
    check(off_union, 0.0, rebuilt=True)
    state.collision = CollisionState(original)
    assembled.setup(state, fem, 0.0)
    check(original, 0.0, rebuilt=True)
    # Shift changes values only: all diagonal slots were structural from start.
    assembled.setup(state, fem, 0.75)
    check(original, 0.75, cache=True)
    fem_dense *= 1.5
    assembled.setup(state, fem, 0.25)
    check(original, 0.25, cache=True)
    print("gpu_free_sparse_hessian CUDA class checks passed")


if __name__ == "__main__":
    main()
    cuda_class_check()
