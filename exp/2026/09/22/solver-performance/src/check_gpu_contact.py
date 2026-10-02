# ruff: noqa: EM101, PT017, SLF001, TRY003
"""CPU checks for the opt-in GPU contact-Hessian adapter.

Use ``--checkpoint`` only on a scheduled CUDA host to exercise the actual saved
Smile state; this default check intentionally performs no CUDA work.
"""

from __future__ import annotations

from types import SimpleNamespace

import attrs
import numpy as np
import scipy.sparse
import torch
from gpu_contact import (
    GpuContactHessian,
    install_adjoint_gpu_contact,
    install_gpu_contact,
)


def check_cpu_path_and_invalidation() -> None:
    matrix = scipy.sparse.csr_matrix(np.diag((2.0, 3.0, 4.0)))

    class Potential:
        def hessian(self, **_kwargs: object) -> scipy.sparse.csr_matrix:
            return matrix

    class Collision:
        indices = torch.tensor([0])
        vertices = torch.zeros((1, 3), dtype=torch.float64)
        potential = Potential()
        collision_mesh = object()

        @staticmethod
        def hess_prod(
            _state: object, _u: torch.Tensor, p: torch.Tensor, out: torch.Tensor
        ) -> None:
            out.add_(p * 2)

    class State:
        collisions = object()
        hess: scipy.sparse.csr_matrix | None = None

    collision, state = Collision(), State()
    adapter = GpuContactHessian(collision, collision.hess_prod)
    p, out = (
        torch.ones((1, 3), dtype=torch.float64),
        torch.zeros((1, 3), dtype=torch.float64),
    )
    adapter.hess_prod(state, torch.zeros_like(p), p, out)
    torch.testing.assert_close(out, 2 * p)
    state.hess = matrix
    adapter.invalidate()
    assert adapter._matrix is None
    assert adapter._indices is None
    model = SimpleNamespace(collision=collision, hess_prod=lambda: None)
    installed = install_gpu_contact(model)
    assert type(collision).__name__.startswith("GpuCollision")
    installed.uninstall()
    assert type(collision) is Collision


def check_attrs_slotted_install_contract() -> None:
    """The production OwnedContact is attrs-slotted, so reclass it directly."""

    @attrs.define
    class SlottedCollision:
        indices: torch.Tensor

        def hess_prod(
            self,
            _state: object,
            _u: torch.Tensor,
            p: torch.Tensor,
            output: torch.Tensor,
        ) -> None:
            output.add_(p)

    collision = SlottedCollision(indices=torch.tensor([0]))
    model = SimpleNamespace(collision=collision, hess_prod=lambda: None)
    installed = install_gpu_contact(model)
    assert type(collision).__name__.startswith("GpuSlottedCollision")
    output = torch.zeros((1, 3), dtype=torch.float64)
    collision.hess_prod(object(), output, torch.ones_like(output), output)
    torch.testing.assert_close(output, torch.ones_like(output))
    installed.uninstall()
    assert type(collision) is SlottedCollision


def check_adjoint_scope_install_lifetime() -> None:
    """The temporary subclass exists only while the wrapped solver runs."""

    class Collision:
        @staticmethod
        def hess_prod(
            _state: object, _u: torch.Tensor, _p: torch.Tensor, _out: torch.Tensor
        ) -> None:
            raise AssertionError("CPU lifetime test does not call the product")

    collision = Collision()
    model = SimpleNamespace(collision=collision, hess_prod=lambda: None)
    runtime = SimpleNamespace()
    observed: list[type[object]] = []

    class Solver:
        def solve(self, *, fail: bool = False) -> str:
            observed.append(type(collision))
            if fail:
                raise RuntimeError("expected solver failure")
            return "solved"

    runtime.solver = Solver()
    installed = install_adjoint_gpu_contact(runtime, model)
    assert type(collision) is Collision
    assert runtime.solver.solve() == "solved"
    assert observed[-1].__name__.startswith("GpuCollision")
    assert type(collision) is Collision
    try:
        runtime.solver.solve(fail=True)
    except RuntimeError as error:
        assert str(error) == "expected solver failure"
    else:
        raise AssertionError("wrapped solver hid its exception")
    assert type(collision) is Collision
    installed.uninstall()
    assert runtime.solver.solve.__self__ is runtime.solver


if __name__ == "__main__":
    check_cpu_path_and_invalidation()
    check_attrs_slotted_install_contract()
    check_adjoint_scope_install_lifetime()
    print("gpu-contact CPU path and cache invalidation checks passed")
