# ruff: noqa: C901, EM101, TRY301
"""CPU contracts for nested timing and patch restoration."""

from __future__ import annotations

import attrs
from inverse_timing import InverseTimer, install_inverse_timing


def check_tree_and_exception_accounting() -> None:
    timer = InverseTimer()
    try:
        with timer.stage("forward/coarse_pncg"), timer.scope("model/grad"):
            raise RuntimeError("expected")
    except RuntimeError:
        pass
    tree = timer.report()["tree"]
    parent = tree["forward/coarse_pncg"]
    child = parent["children"]["model/grad"]
    assert parent["count"] == child["count"] == 1
    assert parent["inclusive_seconds"] >= child["inclusive_seconds"] >= 0
    assert parent["exclusive_seconds"] >= -1e-9


def check_patch_restore() -> None:
    class Collision:
        def state_at(self):
            return "state"

        def update(self):
            return None

        def grad(self):
            return None

        def hess_diag(self):
            return None

        def hess_prod(self):
            return None

        def max_step_size(self):
            return 1

    class Model:
        collision = Collision()

        def fun(self):
            return 1

        def grad(self):
            return 2

        def hess_diag(self):
            return 3

        def hess_prod(self):
            return 4

        def mixed_derivative_prod(self):
            return 5

    model = Model()
    original = model.fun
    installed = install_inverse_timing(model, native_ipc=False)
    assert model.fun() == 1
    assert model.collision.max_step_size() == 1
    assert "model/update" in installed.missing
    installed.uninstall()
    assert model.fun.__func__ is original.__func__


def check_staticmethod_restore() -> None:
    class Backward:
        @staticmethod
        def run(value: int) -> int:
            return value + 1

    timer = InverseTimer()
    timer.patch(Backward, "run", "adjoint/backward")
    assert Backward.run(4) == 5
    timer.restore()
    assert isinstance(Backward.__dict__["run"], staticmethod)
    assert Backward.run(4) == 5


def check_slotted_instance_patch() -> None:
    @attrs.define
    class Slotted:
        def grad(self) -> int:
            return 7

    value = Slotted()
    timer = InverseTimer()
    timer.patch(value, "grad", "model/grad")
    assert value.grad() == 7
    timer.restore()
    assert value.grad() == 7


if __name__ == "__main__":
    check_tree_and_exception_accounting()
    check_patch_restore()
    check_staticmethod_restore()
    check_slotted_instance_patch()
    print("inverse timing CPU checks passed")
