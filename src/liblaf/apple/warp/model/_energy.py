from collections.abc import Mapping
from typing import Annotated, Any

import jarp
import warp as wp
from jaxtyping import Array
from warp._src.codegen import StructInstance

from liblaf.apple import utils

from ._state import WarpEnergyState

type EnergyMaterials = Mapping[str, Array]
type Scalar = Annotated[wp.array, " 1"]
type Vector = Annotated[wp.array, " N"]

vec3 = Any


@jarp.frozen_static
class WarpEnergy:
    name: str = jarp.static(default=utils.name_factory, kw_only=True)
    materials: StructInstance = jarp.static(default=None, kw_only=True)
    requires_grad: tuple[str, ...] = jarp.static(default=(), kw_only=True)

    def init_state(self, u: Vector) -> WarpEnergyState:  # noqa: ARG002
        return WarpEnergyState()

    def update(self, state: WarpEnergyState, u: Vector) -> WarpEnergyState:  # noqa: ARG002
        return state

    def update_materials(self, materials: EnergyMaterials) -> None:
        for name, new_val in materials.items():
            param: wp.array = getattr(self.materials, name)
            wp.copy(param, jarp.to_warp(new_val, param.dtype))

    def fun(self, state: WarpEnergyState, u: Vector, output: Scalar) -> None:
        raise NotImplementedError

    def grad(self, state: WarpEnergyState, u: Vector, output: Vector) -> None:
        raise NotImplementedError

    def hess_prod(
        self, state: WarpEnergyState, u: Vector, v: Vector, output: Vector
    ) -> None:
        raise NotImplementedError

    def hess_diag(self, state: WarpEnergyState, u: Vector, output: Vector) -> None:
        raise NotImplementedError

    def hess_quad(
        self, state: WarpEnergyState, u: Vector, v: Vector, output: Scalar
    ) -> None:
        raise NotImplementedError

    def value_and_grad(
        self, state: WarpEnergyState, u: Vector, value: Scalar, grad: Vector
    ) -> None:
        self.fun(state, u, value)
        self.grad(state, u, grad)

    def mixed_hess_prod(
        self, state: WarpEnergyState, u: Vector, v: Vector
    ) -> dict[str, wp.array]:
        if not self.requires_grad:
            return {}
        output: wp.array = wp.zeros_like(v)
        with wp.Tape() as tape:
            self.grad(state, u, output)
        tape.zero()
        tape.backward(grads={output: v})
        outputs: dict[str, wp.array] = {}
        for name in self.requires_grad:
            param: wp.array = getattr(self.materials, name)
            assert param.grad is not None
            outputs[name] = param.grad
        return outputs
