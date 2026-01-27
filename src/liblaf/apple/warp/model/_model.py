from collections.abc import Mapping
from typing import Annotated

import jarp
import warp as wp
from frozendict import frozendict
from jaxtyping import Array

from ._energy import WarpEnergy
from ._state import WarpEnergyState, WarpModelState

type EnergyMaterials = Mapping[str, Array]
type ModelMaterials = Mapping[str, EnergyMaterials]
type Scalar = Annotated[wp.array, " 1"]
type Vector = Annotated[wp.array, " N"]


@jarp.frozen_static
class WarpModel:
    energies: frozendict[str, WarpEnergy] = jarp.field(factory=lambda: frozendict())

    def init_state(self, u: Vector) -> WarpModelState:
        data: dict[str, WarpEnergyState] = {}
        for name, energy in self.energies.items():
            data[name] = energy.init_state(u)
        return WarpModelState(data=frozendict(data))

    def update(self, state: WarpModelState, u: Vector) -> WarpModelState:
        for name, energy in self.energies.items():
            energy.update(state[name], u)
        return state

    def update_materials(self, materials: ModelMaterials) -> None:
        for name, energy_materials in materials.items():
            self.energies[name].update_materials(energy_materials)

    def fun(self, state: WarpModelState, u: Vector, output: Scalar) -> None:
        output.zero_()
        for name, energy in self.energies.items():
            energy.fun(state[name], u, output)

    def grad(self, state: WarpModelState, u: Vector, output: Vector) -> None:
        output.zero_()
        for name, energy in self.energies.items():
            energy.grad(state[name], u, output)

    def hess_diag(self, state: WarpModelState, u: Vector, output: Vector) -> None:
        output.zero_()
        for name, energy in self.energies.items():
            energy.hess_diag(state[name], u, output)

    def hess_prod(
        self, state: WarpModelState, u: Vector, v: Vector, output: Vector
    ) -> None:
        output.zero_()
        for name, energy in self.energies.items():
            energy.hess_prod(state[name], u, v, output)

    def hess_quad(
        self, state: WarpModelState, u: Vector, v: Vector, output: Scalar
    ) -> None:
        output.zero_()
        for name, energy in self.energies.items():
            energy.hess_quad(state[name], u, v, output)

    def value_and_grad(
        self, state: WarpModelState, u: Vector, value: Scalar, grad: Vector
    ) -> None:
        value.zero_()
        grad.zero_()
        for name, energy in self.energies.items():
            energy.value_and_grad(state[name], u, value, grad)

    def mixed_hess_prod(
        self, state: WarpModelState, u: Vector, v: Vector
    ) -> dict[str, dict[str, wp.array]]:
        outputs: dict[str, dict[str, wp.array]] = {}
        for name, energy in self.energies.items():
            outputs[name] = energy.mixed_hess_prod(state[name], u, v)
        return outputs
