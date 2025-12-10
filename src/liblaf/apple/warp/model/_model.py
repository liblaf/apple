from collections.abc import Mapping

import warp as wp
from jaxtyping import Array, Float
from liblaf.peach import tree

from ._energy import WarpEnergy

type EnergyParams = Mapping[str, Array]
type ModelParams = Mapping[str, EnergyParams]
type Scalar = Float[wp.array, ""]
type Vector = Float[wp.array, " N"]


@tree.define
class WarpModel:
    dim: int = tree.field(default=3, kw_only=True)
    energies: dict[str, WarpEnergy] = tree.field(factory=dict)

    def update(self, u: Vector) -> None:
        for energy in self.energies.values():
            energy.update(u)

    def update_params(self, params: ModelParams) -> None:
        for name, energy_params in params.items():
            self.energies[name].update_params(energy_params)

    def fun(self, u: Vector, output: Scalar) -> None:
        for energy in self.energies.values():
            energy.fun(u, output)

    def grad(self, u: Vector, output: Vector) -> None:
        for energy in self.energies.values():
            energy.grad(u, output)

    def hess_diag(self, u: Vector, output: Vector) -> None:
        for energy in self.energies.values():
            energy.hess_diag(u, output)

    def hess_prod(self, u: Vector, p: Vector, output: Vector) -> None:
        for energy in self.energies.values():
            energy.hess_prod(u, p, output)

    def hess_quad(self, u: Vector, p: Vector, output: Scalar) -> None:
        for energy in self.energies.values():
            energy.hess_quad(u, p, output)

    def mixed_derivative_prod(
        self, u: Vector, p: Vector
    ) -> dict[str, dict[str, wp.array]]:
        output: dict[str, dict[str, wp.array]] = {
            name: energy.mixed_derivative_prod(u, p)
            for name, energy in self.energies.items()
        }
        return output

    def value_and_grad(self, u: Vector, value: Scalar, grad: Vector) -> None:
        for energy in self.energies.values():
            energy.value_and_grad(u, value, grad)

    def grad_and_hess_diag(self, u: Vector, grad: Vector, hess_diag: Vector) -> None:
        for energy in self.energies.values():
            energy.grad_and_hess_diag(u, grad, hess_diag)
