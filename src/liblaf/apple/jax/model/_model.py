from collections.abc import Mapping

import jarp
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from ._energy import JaxEnergy
from ._state import JaxEnergyState, JaxModelState

type EnergyMaterials = Mapping[str, Array]
type Index = Integer[Array, " points"]
type ModelMaterials = Mapping[str, EnergyMaterials]
type Scalar = Float[Array, ""]
type Updates = tuple[Vector, Index]
type Vector = Float[Array, "points dim"]


@jarp.define
class JaxModel:
    energies: dict[str, JaxEnergy] = jarp.field(factory=dict)

    def init_state(self, u: Vector) -> JaxModelState:
        data: dict[str, JaxEnergyState] = {}
        for energy in self.energies.values():
            data[energy.name] = energy.init_state(u)
        return JaxModelState(u=u, data=data)

    def update(self, state: JaxModelState, u: Vector) -> JaxModelState:
        for energy in self.energies.values():
            energy.update(state.data[energy.name], u)
        state.u = u
        return state

    def update_materials(self, materials: ModelMaterials) -> None:
        for energy_name, energy_materials in materials.items():
            self.energies[energy_name].update_materials(energy_materials)

    @jarp.jit(inline=True)
    def fun(self, state: JaxModelState, u: Vector) -> Scalar:
        output: Scalar = jnp.zeros(())
        for energy in self.energies.values():
            output += energy.fun(state.data[energy.name], u)
        return output

    @jarp.jit(inline=True)
    def grad(self, state: JaxModelState, u: Vector) -> Vector:
        output: Vector = jnp.zeros_like(u)
        for energy in self.energies.values():
            grad: Vector
            index: Index
            grad, index = energy.grad(state[energy.name], u)
            output = output.at[index].add(grad)
        return output

    @jarp.jit(inline=True)
    def hess_diag(self, state: JaxModelState, u: Vector) -> Vector:
        output: Vector = jnp.zeros_like(u)
        for energy in self.energies.values():
            diag: Vector
            index: Index
            diag, index = energy.hess_diag(state[energy.name], u)
            output = output.at[index].add(diag)
        return output

    @jarp.jit(inline=True)
    def hess_prod(self, state: JaxModelState, u: Vector, p: Vector) -> Vector:
        output: Vector = jnp.zeros_like(u)
        for energy in self.energies.values():
            prod: Vector
            index: Index
            prod, index = energy.hess_prod(state[energy.name], u, p)
            output = output.at[index].add(prod)
        return output

    @jarp.jit(inline=True)
    def hess_quad(self, state: JaxModelState, u: Vector, p: Vector) -> Scalar:
        output: Scalar = jnp.zeros(())
        for energy in self.energies.values():
            output += energy.hess_quad(state[energy.name], u, p)
        return output

    @jarp.jit(inline=True)
    def mixed_derivative_prod(
        self, state: JaxModelState, u: Vector, p: Vector
    ) -> dict[str, dict[str, Array]]:
        return {
            name: energy.mixed_derivative_prod(state[energy.name], u, p)
            for name, energy in self.energies.items()
        }

    @jarp.jit(inline=True)
    def value_and_grad(self, state: JaxModelState, u: Vector) -> tuple[Scalar, Vector]:
        value: Scalar = jnp.zeros(())
        grad: Vector = jnp.zeros_like(u)
        for energy in self.energies.values():
            value_i: Scalar
            grad_i: Vector
            value_i, (grad_i, index) = energy.value_and_grad(state[energy.name], u)
            value += value_i
            grad = grad.at[index].add(grad_i)
        return value, grad

    @jarp.jit(inline=True)
    def grad_and_hess_diag(
        self, state: JaxModelState, u: Vector
    ) -> tuple[Vector, Vector]:
        grad: Vector = jnp.zeros_like(u)
        hess_diag: Vector = jnp.zeros_like(u)
        for energy in self.energies.values():
            grad_i: Vector
            index_g: Index
            hess_diag_i: Vector
            index_h: Index
            (grad_i, index_g), (hess_diag_i, index_h) = energy.grad_and_hess_diag(
                state[energy.name], u
            )
            grad = grad.at[index_g].add(grad_i)
            hess_diag = hess_diag.at[index_h].add(hess_diag_i)
        return grad, hess_diag
