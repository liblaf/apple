from collections.abc import Container, Mapping

import jarp
import jax.numpy as jnp
import tlz
from jaxtyping import Array, Float

from liblaf.apple.jax import Dirichlet, JaxEnergy, JaxModel, JaxModelState
from liblaf.apple.warp import WarpEnergy, WarpModelAdapter, WarpModelAdapterState

from ._types import MaterialReference, MaterialValues

type EnergyMaterials = Mapping[str, Array]
type ModelMaterials = Mapping[str, EnergyMaterials]
type Scalar = Float[Array, ""]
type Vector = Float[Array, "points dim"]


@jarp.define
class ModelState:
    u: Vector
    jax: JaxModelState
    warp: WarpModelAdapterState


@jarp.define
class Model:
    dirichlet: Dirichlet
    u_full: Vector
    jax: JaxModel
    warp: WarpModelAdapter
    edges_length_mean: Scalar = jarp.array(default=0.0)

    @property
    def dim(self) -> int:
        return self.dirichlet.dim

    @property
    def n_free(self) -> int:
        return self.dirichlet.n_free

    @property
    def n_full(self) -> int:
        return self.dirichlet.n_full

    @property
    def n_points(self) -> int:
        return self.dirichlet.n_points

    @property
    def u_free(self) -> Vector:
        return self.dirichlet.get_free(self.u_full)

    @u_free.setter
    def u_free(self, u_free: Vector) -> None:
        self.u_full = self.dirichlet.to_full(u_free)

    def init_state(self, u: Vector) -> ModelState:
        jax_state: JaxModelState = self.jax.init_state(u)
        warp_state: WarpModelAdapterState = self.warp.init_state(u)
        return ModelState(u=u, jax=jax_state, warp=warp_state)

    @jarp.jit(inline=True)
    def update(self, state: ModelState, u: Vector) -> ModelState:
        state.jax = self.jax.update(state.jax, u)
        if self.warp.energies:
            state.warp, state.u = self.warp.update(state.warp, u)
        else:
            state.u = u
        return state

    def update_materials(self, materials: ModelMaterials) -> None:
        jax_materials: ModelMaterials = {}
        warp_materials: ModelMaterials = {}
        for energy_name, energy_materials in materials.items():
            if energy_name in self.warp.energies:
                warp_materials[energy_name] = energy_materials
            elif energy_name in self.jax.energies:
                jax_materials[energy_name] = energy_materials
            else:
                raise KeyError(energy_name)
        self.jax.update_materials(jax_materials)
        self.warp.update_materials(warp_materials)

    def read_material_values(self) -> dict[MaterialReference, Array]:
        values: dict[MaterialReference, Array] = {}
        for energy_name, energy in self.jax.energies.items():
            for material_name, value in energy.read_materials().items():
                values[MaterialReference(energy_name, material_name)] = value
        for energy_name, energy in self.warp.energies.items():
            for material_name, value in energy.read_materials().items():
                values[MaterialReference(energy_name, material_name)] = value
        return values

    def write_material_values(self, values: MaterialValues) -> None:
        jax_materials: dict[str, dict[str, Array]] = {}
        warp_materials: dict[str, dict[str, Array]] = {}
        for reference, value in values.items():
            if reference.energy_name in self.jax.energies:
                jax_materials.setdefault(reference.energy_name, {})[
                    reference.material_name
                ] = value
            elif reference.energy_name in self.warp.energies:
                warp_materials.setdefault(reference.energy_name, {})[
                    reference.material_name
                ] = value
            else:
                raise KeyError(reference.energy_name)
        if jax_materials:
            self.jax.update_materials(jax_materials)
        if warp_materials:
            self.warp.update_materials(warp_materials)

    @jarp.jit(inline=True)
    def fun(self, state: ModelState) -> Scalar:
        fun_jax: Scalar = self.jax.fun(state.jax, state.u)
        fun_warp: Scalar = (
            self.warp.fun(state.warp) if self.warp.energies else jnp.zeros_like(fun_jax)
        )
        return fun_jax + fun_warp

    @jarp.jit(inline=True)
    def grad(self, state: ModelState) -> Vector:
        grad_jax: Vector = self.jax.grad(state.jax, state.u)
        grad_warp: Vector = (
            self.warp.grad(state.warp)
            if self.warp.energies
            else jnp.zeros_like(grad_jax)
        )
        return grad_jax + grad_warp

    @jarp.jit(inline=True)
    def hess_diag(self, state: ModelState) -> Vector:
        # return jnp.ones_like(state.u)
        hess_diag_jax: Vector = self.jax.hess_diag(state.jax, state.u)
        hess_diag_warp: Vector = (
            self.warp.hess_diag(state.warp)
            if self.warp.energies
            else jnp.zeros_like(hess_diag_jax)
        )
        return hess_diag_jax + hess_diag_warp

    @jarp.jit(inline=True)
    def hess_prod(self, state: ModelState, v: Vector) -> Vector:
        hess_prod_jax: Vector = self.jax.hess_prod(state.jax, state.u, v)
        hess_prod_warp: Vector = (
            self.warp.hess_prod(state.warp, v)
            if self.warp.energies
            else jnp.zeros_like(hess_prod_jax)
        )
        return hess_prod_jax + hess_prod_warp

    @jarp.jit(inline=True)
    def hess_quad(self, state: ModelState, v: Vector) -> Scalar:
        hess_quad_jax: Scalar = self.jax.hess_quad(state.jax, state.u, v)
        hess_quad_warp: Scalar = (
            self.warp.hess_quad(state.warp, v)
            if self.warp.energies
            else jnp.zeros_like(hess_quad_jax)
        )
        return hess_quad_jax + hess_quad_warp

    @jarp.jit(inline=True)
    def value_and_grad(self, state: ModelState) -> tuple[Scalar, Vector]:
        fun_jax, grad_jax = self.jax.value_and_grad(state.jax, state.u)
        if self.warp.energies:
            fun_warp, grad_warp = self.warp.value_and_grad(state.warp)
        else:
            fun_warp = jnp.zeros_like(fun_jax)
            grad_warp = jnp.zeros_like(grad_jax)
        return fun_jax + fun_warp, grad_jax + grad_warp

    def mixed_derivative_prod(self, state: ModelState, p: Vector) -> ModelMaterials:
        mixed_prod_jax: ModelMaterials = self.jax.mixed_derivative_prod(
            state.jax, state.u, p
        )
        mixed_prod_warp: ModelMaterials = (
            self.warp.mixed_derivative_prod(state.warp, p) if self.warp.energies else {}
        )
        return {**mixed_prod_jax, **mixed_prod_warp}

    def get_energy(self, name: str) -> JaxEnergy | WarpEnergy:
        if name in self.warp.energies:
            return self.warp.energies[name]
        if name in self.jax.energies:
            return self.jax.energies[name]
        raise KeyError(name)


def _pick[K, V](allow_list: Container[K], dictionary: Mapping[K, V]) -> dict[K, V]:
    return tlz.keyfilter(lambda key: key in allow_list, dictionary)
