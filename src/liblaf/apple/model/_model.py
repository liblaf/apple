from collections.abc import Container, Mapping

import jarp
import tlz
from jaxtyping import Array, Float

from liblaf.apple.jax import Dirichlet, JaxModel, JaxModelState
from liblaf.apple.warp import WarpEnergy, WarpModelAdapter, WarpModelAdapterState

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
        state.warp, state.u = self.warp.update(state.warp, u)
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

    @jarp.jit(inline=True)
    def fun(self, state: ModelState) -> Scalar:
        fun_jax: Scalar = self.jax.fun(state.jax, state.u)
        fun_warp: Scalar = self.warp.fun(state.warp)
        return fun_jax + fun_warp

    @jarp.jit(inline=True)
    def grad(self, state: ModelState) -> Vector:
        grad_jax: Vector = self.jax.grad(state.jax, state.u)
        grad_warp: Vector = self.warp.grad(state.warp)
        return grad_jax + grad_warp

    @jarp.jit(inline=True)
    def hess_diag(self, state: ModelState) -> Vector:
        # return jnp.ones_like(state.u)
        hess_diag_jax: Vector = self.jax.hess_diag(state.jax, state.u)
        hess_diag_warp: Vector = self.warp.hess_diag(state.warp)
        return hess_diag_jax + hess_diag_warp

    @jarp.jit(inline=True)
    def hess_prod(self, state: ModelState, v: Vector) -> Vector:
        hess_prod_jax: Vector = self.jax.hess_prod(state.jax, state.u, v)
        hess_prod_warp: Vector = self.warp.hess_prod(state.warp, v)
        return hess_prod_jax + hess_prod_warp

    @jarp.jit(inline=True)
    def hess_quad(self, state: ModelState, v: Vector) -> Scalar:
        hess_quad_jax: Scalar = self.jax.hess_quad(state.jax, state.u, v)
        hess_quad_warp: Scalar = self.warp.hess_quad(state.warp, v)
        return hess_quad_jax + hess_quad_warp

    @jarp.jit(inline=True)
    def value_and_grad(self, state: ModelState) -> tuple[Scalar, Vector]:
        fun_jax, grad_jax = self.jax.value_and_grad(state.jax, state.u)
        fun_warp, grad_warp = self.warp.value_and_grad(state.warp)
        return fun_jax + fun_warp, grad_jax + grad_warp

    def get_energy(self, name: str) -> WarpEnergy:
        return self.warp.energies[name]


def _pick[K, V](allow_list: Container[K], dictionary: Mapping[K, V]) -> dict[K, V]:
    return tlz.keyfilter(lambda key: key in allow_list, dictionary)
