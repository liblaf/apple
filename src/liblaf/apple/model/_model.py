from collections.abc import Container, Mapping

import jarp
import tlz
from jaxtyping import Array, Float

from liblaf.apple.jax import Dirichlet
from liblaf.apple.warp import WarpEnergy, WarpModelAdapter, WarpModelAdapterState

type EnergyMaterials = Mapping[str, Array]
type ModelMaterials = Mapping[str, EnergyMaterials]
type Scalar = Float[Array, ""]
type Vector = Float[Array, "points dim"]


@jarp.define
class ModelState:
    u: Vector
    warp: WarpModelAdapterState


@jarp.define
class Model:
    dirichlet: Dirichlet
    u_full: Vector
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
        warp_state: WarpModelAdapterState = self.warp.init_state(u)
        return ModelState(u=u, warp=warp_state)

    @jarp.jit(inline=True)
    def update(self, state: ModelState, u: Vector) -> ModelState:
        state.warp, state.u = self.warp.update(state.warp, u)
        return state

    def update_materials(self, materials: ModelMaterials) -> None:
        warp_materials: ModelMaterials = {}
        for energy_name, energy_materials in materials.items():
            if energy_name in self.warp.energies:
                warp_materials[energy_name] = energy_materials
            else:
                raise KeyError(energy_name)
        self.warp.update_materials(warp_materials)

    @jarp.jit(inline=True)
    def fun(self, state: ModelState) -> Scalar:
        return self.warp.fun(state=state.warp)

    @jarp.jit(inline=True)
    def grad(self, state: ModelState) -> Vector:
        return self.warp.grad(state=state.warp)

    @jarp.jit(inline=True)
    def hess_diag(self, state: ModelState) -> Vector:
        return self.warp.hess_diag(state=state.warp)

    @jarp.jit(inline=True)
    def hess_prod(self, state: ModelState, v: Vector) -> Vector:
        return self.warp.hess_prod(state=state.warp, v=v)

    @jarp.jit(inline=True)
    def hess_quad(self, state: ModelState, v: Vector) -> Scalar:
        return self.warp.hess_quad(state=state.warp, v=v)

    def get_energy(self, name: str) -> WarpEnergy:
        return self.warp.energies[name]


def _pick[K, V](allow_list: Container[K], dictionary: Mapping[K, V]) -> dict[K, V]:
    return tlz.keyfilter(lambda key: key in allow_list, dictionary)
