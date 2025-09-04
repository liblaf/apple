from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.apple.jax import tree
from liblaf.apple.jax.sim.dirichlet import Dirichlet
from liblaf.apple.jax.sim.energy import Energy
from liblaf.apple.jax.typing import Scalar, UpdatesData, UpdatesIndex, Vector


@tree.pytree
class Model:
    points: Float[Array, "p J"] = tree.array()
    dirichlet: Dirichlet = tree.field(factory=Dirichlet)
    energies: list[Energy] = tree.field(factory=list)

    @staticmethod
    def static_fun(u: Vector, model: "Model") -> Scalar:
        return model.fun(u)

    @staticmethod
    def static_jac(u: Vector, model: "Model") -> Vector:
        return model.jac(u)

    @staticmethod
    def static_hess_prod(u: Vector, p: Vector, model: "Model") -> Vector:
        return model.hess_prod(u, p)

    @staticmethod
    def static_fun_and_jac(u: Vector, model: "Model") -> tuple[Scalar, Vector]:
        return model.fun_and_jac(u)

    @property
    def n_dirichlet(self) -> int:
        return self.dirichlet.n_dirichlet

    @property
    def n_dofs(self) -> int:
        return self.dirichlet.n_dofs

    @property
    def n_free(self) -> int:
        return self.dirichlet.n_free

    def fun(self, u: Vector) -> Scalar:
        u_full: Vector = self.to_full(u)
        outputs: list[Scalar] = [energy.fun(u_full) for energy in self.energies]
        return jnp.sum(jnp.asarray(outputs))

    def jac(self, u: Vector) -> Vector:
        u_full: Vector = self.to_full(u)
        updates_data_list: list[UpdatesData] = []
        updates_index_list: list[UpdatesIndex] = []
        for energy in self.energies:
            data: UpdatesData
            index: UpdatesIndex
            data, index = energy.jac(u_full)
            updates_data_list.append(data)
            updates_index_list.append(index)
        jac: Vector = jax.ops.segment_sum(
            jnp.concat(updates_data_list),
            jnp.concat(updates_index_list),
            num_segments=u_full.shape[0],
        )
        jac = self.reshape_or_extract_free(jac, u.shape)
        return jac

    def hess_prod(self, u: Vector, p: Vector) -> Vector:
        u_full: Vector = self.to_full(u)
        p_full: Vector = self.to_full(p, zero=True)
        updates_data_list: list[UpdatesData] = []
        updates_index_list: list[UpdatesIndex] = []
        for energy in self.energies:
            data: UpdatesData
            index: UpdatesIndex
            data, index = energy.hess_prod(u_full, p_full)
            updates_data_list.append(data)
            updates_index_list.append(index)
        hess_prod: Vector = jax.ops.segment_sum(
            jnp.concat(updates_data_list),
            jnp.concat(updates_index_list),
            num_segments=u_full.shape[0],
        )
        hess_prod = self.reshape_or_extract_free(hess_prod, u.shape)
        return hess_prod

    def fun_and_jac(self, u: Vector) -> tuple[Scalar, Vector]:
        u_full: Vector = self.to_full(u)
        value_list: list[Scalar] = []
        updates_data_list: list[UpdatesData] = []
        updates_index_list: list[UpdatesIndex] = []
        for energy in self.energies:
            value, (data, index) = energy.fun_and_jac(u_full)
            value_list.append(value)
            updates_data_list.append(data)
            updates_index_list.append(index)
        value: Scalar = jnp.sum(jnp.asarray(value_list))
        jac: Vector = jax.ops.segment_sum(
            jnp.concat(updates_data_list),
            jnp.concat(updates_index_list),
            num_segments=u_full.shape[0],
        )
        jac = self.reshape_or_extract_free(jac, u.shape)
        return value, jac

    def free_to_full(self, u_free: Vector, *, zero: bool = False) -> Vector:
        u_full: Vector = jnp.zeros_like(self.points)
        u_full = self.dirichlet.set_free(u_full, u_free)
        u_full = self.dirichlet.zero(u_full) if zero else self.dirichlet.apply(u_full)
        return u_full

    def to_full(self, u: Vector, *, zero: bool = False) -> Vector:
        if u.size == self.n_dofs:
            return u.reshape(self.points.shape)
        return self.free_to_full(u, zero=zero)

    def reshape_or_extract_free(self, u: Vector, shape: Sequence[int]) -> Vector:
        if u.size == np.prod(shape):
            return u.reshape(shape)
        return self.dirichlet.get_free(u)
