from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jaxtyping import Array

from liblaf.apple.jax import tree
from liblaf.apple.jax.sim.energy import Energy
from liblaf.apple.jax.typing import Scalar, UpdatesData, UpdatesIndex, Vector


@tree.pytree
class Model:
    energies: Mapping[str, Energy] = tree.field(factory=dict)

    def fun(self, u: Vector) -> Scalar:
        if not self.energies:
            return jnp.zeros((), u.dtype)
        outputs: list[Scalar] = [energy.fun(u) for energy in self.energies.values()]
        return jnp.sum(jnp.asarray(outputs))

    def jac(self, u: Vector) -> Vector:
        if not self.energies:
            return jnp.zeros_like(u)
        updates_data_list: list[UpdatesData] = []
        updates_index_list: list[UpdatesIndex] = []
        for energy in self.energies.values():
            data: UpdatesData
            index: UpdatesIndex
            data, index = energy.jac(u)
            updates_data_list.append(data)
            updates_index_list.append(index)
        jac: Vector = jax.ops.segment_sum(
            jnp.concat(updates_data_list),
            jnp.concat(updates_index_list),
            num_segments=u.shape[0],
        )
        return jac

    def hess_diag(self, u: Vector) -> Vector:
        if not self.energies:
            return jnp.zeros_like(u)
        updates_data_list: list[UpdatesData] = []
        updates_index_list: list[UpdatesIndex] = []
        for energy in self.energies.values():
            data: UpdatesData
            index: UpdatesIndex
            data, index = energy.hess_diag(u)
            updates_data_list.append(data)
            updates_index_list.append(index)
        hess_diag: Vector = jax.ops.segment_sum(
            jnp.concat(updates_data_list),
            jnp.concat(updates_index_list),
            num_segments=u.shape[0],
        )
        return hess_diag

    def hess_prod(self, u: Vector, p: Vector) -> Vector:
        if not self.energies:
            return jnp.zeros_like(u)
        updates_data_list: list[UpdatesData] = []
        updates_index_list: list[UpdatesIndex] = []
        for energy in self.energies.values():
            data: UpdatesData
            index: UpdatesIndex
            data, index = energy.hess_prod(u, p)
            updates_data_list.append(data)
            updates_index_list.append(index)
        hess_prod: Vector = jax.ops.segment_sum(
            jnp.concat(updates_data_list),
            jnp.concat(updates_index_list),
            num_segments=u.shape[0],
        )
        return hess_prod

    def hess_quad(self, u: Vector, p: Vector) -> Scalar:
        if not self.energies:
            return jnp.zeros((), u.dtype)
        outputs: list[Scalar] = [
            energy.hess_quad(u, p) for energy in self.energies.values()
        ]
        return jnp.sum(jnp.asarray(outputs))

    def fun_and_jac(self, u: Vector) -> tuple[Scalar, Vector]:
        if not self.energies:
            return jnp.zeros((), u.dtype), jnp.zeros_like(u)
        value_list: list[Scalar] = []
        updates_data_list: list[UpdatesData] = []
        updates_index_list: list[UpdatesIndex] = []
        for energy in self.energies.values():
            fun, (data, index) = energy.fun_and_jac(u)
            value_list.append(fun)
            updates_data_list.append(data)
            updates_index_list.append(index)
        fun: Scalar = jnp.sum(jnp.asarray(value_list))
        jac: Vector = jax.ops.segment_sum(
            jnp.concat(updates_data_list),
            jnp.concat(updates_index_list),
            num_segments=u.shape[0],
        )
        return fun, jac

    def jac_and_hess_diag(self, u: Vector) -> tuple[Vector, Vector]:
        if not self.energies:
            return jnp.zeros_like(u), jnp.zeros_like(u)
        updates_data_jac_list: list[UpdatesData] = []
        updates_index_jac_list: list[UpdatesIndex] = []
        updates_data_hess_list: list[UpdatesData] = []
        updates_index_hess_list: list[UpdatesIndex] = []
        for energy in self.energies.values():
            (data_jac, index_jac), (data_hess, index_hess) = energy.jac_and_hess_diag(u)
            updates_data_jac_list.append(data_jac)
            updates_index_jac_list.append(index_jac)
            updates_data_hess_list.append(data_hess)
            updates_index_hess_list.append(index_hess)
        jac: Vector = jax.ops.segment_sum(
            jnp.concat(updates_data_jac_list),
            jnp.concat(updates_index_jac_list),
            num_segments=u.shape[0],
        )
        hess_diag: Vector = jax.ops.segment_sum(
            jnp.concat(updates_data_hess_list),
            jnp.concat(updates_index_hess_list),
            num_segments=u.shape[0],
        )
        return jac, hess_diag

    def mixed_derivative_prod(
        self, u: Vector, p: Vector
    ) -> dict[str, dict[str, Array]]:
        outputs: dict[str, dict[str, Array]] = {
            energy.id: energy.mixed_derivative_prod(u, p)
            for energy in self.energies.values()
        }
        return outputs
