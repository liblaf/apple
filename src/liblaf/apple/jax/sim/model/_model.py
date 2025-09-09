import jax
import jax.numpy as jnp

from liblaf.apple.jax import tree
from liblaf.apple.jax.sim.energy import Energy
from liblaf.apple.jax.typing import Scalar, UpdatesData, UpdatesIndex, Vector


@tree.pytree
class Model:
    energies: list[Energy] = tree.field(factory=list)

    def fun(self, u: Vector) -> Scalar:
        if not self.energies:
            return jnp.zeros((), u.dtype)
        outputs: list[Scalar] = [energy.fun(u) for energy in self.energies]
        return jnp.sum(jnp.asarray(outputs))

    def jac(self, u: Vector) -> Vector:
        if not self.energies:
            return jnp.zeros_like(u)
        updates_data_list: list[UpdatesData] = []
        updates_index_list: list[UpdatesIndex] = []
        for energy in self.energies:
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

    def hess_prod(self, u: Vector, p: Vector) -> Vector:
        if not self.energies:
            return jnp.zeros_like(u)
        updates_data_list: list[UpdatesData] = []
        updates_index_list: list[UpdatesIndex] = []
        for energy in self.energies:
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

    def fun_and_jac(self, u: Vector) -> tuple[Scalar, Vector]:
        if not self.energies:
            return jnp.zeros((), u.dtype), jnp.zeros_like(u)
        value_list: list[Scalar] = []
        updates_data_list: list[UpdatesData] = []
        updates_index_list: list[UpdatesIndex] = []
        for energy in self.energies:
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
