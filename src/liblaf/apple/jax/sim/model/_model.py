import jax
import jax.numpy as jnp

from liblaf.apple.jax import tree
from liblaf.apple.jax.sim.dirichlet import Dirichlet
from liblaf.apple.jax.sim.energy import Energy
from liblaf.apple.jax.typing import Scalar, UpdatesData, UpdatesIndex, Vector


@tree.pytree
class Model:
    energies: list[Energy] = tree.field(factory=list)
    dirichlet: Dirichlet = tree.field(factory=Dirichlet)

    def fun(self, u: Vector) -> Scalar:
        outputs: list[Scalar] = [energy.fun(u) for energy in self.energies]
        return jnp.sum(jnp.asarray(outputs))

    def jac(self, u: Vector) -> Vector:
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

    def value_and_jac(self, u: Vector) -> tuple[Scalar, Vector]:
        value_list: list[Scalar] = []
        updates_data_list: list[UpdatesData] = []
        updates_index_list: list[UpdatesIndex] = []
        for energy in self.energies:
            value, (data, index) = energy.value_and_jac(u)
            value_list.append(value)
            updates_data_list.append(data)
            updates_index_list.append(index)
        value: Scalar = jnp.sum(jnp.asarray(value_list))
        jac: Vector = jax.ops.segment_sum(
            jnp.concat(updates_data_list),
            jnp.concat(updates_index_list),
            num_segments=u.shape[0],
        )
        return value, jac

    def hess_prod(self, u: Vector, p: Vector) -> Vector:
        updates_data_list: list[UpdatesData] = []
        updates_index_list: list[UpdatesIndex] = []
        for energy in self.energies:
            data: UpdatesData
            index: UpdatesIndex
            data, index = energy.hess_prod(u, p)
            updates_data_list.append(data)
            updates_index_list.append(index)
        hess: Vector = jax.ops.segment_sum(
            jnp.concat(updates_data_list),
            jnp.concat(updates_index_list),
            num_segments=u.shape[0],
        )
        return hess
