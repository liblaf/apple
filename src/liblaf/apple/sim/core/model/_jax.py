from collections.abc import Mapping

import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Array

from liblaf.apple import struct
from liblaf.apple.sim.core.energy import EnergyJax
from liblaf.apple.types.jax import Scalar, UpdatesData, UpdatesIndex, Vector


@struct.pytree
class ModelJax:
    energies: list[EnergyJax] = attrs.field(factory=list, kw_only=True)

    def fun(self, x: Vector) -> Scalar:
        outputs: list[Scalar] = [energy.fun(x) for energy in self.energies]
        return jnp.sum(jnp.asarray(outputs))

    def jac(self, x: Vector) -> Vector:
        data: list[UpdatesData] = []
        index: list[UpdatesIndex] = []
        for energy in self.energies:
            energy_data: UpdatesData
            energy_index: UpdatesIndex
            energy_data, energy_index = energy.jac(x)
            data.append(energy_data)
            index.append(energy_index)
        return jax.ops.segment_sum(jnp.concat(data, axis=0), jnp.concat(index, axis=0))

    def hess_prod(self, x: Vector, p: Vector) -> Vector:
        data: list[UpdatesData] = []
        index: list[UpdatesIndex] = []
        for energy in self.energies:
            energy_data: UpdatesData
            energy_index: UpdatesIndex
            energy_data, energy_index = energy.hess_prod(x, p)
            data.append(energy_data)
            index.append(energy_index)
        return jax.ops.segment_sum(jnp.concat(data, axis=0), jnp.concat(index, axis=0))

    def hess_mixed_prod(self, x: Vector, p: Vector) -> dict[str, Mapping[str, Array]]:
        outputs: dict[str, Mapping[str, Array]] = {}
        for energy in self.energies:
            outputs[energy.id] = energy.hess_mixed_prod(x, p)
        return outputs
