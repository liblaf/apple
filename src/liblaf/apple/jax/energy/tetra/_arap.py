from collections.abc import Container
from typing import override

import attrs
import jax
from jaxtyping import Float, PyTree

from liblaf.apple.jax import math

from ._abc import EnergyTetra, EnergyTetraElement


@attrs.frozen
class ArapElement(EnergyTetraElement):
    @property
    def required_params(self) -> Container[str]:
        return {"lambda", "mu"}

    @override
    def energy_density(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        mu: Float[jax.Array, ""] = q["mu"]
        R: Float[jax.Array, "3 3"]
        R, _S = math.polar_rv(F)
        R = jax.lax.stop_gradient(R)
        Psi: Float[jax.Array, ""] = 0.5 * mu * math.frobenius_norm_square(F - R)
        return Psi


@attrs.frozen
class Arap(EnergyTetra):
    elem: EnergyTetraElement = attrs.field(factory=ArapElement)
