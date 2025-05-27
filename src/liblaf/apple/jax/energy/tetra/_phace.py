from collections.abc import Container
from typing import override

import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Float, PyTree

from liblaf.apple.jax import math

from ._abc import EnergyTetra, EnergyTetraElement


@attrs.frozen
class PhaceElement(EnergyTetraElement):
    @property
    def required_params(self) -> Container[str]:
        return {"activation", "active-fraction", "lambda", "mu"}

    @override
    def energy_density(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        A: Float[jax.Array, "3 3"] = q["activation"]
        active_fraction: Float[jax.Array, ""] = q["active-fraction"]
        lambda_: Float[jax.Array, ""] = q["lambda"]
        mu: Float[jax.Array, ""] = q["mu"]
        R: Float[jax.Array, "3 3"]
        R, _S = math.polar_rv(F)
        R = jax.lax.stop_gradient(R)
        Psi_passive: Float[jax.Array, ""] = (
            mu * math.frobenius_norm_square(F - R)
            + lambda_ * (jnp.linalg.det(F) - 1) ** 2
        )
        Psi_active: Float[jax.Array, ""] = (
            mu * math.frobenius_norm_square(F - R @ A)
            + lambda_ * (jnp.linalg.det(F) - jnp.linalg.det(A)) ** 2
        )
        Psi: Float[jax.Array, ""] = (
            1 - active_fraction
        ) * Psi_passive + active_fraction * Psi_active
        return Psi


@attrs.frozen
class Phace(EnergyTetra):
    elem: EnergyTetraElement = attrs.field(factory=PhaceElement)
