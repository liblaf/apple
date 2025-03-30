from collections.abc import Sequence
from typing import override

import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Float, PyTree

from liblaf import apple

from . import MaterialTetra, MaterialTetraElement


@apple.register_dataclass()
@attrs.define(kw_only=True)
class CorotatedElement(MaterialTetraElement):
    @property
    @override
    def required_params(self) -> Sequence[str]:
        return ["lambda", "mu"]

    @override
    def strain_energy_density(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        lmbda: Float[jax.Array, ""] = q["lambda"]
        mu: Float[jax.Array, ""] = q["mu"]
        R: Float[jax.Array, "3 3"]
        R, _S = apple.math.polar_rv(F)
        J: Float[jax.Array, ""] = jnp.linalg.det(F)
        return mu * apple.math.norm_sqr(F - R) + lmbda * (J - 1) ** 2

    @override
    def first_piola_kirchhoff_stress(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> jax.Array:
        lmbda: Float[jax.Array, ""] = q["lambda"]
        mu: Float[jax.Array, ""] = q["mu"]
        R: Float[jax.Array, "3 3"]
        R, _S = apple.math.polar_rv(F)
        J: Float[jax.Array, ""] = jnp.linalg.det(F)
        dJdF: Float[jax.Array, "3 3"] = jax.grad(jnp.linalg.det)(F)
        return 2 * mu * (F - R) + 2 * lmbda * (J - 1) * dJdF


@apple.register_dataclass()
@attrs.define(kw_only=True)
class Corotated(MaterialTetra):
    elem: MaterialTetraElement = attrs.field(
        factory=CorotatedElement, metadata={"static": True}
    )
