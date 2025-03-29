from typing import override

import attrs
import jax
from jaxtyping import Float, PyTree

from liblaf import apple

from . import MaterialTetra


@apple.register_dataclass()
@attrs.define(kw_only=True)
class AsRigidAsPossible(MaterialTetra):
    @override
    def _Psi_elem(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        mu: Float[jax.Array, ""] = q["mu"]
        dV: Float[jax.Array, ""] = aux["dV"]
        R: Float[jax.Array, "3 3"]
        R, _S = apple.math.polar_rv(F)
        R = jax.lax.stop_gradient(R)  # TODO: remove this workaround
        Psi: Float[jax.Array, ""] = 0.5 * mu * apple.math.norm_sqr(F - R)
        return Psi * dV

    @override
    def _PK1_elem(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "3 3"]:
        mu: Float[jax.Array, ""] = q["mu"]
        R: Float[jax.Array, "3 3"]
        R, _S = apple.math.polar_rv(F)
        R = jax.lax.stop_gradient(R)  # TODO: remove this workaround
        PK1: Float[jax.Array, "3 3"] = mu * (F - R)
        return PK1
