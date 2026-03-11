from typing import override

import jarp
import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.apple.model import Full, ModelMaterials

from ._base import Loss

type Scalar = Float[Array, ""]


@jarp.define
class UniformActivationLoss(Loss):
    name: str = jarp.static(default="uniform_activation", kw_only=True)

    @override
    @jarp.jit(inline=True)
    def fun(self, u_full: Full, materials: ModelMaterials) -> Scalar:
        act: Float[Array, "face 6"] = materials["muscle"]["activation"]
        return jnp.sum(jnp.var(act, axis=0))
