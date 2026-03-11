from typing import Self, override

import jarp
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Integer

from liblaf.apple.model import Full, ModelMaterials

from ._base import Loss

type Scalar = Float[Array, ""]


@jarp.define
class PointToPointLoss(Loss):
    name: str = jarp.static(default="point_to_point", kw_only=True)

    @override
    @jarp.jit(inline=True)
    def fun(self, u_full: Full, materials: ModelMaterials) -> Scalar:
        diff: Float[Array, "face 3"] = u_full[self.indices] - self.target
        diff *= 10.0  # centimeter to millimeter
        return jnp.average(jnp.sum(jnp.square(diff), axis=-1), weights=self.weights)
