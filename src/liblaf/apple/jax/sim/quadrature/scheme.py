from typing import Self

import felupe.quadrature
import jax
from jaxtyping import Array, Float

from liblaf.apple import struct
from liblaf.apple.jax import math


@struct.pytree
class Scheme:
    points: Float[Array, "q J"] = struct.array()
    weights: Float[Array, " q"] = struct.array()

    @classmethod
    def from_felupe(cls, scheme: felupe.quadrature.Scheme) -> Self:
        with jax.ensure_compile_time_eval():
            return cls(
                points=math.asarray(scheme.points), weights=math.asarray(scheme.weights)
            )

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]
