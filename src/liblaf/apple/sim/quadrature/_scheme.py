from typing import Self

import felupe
import jax
from jaxtyping import Float

from liblaf.apple import struct


class Scheme(struct.PyTree):
    """A quadrature scheme with integration points $x_q$ and weights $w_q$. It approximates the integral of a function over a region $V$ by a weighted sum of function values $f_q = f(x_q)$, evaluated on the quadrature-points.

    References:
        1. [felupe.quadrature.Schema](https://felupe.readthedocs.io/en/latest/felupe/quadrature.html#felupe.quadrature.Scheme)
    """

    points: Float[jax.Array, "a J"] = struct.array(default=None)
    weights: Float[jax.Array, " q"] = struct.array(default=None)

    @classmethod
    def from_felupe(cls, scheme: felupe.quadrature.Scheme) -> Self:
        return cls(points=scheme.points, weights=scheme.weights)

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]
