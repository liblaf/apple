from typing import Self

import felupe.quadrature
import jax
from jaxtyping import Array, Float

from liblaf.apple import struct


@struct.pytree
class Scheme(struct.PyTreeMixin):
    """A quadrature scheme with integration points $x_q$ and weights $w_q$. It approximates the integral of a function over a region $V$ by a weighted sum of function values $f_q = f(x_q)$, evaluated on the quadrature-points.

    Shape Annotations:
        - `J`: `scheme.dim`
        - `q`: `scheme.n_points`

    References:
        1. [felupe.quadrature.Schema](https://felupe.readthedocs.io/en/latest/felupe/quadrature.html#felupe.quadrature.Scheme)
    """

    points: Float[Array, "q J"] = struct.array(default=None)
    weights: Float[Array, " q"] = struct.array(default=None)

    @classmethod
    def from_felupe(cls, scheme: felupe.quadrature.Scheme) -> Self:
        with jax.ensure_compile_time_eval():
            self: Self = cls(points=scheme.points, weights=scheme.weights)
            return self

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]
