from typing import Self

import felupe
import jax
from jaxtyping import Float

from liblaf.apple import struct


class Scheme(struct.PyTree):
    """A quadrature scheme with integration points $x_q$ and weights $w_q$. It approximates the integral of a function over a region $V$ by a weighted sum of function values $f_q = f(x_q)$, evaluated on the quadrature-points.

    Shape Annotations:
        - `J`: `scheme.dim`
        - `q`: `scheme.n_points`

    References:
        1. [felupe.quadrature.Schema](https://felupe.readthedocs.io/en/latest/felupe/quadrature.html#felupe.quadrature.Scheme)
    """

    _points: Float[jax.Array, "q J"] = struct.array(default=None)
    _weights: Float[jax.Array, " q"] = struct.array(default=None)

    @classmethod
    def from_felupe(cls, scheme: felupe.quadrature.Scheme) -> Self:
        with jax.ensure_compile_time_eval():
            self: Self = cls(_points=scheme.points, _weights=scheme.weights)
            return self

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def points(self) -> Float[jax.Array, "q J"]:
        return self._points

    @property
    def weights(self) -> Float[jax.Array, " q"]:
        return self._weights
