import jarp
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from liblaf.apple.jax.fem.quadrature import Scheme


@jarp.define
class Element:
    """Base-class for a finite element which provides methods for plotting.

    References:
        1. [felupe.Element](https://felupe.readthedocs.io/en/latest/felupe/element.html#felupe.Element)
    """

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def points(self) -> Float[Array, "points dim"]:
        raise NotImplementedError

    @property
    def cells(self) -> Integer[Array, " points"]:
        with jax.ensure_compile_time_eval():
            return jnp.arange(self.n_points)

    @property
    def quadrature(self) -> Scheme:
        return None  # pyright: ignore[reportReturnType]

    def function(self, coords: Float[Array, " dim"]) -> Float[Array, " points"]:
        """Return the shape functions at given coordinates."""
        raise NotImplementedError

    def gradient(self, coords: Float[Array, " dim"]) -> Float[Array, "points dim"]:
        return jax.jacobian(self.function)(coords)

    def hessian(self, coords: Float[Array, " dim"]) -> Float[Array, "points dim dim"]:
        return jax.hessian(self.function)(coords)
