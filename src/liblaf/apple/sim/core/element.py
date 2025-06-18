import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float, Integer

from liblaf.apple import struct, utils
from liblaf.apple.sim.core.quadrature import Scheme


class Element(struct.PyTree):
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
    def cells(self) -> Integer[jax.Array, "points"]:
        with jax.ensure_compile_time_eval():
            return jnp.arange(self.n_points)

    @property
    def points(self) -> Float[jax.Array, "points dim"]:
        raise NotImplementedError

    @property
    def quadrature(self) -> Scheme:
        raise NotImplementedError

    @utils.not_implemented
    def function(
        self, coords: Float[ArrayLike, "dim"], /
    ) -> Float[jax.Array, "points"]:
        """Return the shape functions at given coordinates."""
        raise NotImplementedError

    @utils.jit
    def gradient(
        self, coords: Float[ArrayLike, "dim"], /
    ) -> Float[jax.Array, "points dim"]:
        """Return the gradient of shape functions at given coordinates."""
        if utils.is_implemented(self.function):
            return jax.jacobian(self.function)(coords)
        raise NotImplementedError

    @utils.jit
    def hessian(
        self, coords: Float[ArrayLike, "dim"], /
    ) -> Float[jax.Array, "points dim dim"]:
        """Return the Hessian of shape functions at given coordinates."""
        if utils.is_implemented(self.function):
            return jax.hessian(self.function)(coords)
        return jax.jacobian(self.gradient)(coords)
