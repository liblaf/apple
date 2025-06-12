from typing import Self

import felupe
import jax
import jax.numpy as jnp
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple import struct


class Element(struct.PyTree):
    """Base-class for a finite element which provides methods for plotting.

    References:
        1. [felupe.Element](https://felupe.readthedocs.io/en/latest/felupe/element.html#felupe.Element)
    """

    @property
    def n_points(self) -> int:
        raise NotImplementedError

    def function(self, coords: Float[ArrayLike, " c"], /) -> Float[jax.Array, " a"]:
        """Return the shape functions at given coordinates."""
        raise NotImplementedError

    def gradient(self, coords: Float[ArrayLike, " c"], /) -> Float[jax.Array, "a J"]:
        """Return the gradient of shape functions at given coordinates."""
        raise NotImplementedError

    def hessian(self, coords: Float[ArrayLike, " c"], /) -> Float[jax.Array, "a I J"]:
        """Return the Hessian of shape functions at given coordinates."""
        raise NotImplementedError


class ElementFelupe(Element):
    _elem: felupe.Element = struct.static(default=None)

    @classmethod
    def from_felupe(cls, elem: felupe.Element) -> Self:
        return cls(_elem=elem)

    def function(self, coords: Float[ArrayLike, " c"], /) -> Float[jax.Array, " a"]:
        return jnp.asarray(self._elem.function(coords))  # pyright: ignore[reportAttributeAccessIssue]

    def gradient(self, coords: Float[ArrayLike, " c"], /) -> Float[jax.Array, " a J"]:
        return jnp.asarray(self._elem.gradient(coords))  # pyright: ignore[reportAttributeAccessIssue]

    def hessian(self, coords: Float[ArrayLike, " c"], /) -> Float[jax.Array, " a I J"]:
        return jnp.asarray(self._elem.hessian(coords))  # pyright: ignore[reportAttributeAccessIssue]
