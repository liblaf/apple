from typing import Self, override

import felupe
import jax
import jax.numpy as jnp
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple import struct
from liblaf.apple.sim.abc import Element


class ElementFelupe(Element):
    _elem: felupe.Element = struct.static(default=None)

    @classmethod
    def from_felupe(cls, elem: felupe.Element) -> Self:
        return cls(_elem=elem)

    @property
    @override
    def points(self) -> Float[jax.Array, "points dim"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(self._elem.points)  # pyright: ignore[reportAttributeAccessIssue]

    @override
    def function(
        self, coords: Float[ArrayLike, " dim"], /
    ) -> Float[jax.Array, "points"]:
        return jnp.asarray(self._elem.function(coords))  # pyright: ignore[reportAttributeAccessIssue]

    @override
    def gradient(
        self, coords: Float[ArrayLike, " dim"], /
    ) -> Float[jax.Array, "points dim"]:
        return jnp.asarray(self._elem.gradient(coords))  # pyright: ignore[reportAttributeAccessIssue]

    @override
    def hessian(
        self, coords: Float[ArrayLike, " dim"], /
    ) -> Float[jax.Array, "points dim dim"]:
        return jnp.asarray(self._elem.hessian(coords))  # pyright: ignore[reportAttributeAccessIssue]
