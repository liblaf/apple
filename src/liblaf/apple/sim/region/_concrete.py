from typing import Self, override

import einops
import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import struct, utils
from liblaf.apple.sim import geometry as _g
from liblaf.apple.sim import quadrature as _q

from ._region import Region


class RegionConcrete(Region):
    _h: Float[jax.Array, "q a"] = struct.array(default=None)
    _dhdr: Float[jax.Array, "q a J"] = struct.array(default=None)
    _dXdr: Float[jax.Array, "c q I J"] = struct.array(default=None)
    _drdX: Float[jax.Array, "c q J I"] = struct.array(default=None)
    _dV: Float[jax.Array, "c q"] = struct.array(default=None)
    _dhdX: Float[jax.Array, "c q a J"] = struct.array(default=None)

    @classmethod
    def from_geometry(
        cls, geometry: _g.Geometry, quadrature: _q.Scheme, *, grad: bool = True
    ) -> Self:
        self: Self = cls(_geometry=geometry, _quadrature=quadrature)
        if grad:
            self = self.with_grad()
        return self

    # region FEM

    @property
    @override
    def h(self) -> Float[jax.Array, "q a"]:
        return self._h

    @property
    @override
    def dhdr(self) -> Float[jax.Array, "q a J"]:
        return self._dhdr

    @property
    @override
    def dXdr(self) -> Float[jax.Array, "c q I J"]:
        return self._dXdr

    @property
    @override
    def drdX(self) -> Float[jax.Array, "c q J I"]:
        return self._drdX

    @property
    @override
    def dV(self) -> Float[jax.Array, "c q"]:
        return self._dV

    @property
    @override
    def dhdX(self) -> Float[jax.Array, "c q a J"]:
        return self._dhdX

    # endregion FEM

    @utils.jit
    def with_grad(self) -> Self:
        h: Float[jax.Array, "q a"] = jnp.asarray(
            [self.element.function(q) for q in self.quadrature.points]
        )
        dhdr: Float[jax.Array, "q a J"] = jnp.asarray(
            [self.element.gradient(q) for q in self.quadrature.points]
        )
        dXdr: Float[jax.Array, "c q I J"] = einops.einsum(
            self.points[self.cells], dhdr, "c a I, q a J -> c q I J"
        )
        drdX: Float[jax.Array, "c q J I"] = jnp.linalg.inv(dXdr)
        dV: Float[jax.Array, "c q"] = (
            jnp.linalg.det(dXdr) * self.quadrature.weights[None, :]
        )
        dhdX: Float[jax.Array, "c q a J"] = einops.einsum(
            dhdr, drdX, "q a I, c q I J -> c q a J"
        )
        return self.evolve(_h=h, _dhdr=dhdr, _dXdr=dXdr, _drdX=drdX, _dV=dV, _dhdX=dhdX)
