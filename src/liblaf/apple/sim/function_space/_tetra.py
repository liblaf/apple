from typing import override

import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import struct, utils

from ._abc import FunctionSpace


class FunctionSpaceTetraCell(FunctionSpace):
    w: Float[jax.Array, ""] = struct.array(default=jnp.asarray(1.0 / 6.0), init=False)
    h: Float[jax.Array, "a=4"] = struct.array(
        default=jnp.asarray([0.25, 0.25, 0.25, 0.25]), init=False
    )
    dh_dr: Float[jax.Array, "a=4 J=3"] = struct.array(
        default=jnp.asarray(
            [
                [-1, -1, -1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        ),
        init=False,
    )

    @property
    @override
    def n_points(self) -> int:
        return self.n_cells

    @override
    @utils.jit
    def scatter(
        self, values: Float[jax.Array, " cells *dim"]
    ) -> Float[jax.Array, " cells *dim"]:
        values = jnp.asarray(values)
        return values

    @utils.jit
    def gather(
        self, values: Float[jax.Array, " cells *dim"]
    ) -> Float[jax.Array, " cells *dim"]:
        values = jnp.asarray(values)
        return values


class FunctionSpaceTetraPoint(FunctionSpace):
    w: Float[jax.Array, ""] = struct.array(default=jnp.asarray(1.0 / 6.0), init=False)
    h: Float[jax.Array, "a=4"] = struct.array(
        default=jnp.asarray([0.25, 0.25, 0.25, 0.25]), init=False
    )
    dh_dr: Float[jax.Array, "a=4 J=3"] = struct.array(
        default=jnp.asarray(
            [
                [-1, -1, -1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        ),
        init=False,
    )

    @property
    @override
    def grad(self) -> FunctionSpaceTetraCell:
        return FunctionSpaceTetraCell.from_domain(self.domain)
