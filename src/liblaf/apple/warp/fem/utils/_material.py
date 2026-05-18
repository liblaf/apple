from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import liblaf.jarp.warp.types as wpt
import numpy as np
import warp as wp
from jaxtyping import Array, Float

from liblaf import jarp
from liblaf.apple.common import ACTIVATION, LAMBDA, MU
from liblaf.apple.jax.fem import Region


def get_activation(region: Region) -> wp.array:
    return jarp.to_warp(region.cell_data[ACTIVATION], wpt.vector(6))


def get_fraction(region: Region) -> Float[Array, " cells"]:
    fraction: Float[np.ndarray, " cells"] | None = region.cell_data.get("Fraction")
    if fraction is None:
        return jnp.ones((region.mesh.n_cells,))
    return jnp.asarray(fraction)


def get_lambda(region: Region) -> wp.array:
    return jarp.to_warp(region.cell_data[LAMBDA.vtk], wpt.floating)


def get_mu(region: Region) -> wp.array:
    return jarp.to_warp(region.cell_data[MU.vtk], wpt.floating)


def require_grads(materials: Any, requires_grad: Sequence[str]) -> None:
    for name in requires_grad:
        arr: wp.array = getattr(materials, name)
        arr.requires_grad = True
        setattr(materials, name, arr)
