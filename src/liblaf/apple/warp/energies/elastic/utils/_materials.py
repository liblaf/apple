from collections.abc import Sequence

import jarp
import jarp.warp.types as wpt
import numpy as np
import warp as wp
from jaxtyping import Float
from warp._src.codegen import StructInstance

from liblaf.apple.consts import ACTIVATION, LAMBDA, MU
from liblaf.apple.jax.fem import Region


def get_activation(region: Region) -> wp.array:
    return jarp.to_warp(region.cell_data[ACTIVATION], wpt.vector(6))


def get_fraction(region: Region) -> wp.array:
    fraction_np: Float[np.ndarray, " cells"] | None = region.cell_data.get("Fraction")
    if fraction_np is None:
        return wp.ones((region.mesh.n_cells,), wpt.floating)
    return jarp.to_warp(fraction_np, wpt.floating)


def get_lambda(region: Region) -> wp.array:
    return jarp.to_warp(region.cell_data[LAMBDA], wpt.floating)


def get_mu(region: Region) -> wp.array:
    return jarp.to_warp(region.cell_data[MU], wpt.floating)


def require_grads(materials: StructInstance, requires_grad: Sequence[str]) -> None:
    for name in requires_grad:
        arr: wp.array = getattr(materials, name)
        arr.requires_grad = True
        setattr(materials, name, arr)
