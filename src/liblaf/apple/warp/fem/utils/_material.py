from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import warp as wp
from jaxtyping import Float
from torch import Tensor

from liblaf.apple.common import ACTIVATION, LAMBDA, MU
from liblaf.apple.torch.fem import Region
from liblaf.apple.warp.utils import warp_default_dtype


def get_activation(region: Region) -> wp.array:
    floating: Any = warp_default_dtype()
    vec6: Any = wp.types.vector(6, dtype=floating)
    return wp.from_numpy(region.cell_data[ACTIVATION.vtk], vec6)


def get_fraction(region: Region) -> Float[Tensor, " cells"]:
    fraction: Float[np.ndarray, " cells"] | None = region.cell_data.get("Fraction")
    if fraction is None:
        return torch.ones((region.mesh.n_cells,))
    return torch.as_tensor(fraction)


def get_lambda(region: Region) -> wp.array:
    floating: Any = warp_default_dtype()
    return wp.from_numpy(region.cell_data[LAMBDA.vtk], floating)


def get_mu(region: Region) -> wp.array:
    floating: Any = warp_default_dtype()
    return wp.from_numpy(region.cell_data[MU.vtk], floating)


def require_grads(materials: Any, requires_grad: Sequence[str]) -> None:
    for name in requires_grad:
        arr: wp.array = getattr(materials, name)
        arr.requires_grad = True
        setattr(materials, name, arr)
