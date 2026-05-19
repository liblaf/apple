import numpy as np
import torch
import warp as wp
from jaxtyping import Float
from torch import Tensor

from liblaf.apple.common import ACTIVATION, ACTIVATION_INV, FRACTION
from liblaf.apple.torch.fem import Region
from liblaf.apple.warp.model import ArrayAnnotation


def get_dhdX(region: Region, annotation: ArrayAnnotation) -> wp.array:
    return wp.from_torch(region.dhdX.contiguous(), dtype=annotation.dtype)


def get_dV(region: Region, annotation: ArrayAnnotation) -> wp.array:
    dV: Float[Tensor, "c q"] = region.dV
    fraction: Float[np.ndarray, " c"] | None = region.cell_data.get(FRACTION.vtk)
    if fraction is not None:
        fraction: Float[Tensor, " c"] = torch.as_tensor(
            fraction, dtype=torch.get_default_dtype()
        )
        dV: Float[Tensor, "c q"] = fraction[:, None] * dV
    return wp.from_torch(dV.contiguous(), dtype=annotation.dtype)


def get_activation_inv(region: Region, annotation: ArrayAnnotation) -> wp.array:
    activation_inv: Float[np.ndarray, "c 6"] | None = region.cell_data.get(
        ACTIVATION_INV.vtk
    )
    if activation_inv is not None:
        return wp.from_numpy(activation_inv, annotation.dtype)
    activation: Float[np.ndarray, "c 6"] | None = region.cell_data.get(ACTIVATION.vtk)
    if activation is not None:
        A: Float[np.ndarray, "c 3 3"] = np.zeros((region.mesh.n_cells, 3, 3))
        A[:, 0, 0] = 1.0 + activation[:, 0]
        A[:, 1, 1] = 1.0 + activation[:, 1]
        A[:, 2, 2] = 1.0 + activation[:, 2]
        A[:, 0, 1] = A[:, 1, 0] = activation[:, 3]
        A[:, 0, 2] = A[:, 2, 0] = activation[:, 4]
        A[:, 1, 2] = A[:, 2, 1] = activation[:, 5]
        A_inv: Float[np.ndarray, "c 3 3"] = np.linalg.inv(A)
        activation_inv: Float[np.ndarray, "c 6"] = np.zeros((region.mesh.n_cells, 6))
        activation_inv[:, 0] = A_inv[:, 0, 0] - 1.0
        activation_inv[:, 1] = A_inv[:, 1, 1] - 1.0
        activation_inv[:, 2] = A_inv[:, 2, 2] - 1.0
        activation_inv[:, 3] = A_inv[:, 0, 1]
        activation_inv[:, 4] = A_inv[:, 0, 2]
        activation_inv[:, 5] = A_inv[:, 1, 2]
        return wp.from_numpy(activation_inv, annotation.dtype)
    return wp.zeros((region.mesh.n_cells,), annotation.dtype)
