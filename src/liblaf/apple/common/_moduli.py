import torch
from jaxtyping import Float
from numpy.typing import ArrayLike
from torch import Tensor


def lame_converter(
    E: Float[ArrayLike, "..."], nu: Float[ArrayLike, "..."]
) -> tuple[Float[Tensor, "..."], Float[Tensor, "..."]]:
    E: Float[Tensor, " ..."] = torch.as_tensor(E)
    nu: Float[Tensor, " ..."] = torch.as_tensor(nu)
    la: Float[Tensor, " ..."] = torch.as_tensor(
        E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    )
    mu: Float[Tensor, " ..."] = torch.as_tensor(E / (2.0 * (1.0 + nu)))
    return la, mu


def lame_converter_plane_stress(
    E: Float[ArrayLike, "..."], nu: Float[ArrayLike, "..."]
) -> tuple[Float[Tensor, "..."], Float[Tensor, "..."]]:
    """Convert 3D Young's modulus and Poisson's ratio for a 2D membrane.

    This is the plane-stress reduction for a thin membrane whose transverse
    normal stress vanishes. Volume materials should continue to use
    :func:`lame_converter`.
    """
    E: Float[Tensor, " ..."] = torch.as_tensor(E)
    nu: Float[Tensor, " ..."] = torch.as_tensor(nu)
    la: Float[Tensor, " ..."] = torch.as_tensor(E * nu / (1.0 - nu.square()))
    mu: Float[Tensor, " ..."] = torch.as_tensor(E / (2.0 * (1.0 + nu)))
    return la, mu
