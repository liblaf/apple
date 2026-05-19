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
