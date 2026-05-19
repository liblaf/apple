from typing import Self

import attrs
import felupe.quadrature
import torch
from jaxtyping import Float
from torch import Tensor

from ._scheme import Scheme


@attrs.define
class QuadratureTetra(Scheme):
    points: Float[Tensor, "quadrature dim"] = attrs.field(
        factory=lambda: torch.full((1, 3), 0.25)
    )
    weights: Float[Tensor, " quadrature"] = attrs.field(
        factory=lambda: torch.full((1,), 1.0 / 6.0)
    )

    @classmethod
    def from_order(cls, order: int = 1) -> Self:
        return cls.from_felupe(felupe.quadrature.Tetrahedron(order=order))
