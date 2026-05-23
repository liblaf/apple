from typing import override

import attrs
import torch
from jaxtyping import Float
from torch import Tensor

from liblaf.apple.torch.fem.quadrature import QuadratureTetra

from ._element import Element


@attrs.define
class ElementTetra(Element):
    @property
    @override
    def points(self) -> Float[Tensor, "points dim"]:
        return torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    @property
    @override
    def quadrature(self) -> QuadratureTetra:
        return QuadratureTetra()

    @override
    def function(self, coords: Float[Tensor, " dim"]) -> Float[Tensor, "points=4"]:
        r, s, t = coords
        return torch.tensor([1.0 - r - s - t, r, s, t])

    @override
    def gradient(self, coords: Float[Tensor, " dim"]) -> Float[Tensor, "points dim"]:
        return torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    @override
    def hessian(self, coords: Float[Tensor, " dim"]) -> Float[Tensor, "points dim dim"]:
        return torch.zeros((4, 3, 3))
