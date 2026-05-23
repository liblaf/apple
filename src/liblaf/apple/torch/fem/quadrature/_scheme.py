from typing import Self

import attrs
import felupe
import torch
from jaxtyping import Float
from torch import Tensor


@attrs.define
class Scheme:
    points: Float[Tensor, "quadrature dim"]
    weights: Float[Tensor, " quadrature"]

    @classmethod
    def from_felupe(cls, schema: felupe.quadrature.Scheme) -> Self:
        return cls(
            points=torch.tensor(schema.points),
            weights=torch.tensor(schema.weights),
        )

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]
