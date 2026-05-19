import attrs
import torch
from jaxtyping import Float, Integer
from torch import Tensor

from liblaf.apple.torch.fem.quadrature import Scheme


@attrs.define
class Element:
    """Base-class for a finite element which provides methods for plotting.

    References:
        1. [felupe.Element](https://felupe.readthedocs.io/en/latest/felupe/element.html#felupe.Element)
    """

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def points(self) -> Float[Tensor, "points dim"]:
        raise NotImplementedError

    @property
    def cells(self) -> Integer[Tensor, " points"]:
        return torch.arange(self.n_points)

    @property
    def quadrature(self) -> Scheme:
        return NotImplemented

    def function(self, coords: Float[Tensor, " dim"]) -> Float[Tensor, " points"]:
        """Return the shape functions at given coordinates."""
        raise NotImplementedError

    def gradient(self, coords: Float[Tensor, " dim"]) -> Float[Tensor, "points dim"]:
        return torch.autograd.functional.jacobian(self.function, coords)

    def hessian(self, coords: Float[Tensor, " dim"]) -> Float[Tensor, "points dim dim"]:
        return torch.autograd.functional.hessian(self.function, coords)
