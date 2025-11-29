from typing import Self

import warp as wp
from jaxtyping import Float
from liblaf.peach import tree

from liblaf.apple.utils import IdMixin

type Vector = Float[wp.array, " N"]
type Scalar = Float[wp.array, ""]


@tree.define
class WarpEnergy(IdMixin):
    requires_grad: frozenset[str] = tree.field(default=frozenset(), kw_only=True)

    def update(self, u: Vector) -> Self:  # noqa: ARG002
        return self

    def fun(self, u: Vector, output: Scalar) -> None:
        raise NotImplementedError

    def grad(self, u: Vector, output: Vector) -> None:
        raise NotImplementedError

    def hess_diag(self, u: Vector, output: Vector) -> None:
        raise NotImplementedError

    def hess_prod(self, u: Vector, p: Vector, output: Vector) -> None:
        raise NotImplementedError

    def hess_quad(self, u: Vector, p: Vector, output: Scalar) -> None:
        raise NotImplementedError

    def mixed_derivative_prod(self, u: Vector, p: Vector) -> dict[str, wp.array]:
        raise NotImplementedError

    def value_and_grad(self, u: Vector, value: Scalar, grad: Vector) -> None:
        self.fun(u, value)
        self.grad(u, grad)

    def grad_and_hess_diag(self, u: Vector, grad: Vector, hess_diag: Vector) -> None:
        self.grad(u, grad)
        self.hess_diag(u, hess_diag)
