from collections.abc import Mapping, Sequence

import warp as wp
from jaxtyping import Array, Float
from liblaf.peach import tree

import liblaf.apple.warp.utils as wpu
from liblaf.apple.utils import IdMixin

type EnergyParams = Mapping[str, Array]
type Scalar = Float[wp.array, ""]
type Vector = Float[wp.array, " N"]


@tree.define
class WarpEnergy(IdMixin):
    requires_grad: Sequence[str] = tree.field(default=(), kw_only=True)

    def update(self, u: Vector) -> None:
        pass

    def update_params(self, params: EnergyParams) -> None:
        for name, value in params.items():
            param: wp.array = getattr(self, name)
            wp.copy(param, wpu.to_warp(value, param.dtype))

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
