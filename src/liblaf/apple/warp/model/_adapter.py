from collections.abc import Callable, Mapping

import tlz
import warp as wp
import warp.jax_experimental
import warp.jax_experimental.ffi
from jaxtyping import Array, Float
from liblaf.peach import tree

import liblaf.apple.warp.types as wpt

from ._energy import WarpEnergy
from ._model import WarpModel

type Scalar = Float[Array, ""]
type Vector = Float[Array, "points dim"]
type EnergyParams = Mapping[str, Array]
type ModelParams = Mapping[str, EnergyParams]


@tree.define
class WarpModelAdapter:
    wrapped: WarpModel

    @property
    def dim(self) -> int:
        return self.wrapped.dim

    @property
    def energies(self) -> Mapping[str, WarpEnergy]:
        return self.wrapped.energies

    def update(self, u: Vector) -> None:
        u_wp: wp.array = _to_warp(u)
        self.wrapped.update(u_wp)

    def update_params(self, params: ModelParams) -> None:
        self.wrapped.update_params(params)

    def fun(self, u: Vector) -> Scalar:
        u_wp: wp.array = _to_warp(u)
        output_wp: wp.array = wp.zeros((1,), dtype=wp.dtype_from_jax(u.dtype))
        self.wrapped.fun(u_wp, output_wp)
        return wp.to_jax(output_wp)[0]

    def grad(self, u: Vector) -> Vector:
        u_wp: wp.array = _to_warp(u)
        output_wp: wp.array = wp.zeros_like(u_wp)
        self.wrapped.grad(u_wp, output_wp)
        return wp.to_jax(output_wp)

    def hess_diag(self, u: Vector) -> Vector:
        u_wp: wp.array = _to_warp(u)
        output_wp: wp.array = wp.zeros_like(u_wp)
        self.wrapped.hess_diag(u_wp, output_wp)
        return wp.to_jax(output_wp)

    def hess_prod(self, u: Vector, p: Vector) -> Vector:
        output: Vector
        (output,) = self._hess_prod_callable(u, p, output_dims=u.shape)
        return output

    def hess_quad(self, u: Vector, p: Vector) -> Scalar:
        u_wp: wp.array = _to_warp(u)
        p_wp: wp.array = _to_warp(p)
        output_wp: wp.array = wp.zeros((1,), dtype=wp.dtype_from_jax(u.dtype))
        self.wrapped.hess_quad(u_wp, p_wp, output_wp)
        return wp.to_jax(output_wp)[0]

    def mixed_derivative_prod(
        self, u: Vector, p: Vector
    ) -> dict[str, dict[str, Array]]:
        u_wp: wp.array = _to_warp(u)
        p_wp: wp.array = _to_warp(p)
        outputs_wp: dict[str, dict[str, wp.array]] = self.wrapped.mixed_derivative_prod(
            u_wp, p_wp
        )
        outputs: dict[str, dict[str, Array]] = tlz.valmap(
            lambda energy_dict: tlz.valmap(wp.to_jax, energy_dict), outputs_wp
        )
        return outputs

    def value_and_grad(self, u: Vector) -> tuple[Scalar, Vector]:
        u_wp: wp.array = _to_warp(u)
        value_wp: wp.array = wp.zeros((1,), dtype=wp.dtype_from_jax(u.dtype))
        grad_wp: wp.array = wp.zeros_like(u_wp)
        self.wrapped.value_and_grad(u_wp, value_wp, grad_wp)
        value: Scalar = wp.to_jax(value_wp)[0]
        grad: Vector = wp.to_jax(grad_wp)
        return value, grad

    def grad_and_hess_diag(self, u: Vector) -> tuple[Vector, Vector]:
        u_wp: wp.array = _to_warp(u)
        grad_wp: wp.array = wp.zeros_like(u_wp)
        hess_diag_wp: wp.array = wp.zeros_like(u_wp)
        self.wrapped.grad_and_hess_diag(u_wp, grad_wp, hess_diag_wp)
        grad: Vector = wp.to_jax(grad_wp)
        hess_diag: Vector = wp.to_jax(hess_diag_wp)
        return grad, hess_diag

    @property
    def _hess_prod_callable(self) -> Callable:
        @warp.jax_experimental.jax_callable
        def hess_prod_callable(
            u: wp.array(dtype=wp.types.vector(self.dim, wpt.float_)),
            p: wp.array(dtype=wp.types.vector(self.dim, wpt.float_)),
            output: wp.array(dtype=wp.types.vector(self.dim, wpt.float_)),
        ) -> None:
            output.zero_()
            self.wrapped.hess_prod(u, p, output)

        return hess_prod_callable


def _to_warp(u: Vector) -> wp.array:
    _, dim = u.shape
    u_wp: wp.array = wp.from_jax(u, wp.types.vector(dim, wp.dtype_from_jax(u.dtype)))
    return u_wp
