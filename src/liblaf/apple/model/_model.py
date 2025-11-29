import jax.numpy as jnp
import warp as wp
from jaxtyping import Array, ArrayLike, Float
from liblaf.peach import tree

from liblaf.apple.jax import Dirichlet, JaxModel
from liblaf.apple.warp import WarpModel

type Free = Float[Array, " free"]
type Full = Float[Array, "points dim"]
type FreeOrFull = Free | Full
type Scalar = Float[Array, ""]


@tree.define
class Model:
    dirichlet: Dirichlet
    u_full: Full
    jax: JaxModel
    warp: WarpModel

    @property
    def dim(self) -> int:
        return self.dirichlet.dim

    @property
    def n_free(self) -> int:
        return self.dirichlet.n_free

    @property
    def n_full(self) -> int:
        return self.dirichlet.n_full

    @property
    def n_points(self) -> int:
        return self.dirichlet.n_points

    def to_free(self, u: FreeOrFull) -> Free:
        if u.size == self.n_free:
            return u.reshape((self.n_free,))
        return self.dirichlet.get_free(u)

    def to_full(
        self, u: FreeOrFull, dirichlet: Float[ArrayLike, " dirichlet"] | None = None
    ) -> Full:
        if u.size == self.n_full:
            return u.reshape((self.n_points, self.dim))
        return self.dirichlet.to_full(u, dirichlet)

    def to_shape_like(self, u_full: Full, like: FreeOrFull) -> FreeOrFull:
        if u_full.size == like.size:
            return u_full.reshape(like.shape)
        return self.dirichlet.get_free(u_full)

    def update(self, u: FreeOrFull) -> None:
        u_full: Full = self.to_full(u)
        if jnp.array_equiv(u_full, self.u_full):
            return
        self.u_full = u_full
        self.jax.update(u_full)
        self.warp.update(self._to_warp(u_full))

    def fun(self, u: FreeOrFull) -> Scalar:
        u_full: Full = self.to_full(u)
        self.update(u_full)
        output_jax: Scalar = self.jax.fun(u_full)
        u_wp: wp.array = self._to_warp(u_full)
        output_wp: wp.array = wp.zeros((1,), dtype=wp.dtype_from_jax(u_full.dtype))
        self.warp.fun(u_wp, output_wp)
        output: Scalar = output_jax + wp.to_jax(output_wp)[0]
        return output

    def grad(self, u: FreeOrFull) -> FreeOrFull:
        u_full: Full = self.to_full(u)
        self.update(u_full)
        output_jax: Full = self.jax.grad(u_full)
        u_wp: wp.array = self._to_warp(u_full)
        output_wp: wp.array = wp.zeros_like(u_wp)
        self.warp.grad(u_wp, output_wp)
        output: Full = output_jax + wp.to_jax(output_wp)
        return self.to_shape_like(output, u)

    def hess_diag(self, u: FreeOrFull) -> FreeOrFull:
        u_full: Full = self.to_full(u)
        self.update(u_full)
        output_jax: Full = self.jax.hess_diag(u_full)
        u_wp: wp.array = self._to_warp(u_full)
        output_wp: wp.array = wp.zeros_like(u_wp)
        self.warp.hess_diag(u_wp, output_wp)
        output: Full = output_jax + wp.to_jax(output_wp)
        return self.to_shape_like(output, u)

    def hess_prod(self, u: FreeOrFull, p: FreeOrFull) -> FreeOrFull:
        u_full: Full = self.to_full(u)
        self.update(u_full)
        p_full: Full = self.to_full(p, 0.0)
        output_jax: Full = self.jax.hess_prod(u_full, p_full)
        u_wp: wp.array = self._to_warp(u_full)
        p_wp: wp.array = self._to_warp(p_full)
        output_wp: wp.array = wp.zeros_like(u_wp)
        self.warp.hess_prod(u_wp, p_wp, output_wp)
        output: Full = output_jax + wp.to_jax(output_wp)
        return self.to_shape_like(output, u)

    def hess_quad(self, u: FreeOrFull, p: FreeOrFull) -> Scalar:
        u_full: Full = self.to_full(u)
        self.update(u_full)
        p_full: Full = self.to_full(p, 0.0)
        output_jax: Scalar = self.jax.hess_quad(u_full, p_full)
        u_wp: wp.array = self._to_warp(u_full)
        p_wp: wp.array = self._to_warp(p_full)
        output_wp: wp.array = wp.zeros((1,), dtype=wp.dtype_from_jax(u_full.dtype))
        self.warp.hess_quad(u_wp, p_wp, output_wp)
        output: Scalar = output_jax + wp.to_jax(output_wp)[0]
        return output

    def mixed_derivative_prod(
        self, u: FreeOrFull, p: FreeOrFull
    ) -> dict[str, dict[str, Array]]:
        u_full: Full = self.to_full(u)
        self.update(u_full)
        p_full: Full = self.to_full(p, 0.0)
        outputs_jax: dict[str, dict[str, Array]] = self.jax.mixed_derivative_prod(
            u_full, p_full
        )
        u_wp: wp.array = self._to_warp(u_full)
        p_wp: wp.array = self._to_warp(p_full)
        outputs_wp: dict[str, dict[str, wp.array]] = self.warp.mixed_derivative_prod(
            u_wp, p_wp
        )
        outputs: dict[str, dict[str, Array]] = outputs_jax
        for name, output_wp in outputs_wp.items():
            outputs[name] = {key: wp.to_jax(value) for key, value in output_wp.items()}
        return outputs

    def value_and_grad(self, u: FreeOrFull) -> tuple[Scalar, FreeOrFull]:
        u_full: Full = self.to_full(u)
        self.update(u_full)
        value_jax: Scalar
        grad_jax: Full
        value_jax, grad_jax = self.jax.value_and_grad(u_full)
        u_wp: wp.array = self._to_warp(u_full)
        value_wp: wp.array = wp.zeros((1,), dtype=wp.dtype_from_jax(u_full.dtype))
        grad_wp: wp.array = wp.zeros_like(u_wp)
        self.warp.value_and_grad(u_wp, value_wp, grad_wp)
        value: Scalar = value_jax + wp.to_jax(value_wp)[0]
        grad: Full = grad_jax + wp.to_jax(grad_wp)
        return value, self.to_shape_like(grad, u)

    def grad_and_hess_diag(self, u: FreeOrFull) -> tuple[FreeOrFull, FreeOrFull]:
        u_full: Full = self.to_full(u)
        self.update(u_full)
        grad_jax: Full
        hess_diag_jax: Full
        grad_jax, hess_diag_jax = self.jax.grad_and_hess_diag(u_full)
        u_wp: wp.array = self._to_warp(u_full)
        grad_wp: wp.array = wp.zeros_like(u_wp)
        hess_diag_wp: wp.array = wp.zeros_like(u_wp)
        self.warp.grad_and_hess_diag(u_wp, grad_wp, hess_diag_wp)
        grad: Full = grad_jax + wp.to_jax(grad_wp)
        hess_diag: Full = hess_diag_jax + wp.to_jax(hess_diag_wp)
        return self.to_shape_like(grad, u), self.to_shape_like(hess_diag, u)

    def _to_warp(self, u_full: Full) -> wp.array:
        _, dim = u_full.shape
        return wp.from_jax(
            u_full, wp.types.vector(dim, wp.dtype_from_jax(u_full.dtype))
        )
