from collections.abc import Container, Mapping

import jax.numpy as jnp
import tlz
from jaxtyping import Array, ArrayLike, Float
from liblaf.peach import tree

from liblaf.apple.jax import Dirichlet, JaxModel
from liblaf.apple.warp import WarpModelAdapter

type EnergyParams = Mapping[str, Array]
type Free = Float[Array, " free"]
type FreeOrFull = Free | Full
type Full = Float[Array, "points dim"]
type ModelParams = Mapping[str, EnergyParams]
type Scalar = Float[Array, ""]


@tree.define
class Model:
    dirichlet: Dirichlet
    u_full: Full
    jax: JaxModel
    warp: WarpModelAdapter
    edges_length_mean: Scalar = tree.array(default=jnp.zeros(()))
    frozen: bool = tree.field(default=False)

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

    @property
    def u_free(self) -> Free:
        return self.to_free(self.u_full)

    @u_free.setter
    def u_free(self, value: Free) -> None:
        self.u_full = self.to_full(value)

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
        if self.frozen:
            return
        if jnp.array_equiv(u_full, self.u_full):
            return
        self.u_full = u_full
        self.jax.update(u_full)
        self.warp.update(u_full)

    def update_params(self, params: ModelParams) -> None:
        def pick(allowlist: Container[str], d: ModelParams) -> ModelParams:
            return tlz.keyfilter(lambda name: name in allowlist, d)

        params_jax: ModelParams = pick(self.jax.energies, params)
        params_warp: ModelParams = pick(self.warp.energies, params)
        self.jax.update_params(params_jax)
        self.warp.update_params(params_warp)

    def fun(self, u: FreeOrFull) -> Scalar:
        u_full: Full = self.to_full(u)
        self.update(u_full)
        output_jax: Scalar = self.jax.fun(u_full)
        output_wp: Scalar = self.warp.fun(u_full)
        output: Scalar = output_jax + output_wp
        return output

    def grad(self, u: FreeOrFull) -> FreeOrFull:
        u_full: Full = self.to_full(u)
        self.update(u_full)
        output_jax: Full = self.jax.grad(u_full)
        output_wp: Full = self.warp.grad(u_full)
        output: Full = output_jax + output_wp
        return self.to_shape_like(output, u)

    def hess_diag(self, u: FreeOrFull) -> FreeOrFull:
        u_full: Full = self.to_full(u)
        self.update(u_full)
        output_jax: Full = self.jax.hess_diag(u_full)
        output_wp: Full = self.warp.hess_diag(u_full)
        output: Full = output_jax + output_wp
        return self.to_shape_like(output, u)

    def hess_prod(self, u: FreeOrFull, p: FreeOrFull) -> FreeOrFull:
        u_full: Full = self.to_full(u)
        self.update(u_full)
        p_full: Full = self.to_full(p, 0.0)
        output_jax: Full = self.jax.hess_prod(u_full, p_full)
        output_wp: Full = self.warp.hess_prod(u_full, p_full)
        output: Full = output_jax + output_wp
        return self.to_shape_like(output, u)

    def hess_quad(self, u: FreeOrFull, p: FreeOrFull) -> Scalar:
        u_full: Full = self.to_full(u)
        self.update(u_full)
        p_full: Full = self.to_full(p, 0.0)
        output_jax: Scalar = self.jax.hess_quad(u_full, p_full)
        output_wp: Scalar = self.warp.hess_quad(u_full, p_full)
        output: Scalar = output_jax + output_wp
        return output

    def mixed_derivative_prod(self, u: FreeOrFull, p: FreeOrFull) -> ModelParams:
        u_full: Full = self.to_full(u)
        self.update(u_full)
        p_full: Full = self.to_full(p, 0.0)
        outputs_jax: ModelParams = self.jax.mixed_derivative_prod(u_full, p_full)
        outputs_wp: ModelParams = self.warp.mixed_derivative_prod(u_full, p_full)
        outputs: ModelParams = tlz.merge(outputs_jax, outputs_wp)
        return outputs

    def value_and_grad(self, u: FreeOrFull) -> tuple[Scalar, FreeOrFull]:
        u_full: Full = self.to_full(u)
        self.update(u_full)
        value_jax: Scalar
        grad_jax: Full
        value_jax, grad_jax = self.jax.value_and_grad(u_full)
        value_wp: Scalar
        grad_wp: Full
        value_wp, grad_wp = self.warp.value_and_grad(u_full)
        value: Scalar = value_jax + value_wp
        grad: Full = grad_jax + grad_wp
        return value, self.to_shape_like(grad, u)

    def grad_and_hess_diag(self, u: FreeOrFull) -> tuple[FreeOrFull, FreeOrFull]:
        u_full: Full = self.to_full(u)
        self.update(u_full)
        grad_jax: Full
        hess_diag_jax: Full
        grad_jax, hess_diag_jax = self.jax.grad_and_hess_diag(u_full)
        grad_wp: Full
        hess_diag_wp: Full
        grad_wp, hess_diag_wp = self.warp.grad_and_hess_diag(u_full)
        grad: Full = grad_jax + grad_wp
        hess_diag: Full = hess_diag_jax + hess_diag_wp
        return self.to_shape_like(grad, u), self.to_shape_like(hess_diag, u)
