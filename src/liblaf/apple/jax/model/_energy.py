from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer
from liblaf.peach import tree

from liblaf.apple.utils import IdMixin

type Index = Integer[Array, " points"]
type Scalar = Float[Array, ""]
type Updates = tuple[Vector, Index]
type Vector = Float[Array, "points dim"]


@tree.define
class JaxEnergy(IdMixin):
    requires_grad: frozenset[str] = tree.field(default=frozenset(), kw_only=True)

    @eqx.filter_jit
    def update(self, u: Vector) -> Self:  # noqa: ARG002
        return self

    def fun(self, u: Vector) -> Scalar:
        raise NotImplementedError

    @eqx.filter_jit
    def grad(self, u: Vector) -> Updates:
        values: Vector = eqx.filter_grad(self.fun)(u)
        return values, jnp.arange(u.shape[0])

    def hess_diag(self, u: Vector) -> Updates:
        raise NotImplementedError

    @eqx.filter_jit
    def hess_prod(self, u: Vector, p: Vector) -> Updates:
        values: Vector
        _, values = jax.jvp(jax.grad(self.fun), (u,), (p,))
        return values, jnp.arange(u.shape[0])

    @eqx.filter_jit
    def hess_quad(self, u: Vector, p: Vector) -> Scalar:
        values: Vector
        index: Index
        values, index = self.hess_prod(u, p)
        return jnp.vdot(p[index], values)

    @eqx.filter_jit
    def value_and_grad(self, u: Vector) -> tuple[Scalar, Updates]:
        value: Scalar
        grad: Vector
        value, grad = jax.value_and_grad(self.fun)(u)
        return value, (grad, jnp.arange(u.shape[0]))

    @eqx.filter_jit
    def grad_and_hess_diag(self, u: Vector) -> tuple[Updates, Updates]:
        return self.grad(u), self.hess_diag(u)

    @eqx.filter_jit
    def mixed_derivative_prod(self, u: Vector, p: Vector) -> dict[str, Array]:
        outputs: dict[str, Array] = {}
        for name in self.requires_grad:
            outputs[name] = getattr(self, f"mixed_derivative_prod_{name}")(u, p)
        return outputs
