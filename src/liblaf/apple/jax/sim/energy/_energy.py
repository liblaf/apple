from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array

from liblaf.apple.jax import math, tree
from liblaf.apple.jax.typing import Scalar, Updates, UpdatesIndex, Vector


@tree.pytree
class Energy(tree.IdMixin):
    requires_grad: Sequence[str] = tree.field(default=(), kw_only=True)

    def fun(self, u: Vector) -> Scalar:
        raise NotImplementedError

    def jac(self, u: Vector) -> Updates:
        data: Vector = jax.grad(self.fun)(u)
        index: UpdatesIndex = jnp.arange(data.shape[0])
        return data, index

    def hess_diag(self, u: Vector) -> Updates:
        data: Vector = math.hess_diag(self.fun, u)
        index: UpdatesIndex = jnp.arange(data.shape[0])
        return data, index

    def hess_prod(self, u: Vector, p: Vector) -> Updates:
        data: Vector
        _, data = jax.jvp(jax.grad(self.fun), (u,), (p,))
        index: UpdatesIndex = jnp.arange(data.shape[0])
        return data, index

    def hess_quad(self, u: Vector, p: Vector) -> Scalar:
        data: Vector
        index: UpdatesIndex
        data, index = self.hess_prod(u, p)
        return jnp.vdot(p[index], data)

    def fun_and_jac(self, u: Vector) -> tuple[Scalar, Updates]:
        value: Scalar
        data: Vector
        value, data = jax.value_and_grad(self.fun)(u)
        index: UpdatesIndex = jnp.arange(data.shape[0])
        return value, (data, index)

    def jac_and_hess_diag(self, u: Vector) -> tuple[Updates, Updates]:
        jac: Updates = self.jac(u)
        hess_diag: Updates = self.hess_diag(u)
        return jac, hess_diag

    def mixed_derivative_prod(self, u: Vector, p: Vector) -> dict[str, Array]:
        outputs: dict[str, Array] = {}
        for name in self.requires_grad:
            outputs[name] = getattr(self, f"mixed_derivative_prod_{name}")(u, p)
        return outputs
