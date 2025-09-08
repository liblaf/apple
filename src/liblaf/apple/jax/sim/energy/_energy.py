import jax
import jax.numpy as jnp

from liblaf.apple.jax import math, tree
from liblaf.apple.jax.typing import Scalar, Updates, UpdatesIndex, Vector


@tree.pytree
class Energy:
    def fun(self, u: Vector) -> Scalar:
        raise NotImplementedError

    def jac(self, u: Vector) -> Updates:
        data: Vector = jax.grad(self.fun)(u)
        index: UpdatesIndex = jnp.arange(data.shape[0])
        return data, index

    def fun_and_jac(self, u: Vector) -> tuple[Scalar, Updates]:
        value: Scalar
        data: Vector
        value, data = jax.value_and_grad(self.fun)(u)
        index: UpdatesIndex = jnp.arange(data.shape[0])
        return value, (data, index)

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
