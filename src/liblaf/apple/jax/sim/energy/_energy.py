import jax
import jax.numpy as jnp

from liblaf.apple.jax import tree
from liblaf.apple.jax.typing import Scalar, Updates, UpdatesData, UpdatesIndex, Vector


@tree.pytree
class Energy:
    def fun(self, u: Vector) -> Scalar:
        raise NotImplementedError

    def jac(self, u: Vector) -> Updates:
        data: UpdatesData = jax.grad(self.fun)(u)
        index: UpdatesIndex = jnp.arange(data.shape[0])
        return data, index

    def value_and_jac(self, u: Vector) -> tuple[Scalar, Updates]:
        value: Scalar
        data: UpdatesData
        value, data = jax.value_and_grad(self.fun)(u)
        index: UpdatesIndex = jnp.arange(data.shape[0])
        return value, (data, index)

    def hess_prod(self, u: Vector, p: Vector) -> Updates:
        data: UpdatesData
        _, data = jax.jvp(jax.grad(self.fun), (u,), (p,))
        index: UpdatesIndex = jnp.arange(data.shape[0])
        return data, index
