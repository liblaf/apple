import abc

import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf import apple


class AbstractMinimizeProblem(abc.ABC):
    @abc.abstractmethod
    def fun(
        self, x: Float[jax.Array, " N"], *args, **kwargs
    ) -> Float[jax.Array, ""]: ...

    def jac(self, x: Float[jax.Array, " N"], *args, **kwargs) -> Float[jax.Array, " N"]:
        return jax.grad(self.fun)(x, *args, **kwargs)

    def hess(
        self, x: Float[jax.Array, " N"], *args, **kwargs
    ) -> Float[jax.Array, " N N"]:
        return jax.hessian(self.fun)(x, *args, **kwargs)

    def hessp(
        self, x: Float[jax.Array, " N"], p: Float[jax.Array, " N"], *args, **kwargs
    ) -> Float[jax.Array, " N"]:
        return apple.hvp(lambda x: self.fun(x, *args, **kwargs), x, p)

    def hess_diag(
        self, x: Float[jax.Array, " N"], *args, **kwargs
    ) -> Float[jax.Array, " N"]:
        return apple.hess_diag(lambda x: self.fun(x, *args, **kwargs), x)

    def hess_pHp(
        self,
        x: Float[jax.Array, " N"],
        p: Float[jax.Array, " N"],
        *args,
        **kwargs,
    ) -> Float[jax.Array, ""]:
        return jnp.vdot(p, self.hessp(x, p, *args, **kwargs))
