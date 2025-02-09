import abc

import jax
import jax.numpy as jnp
from jaxtyping import Float, Scalar

from liblaf import apple


class ProblemPrepared(abc.ABC):
    @property
    @abc.abstractmethod
    def n_dof(self) -> int: ...

    @abc.abstractmethod
    def fun(self, u: Float[jax.Array, " DoF"]) -> Scalar: ...

    def jac(self, u: Float[jax.Array, " DoF"]) -> Float[jax.Array, " DoF"]:
        return jax.grad(self.fun)(u)

    def hess(self, u: Float[jax.Array, " DoF"]) -> Float[jax.Array, "DoF DoF"]:
        return jax.hessian(self.fun)(u)

    def hessp(
        self, u: Float[jax.Array, " DoF"], v: Float[jax.Array, " DoF"]
    ) -> Float[jax.Array, " DoF"]:
        v = jnp.asarray(v, dtype=u.dtype)
        return apple.hvp(self.fun, u, v)


class Problem(abc.ABC):
    @property
    @abc.abstractmethod
    def n_dof(self) -> int: ...

    @abc.abstractmethod
    def prepare(self) -> ProblemPrepared: ...
