import abc
from collections.abc import Mapping

import jax
import scipy.optimize
from jaxtyping import Float

from liblaf import apple

from . import AbstractPytreeNodeClass


@jax.tree_util.register_pytree_node_class
class AbstractPhysicsProblem(AbstractPytreeNodeClass):
    q: Mapping[str, Float[jax.Array, "..."]] = {}

    @property
    @abc.abstractmethod
    def n_dof(self) -> int: ...

    def solve(
        self, q: Mapping[str, Float[jax.Array, "..."]] = {}
    ) -> scipy.optimize.OptimizeResult:
        raise NotImplementedError

    @abc.abstractmethod
    def fun(
        self, u: Float[jax.Array, " DoF"], q: Mapping[str, Float[jax.Array, "..."]] = {}
    ) -> Float[jax.Array, ""]: ...

    @apple.jit()
    def jac(
        self, u: Float[jax.Array, " DoF"], q: Mapping[str, Float[jax.Array, "..."]] = {}
    ) -> Float[jax.Array, " DoF"]:
        return jax.jacobian(self.fun)(u, {**self.q, **q})

    @apple.jit()
    def hess(
        self, u: Float[jax.Array, " DoF"], q: Mapping[str, Float[jax.Array, "..."]] = {}
    ) -> Float[jax.Array, "..."]:
        return jax.hessian(self.fun)(u, {**self.q, **q})

    @apple.jit()
    def hessp(
        self,
        u: Float[jax.Array, " DoF"],
        v: Float[jax.Array, " DoF"],
        q: Mapping[str, Float[jax.Array, "..."]] = {},
    ) -> Float[jax.Array, "..."]:
        return apple.hvp(lambda u: self.fun(u, {**self.q, **q}), u, v)

    @apple.jit()
    def dh_dq(
        self, u: Float[jax.Array, " DoF"], q: Mapping[str, Float[jax.Array, "..."]] = {}
    ) -> Mapping[str, Float[jax.Array, "DoF ..."]]:
        return jax.jacobian(lambda q: self.fun(u, {**self.q, **q}))(q)
