import abc
from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Self

import jax
import jax.numpy as jnp
import scipy.optimize
from jaxtyping import Float

from liblaf import apple


@jax.tree_util.register_pytree_node_class
class PhysicsProblem(abc.ABC):
    p: Mapping[str, Float[jax.Array, "..."]] = {}

    @property
    @abc.abstractmethod
    def n_dof(self) -> int: ...

    def solve(
        self, p: Mapping[str, Float[jax.Array, "..."]] = {}
    ) -> scipy.optimize.OptimizeResult:
        result: scipy.optimize.OptimizeResult = apple.minimize(
            x0=jnp.zeros((self.n_dof,)),
            fun=lambda u: self.fun(u, p),
            jac=lambda u: self.jac(u, p),
            hess=lambda u: self.hess(u, p),
        )
        return result

    @apple.jit()
    def fun(
        self, u: Float[jax.Array, " DoF"], p: Mapping[str, Float[jax.Array, "..."]] = {}
    ) -> Float[jax.Array, "..."]:
        return self._fun(u, {**self.p, **p})

    @apple.jit()
    def jac(
        self, u: Float[jax.Array, " DoF"], p: Mapping[str, Float[jax.Array, "..."]] = {}
    ) -> Float[jax.Array, "..."]:
        return self._jac(u, {**self.p, **p})

    @apple.jit()
    def hess(
        self, u: Float[jax.Array, " DoF"], p: Mapping[str, Float[jax.Array, "..."]] = {}
    ) -> Float[jax.Array, "..."]:
        return self._hess(u, {**self.p, **p})

    @apple.jit()
    def hessp(
        self,
        u: Float[jax.Array, " DoF"],
        v: Float[jax.Array, " DoF"],
        p: Mapping[str, Float[jax.Array, "..."]] = {},
    ) -> Float[jax.Array, "..."]:
        return self._hessp(u, v, {**self.p, **p})

    @apple.jit()
    def dJ_dp(
        self, u: Float[jax.Array, " DoF"], p: Mapping[str, Float[jax.Array, "..."]] = {}
    ) -> Mapping[str, Float[jax.Array, "DoF ..."]]:
        return self._dJ_dp(u, p)

    @abc.abstractmethod
    def tree_flatten(self) -> tuple[Iterable[Any], Hashable]: ...

    @classmethod
    @abc.abstractmethod
    def tree_unflatten(cls, aux_data: None, children: Iterable[Any]) -> Self: ...

    @abc.abstractmethod
    def _fun(
        self, u: Float[jax.Array, " DoF"], p: Mapping[str, Float[jax.Array, "..."]]
    ) -> Float[jax.Array, "..."]: ...

    def _jac(
        self, u: Float[jax.Array, " DoF"], p: Mapping[str, Float[jax.Array, "..."]]
    ) -> Float[jax.Array, " DoF"]:
        return jax.grad(self._fun)(u, p)

    def _hess(
        self, u: Float[jax.Array, " DoF"], p: Mapping[str, Float[jax.Array, "..."]]
    ) -> Float[jax.Array, " DoF DoF"]:
        return jax.hessian(self._fun)(u, p)

    def _hessp(
        self,
        u: Float[jax.Array, " DoF"],
        v: Float[jax.Array, " DoF"],
        p: Mapping[str, Float[jax.Array, "..."]],
    ) -> Float[jax.Array, " DoF"]:
        return apple.hvp(lambda u: self._fun(u, p), u, v)

    def _dJ_dp(
        self, u: Float[jax.Array, " DoF"], p: Mapping[str, Float[jax.Array, "..."]] = {}
    ) -> Mapping[str, Float[jax.Array, "DoF ..."]]:
        return jax.jacobian(lambda p: self._jac(u, {**self.p, **p}))(p)
