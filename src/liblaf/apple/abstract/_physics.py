import abc
from collections.abc import Callable
from typing import overload

import attrs
import jax
import jax.flatten_util
import jax.numpy as jnp
import pylops
import scipy.optimize
from jaxtyping import Float, PyTree

from liblaf import apple


@apple.register_dataclass()
@attrs.define(kw_only=True)
class AbstractPhysicsProblem:
    aux: PyTree = attrs.field(factory=dict)
    params: PyTree = attrs.field(factory=dict)
    _q_unravel: Callable[[Float[jax.Array, " Q"]], PyTree] | None = attrs.field(
        default=None, metadata={"static": True}, alias="_q_unravel"
    )
    _u_unravel: Callable[[Float[jax.Array, " DoF"]], PyTree] | None = attrs.field(
        default=None, metadata={"static": True}, alias="_u_unravel"
    )

    @property
    @abc.abstractmethod
    def n_dof(self) -> int: ...

    @property
    def params_flat(self) -> Float[jax.Array, " P"]:
        params_flat: Float[jax.Array, " P"]
        params_flat, _unravel = jax.flatten_util.ravel_pytree(self.params)
        return params_flat

    @params_flat.setter
    def params_flat(self, params_flat: Float[jax.Array, " P"]) -> None:
        self.params = self.unravel_params(params_flat)

    @property
    def _params_unravel(self) -> Callable[[Float[jax.Array, " P"]], PyTree]:
        params_unravel: Callable[[Float[jax.Array, " P"]], PyTree]
        _params_flat, params_unravel = jax.flatten_util.ravel_pytree(self.params)
        return params_unravel

    def solve(
        self, u0: PyTree | None = None, q: PyTree | None = None
    ) -> scipy.optimize.OptimizeResult:
        u0_flat: Float[jax.Array, " DoF"]
        u0_flat = self.ravel_u(u0) if u0 is not None else jnp.zeros((self.n_dof,))
        q_flat: Float[jax.Array, " Q"] | None = self.ravel_q(q)
        result: scipy.optimize.OptimizeResult = apple.minimize(
            x0=u0_flat,
            fun=lambda u_flat: self.fun_flat(u_flat, q_flat),
            jac=lambda u_flat: self.jac_flat(u_flat, q_flat),
            hess=lambda u_flat: self.hess_flat(u_flat, q_flat),
        )
        return result

    @abc.abstractmethod
    def fun(self, u: PyTree, q: PyTree | None = None) -> Float[jax.Array, ""]: ...

    @apple.jit()
    def fun_flat(
        self,
        u_flat: Float[jax.Array, " DoF"],
        q_flat: Float[jax.Array, " Q"] | None = None,
    ) -> Float[jax.Array, ""]:
        return self.fun(self.unravel_u(u_flat), self.unravel_q(q_flat))

    @apple.jit()
    def jac(self, u: PyTree, q: PyTree | None = None) -> Float[jax.Array, " DoF"]:
        u_flat: Float[jax.Array, " DoF"] = self.ravel_u(u)
        q_flat: Float[jax.Array, " Q"] | None = self.ravel_q(q)
        return self.jac_flat(u_flat, q_flat)

    @apple.jit()
    def jac_flat(
        self,
        u_flat: Float[jax.Array, " DoF"],
        q_flat: Float[jax.Array, " Q"] | None = None,
    ) -> Float[jax.Array, " DoF"]:
        return jax.jacobian(self.fun_flat)(u_flat, q_flat)

    @apple.jit()
    def hess(self, u: PyTree, q: PyTree | None = None) -> pylops.LinearOperator:
        u_flat: Float[jax.Array, " DoF"] = self.ravel_u(u)
        q_flat: Float[jax.Array, " Q"] | None = self.ravel_q(q)
        return self.hess_flat(u_flat, q_flat)

    @apple.jit()
    def hess_flat(
        self,
        u_flat: Float[jax.Array, " DoF"],
        q_flat: Float[jax.Array, " Q"] | None = None,
    ) -> pylops.LinearOperator:
        return jax.hessian(self.fun_flat)(u_flat, q_flat)
        # TODO: replace with linear operator
        return apple.hess_as_operator(
            lambda u_flat: self.fun_flat(u_flat, q_flat), u_flat
        )

    @apple.jit()
    def dh_dq(self, u: PyTree, q: PyTree) -> Float[pylops.LinearOperator, "DoF Q"]:
        return self.dh_dq_flat(self.ravel_u(u), self.ravel_q(q))

    @apple.jit()
    def dh_dq_flat(
        self, u_flat: Float[jax.Array, " DoF"], q_flat: Float[jax.Array, " Q"]
    ) -> Float[pylops.LinearOperator, "DoF Q"]:
        # TODO: replace with linear operator
        return jax.jacobian(lambda q_flat: self.jac_flat(u_flat, q_flat))(q_flat)
        return apple.jac_as_operator(
            lambda q_flat: self.jac_flat(u_flat, q_flat), q_flat
        )

    def prepare(self, params: PyTree | None = None) -> None:
        self.params = params or self.params
        self.aux = {}

    def ravel_params(self, params: PyTree) -> Float[jax.Array, " P"]:
        params_flat: Float[jax.Array, " P"]
        params_flat, self._q_unravel = jax.flatten_util.ravel_pytree(params)
        return params_flat

    @overload
    def ravel_q(self, q: PyTree) -> Float[jax.Array, " Q"]: ...
    @overload
    def ravel_q(self, q: None) -> None: ...  # pyright: ignore[reportOverlappingOverload]
    def ravel_q(self, q: PyTree | None) -> Float[jax.Array, " Q"] | None:
        if q is None:
            return None
        q_flat: Float[jax.Array, " Q"]
        q_flat, self._q_unravel = jax.flatten_util.ravel_pytree(q)
        return q_flat

    def ravel_u(self, u: PyTree) -> Float[jax.Array, " DoF"]:
        u_flat: Float[jax.Array, " DoF"]
        u_flat, self._u_unravel = jax.flatten_util.ravel_pytree(u)
        return u_flat

    def unravel_params(self, params_flat: Float[jax.Array, " Q"]) -> PyTree:
        return self._params_unravel(params_flat)

    def unravel_q(self, q: Float[jax.Array, " Q"] | None) -> PyTree | None:
        if q is None:
            return None
        if self._q_unravel is None:
            msg: str = "`q_unravel` not set"
            raise ValueError(msg)
        return self._q_unravel(q)

    def unravel_u(self, u: Float[jax.Array, " DoF"]) -> PyTree:
        if self._u_unravel is None:
            msg: str = "`u_unravel` not set"
            raise ValueError(msg)
        return self._u_unravel(u)
