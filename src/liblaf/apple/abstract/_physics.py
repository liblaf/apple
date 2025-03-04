import abc
from collections.abc import Callable
from typing import overload

import attrs
import glom
import jax
import jax.flatten_util
import jax.numpy as jnp
import pylops
import scipy.optimize
from jaxtyping import Float, PyTree

from liblaf import apple


class NotSetError(ValueError):
    def __init__(self, name: str) -> None:
        super().__init__(f"`{name}` not set")


@apple.register_dataclass()
@attrs.define(kw_only=True, on_setattr=attrs.setters.convert)
class AbstractPhysicsProblem(abc.ABC):
    name: str = attrs.field(default="physics", metadata={"static": True})
    _q_unravel: Callable[[Float[jax.Array, " Q"]], PyTree] | None = attrs.field(
        default=None, metadata={"static": True}, alias="_q_unravel"
    )

    @property
    @abc.abstractmethod
    def n_dof(self) -> int: ...

    def solve(
        self, q: PyTree | None = None, u0: PyTree | None = None
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

    def fun(self, u: PyTree, q: PyTree | None = None) -> Float[jax.Array, ""]:
        if type(self).fun_flat is not AbstractPhysicsProblem.fun_flat:
            # `fun_flat()` is overridden
            return self.fun_flat(self.ravel_u(u), self.ravel_q(q))
        raise NotImplementedError

    def fun_flat(
        self,
        u_flat: Float[jax.Array, " DoF"],
        q_flat: Float[jax.Array, " Q"] | None = None,
    ) -> Float[jax.Array, ""]:
        if type(self).fun is not AbstractPhysicsProblem.fun:
            # `fun()` is overridden
            return self.fun(self.unravel_u(u_flat), self.unravel_q(q_flat))
        raise NotImplementedError

    def jac(self, u: PyTree, q: PyTree | None = None) -> Float[jax.Array, " DoF"]:
        return self.jac_flat(self.ravel_u(u), self.ravel_q(q))

    def jac_flat(
        self,
        u_flat: Float[jax.Array, " DoF"],
        q_flat: Float[jax.Array, " Q"] | None = None,
    ) -> Float[jax.Array, " DoF"]:
        if type(self).jac is not AbstractPhysicsProblem.jac:
            # `jac()` is overridden
            return self.jac(self.unravel_u(u_flat), self.unravel_q(q_flat))
        return jax.jacobian(self.fun_flat)(u_flat, q_flat)

    @apple.jit()
    def hess(
        self, u: PyTree, q: PyTree | None = None
    ) -> Float[pylops.LinearOperator, "DoF DoF"]:
        return self.hess_flat(self.ravel_u(u), self.ravel_q(q))

    @apple.jit()
    def hess_flat(
        self,
        u_flat: Float[jax.Array, " DoF"],
        q_flat: Float[jax.Array, " Q"] | None = None,
    ) -> Float[pylops.LinearOperator, "DoF DoF"]:
        if type(self).hess is not AbstractPhysicsProblem.hess:
            # `hess()` is overridden
            return self.hess(self.unravel_u(u_flat), self.unravel_q(q_flat))
        # TODO: replace with linear operator
        return jax.hessian(self.fun_flat)(u_flat, q_flat)

    @apple.jit()
    def dh_dq(self, u: PyTree, q: PyTree) -> Float[pylops.LinearOperator, "DoF Q"]:
        return self.dh_dq_flat(self.ravel_u(u), self.ravel_q(q))

    @apple.jit()
    def dh_dq_flat(
        self, u_flat: Float[jax.Array, " DoF"], q_flat: Float[jax.Array, " Q"]
    ) -> Float[pylops.LinearOperator, "DoF Q"]:
        if type(self).dh_dq is not AbstractPhysicsProblem.dh_dq:
            # `dh_dq()` is overridden
            return self.dh_dq(self.unravel_u(u_flat), self.unravel_q(q_flat))
        # TODO: replace with linear operator
        return jax.jacobian(lambda q_flat: self.jac_flat(u_flat, q_flat))(q_flat)

    def get_param(self, name: str, q: PyTree | None = None) -> Float[jax.Array, "..."]:
        return glom.glom(
            {"q": q, "self": self},
            glom.Coalesce(glom.Path("q", self.name, name), glom.Path("self", name)),
        )

    @overload
    def ravel_q(self, q: PyTree) -> Float[jax.Array, " Q"]: ...
    @overload
    def ravel_q(self, q: None) -> None: ...  # pyright: ignore[reportOverlappingOverload]
    def ravel_q(self, q: PyTree | None) -> Float[jax.Array, " Q"] | None:
        q_flat: Float[jax.Array, " Q"]
        q_flat, self._q_unravel = jax.flatten_util.ravel_pytree(q)
        return q_flat

    def ravel_u(self, u: PyTree) -> Float[jax.Array, " DoF"]:
        u_flat: Float[jax.Array, " DoF"]
        u_flat, _u_unravel = jax.flatten_util.ravel_pytree(u)
        return u_flat

    def unravel_q(self, q_flat: Float[jax.Array, " Q"] | None) -> PyTree | None:
        if q_flat is None:
            return None
        if self._q_unravel is None:
            raise NotSetError("_q_unravel")  # noqa: EM101
        return self._q_unravel(q_flat)

    def unravel_u(self, u_flat: Float[jax.Array, " DoF"]) -> PyTree:
        return u_flat


@attrs.define(kw_only=True, on_setattr=attrs.setters.convert)
class AbstractPhysicsProblemBuilder(abc.ABC):
    name: str = attrs.field(default="physics", metadata={"static": True})
    _q_unravel: Callable[[Float[jax.Array, " Q"]], PyTree] | None = attrs.field(
        default=None, metadata={"static": True}, alias="_q_unravel"
    )

    @abc.abstractmethod
    def build(self, q: PyTree | None = None) -> AbstractPhysicsProblem:
        self.ravel_q(q)

    def get_param(self, name: str, q: PyTree | None = None) -> PyTree:
        return glom.glom(
            {"q": q, "self": self},
            glom.Coalesce(glom.Path("q", self.name, name), glom.Path("self", name)),
        )

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

    def unravel_q(self, q_flat: Float[jax.Array, " Q"] | None) -> PyTree | None:
        if q_flat is None:
            return None
        if self._q_unravel is None:
            raise NotSetError("_q_unravel")  # noqa: EM101
        return self._q_unravel(q_flat)
