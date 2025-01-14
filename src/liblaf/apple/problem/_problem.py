import abc
import functools
from collections.abc import Callable, Mapping
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy.typing as npt
from jaxtyping import Float

from liblaf import apple


class Function(Protocol):
    def __call__(
        self,
        u: Float[jax.Array, " DoF"],
        p: Mapping[str, Float[jax.Array, "..."]],
        s: Mapping[str, Float[jax.Array, "..."]],
    ) -> Float[jax.Array, ""]: ...


class Jacobian(Protocol):
    def __call__(
        self,
        u: Float[jax.Array, " DoF"],
        p: Mapping[str, Float[jax.Array, "..."]],
        s: Mapping[str, Float[jax.Array, "..."]],
    ) -> Float[jax.Array, " DoF"]: ...


class Hessian(Protocol):
    def __call__(
        self,
        u: Float[jax.Array, " DoF"],
        p: Mapping[str, Float[jax.Array, "..."]],
        s: Mapping[str, Float[jax.Array, "..."]],
    ) -> Float[jax.Array, " DoF DoF"]: ...


class HVP(Protocol):
    """Hessian-Vector Product."""

    def __call__(
        self,
        u: Float[jax.Array, " DoF"],
        v: Float[jax.Array, " DoF"],
        p: Mapping[str, Float[jax.Array, "..."]],
        s: Mapping[str, Float[jax.Array, "..."]],
        method: apple.HVPMethod = apple.HVPMethod.FORWARD_OVER_REVERSE,
    ) -> Float[jax.Array, " DoF"]: ...


class VHP(Protocol):
    """Vector-Hessian Product."""

    def __call__(
        self,
        u: Float[jax.Array, " DoF"],
        v: Float[jax.Array, " DoF"],
        p: Mapping[str, Float[jax.Array, "..."]],
        s: Mapping[str, Float[jax.Array, "..."]],
    ) -> Float[jax.Array, " DoF"]: ...


class Problem(abc.ABC):
    p: Mapping[str, Float[jax.Array, "..."]]

    def prepare(self, p: Mapping[str, Float[npt.ArrayLike, "..."]]) -> None:
        self.p = apple.as_array_dict(p)
        if "s" in self.__dict__:
            del self.s

    def fun(
        self,
        u: Float[npt.ArrayLike, " DoF"],
        p: Mapping[str, Float[npt.ArrayLike, "..."]] | None = None,
        s: Mapping[str, Float[npt.ArrayLike, "..."]] | None = None,
    ) -> Float[jax.Array, ""]:
        p, s = self._prepare_p_s(p, s)
        return self._fun_jit(
            u=jnp.asarray(u), p=apple.as_array_dict(p), s=apple.as_array_dict(s)
        )

    def jac(
        self,
        u: Float[npt.ArrayLike, " DoF"],
        p: Mapping[str, Float[npt.ArrayLike, "..."]] | None = None,
        s: Mapping[str, Float[npt.ArrayLike, "..."]] | None = None,
    ) -> Float[jax.Array, " DoF"]:
        p, s = self._prepare_p_s(p, s)
        return self._jac_jit(
            u=jnp.asarray(u), p=apple.as_array_dict(p), s=apple.as_array_dict(s)
        )

    def hess(
        self,
        u: Float[npt.ArrayLike, " DoF"],
        p: Mapping[str, Float[npt.ArrayLike, "..."]] | None = None,
        s: Mapping[str, Float[npt.ArrayLike, "..."]] | None = None,
    ) -> Float[jax.Array, " DoF DoF"]:
        p, s = self._prepare_p_s(p, s)
        return self._hess_jit(
            u=jnp.asarray(u), p=apple.as_array_dict(p), s=apple.as_array_dict(s)
        )

    def hvp(
        self,
        u: Float[npt.ArrayLike, " DoF"],
        v: Float[npt.ArrayLike, " DoF"],
        p: Mapping[str, Float[npt.ArrayLike, "..."]] | None = None,
        s: Mapping[str, Float[npt.ArrayLike, "..."]] | None = None,
        method: apple.HVPMethod = apple.HVPMethod.FORWARD_OVER_REVERSE,
    ) -> Float[jax.Array, " DoF"]:
        p, s = self._prepare_p_s(p, s)
        return self._hvp_jit(
            u=jnp.asarray(u),
            v=jnp.asarray(v),
            p=apple.as_array_dict(p),
            s=apple.as_array_dict(s),
            method=method,
        )

    @functools.cached_property
    def s(self) -> Mapping[str, Float[jax.Array, "..."]]:
        return self._prepare_jit(self.p)

    def _prepare_p_s(
        self,
        p: Mapping[str, Float[npt.ArrayLike, "..."]] | None = None,
        s: Mapping[str, Float[npt.ArrayLike, "..."]] | None = None,
    ) -> tuple[
        Mapping[str, Float[jax.Array, "..."]], Mapping[str, Float[jax.Array, "..."]]
    ]:
        if p is None:
            p = self.p
        else:
            self.prepare(p)
        if s is None:
            s = self.s
        return apple.as_array_dict(p), apple.as_array_dict(s)

    def _prepare(
        self,
        p: Mapping[str, Float[jax.Array, "..."]],  # noqa: ARG002
    ) -> Mapping[str, Float[jax.Array, "..."]]:
        return {}

    @functools.cached_property
    def _prepare_jit(
        self,
    ) -> Callable[
        [Mapping[str, Float[jax.Array, "..."]]], Mapping[str, Float[jax.Array, "..."]]
    ]:
        return jax.jit(self._prepare)

    @abc.abstractmethod
    def _fun(
        self,
        u: Float[jax.Array, " DoF"],
        p: Mapping[str, Float[jax.Array, "..."]],
        s: Mapping[str, Float[jax.Array, "..."]],
    ) -> Float[jax.Array, ""]: ...

    @functools.cached_property
    def _fun_jit(self) -> Function:
        return jax.jit(self._fun)

    def _jac(
        self,
        u: Float[jax.Array, " DoF"],
        p: Mapping[str, Float[jax.Array, "..."]],
        s: Mapping[str, Float[jax.Array, "..."]],
    ) -> Float[jax.Array, " DoF"]:
        return jax.grad(self._fun)(u, p, s)

    @functools.cached_property
    def _jac_jit(self) -> Jacobian:
        return jax.jit(self._jac)

    def _hess(
        self,
        u: Float[jax.Array, " DoF"],
        p: Mapping[str, Float[jax.Array, "..."]],
        s: Mapping[str, Float[jax.Array, "..."]],
    ) -> Float[jax.Array, " DoF DoF"]:
        return jax.hessian(self._fun)(u, p, s)

    @functools.cached_property
    def _hess_jit(self) -> Hessian:
        return jax.jit(self._hess)

    def _hvp(
        self,
        u: Float[jax.Array, " DoF"],
        v: Float[jax.Array, " DoF"],
        p: Mapping[str, Float[jax.Array, "..."]],
        s: Mapping[str, Float[jax.Array, "..."]],
        method: apple.HVPMethod = apple.HVPMethod.FORWARD_OVER_REVERSE,
    ) -> Float[jax.Array, " DoF"]:
        return apple.hvp(lambda u: self._fun(u, p, s), u, v, method=method)

    @functools.cached_property
    def _hvp_jit(self) -> HVP:
        return jax.jit(self._hvp, static_argnames=["method"])
