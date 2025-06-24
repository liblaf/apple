from collections.abc import Callable
from typing import Any, Self

import attrs
import cytoolz as toolz
import jax
import jax.numpy as jnp

from .base import BaseProblem
from .utils import implemented


class ImplementMixin(BaseProblem):
    def implement(self) -> Self:
        return (
            self._implement_fun()  # noqa: SLF001
            ._implement_jac()
            ._implement_hess()
            ._implement_hessp()
            ._implement_hess_diag()
            ._implement_hess_quad()
            ._implement_fun_and_jac()
            ._implement_jac_and_hess_diag()
        )

    def _implement_fun(self) -> Self:
        fun: Callable | None = self.fun
        if implemented(fun):
            return self
        if implemented(self.fun_and_jac):
            fun = index(self.fun_and_jac, 0)
        return attrs.evolve(self, fun=fun)

    def _implement_jac(self) -> Self:
        jac: Callable | None = self.jac
        if implemented(jac):
            return self
        if implemented(self.fun_and_jac):
            jac = index(self.fun_and_jac, 1)
        elif implemented(self.jac_and_hess_diag):
            jac = index(self.jac_and_hess_diag, 0)
        return attrs.evolve(self, jac=jac)

    def _implement_hess(self) -> Self:
        return self

    def _implement_hessp(self) -> Self:
        hessp: Callable | None = self.hessp
        if implemented(hessp):
            return self
        if implemented(self.hess):
            hessp = matmul(self.hess)
        return attrs.evolve(self, hessp=hessp)

    def _implement_hess_diag(self) -> Self:
        hess_diag: Callable | None = self.hess_diag
        if implemented(hess_diag):
            return self
        if implemented(self.jac_and_hess_diag):
            hess_diag = index(self.jac_and_hess_diag, 1)
        elif implemented(self.hess):
            hess_diag = diagonal(self.hess)
        return attrs.evolve(self, hess_diag=hess_diag)

    def _implement_hess_quad(self) -> Self:
        hess_quad: Callable | None = self.hess_quad
        if implemented(hess_quad):
            return self
        if implemented(self.hessp):
            hess_quad = vdot(self.hessp)
        elif implemented(self.hess):
            hess_quad = quad(self.hess)
        return attrs.evolve(self, hess_quad=hess_quad)

    def _implement_fun_and_jac(self) -> Self:
        fun_and_jac: Callable | None = self.fun_and_jac
        if implemented(fun_and_jac):
            return self
        if implemented(self.fun) and implemented(self.jac):
            fun_and_jac = toolz.juxt(self.fun, self.jac)
        return attrs.evolve(self, fun_and_jac=fun_and_jac)

    def _implement_jac_and_hess_diag(self) -> Self:
        jac_and_hess_diag: Callable | None = self.jac_and_hess_diag
        if implemented(jac_and_hess_diag):
            return self
        if implemented(self.jac) and implemented(self.hess_diag):
            jac_and_hess_diag = toolz.juxt(self.jac, self.hess_diag)
        return attrs.evolve(self, jac_and_hess_diag=jac_and_hess_diag)


def diagonal(func: Callable, /) -> Callable:
    def wrapper(x: jax.Array, *args, **kwargs) -> jax.Array:
        return jnp.diagonal(func(x, *args, **kwargs))

    return wrapper


def index(func: Callable, /, index: int) -> Callable:
    def wrapper(*args, **kwargs) -> Any:
        return func(*args, **kwargs)[index]

    return wrapper


def matmul(hess: Callable, /) -> Callable:
    def wrapper(x: jax.Array, p: jax.Array, *args, **kwargs) -> Any:
        return hess(x, *args, **kwargs) @ p

    return wrapper


def quad(func: Callable, /) -> Callable:
    def wrapper(x: jax.Array, p: jax.Array, *args, **kwargs) -> Any:
        return jnp.vdot(func(x, *args, **kwargs) @ p, p)

    return wrapper


def vdot(hessp: Callable, /) -> Callable:
    def wrapper(x: jax.Array, p: jax.Array, *args, **kwargs) -> Any:
        return jnp.vdot(hessp(x, p, *args, **kwargs), p)

    return wrapper
