from collections.abc import Callable
from typing import Self

import attrs
import jax

from .base import BaseProblem
from .utils import implemented


class AutodiffMixin(BaseProblem):
    def autodiff(self) -> Self:
        raise NotImplementedError

    def _autodiff_fun(self) -> Self:
        return self

    def _autodiff_jac(self) -> Self:
        jac: Callable | None = self.jac
        if implemented(jac):
            return self
        if implemented(self.fun):
            jac = jax.grad(self.fun)
        return attrs.evolve(self, jac=jac)

    def _autodiff_hess(self) -> Self:
        hess: Callable | None = self.hess
        if implemented(hess):
            return self
        if implemented(self.jac):
            hess = jax.jacobian(self.jac)
        elif implemented(self.fun):
            hess = jax.hessian(self.fun)
        return attrs.evolve(self, hess=hess)

    def _autodiff_hessp(self) -> Self:
        hessp: Callable | None = self.hessp
        if implemented(hessp):
            return self
        if implemented(self.jac):
            hessp = jvp(self.jac)
        return attrs.evolve(self, hessp=hessp)


def jvp(func: Callable) -> Callable:
    def fun_jvp(x: jax.Array, p: jax.Array, *args, **kwargs) -> jax.Array:
        _, tangents_out = jax.jvp(lambda x: func(x, *args, **kwargs), (x,), (p,))
        return tangents_out

    return fun_jvp
