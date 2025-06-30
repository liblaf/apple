import jax
from jaxtyping import Float, PyTree

from liblaf.apple import utils
from liblaf.apple.math import tree as math_tree

from . import functional

type FloatScalar = Float[jax.Array, ""]
type X = PyTree
type Y = FloatScalar


class AutoDiffMixin:
    @utils.not_implemented
    @utils.jit_method(inline=True)
    def fun(self, x: X, /, *args, **kwargs) -> FloatScalar:
        if utils.is_implemented(self.fun_and_jac):
            fun, _ = self.fun_and_jac(x, *args, **kwargs)
            return fun
        raise NotImplementedError

    @utils.not_implemented
    @utils.jit_method(inline=True)
    def jac(self, x: X, /, *args, **kwargs) -> X:
        if utils.is_implemented(self.fun_and_jac):
            _, jac = self.fun_and_jac(x, *args, **kwargs)
            return jac
        if utils.is_implemented(self.jac_and_hess_diag):
            _, hess_diag = self.jac_and_hess_diag(x, *args, **kwargs)
            return hess_diag
        return jax.grad(self.fun)(x, *args, **kwargs)

    @utils.not_implemented
    @utils.jit_method(inline=True)
    def hessp(self, x: X, p: X, /, *args, **kwargs) -> X:
        def grad(x: X, /) -> X:
            return self.jac(x, *args, **kwargs)

        tangents_out: X
        _, tangents_out = jax.jvp(grad, (x,), (p,))
        return tangents_out

    @utils.not_implemented
    @utils.jit_method(inline=True)
    def hess_diag(self, x: X, /, *args, **kwargs) -> X:
        if utils.is_implemented(self.jac_and_hess_diag):
            jac, hess_diag = self.jac_and_hess_diag(x, *args, **kwargs)
            return hess_diag
        return functional.hess_diag(self.fun)(x, *args, **kwargs)

    @utils.not_implemented
    @utils.jit_method(inline=True)
    def hess_quad(self, x: X, p: X, /, *args, **kwargs) -> FloatScalar:
        hessp: X = self.hessp(x, p, *args, **kwargs)
        return math_tree.tree_vdot(p, hessp)

    @utils.not_implemented
    @utils.jit_method(inline=True)
    def fun_and_jac(self, x: X, /, *args, **kwargs) -> tuple[FloatScalar, X]:
        return self.fun(x, *args, **kwargs), self.jac(x, *args, **kwargs)

    @utils.not_implemented
    @utils.jit_method(inline=True)
    def jac_and_hess_diag(self, x: X, /, *args, **kwargs) -> tuple[X, X]:
        return self.jac(x, *args, **kwargs), self.hess_diag(x, *args, **kwargs)
