import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, PyTree

from liblaf.apple import utils


class DerivativeMixin:
    @utils.not_implemented
    @eqx.filter_jit
    def fun(self, x: PyTree, /, *args, **kwargs) -> Float[jax.Array, ""]:
        fun: Float[jax.Array, ""]
        if utils.is_implemented(self.fun_and_jac):
            fun, _jac = self.fun_and_jac(x, *args, **kwargs)
            return fun
        raise NotImplementedError

    @utils.not_implemented
    @eqx.filter_jit
    def jac[T: PyTree](self, x: T, /, *args, **kwargs) -> T:
        jac: T
        if utils.is_implemented(self.fun_and_jac):
            _fun, jac = self.fun_and_jac(x, *args, **kwargs)
            return jac
        if utils.is_implemented(self.jac_and_hess_diag):
            jac, _hess_diag = self.jac_and_hess_diag(x, *args, **kwargs)
            return jac
        return eqx.filter_grad(self.fun)(x, *args, **kwargs)

    @utils.not_implemented
    @eqx.filter_jit
    def hessp[T: PyTree](self, x: T, p: T, /, *args, **kwargs) -> T:
        return eqx.filter_jvp(lambda x: self.fun(x, *args, **kwargs), (x,), (p,))[1]

    @utils.not_implemented
    @eqx.filter_jit
    def hess_quad[T: PyTree](self, x: T, p: T, /) -> Float[jax.Array, ""]:
        return jnp.vdot(jnp.asarray(p), jnp.asarray(self.hessp(x, p)))

    @utils.not_implemented
    @eqx.filter_jit
    def hess_diag[T: PyTree](self, x: T, /) -> T:
        hess_diag: T
        if utils.is_implemented(self.jac_and_hess_diag):
            _jac, hess_diag = self.jac_and_hess_diag(x)
            return hess_diag
        raise NotImplementedError

    @utils.not_implemented
    @eqx.filter_jit
    def fun_and_jac[T: PyTree](
        self, x: T, /, *args, **kwargs
    ) -> tuple[Float[jax.Array, ""], T]:
        return self.fun(x, *args, **kwargs), self.jac(x, *args, **kwargs)

    @utils.not_implemented
    @eqx.filter_jit
    def jac_and_hess_diag[T: PyTree](self, x: T, /, *args, **kwargs) -> tuple[T, T]:
        return self.jac(x, *args, **kwargs), self.hess_diag(x, *args, **kwargs)
