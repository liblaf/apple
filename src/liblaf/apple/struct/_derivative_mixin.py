import jax
import jax.numpy as jnp
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple import utils


class DerivativeMixin:
    @utils.not_implemented
    @utils.jit
    def fun(
        self, x: Float[ArrayLike, " N"], /, *args, **kwargs
    ) -> Float[jax.Array, ""]:
        fun: Float[jax.Array, ""]
        fun, _jac = self.fun_and_jac(x, *args, **kwargs)
        return fun

    @utils.not_implemented
    @utils.jit
    def jac(
        self, x: Float[ArrayLike, " N"], /, *args, **kwargs
    ) -> Float[jax.Array, " N"]:
        if utils.is_implemented(self.fun_and_jac):
            _fun, jac = self.fun_and_jac(x, *args, **kwargs)
            return jac
        if utils.is_implemented(self.jac_and_hess_diag):
            jac, _hess_diag = self.jac_and_hess_diag(x, *args, **kwargs)
            return jac
        x = jnp.asarray(x)
        return jax.grad(self.fun)(x, *args, **kwargs)

    @utils.not_implemented
    @utils.jit
    def hessp(
        self, x: Float[ArrayLike, " N"], p: Float[ArrayLike, " N"], /, *args, **kwargs
    ) -> Float[jax.Array, " N"]:
        x = jnp.asarray(x)
        p = jnp.asarray(p)
        return jax.jvp(lambda x: self.fun(x, *args, **kwargs), (x,), (p,))[1]

    @utils.not_implemented
    @utils.jit
    def hess_quad(
        self, x: Float[ArrayLike, " N"], p: Float[ArrayLike, " N"], /
    ) -> Float[jax.Array, " N"]:
        p = jnp.asarray(p)
        return jnp.dot(p, self.hessp(x, p))

    @utils.not_implemented
    @utils.jit
    def hess_diag(self, x: Float[ArrayLike, " N"], /) -> Float[jax.Array, " N"]:
        hess_diag: Float[jax.Array, " N"]
        _jac, hess_diag = self.jac_and_hess_diag(x)
        return hess_diag

    @utils.not_implemented
    @utils.jit
    def fun_and_jac(
        self, x: Float[ArrayLike, " N"], /, *args, **kwargs
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, " N"]]:
        return self.fun(x, *args, **kwargs), self.jac(x, *args, **kwargs)

    @utils.not_implemented
    @utils.jit
    def jac_and_hess_diag(
        self, x: Float[ArrayLike, " N"], /, *args, **kwargs
    ) -> tuple[Float[jax.Array, " N"], Float[jax.Array, " N"]]:
        return self.jac(x, *args, **kwargs), self.hess_diag(x, *args, **kwargs)
