import operator

import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import struct, utils
from liblaf.apple.sim.core.params import GlobalParams


class Energy(struct.GraphNode):
    @utils.not_implemented
    @utils.jit
    def fun(self, x: struct.DictArray, /, params: GlobalParams) -> Float[jax.Array, ""]:
        fun: Float[jax.Array, ""]
        if utils.is_implemented(self.fun_and_jac):
            fun, _jac = self.fun_and_jac(x, params)
            return fun
        raise NotImplementedError

    @utils.not_implemented
    @utils.jit
    def jac(self, x: struct.DictArray, /, params: GlobalParams) -> struct.DictArray:
        jac: struct.DictArray
        if utils.is_implemented(self.fun_and_jac):
            _, jac = self.fun_and_jac(x, params)
            return jac
        if utils.is_implemented(self.jac_and_hess_diag):
            jac, _hess_diag = self.jac_and_hess_diag(x, params)
            return jac
        return jax.grad(self.fun)(x, params)

    @utils.not_implemented
    @utils.jit
    def hessp(
        self, x: struct.DictArray, p: struct.DictArray, /, params: GlobalParams
    ) -> struct.DictArray:
        return jax.jvp(lambda x: self.jac(x, params), (x,), (p,))[1]

    @utils.not_implemented
    @utils.jit
    def hess_diag(
        self, x: struct.DictArray, /, params: GlobalParams
    ) -> struct.DictArray:
        hess_diag: struct.DictArray
        if utils.is_implemented(self.jac_and_hess_diag):
            _jac, hess_diag = self.jac_and_hess_diag(x, params)
            return hess_diag
        raise NotImplementedError

    @utils.not_implemented
    @utils.jit
    def hess_quad(
        self, x: struct.DictArray, p: struct.DictArray, /, params: GlobalParams
    ) -> Float[jax.Array, ""]:
        return jax.tree.reduce(
            operator.add, jax.tree.map(jnp.vdot, x, self.hessp(x, p, params))
        )

    @utils.not_implemented
    @utils.jit
    def fun_and_jac(
        self, x: struct.DictArray, /, params: GlobalParams
    ) -> tuple[Float[jax.Array, ""], struct.DictArray]:
        return self.fun(x, params), self.jac(x, params)

    @utils.not_implemented
    @utils.jit
    def jac_and_hess_diag(
        self, x: struct.DictArray, /, params: GlobalParams
    ) -> tuple[struct.DictArray, struct.DictArray]:
        return self.jac(x, params), self.hess_diag(x, params)
