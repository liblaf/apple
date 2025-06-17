from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import struct, utils

from .field import FieldCollection
from .params import GlobalParams


class Energy(struct.GraphNode):
    # region Optimization

    def prepare(self) -> Self:
        return self

    @utils.not_implemented
    @utils.jit
    def fun(self, x: FieldCollection, /, params: GlobalParams) -> Float[jax.Array, ""]:
        fun: Float[jax.Array, ""]
        fun, _jac = self.fun_and_jac(x, params)
        return fun

    @utils.not_implemented
    @utils.jit
    def jac(self, x: FieldCollection, /, params: GlobalParams) -> FieldCollection:
        jac: FieldCollection
        if utils.is_implemented(self.fun_and_jac):
            _, jac = self.fun_and_jac(x, params)
            return jac
        if utils.is_implemented(self.jac_and_hess_diag):
            jac, _hess_diag = self.jac_and_hess_diag(x, params)
            return jac
        return jax.grad(self.fun)(x)

    @utils.not_implemented
    @utils.jit
    def hessp(
        self, x: FieldCollection, p: FieldCollection, /, params: GlobalParams
    ) -> FieldCollection:
        return jax.jvp(lambda x: self.jac(x, params), (x,), (p,))[1]

    @utils.not_implemented
    @utils.jit
    def hess_diag(self, x: FieldCollection, /, params: GlobalParams) -> FieldCollection:
        hess_diag: FieldCollection
        _jac, hess_diag = self.jac_and_hess_diag(x, params)
        return hess_diag

    @utils.not_implemented
    @utils.jit
    def hess_quad(
        self, x: FieldCollection, p: FieldCollection, /, params: GlobalParams
    ) -> Float[jax.Array, ""]:
        hessp: FieldCollection = self.hessp(x, p, params)
        hess_quad: Float[jax.Array, ""] = jnp.asarray(0.0)
        for key in x:
            hess_quad += jnp.vdot(jnp.asarray(p[key]), jnp.asarray(hessp[key]))
        return hess_quad

    @utils.not_implemented
    def fun_and_jac(
        self, x: FieldCollection, /, params: GlobalParams
    ) -> tuple[Float[jax.Array, ""], FieldCollection]:
        return self.fun(x, params), self.jac(x, params)

    @utils.not_implemented
    def jac_and_hess_diag(
        self, x: FieldCollection, /, params: GlobalParams
    ) -> tuple[FieldCollection, FieldCollection]:
        return self.jac(x, params), self.hess_diag(x, params)

    # endregion Optimization
