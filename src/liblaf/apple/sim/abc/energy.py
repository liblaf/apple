from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import struct, utils

from .field import FieldCollection
from .obj import Object


class Energy(struct.Node):
    @property
    def obj(self) -> Object:
        raise NotImplementedError

    @property
    def objects(self) -> struct.NodeCollection[Object]:
        return struct.NodeCollection(self.obj)

    # region Optimization

    @utils.not_implemented
    @utils.jit
    def fun(self, x: FieldCollection, /) -> Float[jax.Array, ""]:
        fun: Float[jax.Array, ""]
        fun, _jac = self.fun_and_jac(x)
        return fun

    @utils.not_implemented
    @utils.jit
    def jac(self, x: FieldCollection, /) -> FieldCollection:
        jac: FieldCollection
        if utils.is_implemented(self.fun_and_jac):
            _, jac = self.fun_and_jac(x)
            return jac
        if utils.is_implemented(self.jac_and_hess_diag):
            jac, _hess_diag = self.jac_and_hess_diag(x)
            return jac
        return jax.grad(self.fun)(x)

    @utils.not_implemented
    @utils.jit
    def hessp(self, x: FieldCollection, p: FieldCollection, /) -> FieldCollection:
        return jax.jvp(lambda x: self.jac(x), (x,), (p,))[1]

    @utils.not_implemented
    @utils.jit
    def hess_diag(self, x: FieldCollection, /) -> FieldCollection:
        hess_diag: FieldCollection
        _jac, hess_diag = self.jac_and_hess_diag(x)
        return hess_diag

    @utils.not_implemented
    @utils.jit
    def hess_quad(
        self, x: FieldCollection, p: FieldCollection, /
    ) -> Float[jax.Array, ""]:
        hessp: FieldCollection = self.hessp(x, p)
        hess_quad: Float[jax.Array, ""] = jnp.asarray(0.0)
        for key in x:
            hess_quad += jnp.vdot(jnp.asarray(p[key]), jnp.asarray(hessp[key]))
        return hess_quad

    @utils.not_implemented
    def fun_and_jac(
        self, x: FieldCollection, /
    ) -> tuple[Float[jax.Array, ""], FieldCollection]:
        return self.fun(x), self.jac(x)

    @utils.not_implemented
    def jac_and_hess_diag(
        self, x: FieldCollection, /
    ) -> tuple[FieldCollection, FieldCollection]:
        return self.jac(x), self.hess_diag(x)

    # endregion Optimization

    def prepare(self) -> Self:
        return self
