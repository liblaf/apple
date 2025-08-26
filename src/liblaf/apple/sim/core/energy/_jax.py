from collections.abc import Mapping
from typing import NoReturn

import jax
import jax.numpy as jnp
from jaxtyping import Array

from liblaf.apple import struct
from liblaf.apple.types.jax import Scalar, Updates, UpdatesData, UpdatesIndex, Vector


@struct.pytree
class EnergyJax(struct.IdMixin):
    def fun(self, u: Vector) -> Scalar:
        raise NotImplementedError

    def jac(self, u: Vector) -> Updates:
        jac: Vector = jax.grad(self.fun)(u)
        return jac, jnp.arange(u.shape[0])

    def hess(self, u: Vector) -> NoReturn:
        raise NotImplementedError

    def hess_diag(self, u: Vector) -> Updates:
        raise NotImplementedError

    def hess_prod(self, u: Vector, p: Vector) -> Updates:
        Hp: Vector
        _, Hp = jax.jvp(jax.grad(self.fun), (u,), (p,))
        return Hp, jnp.arange(u.shape[0])

    def hess_quad(self, u: Vector, p: Vector) -> Scalar:
        data: UpdatesData
        index: UpdatesIndex
        data, index = self.hess_prod(u, p)
        Hp: Vector = jax.ops.segment_sum(data, index, num_segments=u.shape[0])
        return jnp.vdot(Hp, p)

    def hess_mixed(self, u: Vector) -> NoReturn:
        raise NotImplementedError

    def hess_mixed_prod(self, u: Vector, p: Vector) -> Mapping[str, Array]:
        raise NotImplementedError
