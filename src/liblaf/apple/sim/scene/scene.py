from typing import Self

import jax
from jaxtyping import Float

from liblaf.apple import struct, utils


@struct.pytree
class Scene(struct.PyTreeMixin):
    # region Optimization

    @utils.jit
    def fun(self, x: Float[jax.Array, " DOF"], /) -> Float[jax.Array, ""]:
        raise NotImplementedError

    @utils.jit
    def jac(self, x: Float[jax.Array, " DOF"], /) -> Float[jax.Array, " DOF"]:
        raise NotImplementedError

    @utils.jit
    def hessp(
        self, x: Float[jax.Array, " DOF"], p: Float[jax.Array, " DOF"], /
    ) -> Float[jax.Array, " DOF"]:
        raise NotImplementedError

    @utils.jit
    def hess_diag(self, x: Float[jax.Array, " DOF"], /) -> Float[jax.Array, " DOF"]:
        raise NotImplementedError

    @utils.jit
    def hess_quad(
        self, x: Float[jax.Array, " DOF"], p: Float[jax.Array, " DOF"], /
    ) -> Float[jax.Array, ""]:
        raise NotImplementedError

    # endregion Optimization

    # region State Management

    def prepare(self, x: Float[jax.Array, " DOF"], /) -> Self:
        raise NotImplementedError

    def step(self, x: Float[jax.Array, " DOF"], /) -> Self:
        raise NotImplementedError

    def update(self, x: Float[jax.Array, " DOF"], /) -> Self:
        raise NotImplementedError

    # endregion State Management
