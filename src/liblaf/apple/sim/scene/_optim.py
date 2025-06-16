from typing import Protocol, Self

import attrs
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float

from liblaf.apple import optim


class FrozenProblem(Protocol):
    def prepare(self, x: Float[jax.Array, " DoF"], /) -> Self: ...
    def fun(self, x: Float[ArrayLike, " DoF"], /) -> Float[jax.Array, ""]: ...
    def jac(self, x: Float[ArrayLike, " DoF"], /) -> Float[jax.Array, " DoF"]: ...
    def hessp(
        self, x: Float[ArrayLike, " DoF"], p: Float[ArrayLike, " DoF"], /
    ) -> Float[jax.Array, " DoF"]: ...
    def hess_diag(self, x: Float[ArrayLike, " DoF"], /) -> Float[jax.Array, " DoF"]: ...
    def hess_quad(
        self, x: Float[ArrayLike, " DoF"], p: Float[ArrayLike, " DoF"], /
    ) -> Float[jax.Array, ""]: ...
    def fun_and_jac(
        self, x: Float[ArrayLike, " DoF"], /
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, " DoF"]]: ...
    def jac_and_hess_diag(
        self, x: Float[ArrayLike, " DoF"], /
    ) -> tuple[Float[jax.Array, " DoF"], Float[jax.Array, " DoF"]]: ...


@attrs.define
class OptimizationProblem:
    scene: FrozenProblem = attrs.field()
    _callback: optim.Callback = attrs.field(default=None, alias="callback")

    def callback(self, result: optim.OptimizeResult, /) -> None:
        x: Float[jax.Array, " DoF"] = jnp.asarray(result["x"])
        self.scene = self.scene.prepare(x)
        if callable(self._callback):
            self._callback(result)

    def fun(self, x: Float[ArrayLike, " DoF"], /) -> Float[jax.Array, ""]:
        return self.scene.fun(x)

    def jac(self, x: Float[ArrayLike, " DoF"], /) -> Float[jax.Array, " DoF"]:
        return self.scene.jac(x)

    def hessp(
        self, x: Float[ArrayLike, " DoF"], p: Float[ArrayLike, " DoF"], /
    ) -> Float[jax.Array, " DoF"]:
        return self.scene.hessp(x, p)

    def hess_diag(self, x: Float[ArrayLike, " DoF"], /) -> Float[jax.Array, " DoF"]:
        return self.scene.hess_diag(x)

    def hess_quad(
        self, x: Float[ArrayLike, " DoF"], p: Float[ArrayLike, " DoF"], /
    ) -> Float[jax.Array, ""]:
        return self.scene.hess_quad(x, p)

    def fun_and_jac(
        self, x: Float[ArrayLike, " DoF"], /
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, " DoF"]]:
        return self.scene.fun_and_jac(x)

    def jac_and_hess_diag(
        self, x: Float[ArrayLike, " DoF"], /
    ) -> tuple[Float[jax.Array, " DoF"], Float[jax.Array, " DoF"]]:
        return self.scene.jac_and_hess_diag(x)
