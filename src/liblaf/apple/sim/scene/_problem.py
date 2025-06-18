from typing import Protocol, Self

import attrs
import jax
from jaxtyping import ArrayLike, Float

from liblaf.apple import optim


class FrozenProblem(Protocol):
    def prepare(self, x: Float[ArrayLike, " dof"], /) -> Self: ...
    def fun(self, x: Float[ArrayLike, " dof"], /) -> Float[jax.Array, ""]: ...
    def jac(self, x: Float[ArrayLike, " dof"], /) -> Float[jax.Array, " dof"]: ...
    def hessp(
        self, x: Float[ArrayLike, " dof"], p: Float[ArrayLike, " dof"], /
    ) -> Float[jax.Array, " dof"]: ...
    def hess_diag(self, x: Float[ArrayLike, " dof"], /) -> Float[jax.Array, " dof"]: ...
    def hess_quad(
        self, x: Float[ArrayLike, " dof"], p: Float[ArrayLike, " dof"], /
    ) -> Float[jax.Array, ""]: ...
    def fun_and_jac(
        self, x: Float[ArrayLike, " dof"], /
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, " dof"]]: ...
    def jac_and_hess_diag(
        self, x: Float[ArrayLike, " dof"], /
    ) -> tuple[Float[jax.Array, " dof"], Float[jax.Array, " dof"]]: ...


@attrs.define
class OptimizationProblem:
    scene: FrozenProblem = attrs.field()
    _callback: optim.Callback | None = attrs.field(default=None, alias="callback")

    def callback(self, result: optim.OptimizeResult, /) -> None:
        self.scene = self.scene.prepare(result["x"])
        if callable(self._callback):
            self._callback(result)

    def fun(self, x: Float[ArrayLike, " dof"], /) -> Float[jax.Array, ""]:
        return self.scene.fun(x)

    def jac(self, x: Float[ArrayLike, " dof"], /) -> Float[jax.Array, " dof"]:
        return self.scene.jac(x)

    def hessp(
        self, x: Float[ArrayLike, " dof"], p: Float[ArrayLike, " dof"], /
    ) -> Float[jax.Array, " dof"]:
        return self.scene.hessp(x, p)

    def hess_diag(self, x: Float[ArrayLike, " dof"], /) -> Float[jax.Array, " dof"]:
        return self.scene.hess_diag(x)

    def hess_quad(
        self, x: Float[ArrayLike, " dof"], p: Float[ArrayLike, " dof"], /
    ) -> Float[jax.Array, ""]:
        return self.scene.hess_quad(x, p)

    def fun_and_jac(
        self, x: Float[ArrayLike, " dof"], /
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, " dof"]]:
        return self.scene.fun_and_jac(x)

    def jac_and_hess_diag(
        self, x: Float[ArrayLike, " dof"], /
    ) -> tuple[Float[jax.Array, " dof"], Float[jax.Array, " dof"]]:
        return self.scene.jac_and_hess_diag(x)
