import abc
from collections.abc import Mapping

import einops
import jax
import jax.numpy as jnp
import scipy.optimize
from jaxtyping import Float

from liblaf import apple


class InversePhysicsProblem(abc.ABC):
    forward_problem: apple.AbstractPhysicsProblem

    def forward(
        self, p: Mapping[str, Float[jax.Array, "..."]]
    ) -> Float[jax.Array, " DoF"]:
        result: scipy.optimize.OptimizeResult = self.forward_problem.solve(p)
        return result["x"]

    def fun(self, p: Mapping[str, Float[jax.Array, "..."]]) -> Float[jax.Array, ""]:
        u: Float[jax.Array, " DoF"] = self.forward(p)
        return self._fun(u, p)

    def jac(
        self, p: Mapping[str, Float[jax.Array, "..."]]
    ) -> Mapping[str, Float[jax.Array, "..."]]:
        u: Float[jax.Array, " DoF"] = self.forward(p)
        return self._jac(u, p)

    @abc.abstractmethod
    def _fun(
        self, u: Float[jax.Array, " DoF"], p: Mapping[str, Float[jax.Array, "..."]]
    ) -> Float[jax.Array, ""]: ...

    def _jac(
        self, u: Float[jax.Array, " DoF"], p: Mapping[str, Float[jax.Array, "..."]]
    ) -> Mapping[str, Float[jax.Array, "..."]]:
        hess: Float[jax.Array, "DoF DoF"] = self.forward_problem.hess(u, p)
        dJ_du: Float[jax.Array, " DoF"] = self.dJ_du(u, p)
        q: Float[jax.Array, " DoF"] = -jnp.linalg.solve(hess, dJ_du)
        dh_dp: Mapping[str, Float[jax.Array, " ..."]] = self.forward_problem.dh_dq(u, p)
        dJ_dp: Mapping[str, Float[jax.Array, " ..."]] = self.dJ_dp(u, p)
        return jax.tree.map(
            lambda dJ_dq, dh_dq: dJ_dq
            + einops.einsum(dh_dq, q, "DoF ..., DoF ... -> ..."),
            dJ_dp,
            dh_dp,
        )

    def dJ_du(
        self, u: Float[jax.Array, " DoF"], p: Mapping[str, Float[jax.Array, "..."]]
    ) -> Float[jax.Array, " DoF"]:
        return jax.grad(self._fun)(u, p)

    def dJ_dp(
        self, u: Float[jax.Array, " DoF"], p: Mapping[str, Float[jax.Array, "..."]]
    ) -> Mapping[str, Float[jax.Array, "..."]]:
        return jax.grad(lambda p: self._fun(u, p))(p)
