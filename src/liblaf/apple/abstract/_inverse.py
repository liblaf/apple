import abc

import attrs
import jax
import jax.flatten_util
import pylops
import pylops.optimization.basic
import scipy.optimize
from jaxtyping import Float, PyTree

from liblaf import apple


@apple.register_dataclass()
@attrs.define(kw_only=True)
class InversePhysicsProblem(abc.ABC):
    forward_problem: apple.PhysicsProblem
    forward_algo: apple.MinimizeAlgorithm | None = attrs.field(
        default=apple.MinimizeScipy(method="trust-constr"), metadata={"static": True}
    )

    def forward(self, q: PyTree) -> Float[jax.Array, " DoF"]:
        result: scipy.optimize.OptimizeResult = self.forward_problem.solve(
            q=q, algo=self.forward_algo
        )
        return result["x"]

    def forward_flat(self, q_flat: Float[jax.Array, " Q"]) -> Float[jax.Array, " DoF"]:
        q: PyTree = self.unravel_q(q_flat)
        return self.forward(q)

    def fun(self, q: PyTree) -> Float[jax.Array, ""]:
        u: Float[jax.Array, " DoF"] = self.forward(q)
        return self.objective(u, q)

    def fun_flat(self, q_flat: Float[jax.Array, " Q"]) -> Float[jax.Array, ""]:
        q: PyTree = self.unravel_q(q_flat)
        return self.fun(q)

    def jac(self, q: PyTree) -> Float[jax.Array, " Q"]:
        q_flat: Float[jax.Array, " Q"] = self.ravel_q(q)
        return self.jac_flat(q_flat)

    def jac_flat(self, q_flat: Float[jax.Array, " Q"]) -> Float[jax.Array, " Q"]:
        u_flat: Float[jax.Array, " DoF"] = self.forward_flat(q_flat)
        return self._jac(u_flat, q_flat)

    @abc.abstractmethod
    def objective(self, u: PyTree, q: PyTree) -> Float[jax.Array, ""]: ...

    @apple.jit()
    def objective_flat(
        self, u_flat: Float[jax.Array, " DoF"], q_flat: Float[jax.Array, " P"]
    ) -> Float[jax.Array, ""]:
        u: PyTree = self.unravel_u(u_flat)
        p: PyTree = self.unravel_q(q_flat)
        return self.objective(u, p)

    def _jac(
        self, u_flat: Float[jax.Array, " DoF"], q_flat: Float[jax.Array, " Q"]
    ) -> Float[jax.Array, " Q"]:
        hess: pylops.LinearOperator = self.forward_problem.hess_flat(u_flat, q_flat)
        dJ_du: Float[jax.Array, " DoF"] = self.dJ_du_flat(u_flat, q_flat)
        result: apple.LinearResult = apple.cgls(hess, -dJ_du)
        p: Float[jax.Array, " DoF"] = result["x"]
        dh_dq: Float[pylops.LinearOperator, "DoF Q"] = self.forward_problem.dh_dq_flat(
            u_flat, q_flat
        )
        dJ_dq: Float[jax.Array, " Q"] = self.dJ_dq_flat(u_flat, q_flat)
        return dJ_dq + dh_dq.rmatvec(p)  # pyright: ignore[reportReturnType]

    @apple.jit()
    def dJ_du_flat(
        self, u_flat: Float[jax.Array, " DoF"], q_flat: Float[jax.Array, " Q"]
    ) -> Float[jax.Array, " DoF"]:
        return jax.grad(self.objective_flat)(u_flat, q_flat)

    @apple.jit()
    def dJ_dq_flat(
        self, u_flat: Float[jax.Array, " DoF"], q_flat: Float[jax.Array, " Q"]
    ) -> Float[jax.Array, " Q"]:
        return jax.grad(lambda q_flat: self.objective_flat(u_flat, q_flat))(q_flat)

    def ravel_q(self, q: PyTree) -> Float[jax.Array, " Q"]:
        return self.forward_problem.ravel_q(q)

    def ravel_u(self, u: PyTree) -> Float[jax.Array, " DoF"]:
        return self.forward_problem.ravel_u(u)

    def unravel_q(self, q_flat: Float[jax.Array, " Q"]) -> PyTree:
        return self.forward_problem.unravel_q(q_flat)

    def unravel_u(self, u_flat: Float[jax.Array, " DoF"]) -> PyTree:
        return self.forward_problem.unravel_u(u_flat)
