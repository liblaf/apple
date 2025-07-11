from collections.abc import Sequence
from typing import override

import jax
import jax.numpy as jnp
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple import struct, utils
from liblaf.apple.optim.optimizer import OptimizeResult
from liblaf.apple.optim.problem.problem import OptimizationProblem

from .optimizer import Optimizer
from .problem import X

type FloatScalar = Float[jax.Array, ""]


@struct.pytree
class State(struct.PyTreeMixin):
    alpha: FloatScalar = struct.array(default=None)
    beta: FloatScalar = struct.array(default=None)
    Delta_E: FloatScalar = struct.array(default=None)
    g: X = struct.array(default=None)
    hess_diag: X = struct.array(default=None)
    hess_quad: FloatScalar = struct.array(default=None)
    p: X = struct.array(default=None)
    P: X = struct.array(default=None)
    x: X = struct.array(default=None)

    first: bool = struct.static(default=True)


@struct.pytree
class PNCG(Optimizer):
    """Preconditioned Nonlinear Conjugate Gradient Method.

    References:
        1. Xing Shen, Runyuan Cai, Mengxiao Bi, and Tangjie Lv. 2024. Preconditioned Nonlinear Conjugate Gradient Method for Real-time Interior-point Hyperelasticity. In ACM SIGGRAPH 2024 Conference Papers (SIGGRAPH '24). Association for Computing Machinery, New York, NY, USA, Article 96, 1–11. https://doi.org/10.1145/3641519.3657490
    """

    atol: float = struct.data(default=0.0)
    d_hat: float = struct.data(default=jnp.inf)
    maxiter: int = struct.data(default=150)
    rtol: float = struct.data(default=5e-5)

    @override
    def _minimize_impl(
        self,
        problem: OptimizationProblem,
        x0: Float[ArrayLike, " N"],
        args: Sequence,
        **kwargs,
    ) -> OptimizeResult:
        assert callable(problem.hess_quad)
        assert callable(problem.jac_and_hess_diag)

        x: X = jnp.asarray(x0)
        Delta_E0: FloatScalar = None  # pyright: ignore[reportAssignmentType]
        result = OptimizeResult(success=False, x=x)
        state: State = State(x=x, first=True)

        n_iter: int = 0
        for it in range(self.maxiter):
            state = self.step(problem, state, args=args)
            if it == 0:
                Delta_E0 = state.Delta_E
            n_iter = it + 1
            result.update(
                alpha=state.alpha,
                beta=state.beta,
                Delta_E=state.Delta_E,
                Delta_E0=Delta_E0,
                hess_diag=state.hess_diag,
                hess_quad=state.hess_quad,
                jac=state.g,
                n_iter=n_iter,
                p=state.p,
                P=state.P,
                x=state.x,
            )
            if not jnp.isfinite(state.x).all() or not jnp.isfinite(state.Delta_E).all():
                break
            if callable(problem.callback):
                problem.callback(result)
            if state.Delta_E <= self.atol:
                result["success"] = True
                break
            if it > 0 and (state.Delta_E <= self.rtol * Delta_E0):
                result["success"] = True
                break

        result["n_iter"] = n_iter
        result["n_jac_and_hess_diag"] = n_iter
        result["n_hess_quad"] = n_iter

        return result

    @utils.jit_method(inline=True)
    def compute_alpha(self, g: X, p: X, pHp: FloatScalar) -> FloatScalar:
        alpha_1: FloatScalar = self.d_hat / (2 * jnp.linalg.norm(p, ord=jnp.inf))
        alpha_2: FloatScalar = -jnp.vdot(g, p) / pHp
        # alpha_2: FloatScalar = jnp.nan_to_num(alpha_2, nan=0.0)
        alpha: FloatScalar = jnp.minimum(alpha_1, alpha_2)
        # alpha = jnp.nan_to_num(alpha, nan=0.0)
        return alpha

    @utils.jit_method(inline=True)
    def compute_beta(self, g_prev: X, g: X, p: X, P: X) -> FloatScalar:
        y: X = g - g_prev
        yTp: FloatScalar = jnp.vdot(y, p)
        beta: FloatScalar = jnp.vdot(g, P * y) / yTp - (jnp.vdot(y, P * y) / yTp) * (
            jnp.vdot(p, g) / yTp
        )
        # beta = jnp.nan_to_num(beta, nan=0.0)
        return beta

    # @utils.jit_method(static_argnames=("problem", "args"), inline=True)
    def step(self, problem: OptimizationProblem, state: State, args: Sequence) -> State:
        assert callable(problem.hess_quad)
        assert callable(problem.jac_and_hess_diag)

        g: X
        hess_diag: X
        p: X = state.p
        beta: FloatScalar = state.beta
        x: X = state.x
        g, hess_diag = problem.jac_and_hess_diag(x, *args)
        P: X = jnp.reciprocal(hess_diag)
        P: X = jnp.nan_to_num(P, nan=1.0, posinf=1.0, neginf=1.0)

        if state.first:
            p = -P * g
        else:
            beta = self.compute_beta(g_prev=state.g, g=g, p=p, P=P)
            p = -P * g + beta * p
        pHp: FloatScalar = problem.hess_quad(x, p, *args)
        pHp = jnp.nan_to_num(pHp, nan=0.0)
        pHp = jnp.where(pHp > 0, pHp, 1.0)
        alpha: FloatScalar = self.compute_alpha(g=g, p=p, pHp=pHp)
        x += alpha * p
        Delta_E: FloatScalar = -alpha * jnp.vdot(g, p) - 0.5 * jnp.square(alpha) * pHp
        return state.evolve(
            alpha=alpha,
            beta=beta,
            Delta_E=Delta_E,
            g=g,
            hess_diag=hess_diag,
            hess_quad=pHp,
            p=p,
            P=P,
            x=x,
            first=False,
        )
