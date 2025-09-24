from collections.abc import Callable, Mapping, Sequence
from typing import Any, override

import attrs
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from liblaf.apple.jax import tree

from ._minimizer import Minimizer, Solution
from ._objective import Objective

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


@tree.pytree
class State:
    alpha: Scalar = tree.array(default=None)
    beta: Scalar = tree.array(default=0.0)
    DeltaE: Scalar = tree.array(default=None)
    g: Vector = tree.array(default=None)
    hess_diag: Vector = tree.array(default=None)
    hess_quad: Vector = tree.array(default=None)
    p: Vector = tree.array(default=None)
    P: Vector = tree.array(default=None)
    x: Vector = tree.array(default=None)


@tree.pytree
class MinimizerPNCG(Minimizer):
    """Preconditioned Nonlinear Conjugate Gradient Method.

    References:
        1. Xing Shen, Runyuan Cai, Mengxiao Bi, and Tangjie Lv. 2024. Preconditioned Nonlinear Conjugate Gradient Method for Real-time Interior-point Hyperelasticity. In ACM SIGGRAPH 2024 Conference Papers (SIGGRAPH '24). Association for Computing Machinery, New York, NY, USA, Article 96, 1–11. https://doi.org/10.1145/3641519.3657490
    """

    atol: float = tree.field(default=0.0)
    d_hat: float = tree.field(default=jnp.inf)
    maxiter: int = tree.field(default=150)
    rtol: float = tree.field(default=5e-5)

    @override
    def _minimize_impl(
        self,
        objective: Objective,
        x0: PyTree,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] = {},
        bounds: Any = None,
        callback: Callable[..., Any] | None = None,
    ) -> Solution:
        assert bounds is None
        assert callable(objective.hess_quad)
        assert callable(objective.jac_and_hess_diag)

        x0_flat: Vector
        unflatten: Callable[[Array], PyTree]
        x0_flat, unflatten = tree.flatten(x0)
        objective = objective.flatten(unflatten)

        DeltaE0: Scalar = jnp.zeros(())
        solution = Solution()
        state: State = State(x=x0_flat)
        for it in range(self.maxiter):
            state = self.step(
                jac_and_hess_diag=objective.jac_and_hess_diag,  # pyright: ignore[reportArgumentType]
                hess_quad=objective.hess_quad,  # pyright: ignore[reportArgumentType]
                state=state,
                args=args,
                kwargs=kwargs,
            )
            if it == 0:
                DeltaE0 = state.DeltaE
            solution.update(
                alpha=state.alpha,
                beta=state.beta,
                DeltaE=state.DeltaE,
                DeltaE0=DeltaE0,
                hess_diag=unflatten(state.hess_diag),
                hess_quad=state.hess_quad,
                jac=unflatten(state.g),
                n_iter=it + 1,
                P=unflatten(state.P),
                x=unflatten(state.x),
            )
            if callable(callback):
                callback(solution)
            if state.DeltaE <= self.atol:
                solution["success"] = True
                break
            if state.DeltaE <= self.rtol * DeltaE0:
                solution["success"] = True
                break
        return solution

    @eqx.filter_jit
    def compute_alpha(self, g: Vector, p: Vector, pHp: Scalar) -> Scalar:
        alpha_1: Scalar = self.d_hat / (2.0 * jnp.linalg.norm(p, ord=jnp.inf))
        alpha_2: Scalar = -jnp.vdot(g, p) / pHp
        alpha: Scalar = jnp.minimum(alpha_1, alpha_2)
        alpha = jnp.nan_to_num(alpha)
        return alpha

    @eqx.filter_jit
    def compute_beta(self, g_prev: Vector, g: Vector, p: Vector, P: Vector) -> Scalar:
        y: Vector = g - g_prev
        yTp: Scalar = jnp.vdot(y, p)
        beta: Scalar = jnp.vdot(g, P * y) / yTp - (jnp.vdot(y, P * y) / yTp) * (
            jnp.vdot(p, g) / yTp
        )
        return beta

    def step(
        self,
        jac_and_hess_diag: Callable[..., tuple[Vector, Vector]],
        hess_quad: Callable[..., Scalar],
        state: State,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] = {},
    ) -> State:
        x: Vector = state.x
        g: Vector
        hess_diag: Vector
        g, hess_diag = jac_and_hess_diag(x, *args, **kwargs)
        P: Vector = jnp.reciprocal(hess_diag)
        p: Vector
        if state.p is None:  # first iteration
            beta: Scalar = 0.0
            p = -P * g
        else:
            beta: Scalar = self.compute_beta(g_prev=state.g, g=g, p=state.p, P=P)
            p = -P * g + beta * state.p
        pHp: Scalar = hess_quad(x, p, *args, **kwargs)
        alpha: Scalar = self.compute_alpha(g=g, p=p, pHp=pHp)
        x += alpha * p
        DeltaE: Scalar = -alpha * jnp.vdot(g, p) - 0.5 * jnp.square(alpha) * pHp
        return attrs.evolve(
            state,
            alpha=alpha,
            beta=beta,
            DeltaE=DeltaE,
            g=g,
            hess_diag=hess_diag,
            hess_quad=pHp,
            p=p,
            P=P,
            x=x,
        )
