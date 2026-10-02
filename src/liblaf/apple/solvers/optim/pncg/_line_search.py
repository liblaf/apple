import math

import attrs
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from liblaf.apple.solvers.optim.base import Problem
from liblaf.apple.solvers.utils import is_implemented

type Scalar = Float[Tensor, ""]
type Vector = Float[Tensor, " N"]


@attrs.define
class LineSearchState:
    """Mutable diagnostics from Armijo backtracking."""

    f_alpha: Scalar
    alpha: Scalar = attrs.field(default=None)
    f0: Scalar = attrs.field(default=None)
    ok: bool = False
    step: int = 0


@attrs.frozen
class LineSearch:
    """Armijo backtracking with Newton, norm, and problem-specific step bounds."""

    armijo: float = attrs.field(default=1e-4)
    max_step_norm: float = attrs.field(default=torch.inf)
    max_steps: int = attrs.field(default=10)

    def __call__[X](
        self,
        state: LineSearchState,
        problem: Problem[X],
        model_state: X,
        m: Scalar,
        p: Vector,
        params: Vector,
        pHp: Scalar,
    ) -> None:
        """Run line search along direction `p` from `params`.

        The initial step length is the smaller of the Newton proposal and the
        configured infinity-norm bound. If the problem implements
        [`Problem.max_step_size`][liblaf.apple.solvers.optim.base.Problem.max_step_size],
        that hook receives the proposed displacement `alpha * p` and returns a
        safe fraction of it. Every accepted or rejected trial is materialized
        through [`Problem.update`][liblaf.apple.solvers.optim.base.Problem.update].
        """
        alpha_upper: Scalar = self.line_search_upper(
            p=p, max_step_norm=self.max_step_norm
        )
        alpha_newton: Scalar = self.line_search_newton(m=m, pHp=pHp)
        alpha: Scalar = torch.minimum(alpha_upper, alpha_newton)
        if is_implemented(problem, Problem.max_step_size):
            step_fraction: Scalar = problem.max_step_size(model_state, alpha * p)
            step_fraction: Scalar = torch.as_tensor(step_fraction)
            alpha *= torch.clamp(step_fraction, 0.0, 1.0)

        f0: Scalar = state.f_alpha
        for step in range(self.max_steps + 1):
            if step > 0:
                alpha *= 0.5
            problem.update(model_state, params + alpha * p)
            f_alpha: Scalar = problem.fun(model_state)
            if self.armijo_condition(f_alpha=f_alpha, f0=f0, alpha=alpha, m=m):
                state.ok = True
                break
        else:
            state.ok = False
        state.alpha = alpha
        state.f_alpha = f_alpha
        state.f0 = f0
        state.step = step

    def init(self, fun: Scalar) -> LineSearchState:
        """Create an empty line-search state."""
        return LineSearchState(f_alpha=fun)

    @staticmethod
    def line_search_newton(m: Scalar, pHp: Scalar) -> Scalar:
        """Return the Newton step length or `1.0` when curvature is unsuitable."""
        alpha: Scalar = -m / pHp
        return torch.where((pHp <= 0.0) | (m >= 0.0), 1.0, alpha)

    @staticmethod
    def line_search_upper(p: Vector, max_step_norm: float) -> Scalar:
        """Return the largest step satisfying the infinity-norm bound."""
        if not math.isfinite(max_step_norm):
            return torch.as_tensor(torch.inf)
        p_norm: Scalar = torch.linalg.vector_norm(p, ord=torch.inf)
        return torch.where(p_norm == 0.0, 0.0, max_step_norm / p_norm)

    def armijo_condition(
        self, f_alpha: Scalar, f0: Scalar, alpha: Scalar, m: Scalar
    ) -> Bool[Tensor, ""]:
        """Evaluate the Armijo sufficient-decrease condition."""
        return f_alpha <= f0 + alpha * self.armijo * m
