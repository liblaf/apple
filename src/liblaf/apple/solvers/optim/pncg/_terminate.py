import attrs
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from liblaf.apple.solvers.optim.base import Result

type Scalar = Float[Tensor, ""]
type Vector = Float[Tensor, " N"]


@attrs.define(kw_only=True)
class ConvergenceState:
    """Gradient-norm convergence state."""

    grad_norm: Scalar = attrs.field(default=None)
    grad_norm_first: Scalar = attrs.field(default=None)
    stagnation_count: int = 0
    step: int = 0


@attrs.define(kw_only=True)
class ConvergenceCriteria:
    """Gradient-norm stopping criteria for `Pncg`."""

    def _default_atol_secondary(self) -> float:
        return 1e3 * self.atol_primary

    def _default_rtol_secondary(self) -> float:
        return 1e3 * self.rtol_primary

    max_steps: int = 1000

    atol_primary: float = 0.0
    rtol_primary: float = 1e-6
    atol_secondary: float = attrs.field(
        default=attrs.Factory(_default_atol_secondary, takes_self=True)
    )
    rtol_secondary: float = attrs.field(
        default=attrs.Factory(_default_rtol_secondary, takes_self=True)
    )

    patience: int = 20

    def init(self) -> ConvergenceState:
        """Create an empty convergence state shaped like `params`."""
        return ConvergenceState()

    def update(
        self, state: ConvergenceState, g: Vector, *, line_search_ok: bool
    ) -> ConvergenceState:
        """Record the current gradient and gradient norms."""
        grad_norm: Scalar = torch.linalg.vector_norm(g)
        if state.step == 0:
            state.grad_norm_first = grad_norm
        if line_search_ok:
            state.stagnation_count = 0
        else:
            state.stagnation_count += 1
        state.grad_norm = grad_norm
        state.step += 1
        return state

    def terminate(self, state: ConvergenceState) -> tuple[bool, Result]:
        """Return the convergence decision and result code."""
        if state.step < self.max_steps:
            if self.primary_success(state):
                return True, Result.PRIMARY_SUCCESS
            if state.stagnation_count > self.patience:
                return True, Result.STAGNATION
            return False, Result.INTERRUPT
        if self.secondary_success(state):
            return True, Result.SECONDARY_SUCCESS
        return True, Result.MAX_STEPS_REACHED

    def primary_success(self, state: ConvergenceState) -> Bool[Tensor, ""]:
        """Check the primary absolute-relative gradient tolerance."""
        return state.grad_norm <= max(
            self.atol_primary, self.rtol_primary * state.grad_norm_first
        )

    def secondary_success(self, state: ConvergenceState) -> Bool[Tensor, ""]:
        """Check the looser secondary gradient tolerance."""
        return state.grad_norm <= max(
            self.atol_secondary, self.rtol_secondary * state.grad_norm_first
        )
