import attrs
from jaxtyping import Float
from torch import Tensor

from liblaf.apple.solvers.optim.base import State, Stats

from ._hess_damping import HessianDampingState
from ._line_search import LineSearchState
from ._terminate import ConvergenceState

type Scalar = Float[Tensor, ""]
type Vector = Float[Tensor, " N"]


@attrs.define(kw_only=True)
class PncgState(State):
    """State tracked by [`Pncg`][liblaf.apple.solvers.optim.pncg.Pncg]."""

    fun: Scalar
    params: Vector
    convergence_state: ConvergenceState
    hess_damping_state: HessianDampingState
    line_search_state: LineSearchState
    direction: Vector = attrs.field(default=None)
    grad: Vector = attrs.field(default=None)
    hess_diag: Vector = attrs.field(default=None)
    hess_quad: Vector = attrs.field(default=None)
    slope: Vector = attrs.field(default=None)

    @property
    def step(self) -> int:
        return self.convergence_state.step


@attrs.define(kw_only=True)
class PncgStats(Stats):
    """Stats placeholder for [`Pncg`][liblaf.apple.solvers.optim.pncg.Pncg]."""
