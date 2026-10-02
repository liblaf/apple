import attrs
from jaxtyping import Float
from torch import Tensor

from liblaf.apple.solvers.linalg.base import State, Stats

type Scalar = Float[Tensor, ""]
type Vector = Float[Tensor, " N"]


@attrs.define
class CupyState(State):
    """State recorded by CuPy-backed linear solvers."""

    params: Vector
    info: int = -1
    step: int | None = None


@attrs.define
class CupyStats(Stats):
    """Stats placeholder for CuPy-backed linear solvers."""

    absolute_residual: Scalar = attrs.field(default=None)
    relative_residual: Scalar = attrs.field(default=None)
