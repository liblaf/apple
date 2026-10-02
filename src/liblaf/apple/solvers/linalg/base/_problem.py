from typing import Protocol

from jaxtyping import Float
from torch import Tensor

from liblaf.apple.solvers.utils import not_implemented

type Vector = Float[Tensor, " N"]


class BaseProblem(Protocol):
    """Marker protocol for linear-system problems."""


class Problem(Protocol):
    """Protocol for a linear system `A x = b`."""

    @property
    @not_implemented
    def b(self) -> Vector:
        """Right-hand-side vector."""

    @not_implemented
    def matvec(self, x: Vector, /) -> Vector:
        """Apply the system matrix to `x`."""

    @not_implemented
    def rmatvec(self, x: Vector, /) -> Vector:
        """Apply the transpose or adjoint system matrix to `x`."""

    @not_implemented
    def precondition(self, x: Vector, /) -> Vector:
        """Apply an optional left preconditioner to `x`."""

    @not_implemented
    def rprecondition(self, x: Vector, /) -> Vector:
        """Apply an optional transpose or adjoint preconditioner to `x`."""
