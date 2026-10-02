import enum
from typing import Protocol

import attrs
from jaxtyping import Float
from torch import Tensor

type Vector = Float[Tensor, " N"]


class Result(enum.StrEnum):
    """Result code returned by linear solvers."""

    SUCCESS = enum.auto()
    PRIMARY_SUCCESS = enum.auto()
    SECONDARY_SUCCESS = enum.auto()

    BREAKDOWN = enum.auto()
    MAX_STEPS_REACHED = enum.auto()
    UNKNOWN_ERROR = enum.auto()

    @property
    def success(self) -> bool:
        """Whether the result represents an accepted solution."""
        return self in {
            Result.SUCCESS,
            Result.PRIMARY_SUCCESS,
            Result.SECONDARY_SUCCESS,
        }


class State(Protocol):
    """Protocol for solver states that expose solution parameters."""

    @property
    def params(self) -> Vector:
        """Current solution estimate."""


class Stats(Protocol):
    """Protocol for solver-specific summary statistics."""


@attrs.define
class Solution[S: State, T: Stats]:
    """Linear-solver output bundle."""

    result: Result
    state: S
    stats: T

    @property
    def params(self) -> Vector:
        """Final solution parameters."""
        return self.state.params

    @property
    def success(self) -> bool:
        """Whether [`result`][liblaf.apple.solvers.linalg.base.Solution.result] is successful."""
        return self.result.success
