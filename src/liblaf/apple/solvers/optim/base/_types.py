from __future__ import annotations

import enum

import attrs
from jaxtyping import Float
from torch import Tensor

from ._protocols import State, Stats

type Vector = Float[Tensor, " N"]


class Result(enum.StrEnum):
    """Result code returned by optimizers."""

    SUCCESS = enum.auto()
    PRIMARY_SUCCESS = enum.auto()
    SECONDARY_SUCCESS = enum.auto()

    INTERRUPT = enum.auto()
    MAX_STEPS_REACHED = enum.auto()
    NAN = enum.auto()
    STAGNATION = enum.auto()
    UNKNOWN_ERROR = enum.auto()

    @property
    def success(self) -> bool:
        """Whether the result represents an accepted optimization outcome."""
        return self in {
            Result.SUCCESS,
            Result.PRIMARY_SUCCESS,
            Result.SECONDARY_SUCCESS,
        }


@attrs.define
class Solution[S: State, T: Stats]:
    """Optimizer output bundle."""

    result: Result
    state: S
    stats: T

    @property
    def params(self) -> Vector:
        """Final optimizer parameters."""
        return self.state.params

    @property
    def success(self) -> bool:
        """Whether [`result`][liblaf.apple.solvers.optim.base.Solution.result] is successful."""
        return self.result.success
