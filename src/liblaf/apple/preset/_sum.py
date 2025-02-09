from collections.abc import Sequence

import attrs
import jax
from jaxtyping import Float, Scalar

from liblaf import apple


@attrs.frozen
class SumPrepared(apple.ProblemPrepared):
    problems: Sequence[apple.ProblemPrepared]

    @property
    def n_dof(self) -> int:
        return self.problems[0].n_dof

    def fun(self, u: Float[jax.Array, " DoF"]) -> Scalar:
        return sum(problem.fun(u) for problem in self.problems)  # pyright: ignore[reportReturnType]


@attrs.define
class Sum(apple.Problem):
    problems: Sequence[apple.Problem]

    @property
    def n_dof(self) -> int:
        return self.problems[0].n_dof

    def prepare(self) -> apple.ProblemPrepared:
        return SumPrepared(problems=[problem.prepare() for problem in self.problems])
