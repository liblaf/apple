from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Self

import attrs
import jax
import numpy as np
from jaxtyping import Bool, Float

from liblaf import apple


@jax.tree_util.register_pytree_node_class
@attrs.define
class Fixed(apple.PhysicsProblem):
    fixed_mask: Bool[np.ndarray, " D"]
    fixed_values: Float[jax.Array, " D"]
    problem: apple.PhysicsProblem

    @property
    def free_mask(self) -> Bool[np.ndarray, " D"]:
        return ~self.fixed_mask

    @property
    def n_dof(self) -> int:
        return self.problem.n_dof - self.n_fixed

    @property
    def n_fixed(self) -> int:
        return np.count_nonzero(self.fixed_mask)

    def fill(self, u: Float[jax.Array, " DoF"]) -> Float[jax.Array, " D"]:
        u_new: Float[jax.Array, " D"] = self.fixed_values.copy()
        u_new: Float[jax.Array, " D"] = u_new.at[self.free_mask].set(u)
        return u_new

    def tree_flatten(self) -> tuple[Iterable[Any], Hashable]:
        return (self.fixed_values, self.problem), (self.fixed_mask,)

    @classmethod
    def tree_unflatten(cls, aux_data: Hashable, children: Iterable[Any]) -> Self:
        (fixed_values, problem) = children
        (fixed_mask,) = aux_data  # pyright: ignore[reportGeneralTypeIssues]
        return cls(fixed_mask=fixed_mask, fixed_values=fixed_values, problem=problem)

    def _fun(
        self, u: Float[jax.Array, " DoF"], p: Mapping[str, Float[jax.Array, "..."]]
    ) -> Float[jax.Array, ""]:
        u: Float[jax.Array, " D"] = self.fill(u)
        return self.problem.fun(u, p)
