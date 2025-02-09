import attrs
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Bool, Float, Scalar

from liblaf import apple


@attrs.frozen
class FixedPrepared(apple.ProblemPrepared):
    problem: apple.ProblemPrepared
    fixed_mask: Bool[np.ndarray, " N"] = attrs.field(converter=np.asarray)
    fixed_values: Float[jax.Array, " N"] = attrs.field(converter=jnp.asarray)

    @property
    def n_dof(self) -> int:
        return self.problem.n_dof - self.n_fixed

    def fill(self, u: Float[jax.Array, " DoF"]) -> Float[jax.Array, " N"]:
        values: Float[jax.Array, " N"] = self.fixed_values.copy()
        values: Float[jax.Array, " N"] = values.at[~self.fixed_mask].set(u)
        return values

    @property
    def n_fixed(self) -> int:
        return np.count_nonzero(self.fixed_mask)

    def fun(self, u: Float[jax.Array, " DoF"]) -> Scalar:
        u: Float[jax.Array, " N"] = self.fill(u)
        return self.problem.fun(u)


@attrs.define
class Fixed(apple.Problem):
    problem: apple.Problem
    fixed_mask: Bool[np.ndarray, " N"] = attrs.field(converter=np.asarray)
    fixed_values: Float[jax.Array, " N"] = attrs.field(converter=jnp.asarray)

    @property
    def n_dof(self) -> int:
        return self.problem.n_dof - self.n_fixed

    @property
    def n_fixed(self) -> int:
        return np.count_nonzero(self.fixed_mask)

    def fill(self, u: Float[jax.Array, " DoF"]) -> Float[jax.Array, " N"]:
        values: Float[jax.Array, " N"] = self.fixed_values.copy()
        values: Float[jax.Array, " N"] = values.at[~self.fixed_mask].set(u)
        return values

    def prepare(self) -> apple.ProblemPrepared:
        return FixedPrepared(
            problem=self.problem.prepare(),
            fixed_mask=self.fixed_mask,
            fixed_values=self.fixed_values,
        )
