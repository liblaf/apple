import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from liblaf import jarp

type Free = Float[Array, " free"]
type Full = Float[Array, "points dim"]


@jarp.define
class DofMap:
    dim: int = jarp.static()
    n_points: int = jarp.static()
    fixed_indices: Integer[Array, " fixed"] = jarp.array()
    fixed_values: Float[Array, " fixed"] = jarp.array()
    free_indices: Integer[Array, " free"] = jarp.array()

    @property
    def n_fixed(self) -> int:
        return self.fixed_indices.size

    @property
    def n_free(self) -> int:
        return self.free_indices.size

    @property
    def n_full(self) -> int:
        return self.n_points * self.dim

    def to_free(self, full: Full) -> Free:
        return full.flatten()[self.free_indices]

    def to_free_grad(self, full: Full) -> Free:
        return full.flatten()[self.free_indices]

    def to_free_hess_diag(self, full: Full) -> Free:
        return full.flatten()[self.free_indices]

    def to_full(self, free: Free) -> Full:
        result: Full = jnp.empty((self.n_full,))
        result: Full = result.at[self.fixed_indices].set(self.fixed_values)
        result: Full = result.at[self.free_indices].set(free)
        return result.reshape((self.n_points, self.dim))

    def to_full_grad(self, grad_free: Free) -> Full:
        result: Full = jnp.empty((self.n_full,))
        result: Full = result.at[self.fixed_indices].set(0.0)
        result: Full = result.at[self.free_indices].set(grad_free)
        return result.reshape((self.n_points, self.dim))
