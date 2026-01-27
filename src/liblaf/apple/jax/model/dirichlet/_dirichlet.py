import jarp
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, Integer


@jarp.define
class Dirichlet:
    dim: int = jarp.static()
    dirichlet_index: Integer[Array, " dirichlet"]
    dirichlet_value: Float[Array, " dirichlet"]
    fixed_mask: Bool[Array, "points dim"]
    free_index: Integer[Array, " free"]
    n_points: int = jarp.static()

    @property
    def n_dirichlet(self) -> int:
        return self.dirichlet_index.size

    @property
    def n_free(self) -> int:
        return self.free_index.size

    @property
    def n_full(self) -> int:
        return self.n_points * self.dim

    @jarp.jit(inline=True)
    def get_fixed(self, full: Float[Array, "points dim"]) -> Float[Array, " dirichlet"]:
        return full.flatten()[self.dirichlet_index]

    @jarp.jit(inline=True)
    def get_free(self, full: Float[Array, "points dim"]) -> Float[Array, " free"]:
        return full.flatten()[self.free_index]

    @jarp.jit(inline=True)
    def set_fixed(
        self,
        full: Float[Array, "points dim"],
        values: Float[ArrayLike, " dirichlet"] | None = None,
    ) -> Float[Array, "points dim"]:
        if values is None:
            values = self.dirichlet_value
        return full.flatten().at[self.dirichlet_index].set(values).reshape(full.shape)

    @jarp.jit(inline=True)
    def set_free(
        self, full: Float[Array, "points dim"], values: Float[ArrayLike, " free"]
    ) -> Float[Array, "points dim"]:
        return full.flatten().at[self.free_index].set(values).reshape(full.shape)

    @jarp.jit(inline=True)
    def to_full(
        self,
        free: Float[Array, " free"],
        dirichlet: Float[ArrayLike, " dirichlet"] | None = None,
    ) -> Float[Array, "points dim"]:
        full: Float[Array, "points dim"] = jnp.empty(
            (self.n_points, self.dim), free.dtype
        )
        full = self.set_free(full, free)
        full = self.set_fixed(full, dirichlet)
        return full
