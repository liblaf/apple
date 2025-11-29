import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Integer
from liblaf.peach import tree


@tree.define
class Dirichlet:
    dim: int
    dirichlet_index: Integer[Array, " dirichlet"]
    dirichlet_value: Float[Array, " dirichlet"]
    free_index: Integer[Array, " free"]
    n_points: int

    @property
    def n_dirichlet(self) -> int:
        return self.dirichlet_index.size

    @property
    def n_free(self) -> int:
        return self.free_index.size

    @property
    def n_full(self) -> int:
        return self.n_points * self.dim

    @eqx.filter_jit
    def get_dirichlet(
        self, full: Float[Array, "points dim"]
    ) -> Float[Array, " dirichlet"]:
        return full.flatten()[self.dirichlet_index]

    @eqx.filter_jit
    def get_free(self, full: Float[Array, "points dim"]) -> Float[Array, " free"]:
        return full.flatten()[self.free_index]

    @eqx.filter_jit
    def set_dirichlet(
        self,
        full: Float[Array, "points dim"],
        values: Float[ArrayLike, " dirichlet"] | None = None,
    ) -> Float[Array, "points dim"]:
        if values is None:
            values = self.dirichlet_value
        return full.flatten().at[self.dirichlet_index].set(values).reshape(full.shape)

    @eqx.filter_jit
    def set_free(
        self, full: Float[Array, "points dim"], values: Float[ArrayLike, " free"]
    ) -> Float[Array, "points dim"]:
        return full.flatten().at[self.free_index].set(values).reshape(full.shape)

    @eqx.filter_jit
    def to_full(
        self,
        free: Float[Array, " free"],
        dirichlet: Float[ArrayLike, " dirichlet"] | None = None,
    ) -> Float[Array, "points dim"]:
        full: Float[Array, "points dim"] = jnp.empty(
            (self.n_points, self.dim), free.dtype
        )
        full = self.set_free(full, free)
        full = self.set_dirichlet(full, dirichlet)
        return full
