import attrs
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Integer, Scalar

from liblaf import apple


@attrs.frozen
class GravityPrepared(apple.ProblemPrepared):
    region: apple.Region
    density: Float[jax.Array, "c q"] = attrs.field(converter=jnp.asarray)
    gravity: Float[jax.Array, "3"] = attrs.field(converter=jnp.asarray)

    @property
    def n_dof(self) -> int:
        return 3 * self.region.n_points

    @property
    def n_points(self) -> int:
        return self.region.n_points

    @property
    def n_cells(self) -> int:
        return self.region.n_cells

    @property
    def cells(self) -> Integer[np.ndarray, "c a"]:
        return self.region.cells

    @property
    def dV(self) -> Float[jax.Array, "c q"]:
        return self.region.dV.sum(axis=1)

    def fun(self, u: Float[jax.Array, " DoF"]) -> Scalar:
        u: Float[jax.Array, "P I=3"] = u.reshape(self.n_points, 3)
        u: Float[jax.Array, "c a I=3"] = u[self.cells]
        u: Float[jax.Array, "c q I=3"] = jnp.einsum("aq,caI->cqI", self.region.h, u)
        return -jnp.sum(self.density * self.dV * u * self.gravity)


@attrs.define
class Gravity(apple.Problem):
    region: apple.Region
    density: Float[jax.Array, "c q"] = attrs.field(converter=jnp.asarray)
    gravity: Float[jax.Array, "3"] = attrs.field(converter=jnp.asarray)

    @property
    def n_dof(self) -> int:
        return 3 * self.region.n_points

    def prepare(self) -> apple.ProblemPrepared:
        return GravityPrepared(
            region=self.region, density=self.density, gravity=self.gravity
        )
