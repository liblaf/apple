import attrs
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Integer, Scalar
from numpy.typing import ArrayLike

from liblaf import apple


@attrs.frozen
class SaintVenantKirchhoffPrepared(apple.ProblemPrepared):
    region: apple.Region
    lambda_: Float[jax.Array, "c q"] = attrs.field(converter=jnp.asarray)
    mu: Float[jax.Array, "c q"] = attrs.field(converter=jnp.asarray)

    @property
    def n_points(self) -> int:
        return self.region.n_points

    @property
    def n_cells(self) -> int:
        return self.region.n_cells

    @property
    def n_dof(self) -> int:
        return 3 * self.n_points

    @property
    def cells(self) -> Integer[np.ndarray, "c 3"]:
        return self.region.cells

    def fun(self, u: Float[ArrayLike, " DoF"]) -> Scalar:
        u: Float[jax.Array, " DoF"] = jnp.asarray(u)
        u: Float[jax.Array, "P 3"] = u.reshape(self.n_points, 3)
        u: Float[jax.Array, "c a=4 3"] = u[self.cells]
        F: Float[jax.Array, "c q I=3 J=3"] = jnp.einsum(
            "aJqc,caI->cqIJ", self.region.dhdX, u
        )
        Psi: Float[jax.Array, "c q"] = jax.vmap(jax.vmap(saint_venant_kirchhoff))(
            F, self.lambda_, self.mu
        )
        return jnp.sum(Psi * self.region.dV)


@attrs.define
class SaintVenantKirchhoff(apple.Problem):
    region: apple.Region
    lambda_: Float[jax.Array, "c q"] = attrs.field(converter=jnp.asarray)
    mu: Float[jax.Array, "c q"] = attrs.field(converter=jnp.asarray)

    def __attrs_post_init__(self) -> None:
        self.lambda_ = jnp.broadcast_to(
            self.lambda_, (self.region.n_cells, self.region.h.shape[1])
        )
        self.mu = jnp.broadcast_to(
            self.mu, (self.region.n_cells, self.region.h.shape[1])
        )

    @property
    def n_dof(self) -> int:
        return 3 * self.region.n_points

    def prepare(self) -> SaintVenantKirchhoffPrepared:
        return SaintVenantKirchhoffPrepared(
            region=self.region, lambda_=self.lambda_, mu=self.mu
        )


def saint_venant_kirchhoff(
    F: Float[jax.Array, "3 3"], lambda_: Scalar, mu: Scalar
) -> Scalar:
    E: Float[jax.Array, "3 3"] = 0.5 * (F.T + F + F.T @ F)
    Psi: Scalar = 0.5 * lambda_ * jnp.trace(E) ** 2 + mu * jnp.trace(E.T @ E)
    return Psi
