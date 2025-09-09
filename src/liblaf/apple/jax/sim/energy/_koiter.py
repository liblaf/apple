from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.apple.jax import math, tree
from liblaf.apple.jax.sim.energy._energy import Energy
from liblaf.apple.jax.sim.geometry import Geometry
from liblaf.apple.jax.typing import Scalar, Vector


@tree.pytree
class Koiter(Energy):
    alpha: Float[Array, " c"] = tree.array()
    """Lamé's first parameter."""

    beta: Float[Array, " c"] = tree.array()
    """Lamé's second parameter."""

    det_Iu: Float[Array, " c"] = tree.array()
    """det(Iu)."""

    h: Float[Array, " c"] = tree.array()
    """Thickness."""

    Iu_inv: Float[Array, "c 2 2"] = tree.array()
    """Inverse of the midsurface first fundamental form."""

    pre_strain: Float[Array, " c"] = tree.array()

    geometry: Geometry = tree.field()

    @classmethod
    def from_geometry(cls, geometry: Geometry) -> Self:
        alpha: Float[Array, " c"] = math.asarray(
            geometry.cell_data["alpha"], dtype=float
        )
        beta: Float[Array, " c"] = math.asarray(geometry.cell_data["beta"], dtype=float)
        h: Float[Array, " c"] = math.asarray(geometry.cell_data["h"], dtype=float)
        Iu: Float[Array, "c 2 2"] = _first_fundamental_form(
            geometry.points[geometry.cells_global]
        )
        pre_strain: Float[Array, " c"] = math.asarray(
            geometry.cell_data["pre-strain"], dtype=float
        )
        self: Self = cls(
            alpha=alpha,
            beta=beta,
            det_Iu=jnp.linalg.det(Iu),
            Iu_inv=jnp.linalg.inv(Iu),
            h=h,
            pre_strain=pre_strain,
            geometry=geometry,
        )
        return self

    def fun(self, u: Vector) -> Scalar:
        I: Float[Array, "c 2 2"] = _first_fundamental_form(  # noqa: E741
            u[self.geometry.cells_global]
            + self.geometry.points[self.geometry.cells_global]
        )
        M: Float[Array, "c 2 2"] = (
            jnp.matmul(self.Iu_inv, I)
            - self.pre_strain[:, jnp.newaxis, jnp.newaxis]
            * jnp.eye(2)[jnp.newaxis, ...]
        )
        Ws: Float[Array, ""] = self._norm_SV(M)
        E: Float[Array, " c"] = 0.5 * (0.25 * self.h * Ws) * jnp.sqrt(self.det_Iu)
        return jnp.sum(E)

    def _norm_SV(self, M: Float[Array, "c 2 2"]) -> Float[Array, " c"]:
        return 0.5 * self.alpha * jnp.trace(
            M, axis1=-2, axis2=-1
        ) ** 2 + self.beta * jnp.trace(jnp.matmul(M, M), axis1=-2, axis2=-1)


def _first_fundamental_form(points: Float[Array, "c 3 3"]) -> Float[Array, "c 2 2"]:
    return jax.vmap(_first_fundamental_form_single)(points)


def _first_fundamental_form_single(points: Float[Array, "3 3"]) -> Float[Array, "2 2"]:
    vi: Float[Array, " 3"] = points[0]
    vj: Float[Array, " 3"] = points[1]
    vk: Float[Array, " 3"] = points[2]
    I00: Float[Array, ""] = jnp.sum((vj - vi) ** 2)
    I11: Float[Array, ""] = jnp.sum((vk - vi) ** 2)
    I01: Float[Array, ""] = jnp.dot(vj - vi, vk - vi)
    return jnp.asarray([[I00, I01], [I01, I11]])
