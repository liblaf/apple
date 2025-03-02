from collections.abc import Mapping

import attrs
import felupe
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Integer, PyTree

from liblaf import apple


@apple.register_dataclass()
@attrs.define(kw_only=True)
class Corotated(apple.AbstractPhysicsProblem):
    mesh: felupe.Mesh = attrs.field(metadata={"static": True})
    params: Mapping[str, Float[jax.Array, "..."]] = attrs.field(
        factory=lambda: {"lambda": jnp.asarray(3.0), "mu": jnp.asarray(1.0)}
    )

    @property
    def n_cells(self) -> int:
        return self.mesh.ncells

    @property
    def n_points(self) -> int:
        return self.mesh.npoints

    @property
    def n_dof(self) -> int:
        return self.n_points * 3

    @property
    def cell_points(self) -> Float[jax.Array, "C 4 3"]:
        return self.points[self.cells]

    @property
    def cells(self) -> Integer[np.ndarray, "C 4"]:
        return self.mesh.cells

    @property
    def points(self) -> Float[jax.Array, "P 3"]:
        return jnp.asarray(self.mesh.points)

    @property
    def dV(self) -> Float[jax.Array, " C"]:
        return self.aux["dV"]

    @property
    def dh_dX(self) -> Float[jax.Array, " C 4 3"]:
        return self.aux["dh_dX"]

    @apple.jit()
    def fun(self, u: PyTree, q: PyTree | None = None) -> Float[jax.Array, ""]:
        params: Mapping = apple.merge(self.params, q)
        u: Float[jax.Array, "P 3"] = jnp.asarray(u).reshape(self.n_points, 3)
        u: Float[jax.Array, "C 4 3"] = u[self.cells]
        lambda_: Float[jax.Array, " C"] = jnp.broadcast_to(
            params["lambda"], (self.n_cells,)
        )
        mu: Float[jax.Array, " C"] = jnp.broadcast_to(params["mu"], (self.n_cells,))
        F: Float[jax.Array, "C 3 3"] = apple.elem.tetra.deformation_gradient(
            u, self.dh_dX
        )
        Psi: Float[jax.Array, " C"] = jax.vmap(corotational)(F, lambda_, mu)
        return jnp.sum(Psi * self.dV)

    def prepare(self, params: PyTree | None = None) -> None:
        super().prepare(params)
        self.aux["dh_dX"] = apple.elem.tetra.dh_dX(self.cell_points)
        self.aux["dV"] = apple.elem.tetra.dV(self.cell_points)

    def unravel_u(self, u: Float[jax.Array, " DoF"]) -> Float[jax.Array, "P 3"]:
        return u.reshape(self.n_points, 3)


def corotational(
    F: Float[jax.Array, "3 3"], lambda_: Float[jax.Array, ""], mu: Float[jax.Array, ""]
) -> Float[jax.Array, ""]:
    R: Float[jax.Array, "3 3"]
    R, _S = apple.polar_rv(F)
    R = jax.lax.stop_gradient(R)
    Psi: Float[jax.Array, ""] = (
        mu * jnp.sum((F - R) ** 2) + lambda_ * (jnp.linalg.det(F) - 1) ** 2
    )
    return Psi
