from typing import override

import attrs
import felupe
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Integer, PyTree

from liblaf import apple


@apple.register_dataclass()
@attrs.define(kw_only=True, on_setattr=attrs.setters.convert)
class Corotated(apple.AbstractPhysicsProblem):
    name: str = attrs.field(default="corotated", metadata={"static": True})
    mesh: felupe.Mesh = attrs.field(metadata={"static": True})
    # Auxiliaries
    dh_dX: Float[jax.Array, " C 4 3"] = attrs.field(converter=jnp.asarray)
    dV: Float[jax.Array, " C"] = attrs.field(converter=jnp.asarray)
    # Parameters
    lambda_: Float[jax.Array, "..."] = attrs.field(
        converter=jnp.asarray, factory=lambda: jnp.asarray(3.0)
    )
    """Lamé's first parameter."""
    mu: Float[jax.Array, "..."] = attrs.field(
        converter=jnp.asarray, factory=lambda: jnp.asarray(1.0)
    )
    """Lamé's second parameter."""

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

    @apple.jit()
    @override
    def fun(self, u: PyTree, q: PyTree | None = None) -> Float[jax.Array, ""]:
        u: Float[jax.Array, "C 4 3"] = u[self.cells]
        lambda_: Float[jax.Array, " ..."] = self.get_param("lambda_", q)
        mu: Float[jax.Array, " ..."] = self.get_param("mu", q)
        lambda_: Float[jax.Array, " C"] = jnp.broadcast_to(lambda_, (self.n_cells,))
        mu: Float[jax.Array, " C"] = jnp.broadcast_to(mu, (self.n_cells,))
        F: Float[jax.Array, "C 3 3"] = apple.elem.tetra.deformation_gradient(
            u, self.dh_dX
        )
        Psi: Float[jax.Array, " C"] = jax.vmap(corotated)(F, lambda_, mu)
        return jnp.sum(Psi * self.dV)

    @override
    def unravel_u(self, u_flat: Float[jax.Array, " DoF"]) -> Float[jax.Array, "P 3"]:
        return u_flat.reshape(self.n_points, 3)


@attrs.define(kw_only=True, on_setattr=attrs.setters.convert)
class CorotatedBuilder(apple.AbstractPhysicsProblemBuilder):
    name: str = attrs.field(default="corotated", metadata={"static": True})
    mesh: felupe.Mesh = attrs.field(metadata={"static": True})
    lambda_: Float[jax.Array, "..."] = attrs.field(
        converter=jnp.asarray, factory=lambda: jnp.asarray(3e3)
    )
    mu: Float[jax.Array, "..."] = attrs.field(
        converter=jnp.asarray, factory=lambda: jnp.asarray(1e3)
    )

    @override
    def build(self, q: PyTree | None = None) -> Corotated:
        super().build(q)
        cell_points: Float[jax.Array, "C 4 3"] = jnp.asarray(
            self.mesh.points[self.mesh.cells]
        )
        return Corotated(
            name=self.name,
            mesh=self.mesh,
            dh_dX=apple.elem.tetra.dh_dX(cell_points),
            dV=apple.elem.tetra.dV(cell_points),
            lambda_=self.get_param("lambda_", q),
            mu=self.get_param("mu", q),
            _q_unravel=self._q_unravel,
        )


def corotated(
    F: Float[jax.Array, "3 3"], lambda_: Float[jax.Array, ""], mu: Float[jax.Array, ""]
) -> Float[jax.Array, ""]:
    R: Float[jax.Array, "3 3"]
    R, _S = apple.polar_rv(F)
    R = jax.lax.stop_gradient(R)  # TODO: support gradient of `polar_rv()`
    Psi: Float[jax.Array, ""] = (
        mu * jnp.sum((F - R) ** 2) + lambda_ * (jnp.linalg.det(F) - 1) ** 2
    )
    return Psi
