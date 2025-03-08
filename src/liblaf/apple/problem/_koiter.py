from typing import override

import attrs
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Float, PyTree

from liblaf import apple


@apple.register_dataclass()
@attrs.define(kw_only=True, on_setattr=attrs.setters.convert)
class Koiter(apple.AbstractPhysicsProblem):
    name: str = attrs.field(default="koiter", metadata={"static": True})
    mesh: pv.PolyData = attrs.field(metadata={"static": True})
    # auxiliaries
    dA: Float[jax.Array, " C"] = attrs.field(default=None, converter=jnp.asarray)
    I_u_inv: Float[jax.Array, "C 2 2"] = attrs.field(
        default=None, converter=jnp.asarray
    )
    # material properties
    thickness: Float[jax.Array, " C"] = attrs.field(
        converter=jnp.asarray, factory=lambda: jnp.asarray(1e-3)
    )
    lmbda: Float[jax.Array, " C"] = attrs.field(
        converter=jnp.asarray, factory=lambda: jnp.asarray(3e3)
    )
    mu: Float[jax.Array, " C"] = attrs.field(
        converter=jnp.asarray, factory=lambda: jnp.asarray(1e3)
    )
    pre_strain: Float[jax.Array, "C 2 2"] = attrs.field(
        converter=jnp.asarray, factory=lambda: jnp.asarray(1.0)
    )

    @property
    def cell_points(self) -> Float[jax.Array, "C 3 3"]:
        return self.points[self.cells]

    @property
    def cells(self) -> Float[np.ndarray, "C 3"]:
        return self.mesh.regular_faces

    @property
    def n_cells(self) -> int:
        return self.mesh.n_faces_strict

    @property
    def n_dof(self) -> int:
        return 3 * self.n_points

    @property
    def n_points(self) -> int:
        return self.mesh.n_points

    @property
    def points(self) -> Float[jax.Array, "P 3"]:
        return jnp.asarray(self.mesh.points)

    @override
    def fun(self, u: PyTree, q: PyTree | None = None) -> Float[jax.Array, ""]:
        v: Float[jax.Array, "P 3"] = self.points + u
        v: Float[jax.Array, "C 3 3"] = v[self.cells]
        I: Float[jax.Array, "C 3 3"] = apple.elem.triangle.first_fundamental_form(v)  # noqa: E741
        thickness: Float[jax.Array, " ..."] = self.get_param("thickness", q)
        thickness: Float[jax.Array, " C"] = jnp.broadcast_to(thickness, (self.n_cells,))
        lmbda: Float[jax.Array, " ..."] = self.get_param("lmbda", q)
        lmbda: Float[jax.Array, " C"] = jnp.broadcast_to(lmbda, (self.n_cells,))
        mu: Float[jax.Array, " ..."] = self.get_param("mu", q)
        mu: Float[jax.Array, " C"] = jnp.broadcast_to(mu, (self.n_cells,))
        pre_strain: Float[jax.Array, " ..."] = self.get_param("pre_strain", q)
        pre_strain: Float[jax.Array, " C"] = jnp.broadcast_to(
            pre_strain, (self.n_cells,)
        )
        Psi: Float[jax.Array, " C"] = jax.vmap(koiter)(
            pre_strain[:, None, None] * I, self.I_u_inv, thickness, lmbda, mu
        )
        return jnp.sum(Psi * self.dA)

    @override
    def prepare(self, q: PyTree | None = None) -> None:
        super().prepare(q)
        I_u: Float[jax.Array, "C 2 2"] = apple.elem.triangle.first_fundamental_form(
            self.cell_points
        )
        self.dA = apple.elem.triangle.area(I_u)
        self.I_u_inv = jnp.linalg.pinv(I_u)

    @override
    def unravel_u(self, u_flat: Float[jax.Array, " DoF"]) -> Float[jax.Array, "P 2"]:
        return u_flat.reshape(self.n_points, 3)


def koiter(
    I: Float[jax.Array, "2 2"],  # noqa: E741
    I_u_inv: Float[jax.Array, "2 2"],
    thickness: Float[jax.Array, ""],
    lmbda: Float[jax.Array, ""],
    mu: Float[jax.Array, ""],
) -> Float[jax.Array, ""]:
    M: Float[jax.Array, "2 2"] = I_u_inv @ I - jnp.eye(2)
    Ws: Float[jax.Array, ""] = 0.5 * lmbda * jnp.trace(M) ** 2 + mu * jnp.trace(M @ M)
    return 0.25 * thickness * Ws
