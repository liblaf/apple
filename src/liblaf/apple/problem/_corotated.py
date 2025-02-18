from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Self

import attrs
import felupe
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Integer

from liblaf import apple


@jax.tree_util.register_pytree_node_class
@attrs.define
class Corotated(apple.PhysicsProblem):
    mesh: felupe.Mesh
    p: Mapping[str, Float[jax.Array, "..."]] = attrs.field(
        factory=lambda: {"lambda": jnp.asarray(3.0), "mu": jnp.asarray(1.0)}
    )

    @property
    def cells(self) -> Integer[np.ndarray, "C 4"]:
        return self.mesh.cells

    @property
    def n_cells(self) -> int:
        return self.mesh.ncells

    @property
    def n_points(self) -> int:
        return self.mesh.npoints

    @property
    def n_dof(self) -> int:
        return self.mesh.ndof

    def tree_flatten(self) -> tuple[Iterable[Any], Hashable]:
        return (self.p,), (self.mesh,)

    @classmethod
    def tree_unflatten(cls, aux_data: Hashable, children: Iterable[Any]) -> Self:
        (p,) = children
        (mesh,) = aux_data
        return cls(mesh=mesh, p=p)

    def _fun(
        self, u: Float[jax.Array, " DoF"], p: Mapping[str, Float[jax.Array, "..."]]
    ) -> Float[jax.Array, ""]:
        u: Float[jax.Array, "P 3"] = u.reshape(self.n_points, 3)
        u: Float[jax.Array, "C 4 3"] = u[self.cells]
        lambda_: Float[jax.Array, " C"] = jnp.broadcast_to(p["lambda"], (self.n_cells,))
        mu: Float[jax.Array, " C"] = jnp.broadcast_to(p["mu"], (self.n_cells,))
        dV: Float[jax.Array, " C"] = apple.elem.tetra.dV(
            jnp.asarray(self.mesh.points)[self.cells]
        )
        dh_dX: Float[jax.Array, "C 4 3"] = apple.elem.tetra.dh_dX(
            jnp.asarray(self.mesh.points)[self.cells]
        )
        F: Float[jax.Array, "C 3 3"] = apple.elem.tetra.deformation_gradient(u, dh_dX)
        Psi: Float[jax.Array, " C"] = jax.vmap(corotational)(F, lambda_, mu)
        return jnp.sum(Psi * dV)


def corotational(
    F: Float[jax.Array, "3 3"], lambda_: Float[jax.Array, ""], mu: Float[jax.Array, ""]
) -> Float[jax.Array, ""]:
    R: Float[jax.Array, "3 3"]
    S: Float[jax.Array, "3 3"]
    R, S = apple.polar_rv(F)
    R = jax.lax.stop_gradient(R)
    Psi: Float[jax.Array, ""] = (
        mu * jnp.sum((F - R) ** 2) + lambda_ * (jnp.linalg.det(F) - 1) ** 2
    )
    return Psi
