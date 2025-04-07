import felupe
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from liblaf import apple


@apple.jit()
def compute_B(x0: Float[jax.Array, "C 4 3"]) -> Float[jax.Array, "C 3 3"]:
    def _compute_B(x0: Float[jax.Array, "4 3"]) -> Float[jax.Array, "3 3"]:
        Dm: Float[jax.Array, "3 3"] = (x0[1:, :] - x0[:1, :]).T
        return jnp.linalg.pinv(Dm)

    B: Float[jax.Array, "C 3 3"] = jax.vmap(_compute_B)(x0)
    return B


@apple.jit()
def deformation_gradient_naive(
    x: Float[jax.Array, "C 4 3"], x0: Float[jax.Array, "C 4 3"]
) -> Float[jax.Array, "C 3 3"]:
    def _deformation_gradient(
        x: Float[jax.Array, "4 3"], x0: Float[jax.Array, "4 3"]
    ) -> Float[jax.Array, "3 3"]:
        Dm: Float[jax.Array, "3 3"] = (x0[1:, :] - x0[:1, :]).T
        Ds: Float[jax.Array, "3 3"] = (x[1:, :] - x[:1, :]).T
        return Ds @ jnp.linalg.pinv(Dm)

    F: Float[jax.Array, "C 3 3"] = jax.vmap(_deformation_gradient)(x, x0)
    return F


def test_deformation_gradient(
    region: felupe.RegionTetra, displacement: Float[jax.Array, "C 3"]
) -> None:
    mesh: felupe.Mesh = region.mesh  # pyright: ignore[reportAttributeAccessIssue]
    points: Float[jax.Array, "C 4 3"] = jnp.asarray(mesh.points[mesh.cells])
    actual: Float[jax.Array, "C 3 3"] = apple.elem.tetra.deformation_gradient(
        displacement[mesh.cells], apple.elem.tetra.dh_dX(points)
    )
    expected: Float[jax.Array, "C 3 3"] = deformation_gradient_naive(
        points + displacement[mesh.cells], points
    )
    np.testing.assert_allclose(actual, expected, atol=6e-5)
