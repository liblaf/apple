import felupe
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Float

from liblaf import apple


def deformation_gradient_reference(
    x: Float[jax.Array, "c a=4 I=3"], x0: Float[jax.Array, "c a=4 J=3"]
) -> Float[jax.Array, "c I=3 J=3"]:
    def _deformation_gradient(
        x: Float[jax.Array, "a=4 I=3"], x0: Float[jax.Array, "a=4 J=3"]
    ) -> Float[jax.Array, "I=3 J=3"]:
        Dm: Float[jax.Array, "I=3 J=3"] = (x0[1:, :] - x0[0:1, :]).T
        Ds: Float[jax.Array, "I=3 J=3"] = (x[1:, :] - x[0:1, :]).T
        return Ds @ jnp.linalg.pinv(Dm)

    F: Float[jax.Array, "c 3 3"] = jax.vmap(_deformation_gradient)(x, x0)
    return F


def test_deformation_gradient(
    region: felupe.RegionTetra, displacement: Float[jax.Array, "c I=3"]
) -> None:
    mesh: felupe.Mesh = region.mesh  # pyright: ignore[reportAttributeAccessIssue]
    points: Float[jax.Array, "*c a=4 I=3"] = jnp.asarray(mesh.points[mesh.cells])
    actual: Float[jax.Array, "*c I=3 J=3"] = apple.elem.tetra.deformation_gradient(
        displacement[mesh.cells], apple.elem.tetra.dh_dX(points)
    )
    expected: Float[jax.Array, "*c I=3 J=3"] = deformation_gradient_reference(
        points + displacement[mesh.cells], points
    )
    assert actual == pytest.approx(expected, abs=1e-4)
