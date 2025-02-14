import einops
import felupe
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Float

from liblaf import apple


def test_dX_dr(region: felupe.RegionTetra) -> None:
    mesh: felupe.Mesh = region.mesh  # pyright: ignore[reportAttributeAccessIssue]
    points: Float[jax.Array, "c a=4 I=3"] = jnp.asarray(mesh.points[mesh.cells])
    actual: Float[jax.Array, "c I=3 J=3"] = apple.elem.tetra.dX_dr(points)
    expected: Float[np.ndarray, "c I=3 J=3"] = einops.rearrange(
        region.dXdr,  # pyright: ignore[reportAttributeAccessIssue]
        "I J q c -> (c q) I J",
    )
    assert actual == pytest.approx(expected)


def test_dr_dX(region: felupe.RegionTetra) -> None:
    mesh: felupe.Mesh = region.mesh  # pyright: ignore[reportAttributeAccessIssue]
    points: Float[jax.Array, "c a=4 I=3"] = jnp.asarray(mesh.points[mesh.cells])
    actual: Float[jax.Array, "c J=3 I=3"] = apple.elem.tetra.dr_dX(points)
    expected: Float[np.ndarray, "c J=3 I=3"] = einops.rearrange(
        region.drdX,  # pyright: ignore[reportAttributeAccessIssue]
        "J I q c -> (c q) J I",
    )
    assert actual == pytest.approx(expected)


def test_dV(region: felupe.RegionTetra) -> None:
    mesh: felupe.Mesh = region.mesh  # pyright: ignore[reportAttributeAccessIssue]
    points: Float[jax.Array, "c a=4 I=3"] = jnp.asarray(mesh.points[mesh.cells])
    actual: Float[jax.Array, " c"] = apple.elem.tetra.dV(points)
    expected: Float[np.ndarray, " c"] = einops.rearrange(region.dV, "q c -> (c q)")  # pyright: ignore[reportAttributeAccessIssue]
    assert actual == pytest.approx(expected)


def test_dh_dX(region: felupe.RegionTetra) -> None:
    mesh: felupe.Mesh = region.mesh  # pyright: ignore[reportAttributeAccessIssue]
    points: Float[jax.Array, "c a=4 I=3"] = jnp.asarray(mesh.points[mesh.cells])
    actual: Float[jax.Array, "c a=4 J=3"] = apple.elem.tetra.dh_dX(points)
    expected: Float[np.ndarray, "c a=4 J=3"] = einops.rearrange(
        region.dhdX,  # pyright: ignore[reportAttributeAccessIssue]
        "a J q c -> (c q) a J",
    )
    assert actual == pytest.approx(expected)
