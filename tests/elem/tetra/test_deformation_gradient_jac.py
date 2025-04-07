import felupe
import jax
import numpy as np
from jaxtyping import Float, PRNGKeyArray

from liblaf import apple


def test_deformation_gradient_jvp(mesh_felupe: felupe.Mesh) -> None:
    n_cells: int = mesh_felupe.ncells
    key: PRNGKeyArray = jax.random.key(0)
    subkey: PRNGKeyArray
    key, subkey = jax.random.split(key)
    u: Float[jax.Array, "C 4 3"] = jax.random.uniform(subkey, (n_cells, 4, 3))
    key, subkey = jax.random.split(key)
    p: Float[jax.Array, "C 4 3"] = jax.random.uniform(subkey, (n_cells, 4, 3))
    points: Float[jax.Array, "C 4 3"] = jax.numpy.asarray(
        mesh_felupe.points[mesh_felupe.cells]
    )
    dh_dX: Float[jax.Array, "C 4 3"] = apple.elem.tetra.dh_dX(points)
    actual: Float[jax.Array, "C 3 3"] = apple.elem.tetra.deformation_gradient_jvp(
        dh_dX, p
    )
    expected: Float[jax.Array, "C 3 3"] = apple.math.jvp(
        apple.elem.tetra.deformation_gradient, u, p, args=(dh_dX,)
    )
    np.testing.assert_allclose(actual, expected)


def test_deformation_gradient_vjp(mesh_felupe: felupe.Mesh) -> None:
    n_cells: int = mesh_felupe.ncells
    key: PRNGKeyArray = jax.random.key(0)
    subkey: PRNGKeyArray
    key, subkey = jax.random.split(key)
    u: Float[jax.Array, "C 4 3"] = jax.random.uniform(subkey, (n_cells, 4, 3))
    key, subkey = jax.random.split(key)
    p: Float[jax.Array, "C 4 3"] = jax.random.uniform(subkey, (n_cells, 4, 3))
    points: Float[jax.Array, "C 4 3"] = jax.numpy.asarray(
        mesh_felupe.points[mesh_felupe.cells]
    )
    dh_dX: Float[jax.Array, "C 4 3"] = apple.elem.tetra.dh_dX(points)
    actual: Float[jax.Array, "C 3 3"] = apple.elem.tetra.deformation_gradient_vjp(
        dh_dX, p
    )
    expected: Float[jax.Array, "C 3 3"] = apple.math.vjp(
        apple.elem.tetra.deformation_gradient, u, p, args=(dh_dX,)
    )
    np.testing.assert_allclose(actual, expected)
