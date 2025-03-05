import jax
import jax.numpy as jnp
import pytest
import pyvista as pv
from jaxtyping import Float

from liblaf import apple


def first_fundamental_form_naive(
    points: Float[jax.Array, "*C 3 3"],
) -> Float[jax.Array, "*C 2 2"]:
    points = points.reshape(-1, 3, 3)
    I: Float[jax.Array, "C 2 2"] = jax.vmap(first_fundamental_form_naive_single)(points)  # noqa: E741
    return I


def first_fundamental_form_naive_single(
    points: Float[jax.Array, "3 3"],
) -> Float[jax.Array, "2 2"]:
    vi: Float[jax.Array, " 3"]
    vj: Float[jax.Array, " 3"]
    vk: Float[jax.Array, " 3"]
    vi, vj, vk = points
    return jnp.asarray(
        [
            [jnp.sum((vj - vi) ** 2), jnp.dot(vj - vi, vk - vi)],
            [jnp.dot(vk - vi, vj - vi), jnp.sum((vk - vi) ** 2)],
        ]
    )


def test_first_fundamental_form(mesh: pv.PolyData) -> None:
    points: Float[jax.Array, "P 3"] = jnp.asarray(mesh.points)
    points: Float[jax.Array, "C 3 3"] = points[mesh.regular_faces]
    actual: Float[jax.Array, "C 2 2"] = apple.elem.triangle.first_fundamental_form(
        points
    )
    expected: Float[jax.Array, "C 2 2"] = first_fundamental_form_naive(points)
    assert actual == pytest.approx(expected)


def test_first_fundamental_form_single(mesh: pv.PolyData) -> None:
    points: Float[jax.Array, "P 3"] = jnp.asarray(mesh.points)
    points: Float[jax.Array, "3 3"] = points[mesh.regular_faces[0]]
    actual: Float[jax.Array, "2 2"] = apple.elem.triangle.first_fundamental_form(points)
    expected: Float[jax.Array, "2 2"] = first_fundamental_form_naive_single(points)
    assert actual == pytest.approx(expected)
