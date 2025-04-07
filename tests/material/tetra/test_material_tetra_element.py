from typing import override

import beartype
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import pytest
from jaxtyping import Float, PRNGKeyArray, PyTree

from liblaf import apple


class MaterialTetraElementNaive(apple.material.tetra.MaterialTetraElement):
    @jaxtyping.jaxtyped(typechecker=beartype.beartype)
    @override
    def strain_energy_density(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        return apple.math.norm_sqr(F)


@pytest.mark.parametrize(
    "material",
    [MaterialTetraElementNaive(), apple.material.tetra.AsRigidAsPossibleElement()],
)
class TestMaterialTetraElement:
    aux: PyTree
    p: Float[jax.Array, "4 3"]
    points: Float[jax.Array, "4 3"]
    q: PyTree
    u: Float[jax.Array, "4 3"]

    @pytest.fixture(scope="class")
    def aux(self, points: Float[jax.Array, "4 3"]) -> PyTree:
        return {
            "dh_dX": apple.elem.tetra.dh_dX(points),
            "dV": apple.elem.tetra.dV(points),
        }

    @pytest.fixture(scope="class")
    def points(self) -> Float[jax.Array, "4 3"]:
        key: PRNGKeyArray = jax.random.key(1)
        return jax.random.uniform(key, (4, 3))

    @pytest.fixture(scope="class")
    def p(self) -> Float[jax.Array, "4 3"]:
        key: PRNGKeyArray = jax.random.key(2)
        return jax.random.uniform(key, (4, 3))

    @pytest.fixture(scope="class")
    def q(self) -> PyTree:
        q: PyTree = {}
        key: PRNGKeyArray = jax.random.key(3)
        q["lambda"] = jax.random.uniform(key, ())
        key = jax.random.key(4)
        q["mu"] = jax.random.uniform(key, ())
        return q

    @pytest.fixture(scope="class")
    def u(self) -> Float[jax.Array, "4 3"]:
        key: PRNGKeyArray = jax.random.key(5)
        return jax.random.uniform(key, (4, 3))

    def test_jac(
        self,
        material: apple.material.tetra.MaterialTetraElement,
        u: Float[jax.Array, "4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> None:
        actual: Float[jax.Array, "4 3"] = material.jac(u, q, aux)
        expected: Float[jax.Array, "4 3"] = jax.grad(material.fun)(u, q, aux)
        np.testing.assert_allclose(actual, expected)

    def test_hess(
        self,
        material: apple.material.tetra.MaterialTetraElement,
        u: Float[jax.Array, "4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> None:
        actual: Float[jax.Array, "4 3 4 3"] = material.hess(u, q, aux)
        expected: Float[jax.Array, "4 3 4 3"] = jax.hessian(material.fun)(u, q, aux)
        np.testing.assert_allclose(actual, expected)

    def test_hessp(
        self,
        material: apple.material.tetra.MaterialTetraElement,
        u: Float[jax.Array, "4 3"],
        p: Float[jax.Array, "4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> None:
        actual: Float[jax.Array, "4 3"] = material.hessp(u, p, q, aux)
        hess: Float[jax.Array, "4 3 4 3"] = jax.hessian(material.fun)(u, q, aux)
        hess: Float[jax.Array, "12 12"] = hess.reshape(12, 12)
        p: Float[jax.Array, " 12"] = p.reshape(12)
        expected: Float[jax.Array, " 12"] = hess @ p
        expected: Float[jax.Array, "4 3"] = expected.reshape(4, 3)
        np.testing.assert_allclose(actual, expected)

    def test_hess_diag(
        self,
        material: apple.material.tetra.MaterialTetraElement,
        u: Float[jax.Array, "4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> None:
        actual: Float[jax.Array, "4 3"] = material.hess_diag(u, q, aux)
        hess: Float[jax.Array, "4 3 4 3"] = jax.hessian(material.fun)(u, q, aux)
        hess: Float[jax.Array, "12 12"] = hess.reshape(12, 12)
        expected: Float[jax.Array, " 12"] = jnp.diagonal(hess)
        expected: Float[jax.Array, "4 3"] = expected.reshape(4, 3)
        np.testing.assert_allclose(actual, expected)

    def test_hess_quad(
        self,
        material: apple.material.tetra.MaterialTetraElement,
        u: Float[jax.Array, "4 3"],
        p: Float[jax.Array, "4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> None:
        actual: Float[jax.Array, ""] = material.hess_quad(u, p, q, aux)
        hess: Float[jax.Array, "4 3 4 3"] = jax.hessian(material.fun)(u, q, aux)
        hess: Float[jax.Array, "12 12"] = hess.reshape(12, 12)
        p: Float[jax.Array, " 12"] = p.reshape(12)
        expected: Float[jax.Array, ""] = jnp.vdot(p, hess @ p)
        np.testing.assert_allclose(actual, expected)
