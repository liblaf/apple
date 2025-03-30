import beartype
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import pytest
from jaxtyping import Float, PRNGKeyArray, PyTree

from liblaf import apple


class MaterialElement(apple.material.tetra.MaterialTetraElement):
    @jaxtyping.jaxtyped(typechecker=beartype.beartype)
    def strain_energy_density(
        self,
        F: Float[jax.Array, "3 3"],
        q: PyTree,  # noqa: ARG002
        aux: PyTree,  # noqa: ARG002
    ) -> Float[jax.Array, ""]:
        return apple.math.norm_sqr(F)


@pytest.mark.parametrize(
    "material",
    [
        apple.material.tetra.MaterialTetra(elem=MaterialElement()),
        apple.material.tetra.Corotated(),
    ],
)
class TestMaterial:
    aux: PyTree
    n_cells: int = 7
    p: Float[jax.Array, "C 4 3"]
    points: Float[jax.Array, "C 4 3"]
    q: PyTree
    u: Float[jax.Array, "C 4 3"]

    @pytest.fixture(scope="class")
    def aux(self, points: Float[jax.Array, "C 4 3"]) -> PyTree:
        return {
            "dh_dX": apple.elem.tetra.dh_dX(points),
            "dV": apple.elem.tetra.dV(points),
        }

    @pytest.fixture(scope="class")
    def points(self, rng: PRNGKeyArray) -> Float[jax.Array, "C 4 3"]:
        return jax.random.uniform(rng, (self.n_cells, 4, 3))

    @pytest.fixture(scope="class")
    def p(self, rng: PRNGKeyArray) -> Float[jax.Array, "C 4 3"]:
        return jax.random.uniform(rng, (self.n_cells, 4, 3))

    @pytest.fixture(scope="class")
    def q(self, rng: PRNGKeyArray) -> PyTree:
        q: PyTree = {}
        subkey: PRNGKeyArray
        rng, subkey = jax.random.split(rng)
        q["lambda"] = jax.random.uniform(subkey, (self.n_cells,))
        rng, subkey = jax.random.split(rng)
        q["mu"] = jax.random.uniform(subkey, (self.n_cells,))
        return q

    @pytest.fixture(scope="class")
    def u(self, rng: PRNGKeyArray) -> Float[jax.Array, "C 4 3"]:
        return jax.random.uniform(rng, (self.n_cells, 4, 3))

    def hessian(
        self,
        material: apple.material.tetra.MaterialTetra,
        u: Float[jax.Array, "C 4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> Float[jax.Array, "C 4 3 4 3"]:
        hess: Float[jax.Array, "C 4 3 4 3"] = jax.vmap(jax.hessian(material.elem.fun))(
            u, q, aux
        )
        return hess

    def test_jac(
        self,
        material: apple.material.tetra.MaterialTetra,
        u: Float[jax.Array, "C 4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> None:
        actual: Float[jax.Array, "C 4 3"] = material.jac(u, q, aux)
        expected: Float[jax.Array, "C 4 3"] = jax.grad(material.fun)(u, q, aux)
        np.testing.assert_allclose(actual, expected)

    def test_hess(
        self,
        material: apple.material.tetra.MaterialTetra,
        u: Float[jax.Array, "C 4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> None:
        actual: Float[jax.Array, "C 4 3 4 3"] = material.hess(u, q, aux)
        expected: Float[jax.Array, "C 4 3 4 3"] = self.hessian(material, u, q, aux)
        np.testing.assert_allclose(actual, expected)

    def test_hessp(
        self,
        material: apple.material.tetra.MaterialTetra,
        u: Float[jax.Array, "C 4 3"],
        p: Float[jax.Array, "C 4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> None:
        actual: Float[jax.Array, "C 4 3"] = material.hessp(u, p, q, aux)
        hess: Float[jax.Array, "C 4 3 4 3"] = self.hessian(material, u, q, aux)
        hess: Float[jax.Array, "C 12 12"] = hess.reshape(self.n_cells, 12, 12)
        p: Float[jax.Array, "C 12"] = p.reshape(self.n_cells, 12)
        expected: Float[jax.Array, "C 12"] = jnp.matvec(hess, p)
        expected: Float[jax.Array, "C 4 3"] = expected.reshape(self.n_cells, 4, 3)
        np.testing.assert_allclose(actual, expected)

    def test_hess_diag(
        self,
        material: apple.material.tetra.MaterialTetra,
        u: Float[jax.Array, "C 4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> None:
        actual: Float[jax.Array, "C 4 3"] = material.hess_diag(u, q, aux)
        hess: Float[jax.Array, "C 4 3 4 3"] = self.hessian(material, u, q, aux)
        hess: Float[jax.Array, "C 12 12"] = hess.reshape(self.n_cells, 12, 12)
        expected: Float[jax.Array, "C 12"] = jnp.diagonal(hess, axis1=1, axis2=2)
        expected: Float[jax.Array, "C 4 3"] = expected.reshape(self.n_cells, 4, 3)
        np.testing.assert_allclose(actual, expected)

    def test_hess_quad(
        self,
        material: apple.material.tetra.MaterialTetra,
        u: Float[jax.Array, "C 4 3"],
        p: Float[jax.Array, "C 4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> None:
        actual: Float[jax.Array, ""] = material.hess_quad(u, p, q, aux)
        hess: Float[jax.Array, "C 4 3 4 3"] = self.hessian(material, u, q, aux)
        hess: Float[jax.Array, "C 12 12"] = hess.reshape(self.n_cells, 12, 12)
        p: Float[jax.Array, "C 12"] = p.reshape(self.n_cells, 12)
        expected: Float[jax.Array, ""] = jnp.vdot(p, jnp.matvec(hess, p))
        np.testing.assert_allclose(actual, expected)
