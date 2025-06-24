import einops
import felupe
import hypothesis
import hypothesis.extra.array_api
import hypothesis.strategies
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pyvista as pv
from jaxtyping import Float

from liblaf.apple.sim.region import Region

xps = hypothesis.extra.array_api.make_strategies_namespace(jnp)


class TestRegionTetra:
    @pytest.fixture(scope="class")
    def mesh_pyvista(self) -> pv.UnstructuredGrid:
        return pv.examples.download_tetrahedron(load=True)

    @pytest.fixture(scope="class")
    def mesh_felupe(self, mesh_pyvista: pv.UnstructuredGrid) -> felupe.Mesh:
        return felupe.Mesh(
            mesh_pyvista.points, mesh_pyvista.cells_dict[pv.CellType.TETRA], "tetra"
        )

    @pytest.fixture(scope="class")
    def region(self, mesh_pyvista: pv.UnstructuredGrid) -> Region:
        return Region.from_pyvista(mesh_pyvista, grad=True)

    @pytest.fixture(scope="class")
    def region_felupe(self, mesh_felupe: felupe.Mesh) -> felupe.RegionTetra:
        return felupe.RegionTetra(mesh_felupe)

    def test_h(self, region: Region, region_felupe: felupe.RegionTetra) -> None:
        actual: Float[jax.Array, "q a"] = region.h
        expected: Float[np.ndarray, "q a"] = einops.rearrange(
            region_felupe.h,  # pyright: ignore[reportAttributeAccessIssue]
            "a q 1 -> q a",
        )
        np.testing.assert_allclose(actual, expected)

    def test_dhdr(self, region: Region, region_felupe: felupe.RegionTetra) -> None:
        actual: Float[jax.Array, "q a J"] = region.dhdr
        expected: Float[np.ndarray, "q a J"] = einops.rearrange(
            region_felupe.dhdr,  # pyright: ignore[reportAttributeAccessIssue]
            "a J q 1 -> q a J",
        )
        np.testing.assert_allclose(actual, expected)

    def test_dXdr(self, region: Region, region_felupe: felupe.RegionTetra) -> None:
        actual: Float[jax.Array, "c q J J"] = region.dXdr
        expected: Float[np.ndarray, "c q J J"] = einops.rearrange(
            region_felupe.dXdr,  # pyright: ignore[reportAttributeAccessIssue]
            "I J q c -> c q I J",
        )
        np.testing.assert_allclose(actual, expected)

    def test_drdX(self, region: Region, region_felupe: felupe.RegionTetra) -> None:
        actual: Float[jax.Array, "c q J J"] = region.drdX
        expected: Float[np.ndarray, "c q J J"] = einops.rearrange(
            region_felupe.drdX,  # pyright: ignore[reportAttributeAccessIssue]
            "J I q c -> c q J I",
        )
        np.testing.assert_allclose(actual, expected)

    def test_dV(self, region: Region, region_felupe: felupe.RegionTetra) -> None:
        actual: Float[jax.Array, "c q"] = region.dV
        expected: Float[np.ndarray, "c q"] = einops.rearrange(
            region_felupe.dV,  # pyright: ignore[reportAttributeAccessIssue]
            "q c -> c q",
        )
        np.testing.assert_allclose(actual, expected)

    def test_dhdX(self, region: Region, region_felupe: felupe.RegionTetra) -> None:
        actual: Float[jax.Array, "c q a J"] = region.dhdX
        expected: Float[np.ndarray, "c q a J"] = einops.rearrange(
            region_felupe.dhdX,  # pyright: ignore[reportAttributeAccessIssue]
            "a J q c -> c q a J",
        )
        np.testing.assert_allclose(actual, expected)

    @hypothesis.given(
        data=hypothesis.strategies.data(),
        dim=hypothesis.strategies.integers(min_value=1, max_value=16),
    )
    def test_gradient(
        self,
        data: hypothesis.strategies.DataObject,
        dim: int,
        region: Region,
        region_felupe: felupe.RegionTetra,
    ) -> None:
        x: Float[jax.Array, "points dim"] = data.draw(
            xps.arrays(
                dtype=xps.floating_dtypes(sizes=(32,)),
                shape=(region.n_points, dim),
                elements={
                    "min_value": jnp.finfo(jnp.float16).min,
                    "max_value": jnp.finfo(jnp.float16).max,
                },
            )
        )
        actual: Float[jax.Array, "c q *dim J"] = region.gradient(x)
        field: felupe.Field = felupe.Field(region_felupe, dim=x.shape[1], values=x)  # pyright: ignore[reportArgumentType]
        expected: Float[np.ndarray, "c q *dim J"] = einops.rearrange(
            field.grad(), "I J q c -> c q I J"
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4)
