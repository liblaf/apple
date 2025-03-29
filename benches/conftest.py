import jax
import pytest
import pyvista as pv
from icecream import ic
from jaxtyping import PRNGKeyArray

from liblaf import apple


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    ic(jax.default_backend())


@pytest.fixture
def key(seed: int = 0) -> PRNGKeyArray:
    return jax.random.key(seed)


@pytest.fixture
def mesh() -> pv.UnstructuredGrid:
    surface: pv.PolyData = pv.Icosphere()
    mesh: pv.UnstructuredGrid = apple.tetwild(surface, edge_length_fac=0.1)
    ic(mesh)
    return mesh
