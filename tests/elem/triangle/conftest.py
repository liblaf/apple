import pytest
import pyvista as pv


@pytest.fixture(scope="package")
def mesh() -> pv.PolyData:
    return pv.Icosphere()
