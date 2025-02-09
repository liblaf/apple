import functools

import felupe
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Float, Integer

from . import Region


class RegionTetra(Region):
    mesh: pv.UnstructuredGrid
    _region: felupe.RegionTetra

    def __init__(self, mesh: pv.UnstructuredGrid) -> None:
        self.mesh = mesh
        mesh_felupe = felupe.Mesh(
            mesh.points, mesh.cells_dict[pv.CellType.TETRA], cell_type="tetra"
        )
        region = felupe.RegionTetra(mesh_felupe, grad=True, hess=True)
        self._region = region

    @property
    def cells(self) -> Integer[np.ndarray, "c a"]:
        return self.mesh.cells_dict[pv.CellType.TETRA]

    @property
    def n_points(self) -> int:
        return self.mesh.n_points

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells

    @functools.cached_property
    def h(self) -> Float[jax.Array, "a q"]:
        """Element shape function array `h_aq` of shape function `a` evaluated at quadrature point `q`."""
        return jnp.asarray(self._region.h).reshape(4, 1)  # pyright: ignore[reportAttributeAccessIssue]

    @functools.cached_property
    def dV(self) -> Float[jax.Array, "q c"]:
        """Numeric *differential volume element* as product of determinant of geometric gradient `dV_qc = det(dXdr)_qc w_q` and quadrature weight `w_q`, evaluated at quadrature point `q` for every cell `c`."""
        return jnp.asarray(self._region.dV)  # pyright: ignore[reportAttributeAccessIssue]

    @functools.cached_property
    def dhdX(self) -> Float[jax.Array, "a J q c"]:
        """Partial derivative of element shape functions `dhdX_aJqc` of shape function `a` w.r.t. undeformed coordinate `J` evaluated at quadrature point `q` for every cell `c`."""
        return jnp.asarray(self._region.dhdX)  # pyright: ignore[reportAttributeAccessIssue]

    @functools.cached_property
    def d2hdXdX(self) -> Float[jax.Array, "a I J q c"]:
        return jnp.asarray(self._region.d2hdXdX)  # pyright: ignore[reportAttributeAccessIssue]
