import attrs
import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Float, Integer, PyTree

from liblaf.apple.jax.elem import tetra
from liblaf.apple.jax.energy.tetra import EnergyTetra


@attrs.define
class Object:
    name: str
    energy: EnergyTetra
    geometry: pv.UnstructuredGrid

    @property
    def cells(self) -> Integer[jax.Array, "C 4"]:
        return jnp.asarray(self.geometry.cells_dict[pv.CellType.TETRA])

    @property
    def dof_id(self) -> Integer[jax.Array, "P 3"]:
        if (dof_id := self.geometry.point_data.get("dof-id")) is not None:
            return jnp.asarray(dof_id)
        return jnp.arange(self.geometry.n_points * 3).reshape(self.geometry.n_points, 3)

    @property
    def n_points(self) -> int:
        return self.geometry.n_points

    @property
    def points(self) -> Float[jax.Array, "P 3"]:
        return jnp.asarray(self.geometry.points)

    def prepare(self) -> PyTree:
        return self.energy.prepare(self.points[self.cells])

    def select_dof(self, x: Float[jax.Array, " N"]) -> Float[jax.Array, "P 3"]:
        return x[self.dof_id]

    def fun(
        self, u: Float[jax.Array, "P 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        return self.energy.fun(u[self.cells], q, aux)

    def jac(
        self, u: Float[jax.Array, "P 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "P 3"]:
        jac: Float[jax.Array, "C 4 3"] = self.energy.jac(u[self.cells], q, aux)
        jac: Float[jax.Array, "P 3"] = tetra.segment_sum(
            jac, self.cells, n_points=self.n_points
        )
        return jac

    def hess(
        self, u: Float[jax.Array, "P 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "..."]:
        raise NotImplementedError

    def hessp(
        self,
        u: Float[jax.Array, "P 3"],
        p: Float[jax.Array, "P 3"],
        q: PyTree,
        aux: PyTree,
    ) -> Float[jax.Array, "P 3"]:
        hessp: Float[jax.Array, "C 4 3"] = self.energy.hessp(
            u[self.cells], p[self.cells], q, aux
        )
        hessp: Float[jax.Array, "P 3"] = tetra.segment_sum(
            hessp, self.cells, n_points=self.n_points
        )
        return hessp
