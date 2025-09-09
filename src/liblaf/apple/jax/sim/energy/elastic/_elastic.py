from typing import Self, override

import pyvista as pv
from jaxtyping import Array, Float

from liblaf.apple.jax import tree
from liblaf.apple.jax.sim.energy._energy import Energy
from liblaf.apple.jax.sim.geometry import Geometry
from liblaf.apple.jax.sim.quadrature import Scheme
from liblaf.apple.jax.sim.region import Region
from liblaf.apple.jax.typing import Scalar, Vector


@tree.pytree
class Elastic(Energy):
    region: Region

    @classmethod
    def from_region(cls, region: Region) -> Self:
        return cls(region=region)

    @classmethod
    def from_geometry(
        cls, geometry: Geometry, *, quadrature: Scheme | None = None
    ) -> Self:
        region: Region = Region.from_geometry(
            geometry, grad=True, quadrature=quadrature
        )
        return cls.from_region(region)

    @classmethod
    def from_pyvista(
        cls, mesh: pv.UnstructuredGrid, *, quadrature: Scheme | None = None
    ) -> Self:
        geometry: Geometry = Geometry.from_pyvista(mesh)
        return cls.from_geometry(geometry, quadrature=quadrature)

    @override
    def fun(self, u: Vector) -> Scalar:
        F: Float[Array, "c q J J"] = self.region.deformation_gradient(u)
        Psi: Float[Array, "c q"] = self.energy_density(F)
        return self.region.integrate(Psi).sum()

    def energy_density(self, F: Float[Array, "c q J J"]) -> Float[Array, "c q"]:
        raise NotImplementedError
