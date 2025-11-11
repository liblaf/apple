from typing import Self, override

import pyvista as pv
from jaxtyping import Array, Float
from liblaf.peach import tree

from liblaf.apple.jax.sim.energy._energy import Energy
from liblaf.apple.jax.sim.geometry import Geometry
from liblaf.apple.jax.sim.quadrature import Scheme
from liblaf.apple.jax.sim.region import Region
from liblaf.apple.jax.typing import Scalar, Vector


@tree.define
class Elastic(Energy):
    region: Region

    @classmethod
    def from_region(cls, region: Region, **kwargs) -> Self:
        return cls(region=region, **kwargs)

    @classmethod
    def from_geometry(
        cls, geometry: Geometry, *, quadrature: Scheme | None = None, **kwargs
    ) -> Self:
        region: Region = Region.from_geometry(
            geometry, grad=True, quadrature=quadrature
        )
        return cls.from_region(region, **kwargs)

    @classmethod
    def from_pyvista(
        cls, mesh: pv.UnstructuredGrid, *, quadrature: Scheme | None = None, **kwargs
    ) -> Self:
        geometry: Geometry = Geometry.from_pyvista(mesh)
        return cls.from_geometry(geometry, quadrature=quadrature, **kwargs)

    @override
    def fun(self, u: Vector) -> Scalar:
        F: Float[Array, "c q J J"] = self.region.deformation_gradient(u)
        Psi: Float[Array, "c q"] = self.energy_density(F)
        return self.region.integrate(Psi).sum()

    def energy_density(self, F: Float[Array, "c q J J"]) -> Float[Array, "c q"]:
        raise NotImplementedError
