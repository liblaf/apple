from typing import Self, override

from liblaf.apple.sim import element as _e
from liblaf.apple.sim import geometry as _g
from liblaf.apple.sim import quadrature as _q

from ._region import Region


class RegionTriangle(Region):
    @override
    @classmethod
    def from_geometry(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        geometry: _g.GeometryTriangle,
        element: _e.Element | None = None,
        quadrature: _q.Scheme | None = None,
        *,
        grad: bool = True,
        hess: bool = False,
    ) -> Self:
        if element is None:
            element = _e.ElementTriangle()
        if quadrature is None:
            quadrature = _q.QuadratureTriangle()
        return super().from_geometry(
            geometry, element, quadrature, grad=grad, hess=hess
        )
