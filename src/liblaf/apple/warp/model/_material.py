from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, ClassVar, Protocol

import attrs
import numpy as np
import pyvista as pv
import warp as wp

from liblaf.apple.common import AttrName
from liblaf.apple.torch.fem import Region


class ArrayAnnotation(Protocol):
    @property
    def dtype(self) -> Any: ...


class StructInstance(Protocol): ...


class Struct(Protocol):
    def __call__(self) -> StructInstance: ...


@attrs.frozen
class MaterialVar:
    name: str
    annotation: ArrayAnnotation
    factory: Callable[[Region, ArrayAnnotation], wp.array]

    def from_region(self, region: Region) -> wp.array:
        return self.factory(region, self.annotation)


@attrs.frozen
class MaterialFieldFactory:
    association: pv.FieldAssociation

    def _get_data(self, region: Region, name: str) -> np.ndarray:
        match self.association:
            case pv.FieldAssociation.POINT:
                return region.point_data[name]
            case pv.FieldAssociation.CELL:
                return region.cell_data[name]
            case _:
                raise NotImplementedError

    def floating(self, name: str) -> MaterialField:
        vtk: str = AttrName.to_vtk(name)
        return MaterialField(
            name=name,
            annotation=lambda dtype: wp.array1d(dtype=dtype),
            factory=lambda region, annotation: wp.from_numpy(
                self._get_data(region, vtk), dtype=annotation.dtype
            ),
        )

    def vec3(self, name: str) -> MaterialField:
        vtk: str = AttrName.to_vtk(name)
        return MaterialField(
            name=name,
            annotation=lambda dtype: wp.array1d(dtype=wp.types.vector(3, dtype)),
            factory=lambda region, annotation: wp.from_numpy(
                self._get_data(region, vtk), dtype=annotation.dtype
            ),
        )

    def vec6(self, name: str) -> MaterialField:
        vtk: str = AttrName.to_vtk(name)
        return MaterialField(
            name=name,
            annotation=lambda dtype: wp.array1d(dtype=wp.types.vector(6, dtype)),
            factory=lambda region, annotation: wp.from_numpy(
                self._get_data(region, vtk), dtype=annotation.dtype
            ),
        )


@attrs.frozen
class MaterialField:
    POINT: ClassVar[MaterialFieldFactory] = MaterialFieldFactory(
        pv.FieldAssociation.POINT
    )
    CELL: ClassVar[MaterialFieldFactory] = MaterialFieldFactory(
        pv.FieldAssociation.CELL
    )

    name: str
    annotation: Callable[[Any], ArrayAnnotation]
    factory: Callable[[Region, ArrayAnnotation], wp.array]

    def make(self, dtype: Any) -> MaterialVar:
        return MaterialVar(
            name=self.name, annotation=self.annotation(dtype), factory=self.factory
        )


@functools.cache
def make_struct(
    material_vars: tuple[MaterialVar, ...], module: str, qualname: str
) -> Struct:
    annotations: dict[str, ArrayAnnotation] = {
        var.name: var.annotation for var in material_vars
    }
    c: type = type(
        "Materials",
        (),
        {
            "__module__": module,
            "__qualname__": f"{qualname}.Materials",
            "__annotations__": annotations,
        },
    )
    return wp.struct(c, module="unique")
