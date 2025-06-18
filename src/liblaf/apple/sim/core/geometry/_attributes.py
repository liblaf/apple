from typing import Any

import jax
import pyvista as pv

from liblaf.apple import struct


class GeometryAttributes(struct.DictArray): ...


def data_property(
    name: str, association: pv.FieldAssociation = pv.FieldAssociation.POINT
) -> property:
    attributes: str = {
        pv.FieldAssociation.POINT: "point_data",
        pv.FieldAssociation.CELL: "cell_data",
        pv.FieldAssociation.NONE: "field_data",
    }[association]

    def getter(self: Any) -> jax.Array:
        return getattr(self, attributes)[name]

    def setter(self: Any, value: jax.Array) -> None:
        getattr(self, attributes)[name] = value

    def deleter(self: Any) -> None:
        del getattr(self, attributes)[name]

    return property(fget=getter, fset=setter, fdel=deleter)
