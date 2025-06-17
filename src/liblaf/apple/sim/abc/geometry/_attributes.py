from typing import Any, override

import jax
import jax.numpy as jnp
import pyvista as pv
from numpy.typing import ArrayLike

from liblaf.apple import struct


class GeometryAttributes(struct.PyTreeDict[jax.Array]):
    @override
    def __setitem__(self, key: struct.KeyLike, value: ArrayLike) -> None:
        value = jnp.asarray(value)
        super().__setitem__(key, value)


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
