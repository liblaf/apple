import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Self

import pyvista as pv

from liblaf.apple import struct


@struct.pytree
class GeometryAttributes(struct.ArrayDict):
    association: pv.FieldAssociation = struct.static(kw_only=True)

    if TYPE_CHECKING:

        def __init__(
            self,
            data: struct.MappingLike = None,
            /,
            association: pv.FieldAssociation = ...,
        ) -> None: ...

    @classmethod
    def factory(cls, association: pv.FieldAssociation) -> Callable[[], Self]:
        return functools.partial(cls, association=association)
