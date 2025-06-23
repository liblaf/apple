from typing import TYPE_CHECKING

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
