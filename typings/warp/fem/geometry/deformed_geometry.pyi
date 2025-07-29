import warp as wp
from .geometry import Geometry as Geometry
from _typeshed import Incomplete
from warp.fem import cache as cache
from warp.fem.polynomial import Polynomial as Polynomial
from warp.fem.types import Coords as Coords, ElementIndex as ElementIndex, Sample as Sample, make_free_sample as make_free_sample

class DeformedGeometry(Geometry):
    field: GeometryField
    field_trace: Incomplete
    dimension: Incomplete
    SideIndexArg: Incomplete
    cell_count: Incomplete
    vertex_count: Incomplete
    side_count: Incomplete
    boundary_side_count: Incomplete
    reference_cell: Incomplete
    reference_side: Incomplete
    side_index_arg_value: Incomplete
    fill_side_index_arg: Incomplete
    boundary_side_index: Incomplete
    def __init__(self, field: wp.fem.field.GeometryField, relative: bool = True, build_bvh: bool = False) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def base(self) -> Geometry: ...
    def cell_arg_value(self, device) -> DeformedGeometry.CellArg: ...
    def fill_cell_arg(self, args: DeformedGeometry.CellArg, device): ...
    def side_arg_value(self, device) -> DeformedGeometry.SideArg: ...
    def fill_side_arg(self, args: DeformedGeometry.SideArg, device): ...
