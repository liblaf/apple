import warp as wp
from .closest_point import project_on_seg_at_origin as project_on_seg_at_origin
from .element import LinearEdge as LinearEdge, Square as Square
from .geometry import Geometry as Geometry
from _typeshed import Incomplete
from typing import Any
from warp.fem.cache import TemporaryStore as TemporaryStore, borrow_temporary as borrow_temporary, borrow_temporary_like as borrow_temporary_like, cached_arg_value as cached_arg_value
from warp.fem.types import Coords as Coords, ElementIndex as ElementIndex, OUTSIDE as OUTSIDE, Sample as Sample

class QuadmeshCellArg:
    quad_vertex_indices: None
    quad_bvh: wp.uint64

class QuadmeshSideArg:
    cell_arg: QuadmeshCellArg
    edge_vertex_indices: None
    edge_quad_indices: None

class Quadmesh(Geometry):
    quad_vertex_indices: Incomplete
    positions: Incomplete
    cell_closest_point: Incomplete
    cell_coordinates: Incomplete
    side_coordinates: Incomplete
    def __init__(self, quad_vertex_indices: wp.array, positions: wp.array, build_bvh: bool = False, temporary_store: TemporaryStore | None = None) -> None: ...
    def cell_count(self): ...
    def vertex_count(self): ...
    def side_count(self): ...
    def boundary_side_count(self): ...
    def reference_cell(self) -> Square: ...
    def reference_side(self) -> LinearEdge: ...
    @property
    def edge_quad_indices(self) -> wp.array: ...
    @property
    def edge_vertex_indices(self) -> wp.array: ...
    class SideIndexArg:
        boundary_edge_indices: None
    def fill_cell_topo_arg(self, args: QuadmeshCellArg, device): ...
    def fill_side_topo_arg(self, args: QuadmeshSideArg, device): ...
    def cell_arg_value(self, device): ...
    def fill_cell_arg(self, args: Quadmesh.CellArg, device): ...
    def side_arg_value(self, device): ...
    def fill_side_arg(self, args: Quadmesh.SideArg, device): ...
    @cached_arg_value
    def side_index_arg_value(self, device) -> SideIndexArg: ...
    def fill_side_index_arg(self, args: SideIndexArg, device): ...
    @wp.func
    def boundary_side_index(args: SideIndexArg, boundary_side_index: int): ...
    @wp.func
    def cell_position(args: Any, s: Sample): ...
    @wp.func
    def cell_deformation_gradient(cell_arg: Any, s: Sample): ...
    @wp.func
    def side_position(args: Any, s: Sample): ...
    @wp.func
    def side_deformation_gradient(args: Any, s: Sample): ...
    @wp.func
    def side_closest_point(args: Any, side_index: ElementIndex, pos: Any): ...
    @wp.func
    def side_inner_cell_index(arg: Any, side_index: ElementIndex): ...
    @wp.func
    def side_outer_cell_index(arg: Any, side_index: ElementIndex): ...
    @wp.func
    def side_inner_cell_coords(args: Any, side_index: ElementIndex, side_coords: Coords): ...
    @wp.func
    def side_outer_cell_coords(args: Any, side_index: ElementIndex, side_coords: Coords): ...
    @wp.func
    def side_from_cell_coords(args: Any, side_index: ElementIndex, quad_index: ElementIndex, quad_coords: Coords): ...
    @wp.func
    def cell_bvh_id(cell_arg: Any): ...
    @wp.func
    def cell_bounds(cell_arg: Any, cell_index: ElementIndex): ...

class Quadmesh2DCellArg:
    topology: QuadmeshCellArg
    positions: None

class Quadmesh2DSideArg:
    topology: QuadmeshSideArg
    positions: None

class Quadmesh2D(Quadmesh):
    dimension: int
    CellArg = Quadmesh2DCellArg
    SideArg = Quadmesh2DSideArg
    @wp.func
    def side_to_cell_arg(side_arg: SideArg): ...

class Quadmesh3DCellArg:
    topology: QuadmeshCellArg
    positions: None

class Quadmesh3DSideArg:
    topology: QuadmeshSideArg
    positions: None

class Quadmesh3D(Quadmesh):
    dimension: int
    CellArg = Quadmesh3DCellArg
    SideArg = Quadmesh3DSideArg
    @wp.func
    def side_to_cell_arg(side_arg: SideArg): ...
