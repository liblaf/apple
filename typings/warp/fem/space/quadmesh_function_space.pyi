from .shape import SquareShapeFunction as SquareShapeFunction
from .topology import SpaceTopology as SpaceTopology, forward_base_topology as forward_base_topology
from _typeshed import Incomplete
from warp.fem import cache as cache
from warp.fem.geometry import Quadmesh2D as Quadmesh2D
from warp.fem.polynomial import is_closed as is_closed
from warp.fem.types import ElementIndex as ElementIndex, NULL_NODE_INDEX as NULL_NODE_INDEX

class Quadmesh2DTopologyArg:
    edge_vertex_indices: None
    quad_edge_indices: None
    vertex_count: int
    edge_count: int
    cell_count: int

class QuadmeshSpaceTopology(SpaceTopology):
    TopologyArg = Quadmesh2DTopologyArg
    element_node_index: Incomplete
    element_node_sign: Incomplete
    def __init__(self, mesh: Quadmesh2D, shape: SquareShapeFunction) -> None: ...
    @property
    def name(self): ...
    @cache.cached_arg_value
    def topo_arg_value(self, device): ...
    def fill_topo_arg(self, arg: Quadmesh2DTopologyArg, device): ...
    def node_count(self) -> int: ...

def make_quadmesh_space_topology(mesh: Quadmesh2D, shape: SquareShapeFunction): ...
