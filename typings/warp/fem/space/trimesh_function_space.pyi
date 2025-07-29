from .shape import TriangleShapeFunction as TriangleShapeFunction
from .topology import SpaceTopology as SpaceTopology, forward_base_topology as forward_base_topology
from _typeshed import Incomplete
from warp.fem import cache as cache
from warp.fem.geometry import Trimesh as Trimesh
from warp.fem.types import ElementIndex as ElementIndex

class TrimeshTopologyArg:
    edge_vertex_indices: None
    tri_edge_indices: None
    vertex_count: int
    edge_count: int

class TrimeshSpaceTopology(SpaceTopology):
    TopologyArg = TrimeshTopologyArg
    element_node_index: Incomplete
    element_node_sign: Incomplete
    def __init__(self, mesh: Trimesh, shape: TriangleShapeFunction) -> None: ...
    @property
    def name(self): ...
    @cache.cached_arg_value
    def topo_arg_value(self, device): ...
    def fill_topo_arg(self, arg: TrimeshTopologyArg, device): ...
    def node_count(self) -> int: ...

def make_trimesh_space_topology(mesh: Trimesh, shape: TriangleShapeFunction): ...
