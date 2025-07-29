from .shape import CubeShapeFunction as CubeShapeFunction
from .topology import SpaceTopology as SpaceTopology, forward_base_topology as forward_base_topology
from _typeshed import Incomplete
from warp.fem import cache as cache
from warp.fem.geometry import Hexmesh as Hexmesh
from warp.fem.geometry.hexmesh import EDGE_VERTEX_INDICES as EDGE_VERTEX_INDICES, FACE_ORIENTATION as FACE_ORIENTATION, FACE_TRANSLATION as FACE_TRANSLATION
from warp.fem.types import ElementIndex as ElementIndex

class HexmeshTopologyArg:
    hex_edge_indices: None
    hex_face_indices: None
    vertex_count: int
    edge_count: int
    face_count: int

class HexmeshSpaceTopology(SpaceTopology):
    TopologyArg = HexmeshTopologyArg
    element_node_index: Incomplete
    element_node_sign: Incomplete
    def __init__(self, mesh: Hexmesh, shape: CubeShapeFunction) -> None: ...
    @property
    def name(self): ...
    @cache.cached_arg_value
    def topo_arg_value(self, device): ...
    def fill_topo_arg(self, arg: HexmeshTopologyArg, device): ...
    def node_count(self) -> int: ...

def make_hexmesh_space_topology(mesh: Hexmesh, shape: CubeShapeFunction): ...
