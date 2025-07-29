from .shape import ShapeFunction as ShapeFunction, TetrahedronPolynomialShapeFunctions as TetrahedronPolynomialShapeFunctions, TetrahedronShapeFunction as TetrahedronShapeFunction
from .topology import SpaceTopology as SpaceTopology, forward_base_topology as forward_base_topology
from _typeshed import Incomplete
from warp.fem import cache as cache
from warp.fem.geometry import Tetmesh as Tetmesh
from warp.fem.types import ElementIndex as ElementIndex

class TetmeshTopologyArg:
    tet_edge_indices: None
    tet_face_indices: None
    face_vertex_indices: None
    face_tet_indices: None
    vertex_count: int
    edge_count: int
    face_count: int

class TetmeshSpaceTopology(SpaceTopology):
    TopologyArg = TetmeshTopologyArg
    element_node_index: Incomplete
    element_node_sign: Incomplete
    def __init__(self, mesh: Tetmesh, shape: TetrahedronShapeFunction) -> None: ...
    @property
    def name(self): ...
    @cache.cached_arg_value
    def topo_arg_value(self, device): ...
    def fill_topo_arg(self, arg: TetmeshTopologyArg, device): ...
    def node_count(self) -> int: ...

def make_tetmesh_space_topology(mesh: Tetmesh, shape: ShapeFunction): ...
