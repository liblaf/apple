import warp as wp
from .shape import CubeShapeFunction as CubeShapeFunction
from .topology import SpaceTopology as SpaceTopology, forward_base_topology as forward_base_topology
from _typeshed import Incomplete
from warp.fem import cache as cache
from warp.fem.geometry import AdaptiveNanogrid as AdaptiveNanogrid, Nanogrid as Nanogrid
from warp.fem.types import ElementIndex as ElementIndex

class NanogridTopologyArg:
    vertex_grid: wp.uint64
    face_grid: wp.uint64
    edge_grid: wp.uint64
    vertex_count: int
    edge_count: int
    face_count: int

class NanogridSpaceTopology(SpaceTopology):
    TopologyArg = NanogridTopologyArg
    element_node_index: Incomplete
    def __init__(self, grid: Nanogrid | AdaptiveNanogrid, shape: CubeShapeFunction) -> None: ...
    @property
    def name(self): ...
    @cache.cached_arg_value
    def topo_arg_value(self, device): ...
    def fill_topo_arg(self, arg, device) -> None: ...
    def node_count(self) -> int: ...

def make_nanogrid_space_topology(grid: Nanogrid | AdaptiveNanogrid, shape: CubeShapeFunction): ...
