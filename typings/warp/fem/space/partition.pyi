import warp as wp
import warp.fem.cache as cache
from .function_space import FunctionSpace as FunctionSpace
from .topology import SpaceTopology as SpaceTopology
from _typeshed import Incomplete
from typing import Any
from warp.fem.geometry import GeometryPartition as GeometryPartition, WholeGeometryPartition as WholeGeometryPartition
from warp.fem.types import NULL_NODE_INDEX as NULL_NODE_INDEX
from warp.fem.utils import compress_node_indices as compress_node_indices

class SpacePartition:
    class PartitionArg: ...
    space_topology: Incomplete
    geo_partition: Incomplete
    def __init__(self, space_topology: SpaceTopology, geo_partition: GeometryPartition) -> None: ...
    def node_count(self) -> None: ...
    def owned_node_count(self) -> int: ...
    def interior_node_count(self) -> int: ...
    def space_node_indices(self) -> wp.array: ...
    def partition_arg_value(self, device) -> None: ...
    def fill_partition_arg(self, arg, device) -> None: ...
    @staticmethod
    def partition_node_index(args: PartitionArg, space_node_index: int): ...
    @property
    def name(self) -> str: ...

class WholeSpacePartition(SpacePartition):
    class PartitionArg: ...
    def __init__(self, space_topology: SpaceTopology) -> None: ...
    def node_count(self): ...
    def owned_node_count(self) -> int: ...
    def interior_node_count(self) -> int: ...
    def space_node_indices(self): ...
    def partition_arg_value(self, device): ...
    def fill_partition_arg(self, arg, device) -> None: ...
    @wp.func
    def partition_node_index(args: Any, space_node_index: int): ...
    def __eq__(self, other: SpacePartition) -> bool: ...
    @property
    def name(self) -> str: ...

class NodeCategory:
    OWNED_INTERIOR: Incomplete
    OWNED_FRONTIER: Incomplete
    HALO_LOCAL_SIDE: Incomplete
    HALO_OTHER_SIDE: Incomplete
    EXTERIOR: Incomplete
    COUNT: int

class NodePartition(SpacePartition):
    class PartitionArg:
        space_to_partition: None
    def __init__(self, space_topology: SpaceTopology, geo_partition: GeometryPartition, with_halo: bool = True, device=None, temporary_store: cache.TemporaryStore = None) -> None: ...
    def node_count(self) -> int: ...
    def owned_node_count(self) -> int: ...
    def interior_node_count(self) -> int: ...
    def space_node_indices(self): ...
    @cache.cached_arg_value
    def partition_arg_value(self, device): ...
    def fill_partition_arg(self, arg, device) -> None: ...
    @wp.func
    def partition_node_index(args: PartitionArg, space_node_index: int): ...

def make_space_partition(space: FunctionSpace | None = None, geometry_partition: GeometryPartition | None = None, space_topology: SpaceTopology | None = None, with_halo: bool = True, device=None, temporary_store: cache.TemporaryStore = None) -> SpacePartition: ...
