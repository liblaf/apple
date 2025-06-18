from collections.abc import Iterable

import equinox as eqx

from liblaf.apple.struct._utils import MappingLike
from liblaf.apple.struct.dict_util import FrozenDict
from liblaf.apple.struct.graph._node import GraphNode


@eqx.filter_jit
def graph_update(
    nodes: MappingLike[GraphNode], topological: Iterable[str]
) -> FrozenDict[GraphNode]:
    nodes: FrozenDict[GraphNode] = FrozenDict(nodes)
    for key in topological:
        node: GraphNode = nodes[key]
        deps: FrozenDict[GraphNode] = nodes.select(node.deps)
        node = node.with_deps(deps)
        nodes = nodes.copy(node)
    return nodes
