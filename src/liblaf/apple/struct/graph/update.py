from collections.abc import Iterable

import equinox as eqx

from liblaf.apple.struct.graph.node import GraphNode
from liblaf.apple.struct.mapping import MappingLike, PyTreeDict


@eqx.filter_jit
def graph_update(
    nodes: MappingLike[GraphNode], topological: Iterable[str]
) -> PyTreeDict[GraphNode]:
    nodes: PyTreeDict[GraphNode] = PyTreeDict(nodes)
    for key in topological:
        node: GraphNode = nodes[key]
        deps: PyTreeDict[GraphNode] = nodes.select(node.deps)
        node = node.with_deps(deps)
        nodes[key] = node
    return nodes
