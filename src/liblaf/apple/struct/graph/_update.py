from collections.abc import Iterable

from liblaf.apple import utils

from ._collection import NodeCollection
from ._node import Node
from ._utils import NodesLike


@utils.jit(static_argnames=("topological",))
def graph_update[T: Node](
    nodes: NodesLike[T], topological: Iterable[str]
) -> NodeCollection:
    nodes = NodeCollection(nodes)
    for key in topological:
        node: Node = nodes[key]
        deps: NodeCollection = nodes.select(node.deps)
        node = node.with_deps(deps)
        nodes = nodes.add(node)
    return nodes
