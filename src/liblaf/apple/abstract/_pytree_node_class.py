import abc
from collections.abc import Hashable, Iterable
from typing import Any, Self

import jax


@jax.tree_util.register_pytree_node_class
class AbstractPytreeNodeClass[AuxData: Hashable, Children: Iterable[Any]](abc.ABC):
    @abc.abstractmethod
    def tree_flatten(self) -> tuple[Children, AuxData]: ...

    @classmethod
    @abc.abstractmethod
    def tree_unflatten(cls, aux_data: AuxData, children: Children) -> Self: ...
