import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, Self, dataclass_transform, type_check_only

import equinox as eqx

from ._field import array, container, field

type Node = Any

@type_check_only
class PyTreeMixin:
    def replace(self, **changes: Any) -> Self: ...
    def tree_at(
        self,
        where: Callable[[Self], Node | Sequence[Node]],
        replace: Any | Sequence[Any] = ...,
        replace_fn: Callable[[Node], Any] = ...,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> Self: ...

@dataclass_transform(
    frozen_default=True,
    field_specifiers=(dataclasses.field, eqx.field, array, container, field),
)
class PyTree(PyTreeMixin): ...

@dataclass_transform(
    frozen_default=False,
    field_specifiers=(dataclasses.field, eqx.field, array, container, field),
)
class PyTreeMutable(PyTreeMixin): ...
