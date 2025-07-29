import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Self, dataclass_transform

import equinox as eqx

# pyright: enableExperimentalFeatures=true
from typing_extensions import Sentinel

from ._field import array, container, field

MISSING = Sentinel("MISSING")
type Node = Any


class PyTreeMixin:
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]

    def replace(self, **changes: Any) -> Self:
        return dataclasses.replace(self, **changes)

    def tree_at(
        self,
        where: Callable[[Self], Node | Sequence[Node]],
        replace: Any | Sequence[Any] | MISSING = MISSING,
        replace_fn: Callable[[Node], Any] | MISSING = MISSING,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> Self:
        kwargs: dict[str, Any] = {}
        if replace is not MISSING:
            kwargs["replace"] = replace
        if replace_fn is not MISSING:
            kwargs["replace_fn"] = replace_fn
        if is_leaf is not None:
            kwargs["is_leaf"] = is_leaf
        return eqx.tree_at(where, self, **kwargs)


@dataclass_transform(
    frozen_default=True,
    field_specifiers=(dataclasses.field, eqx.field, array, container, field),
)
class PyTree(PyTreeMixin, eqx.Module): ...


@dataclass_transform(
    frozen_default=False,
    field_specifiers=(dataclasses.field, eqx.field, array, container, field),
)
class PyTreeMutable(PyTreeMixin, eqx.Module):
    __setattr__ = object.__setattr__  # pyright: ignore[reportAssignmentType]
