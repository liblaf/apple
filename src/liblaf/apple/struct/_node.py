import collections
from typing import Any, Self

import attrs

from ._pytree import PyTree, class_var, data, static

counter = collections.Counter()


def uniq_id(self: "Node") -> str:
    prefix: str = type(self).__qualname__
    id_: str = f"{prefix}-{counter[prefix]:03d}"
    counter[prefix] += 1
    if self.ref is not None:
        id_ += f" -> {self.ref.id}"
    return id_


def validate_ref(self: "Node", attribute: attrs.Attribute, value: Any) -> None:
    validator = (
        attrs.validators.instance_of(Node)
        if self.is_view
        else attrs.validators.instance_of(type(None))
    )
    validator(self, attribute, value)


class Node[R: "Node"](PyTree):
    is_view: bool = class_var(default=False, init=False)
    ref: R = data(default=None, validator=validate_ref)
    id: str = static(default=attrs.Factory(uniq_id, takes_self=True))

    def with_ref(self, ref: R, /) -> Self:
        assert self.ref.id == ref.id
        return self.evolve(ref=ref)
