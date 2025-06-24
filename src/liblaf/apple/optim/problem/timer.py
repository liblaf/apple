from collections.abc import Callable
from typing import Any, Self

import attrs

from liblaf import grapes

from .base import BaseProblem
from .utils import implemented


class TimerMixin(BaseProblem):
    def timer(self) -> Self:
        changes: dict[str, Any] = {}
        for f in attrs.fields(type(self)):
            f: attrs.Attribute
            v: Callable | None = getattr(self, f.name, None)
            if not implemented(v):
                continue
            v = grapes.timer(v, label=f"{f.name}()")
            changes[f.name] = v
        return attrs.evolve(self, **changes)
