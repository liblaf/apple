import dataclasses
from collections.abc import Callable
from typing import Any, Self

from liblaf import grapes

from .base import BaseProblem
from .utils import implemented


class TimerMixin(BaseProblem):
    def timer(self) -> Self:
        changes: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            f: dataclasses.Field
            v: Callable | None = getattr(self, f.name, None)
            if not implemented(v):
                continue
            v = grapes.timer(v, name=f"{f.name}()")
            changes[f.name] = v
        return self.replace(**changes)
