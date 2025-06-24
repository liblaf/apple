from collections.abc import Callable
from typing import Any, Self

import attrs

from liblaf.apple import utils

from .base import BaseProblem


class JitMixin(BaseProblem):
    def jit(self) -> Self:
        changes: dict[str, Any] = {}
        for f in attrs.fields(type(self)):
            f: attrs.Attribute
            if not f.metadata.get("jit", True):
                continue
            v: Callable | None = getattr(self, f.name, None)
            if v is None:
                continue
            v = utils.jit(v)
            changes[f.name] = v
        return attrs.evolve(self, **changes)
