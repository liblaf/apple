from typing import Any, assert_never, override

import jax
import jax.numpy as jnp
from jaxtyping import Integer

from liblaf.apple.struct.pytree import static

from ._dof_map import DofMap


class DofMapSlice(DofMap):
    _slice: slice = static(default=slice(None))

    def __getitem__(self, idx: Any) -> DofMap:
        if isinstance(idx, slice):
            start: int | None = None
            if idx.start is None:
                start = self.start
            elif idx.start < 0:
                start = (self.stop or 0) + idx.start
            elif idx.start >= 0:
                start = (self.start or 0) + idx.start
            else:
                assert_never(idx.start)
            stop: int | None = None
            if idx.stop is None:
                stop = self.stop
            elif idx.stop < 0:
                stop = (self.stop or 0) + idx.stop
            elif idx.stop >= 0:
                stop = (self.start or 0) + idx.stop
            else:
                assert_never(idx.stop)
            step: int | None = (
                self.step if idx.step is None else (self.step or 1) * idx.step
            )
            return self.replace(_slice=slice(start, stop, step))
        return super().__getitem__(idx)

    @property
    @override
    def index(self) -> slice:
        return self._slice

    @property
    @override
    def integers(self) -> Integer[jax.Array, "..."]:
        with jax.ensure_compile_time_eval():
            assert self.stop is not None
            assert self.stop > 0
            return jnp.arange(self.start or 0, self.stop, self.step)

    @property
    def start(self) -> int | None:
        return self._slice.start

    @property
    def stop(self) -> int | None:
        return self._slice.stop

    @property
    def step(self) -> int | None:
        return self._slice.step
