import dataclasses
from collections.abc import Callable, MutableMapping

from liblaf import grapes

from .autodiff import AutodiffMixin
from .implement import ImplementMixin
from .jit import JitMixin
from .timer import TimerMixin


class OptimizationProblem(AutodiffMixin, ImplementMixin, JitMixin, TimerMixin):
    def update_result[T: MutableMapping](self, result: T) -> T:
        for f in dataclasses.fields(self):
            f: dataclasses.Field
            func: Callable | None = getattr(self, f.name, None)
            try:
                timer: grapes.BaseTimer = grapes.get_timer(func)
            except AttributeError:
                continue
            if len(timer) > 0:
                timer.finish()
            key: str = f.metadata.get("counter_name", f"n_{f.name}")
            if key not in result:
                result[key] = len(timer)
        return result
