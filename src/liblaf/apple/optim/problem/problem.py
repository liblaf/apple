from collections.abc import Callable, MutableMapping

import attrs

from liblaf import grapes

from .autodiff import AutodiffMixin
from .implement import ImplementMixin
from .timer import TimerMixin


@attrs.frozen
class OptimizationProblem(AutodiffMixin, ImplementMixin, TimerMixin):
    fun: Callable | None = attrs.field(default=None)
    jac: Callable | None = attrs.field(default=None)
    hess: Callable | None = attrs.field(default=None)
    hessp: Callable | None = attrs.field(default=None)
    hess_diag: Callable | None = attrs.field(default=None)
    hess_quad: Callable | None = attrs.field(default=None)
    fun_and_jac: Callable | None = attrs.field(default=None)
    jac_and_hess_diag: Callable | None = attrs.field(default=None)
    callback: Callable | None = attrs.field(default=None, metadata={"key": "n_iter"})

    def update_result[T: MutableMapping](self, result: T) -> T:
        for f in attrs.fields(type(self)):
            f: attrs.Attribute
            func: Callable | None = getattr(self, f.name, None)
            if not isinstance(func, grapes.TimedFunction):
                continue
            func.timing.finish()
            key: str = f.metadata.get("key", f"n_{f.name}")
            if key not in result:
                result[key] = func.timing.height
        return result
