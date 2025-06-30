import contextvars
from typing import TypedDict, cast

import jax

from ._jit import JitWrapped


class CostAnalysis(TypedDict):
    flops: float


def cost_analysis(func: JitWrapped, /, *args, **kwargs) -> CostAnalysis:
    lowered: jax.stages.Lowered = func.lower(*args, **kwargs)
    compiled: jax.stages.Compiled = lowered.compile()
    if hasattr(compiled, "compiled"):  # workaround for equinox.filter_jit
        compiled = compiled.compiled  # pyright: ignore[reportAttributeAccessIssue]
    return cast("CostAnalysis", compiled.cost_analysis())


_depth: contextvars.ContextVar[int] = contextvars.ContextVar("depth", default=0)
