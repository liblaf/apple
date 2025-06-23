import contextvars
from typing import TypedDict, cast

import jax

from ._jit import JitWrapped


class CostAnalysis(TypedDict):
    flops: float


def cost_analysis(func: JitWrapped, /, *args, **kwargs) -> CostAnalysis:
    lowered: jax.stages.Lowered = func.lower(*args, **kwargs)
    compiled: jax.stages.Compiled = lowered.compile()
    return cast("CostAnalysis", compiled.cost_analysis())


_depth: contextvars.ContextVar[int] = contextvars.ContextVar("depth", default=0)
