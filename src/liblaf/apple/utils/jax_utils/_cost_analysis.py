import functools
from collections.abc import Callable
from typing import TypedDict, cast

import jax
from equinox import _compile_utils, _jit

from liblaf import grapes


class CostAnalysis(TypedDict):
    flops: float


def cost_analysis[**P, T](
    func: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> CostAnalysis:
    compiled: jax.stages.Compiled = _compile(func, *args, **kwargs)
    return cast("CostAnalysis", compiled.cost_analysis())


@functools.singledispatch
def _compile(*args, **kwargs) -> jax.stages.Compiled:
    raise grapes.error.DispatchLookupError(_compile, args, kwargs)


@_compile.register
def _compile_jax(func: jax.stages.Wrapped, /, *args, **kwargs) -> jax.stages.Compiled:
    lowered: jax.stages.Lowered = func.lower(*args, **kwargs)
    compiled: jax.stages.Compiled = lowered.compile()
    return compiled


@_compile.register
def _compile_equinox(func: _jit._JitWrapper, /, *args, **kwargs) -> jax.stages.Compiled:
    lowered: _compile_utils.Lowered = func.lower(*args, **kwargs)
    compiled: _compile_utils.Compiled = lowered.compile()
    return compiled.compiled
