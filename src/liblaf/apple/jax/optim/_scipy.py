from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import attrs
import scipy.optimize
from jaxtyping import Array, PyTree

from liblaf import grapes
from liblaf.apple import struct
from liblaf.apple.jax import tree

from ._minimizer import Minimizer
from ._solution import Solution


@struct.pytree
class MinimizerScipy(Minimizer):
    method: str = struct.field(default="trust-constr")

    def minimize(
        self,
        x0: PyTree,
        *,
        fun: Callable,
        jac: Callable,
        hessp: Callable,
        args: Iterable[Any] = (),
        kwargs: Mapping[str, Any] = {},
        callback: Callable | None = None,
    ) -> Solution:
        x0_flat: Array
        unflatten: Callable[[Array], PyTree]
        x0_flat, unflatten = tree.flatten(x0)
        wrapper = _ProblemWrapper(args, kwargs, unflatten)
        fun = wrapper.wraps(fun, unflatten_args=(0,))
        jac = wrapper.wraps(jac, unflatten_args=(0,))
        hessp = wrapper.wraps(hessp, unflatten_args=(0, 1))
        result: scipy.optimize.OptimizeResult = scipy.optimize.minimize(
            fun=fun,
            x0=x0_flat,
            method=self.method,
            jac=jac,
            hessp=hessp,
            callback=callback,
            options={"verbose": 3},
        )
        return Solution(result)


@attrs.define
class _ProblemWrapper:
    args: Iterable[Any]
    kwargs: Mapping[str, Any]
    unflatten: Callable[[Array], PyTree]

    def wraps(self, fn: Callable, unflatten_args: Sequence[int] = (0,)) -> Callable:
        @grapes.decorator
        def wrapper(
            wrapped: Callable, _instance: None, args: tuple, kwargs: dict
        ) -> Array:
            args: list = list(args)
            for i in unflatten_args:
                args[i] = self.unflatten(args[i])
            result = wrapped(*args, *self.args, **kwargs, **self.kwargs)
            result_flat, _ = tree.flatten(result)
            return result_flat

        return wrapper(fn)
