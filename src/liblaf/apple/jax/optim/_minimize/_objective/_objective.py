from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Self

import attrs
import cytoolz as toolz
import equinox as eqx
from jaxtyping import Array, PyTree

from liblaf import grapes
from liblaf.apple.jax import tree

FUNCTION_NAMES: list[str] = [
    "fun",
    "jac",
    "hess",
    "hessp",
    "hess_diag",
    "hess_quad",
    "fun_and_jac",
    "jac_and_hess_diag",
]


@tree.pytree
class Objective:
    fun: Callable | None = None
    jac: Callable | None = None
    hess: Callable | None = None
    hessp: Callable | None = None
    hess_diag: Callable | None = None
    hess_quad: Callable | None = None
    fun_and_jac: Callable | None = None
    jac_and_hess_diag: Callable | None = None

    def flatten(self, unflatten: Callable[[Array], PyTree]) -> Self:
        def flatten(
            fn: Callable | None,
            *,
            arg_nums: Iterable[int] = (0,),
            multiple_outputs: bool = False,
        ) -> Callable | None:
            if fn is None:
                return None

            @grapes.decorator
            def wrapper(
                wrapped: Callable,
                _instance: None,
                args: Sequence[Any],
                kwargs: Mapping[str, Any],
            ) -> Any:
                args = list(args)
                for i in arg_nums:
                    args[i] = unflatten(args[i])
                outputs: Any = wrapped(*args, **kwargs)
                if not multiple_outputs:
                    return tree.flatten(outputs)[0]
                outputs = tuple(tree.flatten(r)[0] for r in outputs)
                return outputs

            return wrapper(fn)

        updates: dict[str, Callable | None] = {}
        updates["fun"] = flatten(self.fun, arg_nums=(0,))
        updates["jac"] = flatten(self.jac, arg_nums=(0,))
        if callable(self.hess):
            raise NotImplementedError
        updates["hessp"] = flatten(self.hessp, arg_nums=(0, 1))
        updates["hess_diag"] = flatten(self.hess_diag, arg_nums=(0,))
        updates["hess_quad"] = flatten(self.hess_quad, arg_nums=(0, 1))
        updates["fun_and_jac"] = flatten(
            self.fun_and_jac, arg_nums=(0,), multiple_outputs=True
        )
        updates["jac_and_hess_diag"] = flatten(
            self.jac_and_hess_diag, arg_nums=(0,), multiple_outputs=True
        )
        updates = toolz.valfilter(lambda fn: fn is not None, updates)
        return attrs.evolve(self, **updates)

    def jit(self) -> Self:
        updates: dict[str, Callable] = {}
        for name in FUNCTION_NAMES:
            fn: Callable | None = getattr(self, name)
            if not callable(fn):
                continue
            updates[name] = eqx.filter_jit(fn)
        return attrs.evolve(self, **updates)

    def partial(self, args: Sequence[Any] = (), kwargs: Mapping[str, Any] = {}) -> Self:
        updates: dict[str, Callable] = {}

        def partial(
            fn: Callable, args: Sequence[Any], kwargs: Mapping[str, Any]
        ) -> Callable:
            partial_args: Sequence[Any] = args
            partial_kwargs: Mapping[str, Any] = kwargs

            @grapes.decorator
            def wrapper(
                wrapped: Callable, _instance: None, args: tuple, kwargs: dict
            ) -> Any:
                return wrapped(*args, *partial_args, **partial_kwargs, **kwargs)

            return wrapper(fn)

        for name in FUNCTION_NAMES:
            fn: Callable | None = getattr(self, name)
            if not callable(fn):
                continue
            updates[name] = partial(fn, args, kwargs)
        return attrs.evolve(self, **updates)

    def timer(self) -> Self:
        updates: dict[str, Callable] = {}
        for name in FUNCTION_NAMES:
            fn: Callable | None = getattr(self, name)
            if not callable(fn):
                continue
            updates[name] = grapes.timer(fn, name=f"{name}()")
        return attrs.evolve(self, **updates)
