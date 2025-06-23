import functools
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypedDict, Unpack, cast, overload

import beartype
import jax
from jaxtyping import jaxtyped


class JitKwargs(TypedDict, total=False):
    static_argnums: int | Sequence[int] | None
    static_argnames: str | Iterable[str] | None
    inline: bool
    validate: bool | None


class JitWrapped[**P, T]:
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...
    def eval_shape(self, *args: P.args, **kwargs: P.kwargs) -> jax.ShapeDtypeStruct: ...
    def lower(self, *args: P.args, **kwargs: P.kwargs) -> jax.stages.Lowered: ...
    def trace(self, *args: P.args, **kwargs: P.kwargs) -> jax.stages.Traced: ...


@overload
def jit[**P, T](
    func: Callable[P, T], /, **kwargs: Unpack[JitKwargs]
) -> JitWrapped[P, T]: ...
@overload
def jit[**P, T](
    **kwargs: Unpack[JitKwargs],
) -> Callable[[Callable[P, T]], JitWrapped[P, T]]: ...
def jit(func: Callable | None = None, /, **kwargs) -> Any:
    if func is None:
        return functools.partial(jit, **kwargs)
    validate: bool = kwargs.pop("validate", True)
    if validate:
        func = jaxtyped(func, typechecker=beartype.beartype)
    return cast("JitWrapped", jax.jit(func, **kwargs))


@overload
def jit_method[C: Callable](func: C, /, **kwargs: Unpack[JitKwargs]) -> C: ...
@overload
def jit_method[C: Callable](**kwargs: Unpack[JitKwargs]) -> Callable[[C], C]: ...
def jit_method(*args, **kwargs) -> Any:
    return jit(*args, **kwargs)
