import abc
import functools
from collections.abc import Callable
from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike


def _array_method[C: Callable](func: C, /) -> C:
    @functools.wraps(func)
    def method[T: ArrayMixin](self: T, *args, **kwargs) -> T:
        arr: jax.Array = jnp.asarray(self)
        op: Callable | None = getattr(arr, func.__name__, None)
        if op is None:
            return NotImplemented
        result: jax.Array = op(*args, **kwargs)
        return self.from_values(result)

    return method  # pyright: ignore[reportReturnType]


class ArrayMixin(abc.ABC):
    values: jax.Array

    @abc.abstractmethod
    def from_values(self, values: ArrayLike, /) -> Self:
        raise NotImplementedError

    def __jax_array__(self) -> jax.Array:
        return self.values

    @property
    def dtype(self) -> jnp.dtype:
        return self.values.dtype

    @property
    def ndim(self) -> int:
        return self.values.ndim

    @property
    def size(self) -> int:
        return self.values.size

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @_array_method  # noqa: B027
    def __add__(self, other: ArrayLike) -> Self: ...
    @_array_method  # noqa: B027
    def __sub__(self, other: ArrayLike) -> Self: ...
    @_array_method  # noqa: B027
    def __mul__(self, other: ArrayLike) -> Self: ...
    @_array_method  # noqa: B027
    def __matmul__(self, other: ArrayLike) -> Self: ...
    @_array_method  # noqa: B027
    def __truediv__(self, other: ArrayLike) -> Self: ...
    @_array_method  # noqa: B027
    def __floordiv__(self, other: ArrayLike) -> Self: ...
    @_array_method  # noqa: B027
    def __mod__(self, other: ArrayLike) -> Self: ...
    @_array_method  # noqa: B027
    def __divmod__(self, other: ArrayLike) -> Self: ...
    @_array_method  # noqa: B027
    def __pow__(self, other: ArrayLike) -> Self: ...
    @_array_method  # noqa: B027
    def __lshift__(self, other: ArrayLike) -> Self: ...
    @_array_method  # noqa: B027
    def __rshift__(self, other: ArrayLike) -> Self: ...
    @_array_method  # noqa: B027
    def __and__(self, other: ArrayLike) -> Self: ...
    @_array_method  # noqa: B027
    def __xor__(self, other: ArrayLike) -> Self: ...
    @_array_method  # noqa: B027
    def __or__(self, other: ArrayLike) -> Self: ...
