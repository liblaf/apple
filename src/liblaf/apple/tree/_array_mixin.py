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
        op: Callable | None = getattr(jnp, func.__name__, None)
        if op is None:
            return NotImplemented
        result: jax.Array = op(arr, *args, **kwargs)
        return self.from_values(result)

    return method  # pyright: ignore[reportReturnType]


class ArrayMixin:
    @abc.abstractmethod
    def __jax_array__(self) -> jax.Array:
        raise NotImplementedError

    @abc.abstractmethod
    def from_values(self, values: ArrayLike) -> Self:
        raise NotImplementedError

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.asarray(self).dtype

    @property
    def ndim(self) -> int:
        return jnp.asarray(self).ndim

    @property
    def size(self) -> int:
        return jnp.asarray(self).size

    @property
    def shape(self) -> tuple[int, ...]:
        return jnp.asarray(self).shape

    @_array_method
    def __add__(self, other: ArrayLike) -> Self: ...
    @_array_method
    def __sub__(self, other: ArrayLike) -> Self: ...
    @_array_method
    def __mul__(self, other: ArrayLike) -> Self: ...
    @_array_method
    def __matmul__(self, other: ArrayLike) -> Self: ...
    @_array_method
    def __truediv__(self, other: ArrayLike) -> Self: ...
    @_array_method
    def __floordiv__(self, other: ArrayLike) -> Self: ...
    @_array_method
    def __mod__(self, other: ArrayLike) -> Self: ...
    @_array_method
    def __divmod__(self, other: ArrayLike) -> Self: ...
    @_array_method
    def __pow__(self, other: ArrayLike) -> Self: ...
    @_array_method
    def __lshift__(self, other: ArrayLike) -> Self: ...
    @_array_method
    def __rshift__(self, other: ArrayLike) -> Self: ...
    @_array_method
    def __and__(self, other: ArrayLike) -> Self: ...
    @_array_method
    def __xor__(self, other: ArrayLike) -> Self: ...
    @_array_method
    def __or__(self, other: ArrayLike) -> Self: ...
