import abc
from collections.abc import Callable
from functools import partialmethod
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike


class ArrayMixin:
    @abc.abstractmethod
    def __jax_array__(self) -> jax.Array:
        raise NotImplementedError

    @abc.abstractmethod
    def from_values(self, values: ArrayLike, /) -> Self:
        raise NotImplementedError

    def _op(self, op: str, /, *args, **kwargs) -> Self:
        values: jax.Array = jnp.asarray(self)
        fn: Callable | None = getattr(values, op, None)
        if fn is None:
            return NotImplemented
        result: jax.Array = fn(*args, **kwargs)
        if eqx.is_array(result):
            return self.from_values(result)
        return NotImplemented

    __add__: partialmethod[Self] = partialmethod(_op, "__add__")
    __sub__: partialmethod[Self] = partialmethod(_op, "__sub__")
    __mul__: partialmethod[Self] = partialmethod(_op, "__mul__")
    __matmul__: partialmethod[Self] = partialmethod(_op, "__matmul__")
    __truediv__: partialmethod[Self] = partialmethod(_op, "__truediv__")
    __floordiv__: partialmethod[Self] = partialmethod(_op, "__floordiv__")
    __mod__: partialmethod[Self] = partialmethod(_op, "__mod__")
    __divmod__: partialmethod[Self] = partialmethod(_op, "__divmod__")
    __pow__: partialmethod[Self] = partialmethod(_op, "__pow__")
    __lshift__: partialmethod[Self] = partialmethod(_op, "__lshift__")
    __rshift__: partialmethod[Self] = partialmethod(_op, "__rshift__")
    __and__: partialmethod[Self] = partialmethod(_op, "__and__")
    __xor__: partialmethod[Self] = partialmethod(_op, "__xor__")
    __or__: partialmethod[Self] = partialmethod(_op, "__or__")

    __radd__: partialmethod[Self] = partialmethod(_op, "__radd__")
    __rsub__: partialmethod[Self] = partialmethod(_op, "__rsub__")
    __rmul__: partialmethod[Self] = partialmethod(_op, "__rmul__")
    __rmatmul__: partialmethod[Self] = partialmethod(_op, "__rmatmul__")
    __rtruediv__: partialmethod[Self] = partialmethod(_op, "__rtruediv__")
    __rfloordiv__: partialmethod[Self] = partialmethod(_op, "__rfloordiv__")
    __rmod__: partialmethod[Self] = partialmethod(_op, "__rmod__")
    __rdivmod__: partialmethod[Self] = partialmethod(_op, "__rdivmod__")
    __rpow__: partialmethod[Self] = partialmethod(_op, "__rpow__")
    __rlshift__: partialmethod[Self] = partialmethod(_op, "__rlshift__")
    __rrshift__: partialmethod[Self] = partialmethod(_op, "__rrshift__")
    __rand__: partialmethod[Self] = partialmethod(_op, "__rand__")
    __rxor__: partialmethod[Self] = partialmethod(_op, "__rxor__")
    __ror__: partialmethod[Self] = partialmethod(_op, "__ror__")

    __neg__: partialmethod[Self] = partialmethod(_op, "__neg__")
    __pos__: partialmethod[Self] = partialmethod(_op, "__pos__")
    __abs__: partialmethod[Self] = partialmethod(_op, "__abs__")
    __invert__: partialmethod[Self] = partialmethod(_op, "__invert__")
