import abc
from functools import partialmethod
from typing import Any, Self

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike


class ArrayMixin(abc.ABC):
    @abc.abstractmethod
    def __jax_array__(self) -> jax.Array:
        raise NotImplementedError

    @abc.abstractmethod
    def with_values(self, values: ArrayLike) -> Self:
        raise NotImplementedError

    def _op(self, op: str, /, *args, **kwargs) -> Self:
        values: jax.Array = jnp.asarray(self)
        attr = getattr(values, op, None)
        if attr is None:
            return NotImplemented
        values = attr(*args, **kwargs)
        return self.with_values(values)

    def __getitem__(self, index: Any) -> jax.Array:
        return jnp.asarray(self)[index]

    __add__: partialmethod[jax.Array] = partialmethod(_op, "__add__")
    __sub__: partialmethod[jax.Array] = partialmethod(_op, "__sub__")
    __mul__: partialmethod[jax.Array] = partialmethod(_op, "__mul__")
    __matmul__: partialmethod[jax.Array] = partialmethod(_op, "__matmul__")
    __truediv__: partialmethod[jax.Array] = partialmethod(_op, "__add__")
    __floordiv__: partialmethod[jax.Array] = partialmethod(_op, "__floordiv__")
    __mod__: partialmethod[jax.Array] = partialmethod(_op, "__mod__")
    __divmod__: partialmethod[jax.Array] = partialmethod(_op, "__add__")
    __pow__: partialmethod[jax.Array] = partialmethod(_op, "__pow__")
    __lshift__: partialmethod[jax.Array] = partialmethod(_op, "__lshift__")
    __rshift__: partialmethod[jax.Array] = partialmethod(_op, "__rshift__")
    __and__: partialmethod[jax.Array] = partialmethod(_op, "__and__")
    __xor__: partialmethod[jax.Array] = partialmethod(_op, "__xor__")
    __or__: partialmethod[jax.Array] = partialmethod(_op, "__or__")

    __radd__: partialmethod[jax.Array] = partialmethod(_op, "__radd__")
    __rsub__: partialmethod[jax.Array] = partialmethod(_op, "__rsub__")
    __rmul__: partialmethod[jax.Array] = partialmethod(_op, "__rmul__")
    __rmatmul__: partialmethod[jax.Array] = partialmethod(_op, "__rmatmul__")
    __rtruediv__: partialmethod[jax.Array] = partialmethod(_op, "__rtruediv__")
    __rfloordiv__: partialmethod[jax.Array] = partialmethod(_op, "__rfloordiv__")
    __rmod__: partialmethod[jax.Array] = partialmethod(_op, "__rmod__")
    __rdivmod__: partialmethod[jax.Array] = partialmethod(_op, "__rdivmod__")
    __rpow__: partialmethod[jax.Array] = partialmethod(_op, "__rpow__")
    __rlshift__: partialmethod[jax.Array] = partialmethod(_op, "__rlshift__")
    __rrshift__: partialmethod[jax.Array] = partialmethod(_op, "__rrshift__")
    __rand__: partialmethod[jax.Array] = partialmethod(_op, "__rand__")
    __rxor__: partialmethod[jax.Array] = partialmethod(_op, "__rxor__")
    __ror__: partialmethod[jax.Array] = partialmethod(_op, "__ror__")

    __iadd__: partialmethod[jax.Array] = partialmethod(_op, "__iadd__")
    __isub__: partialmethod[jax.Array] = partialmethod(_op, "__isub__")
    __imul__: partialmethod[jax.Array] = partialmethod(_op, "__imul__")
    __imatmul__: partialmethod[jax.Array] = partialmethod(_op, "__imatmul__")
    __itruediv__: partialmethod[jax.Array] = partialmethod(_op, "__itruediv__")
    __ifloordiv__: partialmethod[jax.Array] = partialmethod(_op, "__ifloordiv__")
    __imod__: partialmethod[jax.Array] = partialmethod(_op, "__imod__")
    __ipow__: partialmethod[jax.Array] = partialmethod(_op, "__ipow__")
    __ilshift__: partialmethod[jax.Array] = partialmethod(_op, "__ilshift__")
    __irshift__: partialmethod[jax.Array] = partialmethod(_op, "__irshift__")
    __iand__: partialmethod[jax.Array] = partialmethod(_op, "__iand__")
    __ixor__: partialmethod[jax.Array] = partialmethod(_op, "__ixor__")
    __ior__: partialmethod[jax.Array] = partialmethod(_op, "__ior__")

    __neg__: partialmethod[jax.Array] = partialmethod(_op, "__neg__")
    __pos__: partialmethod[jax.Array] = partialmethod(_op, "__pos__")
    __abs__: partialmethod[jax.Array] = partialmethod(_op, "__abs__")
    __invert__: partialmethod[jax.Array] = partialmethod(_op, "__invert__")

    __complex__: partialmethod[jax.Array] = partialmethod(_op, "__complex__")
    __int__: partialmethod[jax.Array] = partialmethod(_op, "__int__")
    __float__: partialmethod[jax.Array] = partialmethod(_op, "__float__")

    __index__: partialmethod[jax.Array] = partialmethod(_op, "__index__")

    __round__: partialmethod[jax.Array] = partialmethod(_op, "__round__")
    __trunc__: partialmethod[jax.Array] = partialmethod(_op, "__trunc__")
    __floor__: partialmethod[jax.Array] = partialmethod(_op, "__floor__")
    __ceil__: partialmethod[jax.Array] = partialmethod(_op, "__ceil__")
