import abc
from functools import partialmethod
from typing import Self

import jax
import jax.numpy as jnp


class ArrayMixin(abc.ABC):
    @abc.abstractmethod
    def __jax_array__(self) -> jax.Array:
        raise NotImplementedError

    @abc.abstractmethod
    def with_values(self, values: jax.Array) -> Self:
        raise NotImplementedError

    def _op(self, op: str, /, *args, **kwargs) -> Self:
        values: jax.Array = jnp.asarray(self)
        values = getattr(values, op)(*args, **kwargs)
        return self.with_values(values=values)

    __add__: partialmethod[Self] = partialmethod(_op, "__add__")
    __sub__: partialmethod[Self] = partialmethod(_op, "__sub__")
    __mul__: partialmethod[Self] = partialmethod(_op, "__mul__")
    __matmul__: partialmethod[Self] = partialmethod(_op, "__matmul__")
    __truediv__: partialmethod[Self] = partialmethod(_op, "__add__")
    __floordiv__: partialmethod[Self] = partialmethod(_op, "__floordiv__")
    __mod__: partialmethod[Self] = partialmethod(_op, "__mod__")
    __divmod__: partialmethod[Self] = partialmethod(_op, "__add__")
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

    __iadd__: partialmethod[Self] = partialmethod(_op, "__iadd__")
    __isub__: partialmethod[Self] = partialmethod(_op, "__isub__")
    __imul__: partialmethod[Self] = partialmethod(_op, "__imul__")
    __imatmul__: partialmethod[Self] = partialmethod(_op, "__imatmul__")
    __itruediv__: partialmethod[Self] = partialmethod(_op, "__itruediv__")
    __ifloordiv__: partialmethod[Self] = partialmethod(_op, "__ifloordiv__")
    __imod__: partialmethod[Self] = partialmethod(_op, "__imod__")
    __ipow__: partialmethod[Self] = partialmethod(_op, "__ipow__")
    __ilshift__: partialmethod[Self] = partialmethod(_op, "__ilshift__")
    __irshift__: partialmethod[Self] = partialmethod(_op, "__irshift__")
    __iand__: partialmethod[Self] = partialmethod(_op, "__iand__")
    __ixor__: partialmethod[Self] = partialmethod(_op, "__ixor__")
    __ior__: partialmethod[Self] = partialmethod(_op, "__ior__")

    __neg__: partialmethod[Self] = partialmethod(_op, "__neg__")
    __pos__: partialmethod[Self] = partialmethod(_op, "__pos__")
    __abs__: partialmethod[Self] = partialmethod(_op, "__abs__")
    __invert__: partialmethod[Self] = partialmethod(_op, "__invert__")

    __complex__: partialmethod[Self] = partialmethod(_op, "__complex__")
    __int__: partialmethod[Self] = partialmethod(_op, "__int__")
    __float__: partialmethod[Self] = partialmethod(_op, "__float__")

    __index__: partialmethod[Self] = partialmethod(_op, "__index__")

    __round__: partialmethod[Self] = partialmethod(_op, "__round__")
    __trunc__: partialmethod[Self] = partialmethod(_op, "__trunc__")
    __floor__: partialmethod[Self] = partialmethod(_op, "__floor__")
    __ceil__: partialmethod[Self] = partialmethod(_op, "__ceil__")
