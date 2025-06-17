import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, Self, TypedDict, Unpack, dataclass_transform, overload

import equinox as eqx
import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike


class FieldKwargs(TypedDict, total=False):
    default: Any
    default_factory: Callable[[], Any]
    factory: Callable[[], Any]
    converter: Callable[[Any], Any]
    static: bool
    # dataclasses.field
    init: bool
    repr: bool
    hash: bool | None
    compare: bool
    metadata: Mapping[Any, Any] | None
    kw_only: bool


def array(**kwargs: Unpack[FieldKwargs]) -> Any:
    kwargs.setdefault("converter", _as_array_optional)
    return field(**kwargs)


def field(**kwargs: Unpack[FieldKwargs]) -> Any:
    if "factory" in kwargs:
        kwargs["default_factory"] = kwargs.pop("factory")
    return eqx.field(**kwargs)


def pytree_dict(**kwargs: Unpack[FieldKwargs]) -> Any:
    from .mapping import PyTreeDict

    kwargs.setdefault("converter", PyTreeDict)
    kwargs.setdefault("factory", PyTreeDict)
    return field(**kwargs)


def static(**kwargs: Unpack[FieldKwargs]) -> Any:
    kwargs["static"] = True
    return field(**kwargs)


@dataclass_transform(
    eq_default=False,
    frozen_default=True,
    field_specifiers=(
        dataclasses.field,
        eqx.field,
        eqx.static_field,
        array,
        field,
        static,
    ),
)
class PyTree(eqx.Module):
    def __post_init__(self) -> None: ...

    def replace(self, **changes) -> Self:
        return dataclasses.replace(self, **changes)


@overload
def _as_array_optional(value: None) -> None: ...
@overload
def _as_array_optional(value: ArrayLike) -> jax.Array: ...
def _as_array_optional(value: ArrayLike | None) -> jax.Array | None:
    if value is None:
        return None
    return jnp.asarray(value)
