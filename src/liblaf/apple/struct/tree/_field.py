from collections.abc import Callable, Mapping
from typing import Any, TypedDict, Unpack, overload

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


class FieldKwargs(TypedDict, total=False):
    compare: bool
    converter: Callable[[Any], Any] | None
    default_factory: Callable[[], Any] | None
    default: Any
    factory: Callable[[], Any] | type | None
    hash: bool | None
    init: bool
    kw_only: bool
    metadata: Mapping[Any, Any] | None
    repr: bool
    static: bool


def array(**kwargs: Unpack[FieldKwargs]) -> Any:
    kwargs.setdefault("converter", _optional_as_array)
    return field(**kwargs)


def container(**kwargs: Unpack[FieldKwargs]) -> Any:
    if "converter" in kwargs and "factory" not in kwargs:
        kwargs["factory"] = kwargs["converter"]  # pyright: ignore[reportGeneralTypeIssues]
    elif "converter" not in kwargs and "factory" in kwargs:
        kwargs["converter"] = kwargs["factory"]  # pyright: ignore[reportGeneralTypeIssues]
    elif "converter" not in kwargs and "factory" not in kwargs:
        kwargs["converter"] = _dict_if_none
        kwargs["factory"] = dict
    return field(**kwargs)  # pyright: ignore[reportArgumentType]


def field(**kwargs: Unpack[FieldKwargs]) -> Any:
    if "factory" in kwargs:
        kwargs["default_factory"] = kwargs.pop("factory")
    return eqx.field(**kwargs)


@overload
def _dict_if_none(value: None) -> dict: ...
@overload
def _dict_if_none[T](value: T) -> T: ...
def _dict_if_none(value: Any) -> Any:
    if value is None:
        return {}
    return value


@overload
def _optional_as_array(value: None) -> None: ...
@overload
def _optional_as_array(value: ArrayLike) -> Array: ...
def _optional_as_array(value: Any) -> Any:
    if value is None:
        return None
    return jnp.asarray(value)
