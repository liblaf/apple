from collections.abc import Callable, Mapping
from typing import Any, TypedDict, Unpack

import attrs
import beartype.door
import equinox as eqx
import jax.numpy as jnp


class FieldKwargs(TypedDict, total=False):
    default: Any
    validator: "attrs._ValidatorArgType | None"
    repr: "attrs._ReprArgType"
    metadata: Mapping[Any, Any] | None
    converter: "attrs._ConverterType | list[attrs._ConverterType] | tuple[attrs._ConverterType] | None"
    factory: Callable | None
    kw_only: bool


def array(**kwargs: Unpack[FieldKwargs]) -> Any:
    kwargs.setdefault("converter", attrs.converters.optional(jnp.asarray))
    return data(**kwargs)


def container(**kwargs: Unpack[FieldKwargs]) -> Any:
    if "converter" in kwargs and "factory" not in kwargs:
        kwargs["factory"] = kwargs["converter"]  # pyright: ignore[reportGeneralTypeIssues]
    elif "converter" not in kwargs and "factory" in kwargs:
        kwargs["converter"] = kwargs["factory"]  # pyright: ignore[reportGeneralTypeIssues]
    elif "converter" not in kwargs and "factory" not in kwargs:
        kwargs["converter"] = attrs.converters.default_if_none(factory=dict)
        kwargs["factory"] = dict
    return data(**kwargs)  # pyright: ignore[reportArgumentType]


def data(**kwargs: Unpack[FieldKwargs]) -> Any:
    metadata: dict = kwargs.setdefault("metadata", {})  # pyright: ignore[reportAssignmentType]
    metadata.setdefault("static", False)
    return _field(**kwargs)


def static(**kwargs: Unpack[FieldKwargs]) -> Any:
    metadata: dict = kwargs.setdefault("metadata", {})  # pyright: ignore[reportAssignmentType]
    metadata.setdefault("static", True)
    return _field(**kwargs)


def _field(**kwargs: Unpack[FieldKwargs]) -> Any:
    kwargs.setdefault("repr", eqx.tree_pformat)
    kwargs.setdefault("validator", _validator)
    return attrs.field(**kwargs)


def _validator(_self: Any, attr: attrs.Attribute, value: Any) -> None:
    if value is None:
        return
    if attr.type is None or isinstance(attr.type, str):
        return
    beartype.door.die_if_unbearable(value, attr.type)
