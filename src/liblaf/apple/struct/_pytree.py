import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, dataclass_transform, overload

import jax
import jax.numpy as jnp

from liblaf.apple.struct import converters

type Converter[T] = Callable[[], T]
type Validator[T] = Callable[[T], None]


@overload
def field[T](
    *,
    default: T,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: Mapping[Any, Any] | None = None,
    kw_only: bool = ...,
    converter: Converter[T] | None = None,
    static: bool = False,
    validator: Validator[T] | None = None,
) -> T: ...
@overload
def field[T](
    *,
    default_factory: Callable[[], T],
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: Mapping[Any, Any] | None = None,
    kw_only: bool = ...,
    converter: Converter[T] | None = None,
    static: bool = False,
    validator: Validator[T] | None = None,
) -> T: ...
@overload
def field(
    *,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: Mapping[Any, Any] | None = None,
    kw_only: bool = ...,
    converter: Converter | None = None,
    static: bool = False,
    validator: Validator | None = None,
) -> Any: ...
def field(**kwargs) -> Any:
    if static := kwargs.pop("static", False):
        kwargs.setdefault("metadata", {}).setdefault("static", static)
    if (converter := kwargs.pop("converter", None)) is not None:
        kwargs.setdefault("metadata", {}).setdefault("converter", converter)
    if (validator := kwargs.pop("validator", None)) is not None:
        kwargs.setdefault("metadata", {}).setdefault("validator", validator)
    return dataclasses.field(**kwargs)


@overload
def array[T](
    *,
    default: T,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: Mapping[Any, Any] | None = None,
    kw_only: bool = ...,
    static: bool = False,
    validator: Validator[T] | None = None,
) -> T: ...
@overload
def array[T](
    *,
    default_factory: Callable[[], T],
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: Mapping[Any, Any] | None = None,
    kw_only: bool = ...,
    static: bool = False,
    validator: Validator[T] | None = None,
) -> T: ...
@overload
def array(
    *,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: Mapping[Any, Any] | None = None,
    kw_only: bool = ...,
    static: bool = False,
    validator: Validator | None = None,
) -> Any: ...
def array(**kwargs) -> Any:
    kwargs.setdefault("converter", converters.optional(jnp.asarray))
    return field(**kwargs)


@overload
def static[T](
    *,
    default: T,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: Mapping[Any, Any] | None = None,
    kw_only: bool = ...,
    converter: Converter[T] | None = None,
    validator: Validator[T] | None = None,
) -> T: ...
@overload
def static[T](
    *,
    default_factory: Callable[[], T],
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: Mapping[Any, Any] | None = None,
    kw_only: bool = ...,
    converter: Converter[T] | None = None,
    validator: Validator[T] | None = None,
) -> T: ...
@overload
def static(
    *,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: Mapping[Any, Any] | None = None,
    kw_only: bool = ...,
    converter: Converter | None = None,
    validator: Validator | None = None,
) -> Any: ...
def static(**kwargs) -> Any:
    kwargs.setdefault("static", True)
    return field(**kwargs)


@dataclass_transform(frozen_default=True, field_specifiers=(field, array, static))
class PyTree:
    def __init_subclass__(cls, **kwargs) -> None:
        kwargs.setdefault("frozen", True)
        dataclasses.dataclass(cls, **kwargs)
        jax.tree_util.register_dataclass(cls)

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):  # pyright: ignore[reportArgumentType]
            if callable(validator := f.metadata.get("validator")):
                validator(getattr(self, f.name))

    def replace(self, **changes) -> Any:
        for f in dataclasses.fields(self):  # pyright: ignore[reportArgumentType]
            if (f.name in changes) and callable(
                converter := f.metadata.get("converter")
            ):
                changes[f.name] = converter(changes[f.name])
        return dataclasses.replace(self, **changes)  # pyright: ignore[reportArgumentType]
