import functools
from collections.abc import Callable
from typing import Any, dataclass_transform, overload

import attrs

from liblaf import grapes

from ._register_attrs import register_attrs

type OnSetAttrArgType = attrs._OnSetAttrArgType  # noqa: SLF001
type FieldTransformer = attrs._FieldTransformer  # noqa: SLF001


@overload
@dataclass_transform(field_specifiers=(attrs.field,))
def pytree[C: type](
    cls: C,
    /,
    *,
    these: dict[str, Any] | None = ...,
    repr: bool = ...,
    unsafe_hash: bool | None = ...,
    hash: bool | None = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: bool | None = ...,
    order: bool | None = ...,
    auto_detect: bool = ...,
    getstate_setstate: bool | None = ...,
    on_setattr: OnSetAttrArgType | None = ...,
    field_transformer: FieldTransformer | None = ...,
    match_args: bool = ...,
) -> C: ...
@overload
@dataclass_transform(field_specifiers=(attrs.field,))
def pytree[C: type](
    *,
    these: dict[str, Any] | None = ...,
    repr: bool = ...,
    unsafe_hash: bool | None = ...,
    hash: bool | None = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: bool | None = ...,
    order: bool | None = ...,
    auto_detect: bool = ...,
    getstate_setstate: bool | None = ...,
    on_setattr: OnSetAttrArgType | None = ...,
    field_transformer: FieldTransformer | None = ...,
    match_args: bool = ...,
) -> Callable[[C], C]: ...
def pytree[C: type](cls: type | None = None, /, **kwargs) -> Any:
    if cls is None:
        return functools.partial(pytree, **kwargs)
    kwargs.setdefault("repr", False)
    cls = attrs.define(cls, **kwargs)
    cls = register_attrs(cls)
    cls = grapes.wadler_lindig(cls)
    return cls
