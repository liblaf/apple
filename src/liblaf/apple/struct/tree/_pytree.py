import functools
from collections.abc import Callable
from typing import Any, dataclass_transform, overload

import attrs

from liblaf import grapes

from ._register_attrs import register_attrs


@overload
@dataclass_transform(field_specifiers=(attrs.field,))
def pytree[C: type](
    maybe_cls: C,
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
    on_setattr: attrs._OnSetAttrArgType | None = ...,
    field_transformer: attrs._FieldTransformer | None = ...,
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
    on_setattr: attrs._OnSetAttrArgType | None = ...,
    field_transformer: attrs._FieldTransformer | None = ...,
    match_args: bool = ...,
) -> Callable[[C], C]: ...
def pytree[C: type](maybe_cls: type | None = None, **kwargs) -> Any:
    if maybe_cls is None:
        return functools.partial(pytree, **kwargs)
    kwargs.setdefault("repr", False)
    maybe_cls = attrs.define(maybe_cls, **kwargs)
    maybe_cls = register_attrs(maybe_cls)
    maybe_cls = grapes.wadler_lindig(maybe_cls)
    return maybe_cls
