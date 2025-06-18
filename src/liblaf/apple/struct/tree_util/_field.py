from typing import Any

import attrs
import beartype
import beartype.door
import equinox as eqx
import jax.numpy as jnp

from liblaf.apple.struct._utils import as_dict


def array(**kwargs) -> Any:
    kwargs.setdefault("converter", attrs.converters.optional(jnp.asarray))
    return data(**kwargs)


def data(**kwargs) -> Any:
    metadata: dict = kwargs.setdefault("metadata", {})
    metadata.setdefault("static", False)
    return _field(**kwargs)


def mapping(**kwargs) -> Any:
    if "converter" in kwargs and "factory" not in kwargs:
        kwargs["factory"] = kwargs["converter"]
    elif "converter" not in kwargs and "factory" in kwargs:
        kwargs["converter"] = kwargs["factory"]
    elif "converter" not in kwargs and "factory" not in kwargs:
        kwargs["converter"] = as_dict
        kwargs["factory"] = dict
    return data(**kwargs)


def static(**kwargs) -> Any:
    metadata: dict = kwargs.setdefault("metadata", {})
    metadata.setdefault("static", True)
    return _field(**kwargs)


def _field(**kwargs) -> Any:
    kwargs.setdefault("repr", eqx.tree_pformat)
    kwargs.setdefault("validator", _validator)
    return attrs.field(**kwargs)


def _validator(_self: Any, attr: attrs.Attribute, value: Any) -> None:
    if value is None:
        return
    if attr.type is None or isinstance(attr.type, str):
        return
    beartype.door.die_if_unbearable(value, attr.type)
