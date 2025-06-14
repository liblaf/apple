from typing import Any

import attrs
import jax.numpy as jnp

from ._utils import clone_signature


@clone_signature(attrs.field)
def array(**kwargs) -> Any:
    kwargs.setdefault("converter", attrs.converters.optional(jnp.asarray))
    return data(**kwargs)


@clone_signature(attrs.field)
def data(**kwargs) -> Any:
    metadata: dict[Any, Any] = kwargs.setdefault("metadata", {})
    metadata.setdefault("static", False)
    return attrs.field(**kwargs)


@clone_signature(attrs.field)
def static(**kwargs) -> Any:
    metadata: dict[Any, Any] = kwargs.setdefault("metadata", {})
    metadata.setdefault("static", True)
    return attrs.field(**kwargs)
