from collections.abc import Mapping

import jax.numpy as jnp
from jaxtyping import PyTree


def as_jax(tree: PyTree | None) -> PyTree:
    if tree is None:
        return None
    if isinstance(tree, Mapping):
        return {k: as_jax(v) for k, v in tree.items()}
    return jnp.asarray(tree)
