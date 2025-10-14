from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree


def tree_dot(a: PyTree, b: PyTree) -> Float[Array, ""]:
    a_leaves: list[Any] = jax.tree.leaves(a)
    b_leaves: list[Any] = jax.tree.leaves(b)
    outputs: list[Float[Array, ""]] = []
    for a_leaf, b_leaf in zip(a_leaves, b_leaves, strict=True):
        if eqx.is_array(a_leaf):
            outputs.append(jnp.vdot(a_leaf, b_leaf))
    return jnp.sum(jnp.asarray(outputs))
