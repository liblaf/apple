import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@jax.jit
def make_activation(activation: Float[Array, "cells 6"]) -> Float[Array, "cells 3 3"]:
    A: Float[Array, "cells 3 3"] = jnp.zeros((activation.shape[0], 3, 3))
    A = A.at[:, 0, 0].set(1.0 + activation[:, 0])
    A = A.at[:, 1, 1].set(1.0 + activation[:, 1])
    A = A.at[:, 2, 2].set(1.0 + activation[:, 2])
    A = A.at[:, 0, 1].set(activation[:, 3])
    A = A.at[:, 1, 0].set(activation[:, 3])
    A = A.at[:, 0, 2].set(activation[:, 4])
    A = A.at[:, 2, 0].set(activation[:, 4])
    A = A.at[:, 1, 2].set(activation[:, 5])
    A = A.at[:, 2, 1].set(activation[:, 5])
    return A
