import jax.numpy as jnp

from liblaf.apple.typed.jax import Mat33


def Qs(U: Mat33, V: Mat33) -> tuple[Mat33, Mat33, Mat33]:
    T0: Mat33 = jnp.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    T1: Mat33 = jnp.asarray([[0, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    T2: Mat33 = jnp.asarray([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float)
    Q0: Mat33 = U @ T0 @ V.T / jnp.sqrt(2.0)
    Q1: Mat33 = U @ T1 @ V.T / jnp.sqrt(2.0)
    Q2: Mat33 = U @ T2 @ V.T / jnp.sqrt(2.0)
    return Q0, Q1, Q2
