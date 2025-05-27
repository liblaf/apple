import jax.numpy as jnp

from liblaf.apple.typed.jax import Mat33, Vec3, Vec9


def g3(F: Mat33) -> Vec9:
    f0: Vec3
    f1: Vec3
    f2: Vec3
    f0, f1, f2 = F[:, 0], F[:, 1], F[:, 2]
    g3: Mat33 = jnp.column_stack(
        [jnp.cross(f1, f2), jnp.cross(f2, f0), jnp.cross(f0, f1)]
    )
    return jnp.ravel(g3, order="F")
