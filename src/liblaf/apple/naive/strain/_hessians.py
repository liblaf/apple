import jax
import jax.numpy as jnp

from liblaf.apple.typed.jax import Mat12x12, Mat33, Mat99, Vec3, Vec9


def H1(lambdas: Vec3, Q0: Mat33, Q1: Mat33, Q2: Mat33) -> Mat99:
    q0: Vec9 = Q0.ravel(order="F")
    q1: Vec9 = Q1.ravel(order="F")
    q2: Vec9 = Q2.ravel(order="F")
    return (
        lambdas[0] * jnp.outer(q0, q0)
        + lambdas[1] * jnp.outer(q1, q1)
        + lambdas[2] * jnp.outer(q2, q2)
    )


def H2() -> Mat99:
    return 2.0 * jnp.identity(9)


def H3(F: Mat33) -> Mat12x12:
    def hat(f: Vec3) -> Mat33:
        return jax.numpy.array(
            [
                [0.0, -f[2], f[1]],
                [f[2], 0.0, -f[0]],
                [-f[1], f[0], 0.0],
            ]
        )

    f0: Vec3 = F[:, 0]
    f1: Vec3 = F[:, 1]
    f2: Vec3 = F[:, 2]
    Z3: Mat33 = jnp.zeros((3, 3))
    return jnp.vstack(
        [
            jnp.hstack([Z3, -hat(f2), hat(f1)]),
            jnp.hstack([hat(f2), Z3, -hat(f0)]),
            jnp.hstack([-hat(f1), hat(f0), Z3]),
        ]
    )
