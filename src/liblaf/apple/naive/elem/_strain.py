import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple.typed.jax import Mat9x12, Mat33, Mat43


def deformation_gradient(
    u: Float[jax.Array, "4 3"], points: Float[jax.Array, "4 3"]
) -> Mat33:
    """...

    References:
        1. Kim, T. and Eberle, D. 2022. Dynamic deformables: implementation and production practicalities (now with code!). ACM SIGGRAPH 2022 Courses, ACM, 1-259. Page 121.
    """
    x: Float[jax.Array, "4 3"] = points + u
    Dm: Mat33 = jnp.column_stack([points[i] - points[0] for i in range(1, 4)])
    Ds: Mat33 = jnp.column_stack([x[i] - x[0] for i in range(1, 4)])
    F: Mat33 = Ds @ jnp.linalg.inv(Dm)
    return F


def dFdx(points: Mat43) -> Mat9x12:  # noqa: PLR0915
    """...

    References:
        1. Kim, T. and Eberle, D. 2022. Dynamic deformables: implementation and production practicalities (now with code!). ACM SIGGRAPH 2022 Courses, ACM, 1-259. Page 121.
    """
    Dm_inv: Mat33 = jnp.linalg.inv(_calc_Dm(points))
    m = Dm_inv[0, 0]
    n = Dm_inv[0, 1]
    o = Dm_inv[0, 2]
    p = Dm_inv[1, 0]
    q = Dm_inv[1, 1]
    r = Dm_inv[1, 2]
    s = Dm_inv[2, 0]
    t = Dm_inv[2, 1]
    u = Dm_inv[2, 2]
    t1 = -m - p - s
    t2 = -n - q - t
    t3 = -o - r - u
    dFdx = jnp.zeros((9, 12), dtype=points.dtype)
    dFdx = dFdx.at[0, 0].set(t1)
    dFdx = dFdx.at[0, 3].set(m)
    dFdx = dFdx.at[0, 6].set(p)
    dFdx = dFdx.at[0, 9].set(s)
    dFdx = dFdx.at[1, 1].set(t1)
    dFdx = dFdx.at[1, 4].set(m)
    dFdx = dFdx.at[1, 7].set(p)
    dFdx = dFdx.at[1, 10].set(s)
    dFdx = dFdx.at[2, 2].set(t1)
    dFdx = dFdx.at[2, 5].set(m)
    dFdx = dFdx.at[2, 8].set(p)
    dFdx = dFdx.at[2, 11].set(s)
    dFdx = dFdx.at[3, 0].set(t2)
    dFdx = dFdx.at[3, 3].set(n)
    dFdx = dFdx.at[3, 6].set(q)
    dFdx = dFdx.at[3, 9].set(t)
    dFdx = dFdx.at[4, 1].set(t2)
    dFdx = dFdx.at[4, 4].set(n)
    dFdx = dFdx.at[4, 7].set(q)
    dFdx = dFdx.at[4, 10].set(t)
    dFdx = dFdx.at[5, 2].set(t2)
    dFdx = dFdx.at[5, 5].set(n)
    dFdx = dFdx.at[5, 8].set(q)
    dFdx = dFdx.at[5, 11].set(t)
    dFdx = dFdx.at[6, 0].set(t3)
    dFdx = dFdx.at[6, 3].set(o)
    dFdx = dFdx.at[6, 6].set(r)
    dFdx = dFdx.at[6, 9].set(u)
    dFdx = dFdx.at[7, 1].set(t3)
    dFdx = dFdx.at[7, 4].set(o)
    dFdx = dFdx.at[7, 7].set(r)
    dFdx = dFdx.at[7, 10].set(u)
    dFdx = dFdx.at[8, 2].set(t3)
    dFdx = dFdx.at[8, 5].set(o)
    dFdx = dFdx.at[8, 8].set(r)
    dFdx = dFdx.at[8, 11].set(u)
    return dFdx


def _calc_Dm(points: Mat43) -> Mat33:
    return jnp.column_stack([points[i] - points[0] for i in range(1, 4)])
