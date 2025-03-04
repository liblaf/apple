import jax
import jax.numpy as jnp
import numpy as np
import pylops
from jaxtyping import Float

from . import LinearResult


def cgls(
    op: Float[pylops.LinearOperator, "N M"], y: Float[jax.Array, " N"], **kwargs
) -> LinearResult:
    x: Float[jax.Array, " M"]
    istop: int
    iit: int
    r1norm: float
    r2norm: float
    cost: Float[np.ndarray, " iterations"]
    x, istop, iit, r1norm, r2norm, cost = pylops.cgls(op, y, **kwargs)
    x = jnp.asarray(x)
    return LinearResult(
        {
            "x": x,
            "istop": istop,
            "iit": iit,
            "r1norm": r1norm,
            "r2norm": r2norm,
            "cost": cost,
        }
    )
