import beartype
import einops
import jax
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Float

from liblaf.apple.typed.jax import Mat33, Vec3

from ._svd_rv import _svd_rv_elem


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def polar_rv(
    F: Float[jax.Array, "*B 3 3"],
) -> tuple[Float[jax.Array, "*B 3 3"], Float[jax.Array, "*B 3 3"]]:
    F_packed: Float[jax.Array, "B 3 3"]
    F_packed, packed_shapes = einops.pack([F], "* i j")
    R_packed: Float[jax.Array, "B 3 3"]
    S_packed: Float[jax.Array, "B 3 3"]
    R_packed, S_packed = jax.vmap(_polar_rv_elem)(F_packed)
    [R] = einops.unpack(R_packed, packed_shapes, "* i j")
    [S] = einops.unpack(S_packed, packed_shapes, "* i j")
    return R, S


def _polar_rv_elem(F: Mat33) -> tuple[Mat33, Mat33]:
    """...

    References:
        1. Kim, Theodore, and David Eberle. “Dynamic Deformables: Implementation and Production Practicalities (Now with Code!).” In ACM SIGGRAPH 2022 Courses, 1-259. Vancouver British Columbia Canada: ACM, 2022. https://doi.org/10.1145/3532720.3535628. P227. Figure F.2
    """
    U: Mat33
    sigma: Vec3
    V: Mat33
    U, sigma, V = _svd_rv_elem(F)
    Sigma: Mat33 = jnp.diagflat(sigma)
    R: Mat33 = U @ V.T
    S: Mat33 = V @ Sigma @ V.T
    return R, S
