import beartype
import einops
import jax
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Float

from liblaf.apple import utils
from liblaf.apple.typed.jax import Mat33, Vec3


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@utils.jit
def svd_rv(
    F: Float[jax.Array, "*B 3 3"],
) -> tuple[
    Float[jax.Array, "*B 3 3"], Float[jax.Array, "*B 3"], Float[jax.Array, "*B 3 3"]
]:
    F_packed: Float[jax.Array, "B 3 3"]
    F_packed, packed_shapes = einops.pack([F], "* i j")
    U_packed: Float[jax.Array, "B 3 3"]
    sigma_packed: Float[jax.Array, "B 3"]
    V_packed: Float[jax.Array, "B 3 3"]
    U_packed, sigma_packed, V_packed = jax.vmap(_svd_rv_elem)(F_packed)
    [U] = einops.unpack(U_packed, packed_shapes, "* i j")
    [sigma] = einops.unpack(sigma_packed, packed_shapes, "* i")
    [V] = einops.unpack(V_packed, packed_shapes, "* i j")
    return U, sigma, V


def _svd_rv_elem(F: Mat33) -> tuple[Mat33, Vec3, Mat33]:
    """...

    References:
        1. Kim, Theodore, and David Eberle. “Dynamic Deformables: Implementation and Production Practicalities (Now with Code!).” In ACM SIGGRAPH 2022 Courses, 1-259. Vancouver British Columbia Canada: ACM, 2022. https://doi.org/10.1145/3532720.3535628. Page 227. Figure F.1
    """
    U: Mat33
    sigma: Vec3
    VH: Mat33
    U, sigma, VH = jnp.linalg.svd(F, full_matrices=False)
    Sigma: Mat33 = jnp.diagflat(sigma)
    V: Mat33 = VH.T
    # reflection matrix
    L: Mat33 = jnp.identity(3)
    L = L.at[2, 2].set(jnp.linalg.det(U @ V.T))
    # see where to pull the reflection out of
    detU: Float[jax.Array, ""] = jnp.linalg.det(U)
    detV: Float[jax.Array, ""] = jnp.linalg.det(V)
    U = jnp.where((detU < 0) & (detV > 0), U @ L, U)
    V = jnp.where((detU > 0) & (detV < 0), V @ L, V)
    # push the reflection to the diagonal
    Sigma = Sigma @ L
    sigma = jnp.diagonal(Sigma)
    return U, sigma, V
