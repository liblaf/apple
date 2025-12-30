import einops
import jax.numpy as jnp
from jaxtyping import Array, Float

type Vec3 = Float[Array, "*batch 3"]
type Mat33 = Float[Array, "*batch 3 3"]


def svd_rv(F: Mat33) -> tuple[Mat33, Vec3, Mat33]:
    U, S, Vh = jnp.linalg.svd(F, full_matrices=True)
    detU = jnp.linalg.det(U)
    detV = jnp.linalg.det(Vh)
    U = U.at[..., :, -1].set(U[..., :, -1] * jnp.sign(detU)[..., jnp.newaxis])
    Vh = Vh.at[..., -1, :].set(Vh[..., -1, :] * jnp.sign(detV)[..., jnp.newaxis])
    return U, S, Vh


def polar_rv(F: Mat33) -> tuple[Mat33, Mat33]:
    U, S, Vh = svd_rv(F)
    R = U @ Vh
    # P = Vh.T @ S @ Vh
    P = einops.einsum(Vh, S, Vh, "... j i, ... j, ... j k -> ... i k")
    return R, P
