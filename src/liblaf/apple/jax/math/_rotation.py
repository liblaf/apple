import einops
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def svd_rv(
    a: Float[Array, "*batch N N"],
) -> tuple[
    Float[Array, "*batch N N"], Float[Array, "*batch N"], Float[Array, "*batch N N"]
]:
    u: Float[Array, "*batch N N"]
    s: Float[Array, "*batch N"]
    vh: Float[Array, "*batch N N"]
    u, s, vh = jnp.linalg.svd(a, full_matrices=False)
    det_u: Float[Array, "*batch"] = jnp.linalg.det(u)
    det_v: Float[Array, "*batch"] = jnp.linalg.det(vh)
    u = u.at[..., :, -1].set(u[..., :, -1] * det_u[..., None])
    vh = vh.at[..., -1, :].set(vh[..., -1, :] * det_v[..., None])
    s = s.at[..., -1].set(s[..., -1] * det_u * det_v)
    return u, s, vh


def polar_rv(
    a: Float[Array, "*batch N N"],
) -> tuple[Float[Array, "*batch N N"], Float[Array, "*batch N N"]]:
    u: Float[Array, "*batch N N"]
    s: Float[Array, "*batch N"]
    vh: Float[Array, "*batch N N"]
    u, s, vh = svd_rv(a)
    R: Float[Array, "*batch N N"] = einops.einsum(u, vh, "... i j, ... j k -> ... i k")
    S: Float[Array, "*batch N N"] = einops.einsum(
        vh, s, vh, "... j i, ... j, ... j k -> ... i k"
    )
    R = jax.lax.stop_gradient(R)
    S = jax.lax.stop_gradient(S)
    return R, S
