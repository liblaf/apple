import jax.numpy as jnp
from jaxtyping import Array, Float


def svd_rv(
    F: Float[Array, "*batch N N"],
) -> tuple[
    Float[Array, "*batch N N"], Float[Array, "*batch N"], Float[Array, "*batch N N"]
]:
    u: Float[Array, "*batch N N"]
    s: Float[Array, "*batch N"]
    vh: Float[Array, "*batch N N"]
    u, s, vh = jnp.linalg.svd(F, full_matrices=False)
    det_u: Float[Array, "*batch"] = jnp.linalg.det(u)
    det_v: Float[Array, "*batch"] = jnp.linalg.det(vh)
    u = u.at[..., :, -1].set(u.at[..., :, -1] * det_u[..., None])
    vh = vh.at[..., -1, :].set(vh.at[..., -1, :] * det_v[..., None])
    s = s.at[..., -1].set(s.at[..., -1] * det_u * det_v)
    return u, s, vh
