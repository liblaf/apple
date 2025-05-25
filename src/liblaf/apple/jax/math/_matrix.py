import einops
import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import utils


@utils.jit
def diag(vec: Float[jax.Array, "*B N"]) -> Float[jax.Array, "*B N N"]:
    vec_packed: Float[jax.Array, "B N"]
    vec_packed, packed_shapes = einops.pack([vec], "* i")
    mat_packed: Float[jax.Array, "B N N"] = jax.vmap(jnp.diagflat)(vec_packed)
    [mat] = einops.unpack(mat_packed, packed_shapes, "* i j")
    return mat


@utils.jit
def frobenius_norm_square(x: Float[jax.Array, "..."]) -> Float[jax.Array, ""]:
    return jnp.sum(x**2)


def transpose(a: Float[jax.Array, "*B M N"]) -> Float[jax.Array, "*B N M"]:
    return einops.rearrange(a, "... M N -> ... N M")


def unvec(a: Float[jax.Array, "*B N*M"], m: int, n: int) -> Float[jax.Array, "*B M N"]:
    return einops.rearrange(a, "... (N M) -> ... M N", M=m, N=n)


def unvec_mat(
    a: Float[jax.Array, "*B N*M J*I"], m: int, n: int, i: int, j: int
) -> Float[jax.Array, "*B M N I J"]:
    return einops.rearrange(a, "... (N M) (J I) -> ... M N I J", M=m, N=n, I=i, J=j)


def vec(a: Float[jax.Array, "*B M N"]) -> Float[jax.Array, "*B N*M"]:
    return einops.rearrange(a, "... M N -> ... (N M)")


def vec_mat(a: Float[jax.Array, "*B M N I J"]) -> Float[jax.Array, "*B N*M J*I"]:
    return einops.rearrange(a, "... M N I J -> ... (N M) (J I)")
