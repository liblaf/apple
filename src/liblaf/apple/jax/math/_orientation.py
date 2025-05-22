import beartype
import einops
import jax
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Float

from liblaf.apple import utils
from liblaf.apple.typed.jax import Mat33, Vec3


@utils.jit
@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def orientation_matrix(
    a: Float[jax.Array, "*N 3"], b: Float[jax.Array, "*N 3"]
) -> Float[jax.Array, "*N 3 3"]:
    a_packed: Float[jax.Array, "B 3"]
    a_packed, packed_shapes = einops.pack([a], "* i")
    b_packed: Float[jax.Array, "B 3"]
    b_packed, _packed_shapes = einops.pack([b], "* i")
    Q_packed: Float[jax.Array, "B 3 3"]
    Q_packed = jax.vmap(_orientation_matrix_elem)(a_packed, b_packed)
    [Q] = einops.unpack(Q_packed, packed_shapes, "* i j")
    return Q


def _orientation_matrix_elem(a: Vec3, b: Vec3) -> Mat33:
    aU: Mat33 = _svd_rv_elem(a)
    bU: Mat33 = _svd_rv_elem(b)
    return bU @ aU.T


def _svd_rv_elem(v: Vec3) -> Mat33:
    U: Mat33
    U, _sigma, _VH = jnp.linalg.svd(v[:, None])
    # reflection matrix
    L: Mat33 = jnp.identity(3)
    L = L.at[2, 2].set(jnp.linalg.det(U))
    U = U @ L
    return U
