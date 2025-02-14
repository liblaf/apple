import einops
import jax
import jax.numpy as jnp
from jaxtyping import Float


def svd_rv(
    F: Float[jax.Array, "*c 3 3"],
) -> tuple[
    Float[jax.Array, "*c 3 3"], Float[jax.Array, "*c 3 3"], Float[jax.Array, "*c 3 3"]
]:
    F_packed: Float[jax.Array, "c 3 3"]
    F_packed, packed_shapes = einops.pack([F], "* i j")
    U_packed: Float[jax.Array, "c 3 3"]
    S_diag_packed: Float[jax.Array, "c 3"]
    VH_packed: Float[jax.Array, "c 3 3"]
    U_packed, S_diag_packed, VH_packed = jax.vmap(_svd_rv)(F_packed)
    [U] = einops.unpack(U_packed, packed_shapes, "* i j")
    [S_diag] = einops.unpack(S_diag_packed, packed_shapes, "* i")
    [VH] = einops.unpack(VH_packed, packed_shapes, "* i j")
    return U, S_diag, VH


def _svd_rv(
    F: Float[jax.Array, "3 3"],
) -> tuple[Float[jax.Array, "3 3"], Float[jax.Array, "3 3"], Float[jax.Array, "3 3"]]:
    # Kim, Theodore, and David Eberle. "Dynamic deformables: implementation and production practicalities (now with code!)." ACM SIGGRAPH 2022 Courses. 2022. 1-259.
    # Appendix F. Rotation-Variant SVD and Polar Decomposition
    U: Float[jax.Array, "3 3"]
    S_diag: Float[jax.Array, " 3"]
    VH: Float[jax.Array, "3 3"]
    U, S_diag, VH = jnp.linalg.svd(F, full_matrices=False)
    detU: Float[jax.Array, ""] = jnp.linalg.det(U)
    detV: Float[jax.Array, ""] = jnp.linalg.det(VH)
    L_diag: Float[jax.Array, " 3"] = jnp.asarray([1.0, 1.0, detU * detV])
    L: Float[jax.Array, "3 3"] = jnp.diagflat(L_diag)
    U = jax.lax.cond((detU < 0) & (detV > 0), lambda: U @ L, lambda: U)
    VH = jax.lax.cond((detU > 0) & (detV < 0), lambda: L @ VH, lambda: VH)
    S_diag = S_diag * L_diag
    return U, S_diag, VH


def polar_rv(
    F: Float[jax.Array, "*c 3 3"],
) -> tuple[Float[jax.Array, "*c 3 3"], Float[jax.Array, "*c 3 3"]]:
    F_packed: Float[jax.Array, "c 3 3"]
    F_packed, packed_shapes = einops.pack([F], "* i j")
    R_packed: Float[jax.Array, "c 3 3"]
    S_packed: Float[jax.Array, "c 3 3"]
    R_packed, S_packed = jax.vmap(_polar_rv)(F_packed)
    [R] = einops.unpack(R_packed, packed_shapes, "* i j")
    [S] = einops.unpack(S_packed, packed_shapes, "* i j")
    return R, S


def _polar_rv(
    F: Float[jax.Array, "3 3"],
) -> tuple[Float[jax.Array, "3 3"], Float[jax.Array, "3 3"]]:
    R: Float[jax.Array, "3 3"]
    S: Float[jax.Array, "3 3"]
    R, S = jax.scipy.linalg.polar(F, side="right", method="svd")
    detR: Float[jax.Array, ""] = jnp.linalg.det(R)
    L_diag: Float[jax.Array, " 3"] = jnp.asarray([1.0, 1.0, detR])
    L: Float[jax.Array, "3 3"] = jnp.diagflat(L_diag)
    R = R @ L
    S = L @ S
    return R, S
