import einops
import jax
import jax.numpy as jnp
from jaxtyping import Float

type F3 = Float[jax.Array, "3"]
type F33 = Float[jax.Array, "3 3"]
type F9 = Float[jax.Array, "9"]
type F99 = Float[jax.Array, "9 9"]


def svd_rv(
    F: Float[jax.Array, "*C 3 3"],
) -> tuple[
    Float[jax.Array, "*C 3 3"], Float[jax.Array, "*C 3 3"], Float[jax.Array, "*C 3 3"]
]:
    F_packed: Float[jax.Array, "C 3 3"]
    F_packed, packed_shapes = einops.pack([F], "* i j")
    U_packed: Float[jax.Array, "C 3 3"]
    S_diag_packed: Float[jax.Array, "C 3"]
    VH_packed: Float[jax.Array, "C 3 3"]
    U_packed, S_diag_packed, VH_packed = jax.vmap(_svd_rv)(F_packed)
    [U] = einops.unpack(U_packed, packed_shapes, "* i j")
    [S_diag] = einops.unpack(S_diag_packed, packed_shapes, "* i")
    [VH] = einops.unpack(VH_packed, packed_shapes, "* i j")
    return U, S_diag, VH


def _svd_rv(F: F33) -> tuple[F33, F3, F33]:
    # Kim, Theodore, and David Eberle. "Dynamic deformables: implementation and production practicalities (now with code!)." ACM SIGGRAPH 2022 Courses. 2022. 1-259.
    # Appendix F. Rotation-Variant SVD and Polar Decomposition
    U: F33
    S_diag: F3
    VH: F33
    U, S_diag, VH = jnp.linalg.svd(F, full_matrices=False)
    detU: Float[jax.Array, ""] = jnp.linalg.det(U)
    detV: Float[jax.Array, ""] = jnp.linalg.det(VH)
    L_diag: F3 = jnp.asarray([1.0, 1.0, detU * detV])
    L: F33 = jnp.diagflat(L_diag)
    U = jax.lax.cond((detU < 0) & (detV > 0), lambda: U @ L, lambda: U)
    VH = jax.lax.cond((detU > 0) & (detV < 0), lambda: L @ VH, lambda: VH)
    S_diag = S_diag * L_diag
    return U, S_diag, VH


def polar_rv(
    F: Float[jax.Array, "*C 3 3"],
) -> tuple[Float[jax.Array, "*C 3 3"], Float[jax.Array, "*C 3 3"]]:
    F_packed: Float[jax.Array, "C 3 3"]
    F_packed, packed_shapes = einops.pack([F], "* i j")
    R_packed: Float[jax.Array, "C 3 3"]
    S_packed: Float[jax.Array, "C 3 3"]
    R_packed, S_packed = jax.vmap(_polar_rv_custom)(F_packed)
    [R] = einops.unpack(R_packed, packed_shapes, "* i j")
    [S] = einops.unpack(S_packed, packed_shapes, "* i j")
    return R, S


def _polar_rv_jax(F: F33) -> tuple[F33, F33]:
    R: F33
    S: F33
    R, S = jax.scipy.linalg.polar(F, side="right", method="svd")
    detR: Float[jax.Array, ""] = jnp.linalg.det(R)
    L_diag: F3 = jnp.asarray([1.0, 1.0, detR])
    L: F33 = jnp.diagflat(L_diag)
    R = R @ L
    S = L @ S
    return R, S


@jax.custom_jvp
def _polar_rv_custom(
    F: F33,
) -> tuple[F33, F33]:
    U: F33
    S_diag: F3
    VH: F33
    U, S_diag, VH = _svd_rv(F)
    R: F33 = U @ VH
    Sigma: F33 = jnp.diagflat(S_diag)
    S: F33 = VH.T @ Sigma @ VH
    return R, S


@_polar_rv_custom.defjvp
def _polar_rv_custom_jvp(
    primals: tuple[F33], tangents: tuple[F33]
) -> tuple[tuple[F33, F33], tuple[F33, F33]]:
    F: F33
    (F,) = primals
    dF: F33
    (dF,) = tangents
    U: F33
    S_diag: F3
    VH: F33
    U, S_diag, VH = _svd_rv(F)
    R: F33 = U @ VH
    S: F33 = VH.T @ jnp.diagflat(S_diag) @ VH
    dR_dF: F99 = _dR_dF(U, S_diag, VH)
    dR: F9 = dR_dF @ dF.ravel()
    dR: F33 = dR.reshape(3, 3)
    dS: F33 = _dS_dF(R, S, dF, dR)
    return ((R, S), (dR, dS))


def _dR_dF(U: F33, S_diag: F3, VH: F33) -> F99:
    r"""...

    References:
        \[1\] T. Kim and D. Eberle, “Dynamic deformables: implementation and production practicalities (now with code!),” in ACM SIGGRAPH 2022 Courses, Vancouver British Columbia Canada: ACM, Aug. 2022, pp. 1-259. doi: 10.1145/3532720.3535628.
    """
    # get the twist modes
    T0: F33 = jnp.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    T0 = (1 / jnp.sqrt(2)) * U @ T0 @ VH
    T1: F33 = jnp.asarray([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
    T1 = (1 / jnp.sqrt(2)) * U @ T1 @ VH
    T2: F33 = jnp.asarray([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    T2 = (1 / jnp.sqrt(2)) * U @ T2 @ VH
    # get the flattened versions
    t0: Float[jax.Array, " 9"] = T0.ravel()
    t1: Float[jax.Array, " 9"] = T1.ravel()
    t2: Float[jax.Array, " 9"] = T2.ravel()
    # get the singular values
    sx: Float[jax.Array, ""] = S_diag[0]
    sy: Float[jax.Array, ""] = S_diag[1]
    sz: Float[jax.Array, ""] = S_diag[2]
    H: Float[jax.Array, "9 9"] = (
        (2 / (sx + sy)) * jnp.outer(t0, t0)
        + (2 / (sy + sz)) * (jnp.outer(t1, t1))
        + (2 / (sx + sz)) * (jnp.outer(t2, t2))
    )
    return H


def _dS_dF(R: F33, S: F33, dF: F33, dR: F33) -> F99:
    dS: F33 = R.T @ (dF - dR @ S)
    return dS
