from typing import no_type_check

import beartype
import einops
import jax
import jax.numpy as jnp
import jaxtyping
import warp as wp
from jaxtyping import Float

from liblaf.apple import func, utils
from liblaf.apple.typed.warp import mat33, mat43


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@utils.jit
def gradient(
    u: Float[jax.Array, "*cells a=4 I=3"], dh_dX: Float[jax.Array, "*cells a=4 J=3"]
) -> Float[jax.Array, "*cells I=3 J=3"]:
    return einops.einsum(u, dh_dX, "... a I, ... a J -> ... I J")


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@utils.jit
def deformation_gradient(
    u: Float[jax.Array, "cells 4 3"], dh_dX: Float[jax.Array, "cells 4 3"]
) -> Float[jax.Array, "cells 3 3"]:
    F: Float[jax.Array, "cells 3 3"]
    (F,) = _deformation_gradient_warp(u, dh_dX)
    return F


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@utils.jit
def deformation_gradient_jac(
    dh_dX: Float[jax.Array, "*cells 4 3"],
) -> Float[jax.Array, "*cells 3 3 4 3"]:
    def dFdx_elem(dh_dX: Float[jax.Array, "4 3"]) -> Float[jax.Array, "3 3 4 3"]:
        zeros: Float[jax.Array, " 4"] = jnp.zeros((4,))
        return jnp.asarray(
            [
                [
                    jnp.column_stack([dh_dX[:, 0], zeros, zeros]),
                    jnp.column_stack([dh_dX[:, 1], zeros, zeros]),
                    jnp.column_stack([dh_dX[:, 2], zeros, zeros]),
                ],
                [
                    jnp.column_stack([zeros, dh_dX[:, 0], zeros]),
                    jnp.column_stack([zeros, dh_dX[:, 1], zeros]),
                    jnp.column_stack([zeros, dh_dX[:, 2], zeros]),
                ],
                [
                    jnp.column_stack([zeros, zeros, dh_dX[:, 0]]),
                    jnp.column_stack([zeros, zeros, dh_dX[:, 1]]),
                    jnp.column_stack([zeros, zeros, dh_dX[:, 2]]),
                ],
            ]
        )

    return jax.vmap(dFdx_elem)(dh_dX)


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@utils.jit
def deformation_gradient_jvp(
    dh_dX: Float[jax.Array, "cells 4 3"], p: Float[jax.Array, "cells 4 3"]
) -> Float[jax.Array, "cells 3 3"]:
    results: Float[jax.Array, "cells 3 3"]
    (results,) = _deformation_gradient_jvp_warp(dh_dX, p)
    return results


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@utils.jit
def deformation_gradient_vjp(
    dh_dX: Float[jax.Array, "*cells 4 3"], p: Float[jax.Array, "*cells 3 3"]
) -> Float[jax.Array, "*cells 4 3"]:
    results: Float[jax.Array, "cells 3 3"]
    (results,) = _deformation_gradient_vjp_warp(dh_dX, p)
    return results


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@utils.jit
def deformation_gradient_gram(
    dh_dX: Float[jax.Array, "*cells 4 3"],
) -> Float[jax.Array, "*cells 4 3"]:
    result: Float[jax.Array, "*cells 4"] = jnp.sum(dh_dX**2, axis=-1)
    result: Float[jax.Array, "*cells 4 3"] = einops.repeat(result, "... a -> ... a 3")
    return result


@no_type_check
@utils.jax_kernel
def _deformation_gradient_warp(
    u: wp.array(dtype=mat43),
    dh_dX: wp.array(dtype=mat43),
    F: wp.array(dtype=mat33),
) -> None:
    tid = wp.tid()
    F[tid] = func.deformation_gradient(u[tid], dh_dX[tid])


@no_type_check
@utils.jax_kernel
def _deformation_gradient_jvp_warp(
    dh_dX: wp.array(dtype=mat43),
    p: wp.array(dtype=mat43),
    results: wp.array(dtype=mat33),
) -> None:
    tid = wp.tid()
    results[tid] = func.deformation_gradient_jvp(dh_dX[tid], p[tid])


@no_type_check
@utils.jax_kernel
def _deformation_gradient_vjp_warp(
    dh_dX: wp.array(dtype=mat43),
    p: wp.array(dtype=mat33),
    results: wp.array(dtype=mat43),
) -> None:
    tid = wp.tid()
    results[tid] = func.deformation_gradient_vjp(dh_dX[tid], p[tid])
