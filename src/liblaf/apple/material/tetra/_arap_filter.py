from typing import override

import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Float, PyTree

from liblaf import apple

from ._abc import MaterialTetraElement
from ._arap import AsRigidAsPossible, AsRigidAsPossibleElement


@apple.register_dataclass()
@attrs.frozen(kw_only=True)
class AsRigidAsPossibleFilterElement(AsRigidAsPossibleElement):
    @override
    def strain_energy_density_hess_diag(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "4 3"]:
        mu: Float[jax.Array, ""] = q["mu"]
        dh_dX: Float[jax.Array, "4 3"] = aux["dh_dX"]
        U: Float[jax.Array, "3 3"]
        S_diag: Float[jax.Array, " 3"]
        VH: Float[jax.Array, "3 3"]
        U, S_diag, VH = apple.math.svd_rv(F)
        s0: Float[jax.Array, ""] = S_diag[0]
        s1: Float[jax.Array, ""] = S_diag[1]
        s2: Float[jax.Array, ""] = S_diag[2]
        lambda0: Float[jax.Array, ""] = 2.0 / (s1 + s2)
        lambda0 = jnp.where(s1 + s2 < 2.0, 1.0, lambda0)
        lambda1: Float[jax.Array, ""] = 2.0 / (s0 + s2)
        lambda1 = jnp.where(s0 + s2 < 2.0, 1.0, lambda1)
        lambda2: Float[jax.Array, ""] = 2.0 / (s0 + s1)
        lambda2 = jnp.where(s0 + s1 < 2.0, 1.0, lambda2)
        U0: Float[jax.Array, " 3"] = U[:, 0]
        U1: Float[jax.Array, " 3"] = U[:, 1]
        U2: Float[jax.Array, " 3"] = U[:, 2]
        V0: Float[jax.Array, " 3"] = VH[0, :]
        V1: Float[jax.Array, " 3"] = VH[1, :]
        V2: Float[jax.Array, " 3"] = VH[2, :]
        Q0: Float[jax.Array, "3 3"] = jnp.outer(V1, U2) - jnp.outer(V2, U1)
        Q1: Float[jax.Array, "3 3"] = jnp.outer(V2, U0) - jnp.outer(V0, U2)
        Q2: Float[jax.Array, "3 3"] = jnp.outer(V1, U0) - jnp.outer(V0, U1)
        W0: Float[jax.Array, "4 3"] = (
            apple.elem.tetra.deformation_gradient_vjp(dh_dX, Q0) ** 2
        )
        W1: Float[jax.Array, "4 3"] = (
            apple.elem.tetra.deformation_gradient_vjp(dh_dX, Q1) ** 2
        )
        W2: Float[jax.Array, "4 3"] = (
            apple.elem.tetra.deformation_gradient_vjp(dh_dX, Q2) ** 2
        )
        diag_h4: Float[jax.Array, "4 3"] = lambda0 * W0 + lambda1 * W1 + lambda2 * W2
        X: Float[jax.Array, "4 3"] = apple.elem.tetra.deformation_gradient_gram(dh_dX)
        return mu * (2.0 * X - diag_h4)

    @override
    def strain_energy_density_hess_quad(
        self,
        F: Float[jax.Array, "3 3"],
        p: Float[jax.Array, "4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> Float[jax.Array, ""]:
        mu: Float[jax.Array, ""] = q["mu"]
        dh_dX: Float[jax.Array, "4 3"] = aux["dh_dX"]
        dFdx_p: Float[jax.Array, "3 3"] = apple.elem.tetra.deformation_gradient_jvp(
            dh_dX, p
        )
        ret0: Float[jax.Array, ""] = apple.math.norm_sqr(dFdx_p)
        U: Float[jax.Array, "3 3"]
        S_diag: Float[jax.Array, " 3"]
        VH: Float[jax.Array, "3 3"]
        U, S_diag, VH = apple.math.svd_rv(F)
        s0: Float[jax.Array, ""] = S_diag[0]
        s1: Float[jax.Array, ""] = S_diag[1]
        s2: Float[jax.Array, ""] = S_diag[2]
        lambda0: Float[jax.Array, ""] = 2.0 / (s1 + s2)
        lambda0 = jnp.where(s1 + s2 < 2.0, 1.0, lambda0)
        lambda1: Float[jax.Array, ""] = 2.0 / (s0 + s2)
        lambda1 = jnp.where(s0 + s2 < 2.0, 1.0, lambda1)
        lambda2: Float[jax.Array, ""] = 2.0 / (s0 + s1)
        lambda2 = jnp.where(s0 + s1 < 2.0, 1.0, lambda2)
        V: Float[jax.Array, "3 3"] = VH.T
        U0: Float[jax.Array, " 3"] = U[:, 0]
        U1: Float[jax.Array, " 3"] = U[:, 1]
        U2: Float[jax.Array, " 3"] = U[:, 2]
        V0: Float[jax.Array, " 3"] = V[:, 0]
        V1: Float[jax.Array, " 3"] = V[:, 1]
        V2: Float[jax.Array, " 3"] = V[:, 2]
        Q0: Float[jax.Array, "3 3"] = jnp.outer(V1, U2) - jnp.outer(V2, U1)
        Q1: Float[jax.Array, "3 3"] = jnp.outer(V2, U0) - jnp.outer(V0, U2)
        Q2: Float[jax.Array, "3 3"] = jnp.outer(V1, U0) - jnp.outer(V0, U1)
        ret1: Float[jax.Array, ""] = (
            lambda0 * jnp.vdot(Q0, dFdx_p) ** 2
            + lambda1 * jnp.vdot(Q1, dFdx_p) ** 2
            + lambda2 * jnp.vdot(Q2, dFdx_p) ** 2
        )
        ret: Float[jax.Array, ""] = mu * (2.0 * ret0 - ret1)
        return ret


@apple.register_dataclass()
@attrs.frozen(kw_only=True)
class AsRigidAsPossibleFilter(AsRigidAsPossible):
    elem: MaterialTetraElement = attrs.field(
        factory=AsRigidAsPossibleFilterElement, metadata={"static": True}
    )
