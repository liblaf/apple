import attrs
import einops
import jax
import jax.numpy as jnp
from jaxtyping import Float, PyTree

from liblaf import apple


@apple.register_dataclass()
@attrs.define(kw_only=True)
class MaterialTetra:
    def fun(
        self, u: Float[jax.Array, "C 4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        return jnp.sum(jax.vmap(self._fun_elem)(u, q, aux))

    @apple.jit()
    def jac(
        self, u: Float[jax.Array, "C 4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "C 4 3"]:
        return jax.vmap(self._jac_elem)(u, q, aux)

    @apple.jit()
    def fun_jac(
        self, u: Float[jax.Array, "C 4 3"], q: PyTree, aux: PyTree
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, "C 4 3"]]:
        dW: Float[jax.Array, " C"]
        jac: Float[jax.Array, "C 4 3"]
        dW, jac = jax.vmap(self._fun_jac_elem)(u, q, aux)
        return jnp.sum(dW), jac

    @apple.jit()
    def hess(
        self, u: Float[jax.Array, "C 4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "C 4 3 4 3"]:
        return jax.vmap(self._hess_elem)(u, q, aux)

    @apple.jit()
    def hess_diag(
        self, u: Float[jax.Array, "C 4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "C 4 3"]:
        return jax.vmap(self._hess_diag_elem)(u, q, aux)

    def _fun_elem(
        self, u: Float[jax.Array, "4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        dh_dX: Float[jax.Array, "4 3"] = aux["dh_dX"]
        dV: Float[jax.Array, ""] = aux["dV"]
        F: Float[jax.Array, "3 3"] = apple.elem.tetra.deformation_gradient(u, dh_dX)
        Psi: Float[jax.Array, ""] = self._Psi_elem(F, q, aux)
        return Psi * dV

    def _jac_elem(
        self, u: Float[jax.Array, "4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "4 3"]:
        dh_dX: Float[jax.Array, "4 3"] = aux["dh_dX"]
        dV: Float[jax.Array, ""] = aux["dV"]
        F: Float[jax.Array, "3 3"] = apple.elem.tetra.deformation_gradient(u, dh_dX)
        PK1: Float[jax.Array, "3 3"] = self._PK1_elem(F, q, aux)
        dPsi_du: Float[jax.Array, "3 3"] = einops.einsum(dh_dX, PK1, "i j, j k -> i k")
        return dPsi_du * dV

    def _fun_jac_elem(
        self, u: Float[jax.Array, "4 3"], q: PyTree, aux: PyTree
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, "4 3"]]:
        return self._fun_elem(u, q, aux), self._jac_elem(u, q, aux)

    def _hess_elem(
        self, u: Float[jax.Array, "4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "4 3 4 3"]:
        return jax.jacobian(self._jac_elem)(u, q, aux)

    def _hess_diag_elem(
        self, u: Float[jax.Array, "4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "4 3"]:
        hess: Float[jax.Array, "4 3 4 3"] = self._hess_elem(u, q, aux)
        hess = hess.reshape(12, 12)
        diag: Float[jax.Array, " 12"] = jnp.diagonal(hess)
        diag: Float[jax.Array, "4 3"] = diag.reshape(4, 3)
        return diag

    def _Psi_elem(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        raise NotImplementedError

    def _PK1_elem(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "3 3"]:
        return jax.grad(self._Psi_elem)(F, q, aux)
