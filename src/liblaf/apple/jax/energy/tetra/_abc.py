from collections.abc import Container

import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Float, PyTree

from liblaf.apple import elem, utils
from liblaf.apple.jax import math


@attrs.frozen
class EnergyTetraElement:
    @property
    def required_aux(self) -> Container[str]:
        return {"dh_dX", "dV"}

    @property
    def required_params(self) -> Container[str]:
        return set()

    def prepare(self, points: Float[jax.Array, "4 3"]) -> PyTree:
        return {
            "dh_dX": elem.tetra.dh_dX(points),
            "dV": elem.tetra.dV(points),
        }

    def fun(
        self, u: Float[jax.Array, "4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        dh_dX: Float[jax.Array, "4 3"] = aux["dh_dX"]
        dV: Float[jax.Array, ""] = aux["dV"]
        F: Float[jax.Array, "3 3"] = elem.tetra.deformation_gradient(u, dh_dX)
        Psi: Float[jax.Array, ""] = self.energy_density(F, q, aux)
        return Psi * dV

    def jac(
        self, u: Float[jax.Array, "4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "4 3"]:
        dh_dX: Float[jax.Array, "4 3"] = aux["dh_dX"]
        dV: Float[jax.Array, ""] = aux["dV"]
        F: Float[jax.Array, "3 3"] = elem.tetra.deformation_gradient(u, dh_dX)
        PK1: Float[jax.Array, "3 3"] = self.first_piola_kirchhoff_stress(F, q, aux)
        jac: Float[jax.Array, "4 3"] = elem.tetra.deformation_gradient_vjp(dh_dX, PK1)
        return jac * dV

    def hess(
        self, u: Float[jax.Array, "4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "4 3 4 3"]:
        raise NotImplementedError

    def hessp(
        self,
        u: Float[jax.Array, "4 3"],
        p: Float[jax.Array, "4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> Float[jax.Array, "4 3"]:
        return math.hessp(self.fun, u, p, args=(q, aux))

    def hess_diag(
        self, u: Float[jax.Array, "4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "4 3"]:
        raise NotImplementedError

    def hess_quad(
        self,
        u: Float[jax.Array, "4 3"],
        p: Float[jax.Array, "4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> Float[jax.Array, ""]:
        raise NotImplementedError

    def jac_and_hess_diag(
        self, u: Float[jax.Array, "4 3"], q: PyTree, aux: PyTree
    ) -> tuple[Float[jax.Array, "4 3"], Float[jax.Array, "4 3"]]:
        raise NotImplementedError

    def energy_density(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        raise NotImplementedError

    def first_piola_kirchhoff_stress(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "3 3"]:
        return jax.grad(self.energy_density)(F, q, aux)

    def energy_density_hess(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, "3 3"]]:
        raise NotImplementedError

    def energy_density_hessp(
        self,
        F: Float[jax.Array, "3 3"],
        p: Float[jax.Array, "3 3"],
        q: PyTree,
        aux: PyTree,
    ) -> Float[jax.Array, ""]:
        raise NotImplementedError

    def energy_density_hess_diag(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, "3 3"]]:
        raise NotImplementedError

    def energy_density_hess_quad(
        self,
        F: Float[jax.Array, "3 3"],
        p: Float[jax.Array, "3 3"],
        q: PyTree,
        aux: PyTree,
    ) -> Float[jax.Array, ""]:
        raise NotImplementedError

    def energy_density_jac_and_hess_diag(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, "3 3"]]:
        raise NotImplementedError


@attrs.frozen
class EnergyTetra:
    elem: EnergyTetraElement

    @utils.jit
    def prepare(self, points: Float[jax.Array, "C 4 3"]) -> PyTree:
        return jax.vmap(self.elem.prepare)(points)

    @utils.jit
    def fun(
        self, u: Float[jax.Array, "C 4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        fun: Float[jax.Array, " C"] = jax.vmap(self.elem.fun)(u, q, aux)
        return jnp.sum(fun)

    @utils.jit
    def jac(
        self, u: Float[jax.Array, "C 4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "C 4 3"]:
        return jax.vmap(self.elem.jac)(u, q, aux)

    @utils.jit
    def hess(
        self, u: Float[jax.Array, "C 4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "C 4 3 4 3"]:
        return jax.vmap(self.elem.hess)(u, q, aux)

    @utils.jit
    def hessp(
        self,
        u: Float[jax.Array, "C 4 3"],
        p: Float[jax.Array, "C 4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> Float[jax.Array, "C 4 3"]:
        return jax.vmap(self.elem.hessp)(u, p, q, aux)

    @utils.jit
    def hess_diag(
        self, u: Float[jax.Array, "C 4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "C 4 3"]:
        raise NotImplementedError

    @utils.jit
    def hess_quad(
        self,
        u: Float[jax.Array, "C 4 3"],
        p: Float[jax.Array, "C 4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> Float[jax.Array, ""]:
        raise NotImplementedError

    @utils.jit
    def jac_and_hess_diag(
        self, u: Float[jax.Array, "C 4 3"], q: PyTree, aux: PyTree
    ) -> tuple[Float[jax.Array, "C 4 3"], Float[jax.Array, "C 4 3"]]:
        raise NotImplementedError

    @utils.jit
    def energy_density(
        self, F: Float[jax.Array, "C 3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, " C"]:
        return jax.vmap(self.elem.energy_density)(F, q, aux)

    @utils.jit
    def first_piola_kirchhoff_stress(
        self, F: Float[jax.Array, "C 3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "C 3 3"]:
        return jax.vmap(self.elem.first_piola_kirchhoff_stress)(F, q, aux)
