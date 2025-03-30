import abc
from collections.abc import Sequence

import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Float, PyTree

from liblaf import apple


@apple.register_dataclass()
@attrs.define(kw_only=True)
class MaterialTetraElement(abc.ABC):
    @property
    def required_aux(self) -> Sequence[str]:
        return ("dh_dX", "dV")

    @property
    def required_params(self) -> Sequence[str]:
        return ()

    def prepare(self, points: Float[jax.Array, "4 3"]) -> PyTree:
        return {
            "dh_dX": apple.elem.tetra.dh_dX(points),
            "dV": apple.elem.tetra.dV(points),
        }

    def fun(
        self, u: Float[jax.Array, "4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        dh_dX: Float[jax.Array, "4 3"] = aux["dh_dX"]
        dV: Float[jax.Array, ""] = aux["dV"]
        F: Float[jax.Array, "3 3"] = apple.elem.tetra.deformation_gradient(u, dh_dX)
        Psi: Float[jax.Array, ""] = self.strain_energy_density(F, q, aux)
        return Psi * dV

    def jac(
        self, u: Float[jax.Array, "4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "4 3"]:
        dh_dX: Float[jax.Array, "4 3"] = aux["dh_dX"]
        dV: Float[jax.Array, ""] = aux["dV"]
        F: Float[jax.Array, "3 3"] = apple.elem.tetra.deformation_gradient(u, dh_dX)
        PK1: Float[jax.Array, "3 3"] = self.first_piola_kirchhoff_stress(F, q, aux)
        jac: Float[jax.Array, "4 3"] = dh_dX @ PK1
        return jac * dV

    def hess(
        self, u: Float[jax.Array, "4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "4 3 4 3"]:
        return jax.hessian(self.fun)(u, q, aux)

    def hessp(
        self,
        u: Float[jax.Array, "4 3"],
        p: Float[jax.Array, "4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> Float[jax.Array, "4 3"]:
        return apple.hvp(self.fun, u, p, q, aux)

    def hess_diag(
        self, u: Float[jax.Array, "4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "4 3"]:
        return apple.hess_diag(self.fun, u, q, aux)

    def hess_quad(
        self,
        u: Float[jax.Array, "4 3"],
        p: Float[jax.Array, "4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> Float[jax.Array, ""]:
        return apple.hess_quad(self.fun, u, p, q, aux)

    @abc.abstractmethod
    def strain_energy_density(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]: ...

    def first_piola_kirchhoff_stress(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "3 3"]:
        return jax.jacobian(self.strain_energy_density)(F, q, aux)

    def elasticity_tensor(
        self, F: Float[jax.Array, "3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "3 3 3 3"]:
        return jax.hessian(self.strain_energy_density)(F, q, aux)


@apple.register_dataclass()
@attrs.define(kw_only=True)
class MaterialTetra:
    elem: MaterialTetraElement = attrs.field(default=None, metadata={"static": True})

    @property
    def required_aux(self) -> Sequence[str]:
        return self.elem.required_aux

    @property
    def required_params(self) -> Sequence[str]:
        return self.elem.required_params

    @apple.jit()
    def prepare(self, points: Float[jax.Array, "C 4 3"]) -> PyTree:
        return jax.vmap(self.elem.prepare)(points)

    @apple.jit()
    def fun(
        self, u: Float[jax.Array, "C 4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        q = self.params_filter(q)
        aux = self.aux_filter(aux)
        return jnp.sum(jax.vmap(self.elem.fun)(u, q, aux))

    @apple.jit()
    def jac(
        self, u: Float[jax.Array, "C 4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "C 4 3"]:
        q = self.params_filter(q)
        aux = self.aux_filter(aux)
        return jax.vmap(self.elem.jac)(u, q, aux)

    @apple.jit()
    def hess(
        self, u: Float[jax.Array, "C 4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "C 4 3 4 3"]:
        q = self.params_filter(q)
        aux = self.aux_filter(aux)
        return jax.vmap(self.elem.hess)(u, q, aux)

    @apple.jit()
    def hessp(
        self,
        u: Float[jax.Array, "C 4 3"],
        p: Float[jax.Array, "C 4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> Float[jax.Array, "C 4 3"]:
        q = self.params_filter(q)
        aux = self.aux_filter(aux)
        return jax.vmap(self.elem.hessp)(u, p, q, aux)

    @apple.jit()
    def hess_diag(
        self, u: Float[jax.Array, "C 4 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "C 4 3"]:
        q = self.params_filter(q)
        aux = self.aux_filter(aux)
        return jax.vmap(self.elem.hess_diag)(u, q, aux)

    @apple.jit()
    def hess_quad(
        self,
        u: Float[jax.Array, "C 4 3"],
        p: Float[jax.Array, "C 4 3"],
        q: PyTree,
        aux: PyTree,
    ) -> Float[jax.Array, " C"]:
        q = self.params_filter(q)
        aux = self.aux_filter(aux)
        return jnp.sum(jax.vmap(self.elem.hess_quad)(u, p, q, aux))

    @apple.jit()
    def strain_energy_density(
        self, F: Float[jax.Array, "C 3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, " C"]:
        q = self.params_filter(q)
        aux = self.aux_filter(aux)
        return jax.vmap(self.elem.strain_energy_density)(F, q, aux)

    @apple.jit()
    def first_piola_kirchhoff_stress(
        self, F: Float[jax.Array, "C 3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "C 3 3"]:
        q = self.params_filter(q)
        aux = self.aux_filter(aux)
        return jax.vmap(self.elem.first_piola_kirchhoff_stress)(F, q, aux)

    @apple.jit()
    def elasticity_tensor(
        self, F: Float[jax.Array, "C 3 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "C 3 3 3 3"]:
        q = self.params_filter(q)
        aux = self.aux_filter(aux)
        return jax.vmap(self.elem.elasticity_tensor)(F, q, aux)

    def aux_filter(self, aux: PyTree) -> PyTree:
        return {k: aux[k] for k in self.required_aux}

    def params_filter(self, q: PyTree) -> PyTree:
        return {k: q[k] for k in self.required_params}
