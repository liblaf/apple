import abc
from collections.abc import Sequence

import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Float, PyTree

from liblaf import apple


@apple.register_dataclass()
@attrs.frozen(kw_only=True)
class MaterialPointElement(abc.ABC):
    @property
    def required_aux(self) -> Sequence[str]:
        return ()

    @property
    def required_params(self) -> Sequence[str]:
        return ("mass",)

    def prepare(self, points: Float[jax.Array, "3"]) -> PyTree:  # noqa: ARG002
        return {}

    @abc.abstractmethod
    def fun(
        self, u: Float[jax.Array, "3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]: ...

    def jac(
        self, u: Float[jax.Array, "3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "3"]:
        return jax.grad(self.fun)(u, q, aux)

    def hess(
        self, u: Float[jax.Array, "3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "3 3"]:
        return jax.hessian(self.fun)(u, q, aux)

    def hessp(
        self, u: Float[jax.Array, "3"], p: Float[jax.Array, "3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "3"]:
        return apple.hvp(self.fun, u, p, args=(q, aux))

    def hess_diag(
        self, u: Float[jax.Array, "3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "3"]:
        return apple.hess_diag(self.fun, u, args=(q, aux))

    def hess_quad(
        self, u: Float[jax.Array, "3"], p: Float[jax.Array, "3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        return apple.hess_quad(self.fun, u, p, args=(q, aux))


@apple.register_dataclass()
@attrs.frozen(kw_only=True)
class MaterialPoint:
    elem: MaterialPointElement = attrs.field(default=None, metadata={"static": True})

    @property
    def required_aux(self) -> Sequence[str]:
        return self.elem.required_aux

    @property
    def required_params(self) -> Sequence[str]:
        return self.elem.required_params

    @apple.jit()
    def prepare(self, points: Float[jax.Array, "V 3"]) -> PyTree:
        return jax.vmap(self.elem.prepare)(points)

    @apple.jit()
    def fun(
        self, u: Float[jax.Array, "V 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, ""]:
        q = self.params_filter(q)
        aux = self.aux_filter(aux)
        return jnp.sum(jax.vmap(self.elem.fun)(u, q, aux))

    @apple.jit()
    def jac(
        self, u: Float[jax.Array, "V 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "V 3"]:
        q = self.params_filter(q)
        aux = self.aux_filter(aux)
        return jax.vmap(self.elem.jac)(u, q, aux)

    @apple.jit()
    def hess(
        self, u: Float[jax.Array, "V 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "V 3 3"]:
        q = self.params_filter(q)
        aux = self.aux_filter(aux)
        return jax.vmap(self.elem.hess)(u, q, aux)

    @apple.jit()
    def hessp(
        self,
        u: Float[jax.Array, "V 3"],
        p: Float[jax.Array, "V 3"],
        q: PyTree,
        aux: PyTree,
    ) -> Float[jax.Array, "V 3"]:
        q = self.params_filter(q)
        aux = self.aux_filter(aux)
        return jax.vmap(self.elem.hessp)(u, p, q, aux)

    @apple.jit()
    def hess_diag(
        self, u: Float[jax.Array, "V 3"], q: PyTree, aux: PyTree
    ) -> Float[jax.Array, "V 3"]:
        q = self.params_filter(q)
        aux = self.aux_filter(aux)
        return jax.vmap(self.elem.hess_diag)(u, q, aux)

    @apple.jit()
    def hess_quad(
        self,
        u: Float[jax.Array, "V 3"],
        p: Float[jax.Array, "V 3"],
        q: PyTree,
        aux: PyTree,
    ) -> Float[jax.Array, " C"]:
        q = self.params_filter(q)
        aux = self.aux_filter(aux)
        return jnp.sum(jax.vmap(self.elem.hess_quad)(u, p, q, aux))

    def aux_filter(self, aux: PyTree) -> PyTree:
        return {k: aux[k] for k in self.required_aux}

    def params_filter(self, q: PyTree) -> PyTree:
        return {k: q[k] for k in self.required_params}
