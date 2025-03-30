import abc
from typing import Protocol

import attrs
import glom
import jax
import jax.flatten_util
import jax.numpy as jnp
from jaxtyping import Bool, Float, PyTree

from liblaf import apple


class Unraveler(Protocol):
    def __call__(self, flat: Float[jax.Array, " DoF"]) -> PyTree: ...


type UFull = Float[jax.Array, " F"]
type QFlat = Float[jax.Array, " Q"]
type QTree = PyTree
type Q = QFlat | QTree
type UFlat = Float[jax.Array, " N"]
type UTree = PyTree
type U = UFlat | UTree
type H = Float[jax.Array, " N N"]
type HFull = Float[jax.Array, " F F"]
type Scalar = Float[jax.Array, " "]


@attrs.define(kw_only=True)
class AbstractObject(abc.ABC):
    """...

    Annotations:
        - `F`: number of degrees of freedom in the system after applying constraints
        - `N`: number of degrees of freedom in the object without constraints (usually `N = V * 3`)
        - `Q`: number of parameters in the system
        - `V`: number of points / vertices in the object
    """

    name: str = attrs.field(metadata={"static": True})
    aux: PyTree = attrs.field(factory=dict)
    params: PyTree = attrs.field(factory=dict)
    dof_selection: Bool[jax.Array, " F"] = attrs.field(
        metadata={"static": True}, converter=jnp.asarray
    )
    _unravel_q: Unraveler = attrs.field(
        default=None, metadata={"static": True}, alias="unravel_q"
    )
    _unravel_u: Unraveler = attrs.field(
        default=None, metadata={"static": True}, alias="unravel_u"
    )

    def select_dof(self, u_full: UFull) -> U:
        return u_full[self.dof_selection]

    def get_param(self, key: str, q: PyTree | None = None) -> PyTree:
        return glom.glom(
            {"q": q, "self": self.params},
            glom.Coalesce(glom.Path("q", self.name, key), glom.Path("self", key)),
        )

    def fun(self, u_full: UFull, q: Q | None = None) -> Scalar:
        u_flat: UFlat = self.select_dof(u_full)
        return self._fun(u_flat, q)

    def jac(self, u_full: UFull, q: Q | None = None) -> UFull:
        u_flat: UFlat = self.select_dof(u_full)
        jac: UFlat = self._jac(u_flat, q)
        jac_full: UFull = jnp.zeros_like(u_full)
        jac_full = jac_full.at[self.dof_selection].set(jac)
        return jac

    def hess(self, u_full: UFull, q: Q | None = None) -> HFull:
        u_flat: U = self.select_dof(u_full)
        u: PyTree = self.unravel_u(u_flat)
        hess: H = self._hess(u, q)
        hess_full: HFull = jnp.zeros((u_full.size, u_full.size))
        hess_full = hess_full.at[self.dof_selection, self.dof_selection].set(hess)
        return hess_full

    def hessp(self, u_full: UFull, p_full: UFull, q: Q | None = None) -> UFull:
        u_flat: UFlat = self.select_dof(u_full)
        p_flat: UFlat = self.select_dof(p_full)
        hessp: UFlat = self._hessp(u_flat, p_flat, q)
        hessp_full: UFull = jnp.zeros_like(u_full)
        hessp_full = hessp_full.at[self.dof_selection].set(hessp)
        return hessp_full

    def hess_diag(self, u_full: UFull, q: Q | None = None) -> UFull:
        u_flat: UFlat = self.select_dof(u_full)
        hess_diag: UFlat = self._hess_diag(u_flat, q)
        hess_diag_full: UFull = jnp.zeros_like(u_full)
        hess_diag_full = hess_diag_full.at[self.dof_selection].set(hess_diag)
        return hess_diag_full

    def hess_quad(self, u_full: UFull, p_full: UFull, q: Q | None = None) -> Scalar:
        u_flat: UFlat = self.select_dof(u_full)
        p_flat: UFlat = self.select_dof(p_full)
        hess_quad: UFull = self._hess_quad(u_flat, p_flat, q)
        hess_quad_full: UFull = jnp.zeros_like(u_full)
        hess_quad_full = hess_quad_full.at[self.dof_selection].set(hess_quad)
        return hess_quad_full

    def prepare(self) -> None:
        self.aux = {}

    def ravel_q(self, q: Q | None) -> QFlat | None:
        if q is None:
            return None
        if apple.is_flat(q):
            return q
        q_flat: QFlat
        q_flat, self._unravel_q = jax.flatten_util.ravel_pytree(q)
        return q_flat

    def ravel_u(self, u: U) -> UFlat:
        if apple.is_flat(u):
            return u
        u_flat: UFlat
        u_flat, self._unravel_u = jax.flatten_util.ravel_pytree(u)
        return u_flat

    def unravel_q(self, q: Q | None) -> QTree | None:
        if q is None:
            return None
        if not apple.is_flat(q):
            return q
        return self._unravel_q(q)

    def unravel_u(self, u: U) -> UTree:
        if not apple.is_flat(u):
            return u
        return self._unravel_u(u)

    @abc.abstractmethod
    def _fun(self, u: U, q: Q | None = None) -> Scalar: ...

    def _jac(self, u: U, q: Q | None = None) -> UFlat:
        u_flat: UFlat = self.ravel_u(u)
        return jax.grad(self._fun)(u_flat, q)

    def _hess(self, u: U, q: Q | None = None) -> H:
        u_flat: UFlat = self.ravel_u(u)
        return jax.hessian(self._fun)(u_flat, q)

    def _hessp(self, u: U, p: U, q: Q | None = None) -> UFlat:
        u_flat: UFlat = self.ravel_u(u)
        p_flat: UFlat = self.ravel_u(p)
        return apple.hvp(lambda u_flat: self._fun(u_flat, q), u_flat, p_flat)

    def _hess_diag(self, u: U, q: Q | None = None) -> UFlat:
        u_flat: UFlat = self.ravel_u(u)
        return apple.hess_diag(lambda u_flat: self._fun(u_flat, q), u_flat)

    def _hess_quad(self, u: U, p: U, q: Q | None = None) -> Scalar:
        u_flat: UFlat = self.ravel_u(u)
        p_flat: UFlat = self.ravel_u(p)
        return jnp.dot(p_flat, self._hessp(u_flat, p_flat, q))


class PhysicsProblem:
    objects: list[AbstractObject] = attrs.field(factory=list)
    dirichlet_mask: Bool[jax.Array, " F"] = attrs.field(metadata={"static": True})
    dirichlet_values: Float[jax.Array, " F"] = attrs.field(converter=jnp.asarray)

    def fun(self, u_full: UFull, q: Q | None = None) -> Scalar:
        return jnp.sum(
            jnp.asarray([obj.fun(u_full, q) for obj in self.objects]), axis=0
        )

    def jac(self, u_full: UFull, q: Q | None = None) -> UFull:
        return jnp.sum(
            jnp.asarray([obj.jac(u_full, q) for obj in self.objects]), axis=0
        )

    def hess(self, u_full: UFull, q: Q | None = None) -> Float[jax.Array, "F F"]:
        return jnp.sum(
            jnp.asarray([obj.hess(u_full, q) for obj in self.objects]), axis=0
        )

    def hessp(self, u_full: UFull, p_full: UFull, q: Q | None = None) -> UFull:
        return jnp.sum(
            jnp.asarray([obj.hessp(u_full, p_full, q) for obj in self.objects]), axis=0
        )

    def hess_diag(self, u_full: UFull, q: Q | None = None) -> UFull:
        return jnp.sum(
            jnp.asarray([obj.hess_diag(u_full, q) for obj in self.objects]), axis=0
        )

    def hess_quad(self, u_full: UFull, p_full: UFull, q: Q | None = None) -> Scalar:
        return jnp.sum(
            jnp.asarray([obj.hess_quad(u_full, p_full, q) for obj in self.objects]),
            axis=0,
        )
