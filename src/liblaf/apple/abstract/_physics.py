import abc
from collections.abc import Callable
from typing import Protocol

import attrs
import glom
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from jaxtyping import Bool, Float, PyTree

from liblaf import apple


class Unraveler(Protocol):
    def __call__(self, flat: Float[jax.Array, " DoF"]) -> PyTree: ...


type UFree = Float[jax.Array, " DoF"]
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

    aux: PyTree = attrs.field(converter=apple.as_jax, factory=dict)
    dof_selection: Bool[jax.Array, " F"] | None = attrs.field(
        default=None,
        metadata={"static": True},
        converter=attrs.converters.optional(jnp.asarray),
    )
    name: str = attrs.field(metadata={"static": True})
    params: PyTree = attrs.field(converter=apple.as_jax, factory=dict)
    _unravel_q: Unraveler = attrs.field(
        default=None, metadata={"static": True}, alias="unravel_q"
    )
    _unravel_u: Unraveler = attrs.field(
        default=None, metadata={"static": True}, alias="unravel_u"
    )

    @property
    @abc.abstractmethod
    def n_dof(self) -> int: ...

    def prepare(self) -> None:
        self.aux = {}

    def select_dof(self, u_full: UFull) -> U:
        u: U = u_full if self.dof_selection is None else u_full[self.dof_selection]
        return u

    def get_param(self, key: str, q: PyTree | None = None) -> PyTree:
        return glom.glom(
            {"q": q, "self": self.params},
            glom.Coalesce(glom.Path("q", self.name, key), glom.Path("self", key)),
        )

    def fun(self, u_full: UFull, q: Q | None = None) -> Scalar:
        u_flat: UFlat = self.select_dof(u_full)
        result: Scalar = self._fun(u_flat, q)
        return result

    def jac(self, u_full: UFull, q: Q | None = None) -> UFull:
        u_flat: UFlat = self.select_dof(u_full)
        jac: UFlat = self._jac(u_flat, q)
        jac_full: UFull = jnp.zeros_like(u_full)
        jac_full = jac_full.at[self.dof_selection].set(jac.ravel())
        return jac_full

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
        hessp: U = self._hessp(u_flat, p_flat, q)
        hessp_flat: UFlat = self.ravel_u(hessp)
        hessp_full: UFull = jnp.zeros_like(u_full)
        hessp_full = hessp_full.at[self.dof_selection].set(hessp_flat)
        return hessp_full

    def hess_diag(self, u_full: UFull, q: Q | None = None) -> UFull:
        u_flat: UFlat = self.select_dof(u_full)
        diag: UFlat = self._hess_diag(u_flat, q)
        diag_flat: UFlat = self.ravel_u(diag)
        diag_full: UFull = jnp.zeros_like(u_full)
        diag_full = diag_full.at[self.dof_selection].set(diag_flat)
        return diag_full

    def hess_quad(self, u_full: UFull, p_full: UFull, q: Q | None = None) -> Scalar:
        u_flat: UFlat = self.select_dof(u_full)
        p_flat: UFlat = self.select_dof(p_full)
        quad: Scalar = self._hess_quad(u_flat, p_flat, q)
        return quad

    def ravel_q(self, q: Q | None) -> QFlat | None:
        if q is None:
            return None
        if apple.is_flat(q):
            return q
        q_flat: QFlat
        q_flat, self._unravel_q = jax.flatten_util.ravel_pytree(q)
        return q_flat

    def ravel_u(self, u: U) -> UFlat:
        u_flat: UFlat
        if apple.is_flat(u):
            u_flat = u
        else:
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
        return apple.hvp(self._fun, u_flat, p_flat, args=(q,))

    def _hess_diag(self, u: U, q: Q | None = None) -> UFlat:
        u_flat: UFlat = self.ravel_u(u)
        return apple.hess_diag(self._fun, u_flat, args=(q,))

    def _hess_quad(self, u: U, p: U, q: Q | None = None) -> Scalar:
        u_flat: UFlat = self.ravel_u(u)
        p_flat: UFlat = self.ravel_u(p)
        return jnp.dot(p_flat, self._hessp(u_flat, p_flat, q))


@attrs.define(kw_only=True)
class PhysicsProblem:
    def _default_dirichlet_mask(self) -> Bool[np.ndarray, " F"]:
        return np.zeros((self.n_dof_full,), dtype=bool)

    def _default_dirichlet_values(self) -> Float[jax.Array, " F"]:
        return jnp.zeros((self.n_dof_full,), dtype=jnp.float32)

    objects: list[AbstractObject] = attrs.field(factory=list)
    dirichlet_mask: Bool[np.ndarray, " F"] = attrs.field(
        default=attrs.Factory(_default_dirichlet_mask, takes_self=True),
        metadata={"static": True},
        converter=np.asarray,
    )
    dirichlet_values: Float[jax.Array, " F"] = attrs.field(
        default=attrs.Factory(_default_dirichlet_values, takes_self=True),
        converter=jnp.asarray,
    )

    @property
    def free_mask(self) -> Bool[np.ndarray, " F"]:
        return ~self.dirichlet_mask

    @property
    def n_dof(self) -> int:
        return self.n_dof_full - self.n_dirichlet

    @property
    def n_dof_full(self) -> int:
        return self.dirichlet_mask.size

    @property
    def n_dirichlet(self) -> int:
        return jnp.count_nonzero(self.dirichlet_mask)  # pyright: ignore[reportReturnType]

    def prepare(self) -> None:
        for obj in self.objects:
            obj.prepare()

    def solve(
        self,
        q: Q | None = None,
        u0: Float[jax.Array, " DoF"] | None = None,
        algo: apple.MinimizeAlgorithm | None = None,
        callback: Callable | None = None,
    ) -> apple.MinimizeResult:
        if u0 is None:
            u0 = jnp.zeros((self.n_dof,))
        return apple.minimize(
            x0=u0,
            fun=self.fun,
            jac=self.jac,
            hessp=self.hessp,
            hess_diag=self.hess_diag,
            hess_quad=self.hess_quad,
            args=(q,),
            algo=algo,
            callback=callback,
        )

    def fill(self, u_free: Float[jax.Array, " DoF"]) -> UFull:
        u_full: UFull = self.dirichlet_values.copy()
        u_full = u_full.at[self.free_mask].set(u_free)
        return u_full

    def fill_zeros(self, p_free: Float[jax.Array, " DoF"]) -> UFull:
        p_full: UFull = jnp.zeros_like(self.dirichlet_values)
        p_full = p_full.at[self.free_mask].set(p_free)
        return p_full

    def fun(self, u_free: UFree, q: Q | None = None) -> Scalar:
        u_full: UFull = self.fill(u_free)
        return jnp.sum(
            jnp.asarray([obj.fun(u_full, q) for obj in self.objects]), axis=0
        )

    def jac(self, u_free: UFree, q: Q | None = None) -> UFree:
        u_full: UFull = self.fill(u_free)
        jac_full: UFull = jnp.sum(
            jnp.asarray([obj.jac(u_full, q) for obj in self.objects]), axis=0
        )
        jac_free: UFree = jac_full[self.free_mask]
        return jac_free

    def hess(self, u_free: UFree, q: Q | None = None) -> Float[jax.Array, "F F"]:
        raise NotImplementedError

    def hessp(self, u_full: UFree, p_free: UFree, q: Q | None = None) -> UFull:
        u_full: UFull = self.fill(u_full)
        p_full: UFull = self.fill_zeros(p_free)
        hessp_full: UFull = jnp.sum(
            jnp.asarray([obj.hessp(u_full, p_full, q) for obj in self.objects]), axis=0
        )
        hessp_free: UFree = hessp_full[self.free_mask]
        return hessp_free

    def hess_diag(self, u_full: UFree, q: Q | None = None) -> UFull:
        u_full: UFull = self.fill(u_full)
        diag_full: UFull = jnp.sum(
            jnp.asarray([obj.hess_diag(u_full, q) for obj in self.objects]), axis=0
        )
        diag_free: UFree = diag_full[self.free_mask]
        return diag_free

    def hess_quad(self, u_free: UFree, p_free: UFree, q: Q | None = None) -> Scalar:
        u_full: UFull = self.fill(u_free)
        p_full: UFull = self.fill_zeros(p_free)
        quad: Scalar = jnp.sum(
            jnp.asarray([obj.hess_quad(u_full, p_full, q) for obj in self.objects]),
            axis=0,
        )
        return quad

    def jac_and_hess_diag(
        self,
        u_free: UFree,
        q: Q | None = None,
    ) -> tuple[UFree, UFree]:
        u_full: UFull = self.fill(u_free)
        jac_full: UFull = jnp.sum(
            jnp.asarray([obj.jac(u_full, q) for obj in self.objects]), axis=0
        )
        jac_free: UFree = jac_full[self.free_mask]
        diag_full: UFull = jnp.sum(
            jnp.asarray([obj.hess_diag(u_full, q) for obj in self.objects]), axis=0
        )
        diag_free: UFree = diag_full[self.free_mask]
        return jac_free, diag_free
