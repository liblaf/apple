from typing import override

import attrs
import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Float, PyTree

from liblaf import apple

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
class ObjectPoint(apple.AbstractObject):
    clip: bool = True
    data: pv.PointSet
    material: apple.material.point.MaterialPoint

    @property
    @override
    def n_dof(self) -> int:
        return self.n_points * 3

    @property
    def n_points(self) -> int:
        return self.data.n_points

    @property
    def points(self) -> Float[jax.Array, "V 3"]:
        return jnp.asarray(self.data.points)

    @override
    def ravel_u(self, u: U) -> UFlat:
        return u.ravel()

    @override
    def unravel_u(self, u: U) -> UTree:
        return u.reshape(self.n_points, 3)

    @override
    def _fun(self, u: U, q: Q | None = None) -> Scalar:
        u: UTree = self.unravel_u(u)
        q: QTree | None = self.unravel_q(q)
        params: PyTree = {
            k: self.get_param(k, q) for k in self.material.required_params
        }
        return self.material.fun(u, params, self.aux)

    @override
    def _jac(self, u: U, q: Q | None = None) -> U:
        u: UTree = self.unravel_u(u)
        q: QTree | None = self.unravel_q(q)
        params: PyTree = {
            k: self.get_param(k, q) for k in self.material.required_params
        }
        jac: Float[jax.Array, "V 3"] = self.material.jac(u, params, self.aux)
        return jac

    @override
    def _hessp(self, u: U, p: U, q: Q | None = None) -> U:
        u: UTree = self.unravel_u(u)
        p: UTree = self.unravel_u(p)
        q: QTree | None = self.unravel_q(q)
        params: PyTree = {
            k: self.get_param(k, q) for k in self.material.required_params
        }
        hessp: Float[jax.Array, "V 3"] = self.material.hessp(u, p, params, self.aux)
        return hessp

    @override
    def _hess_diag(self, u: U, q: Q | None = None) -> U:
        u: UTree = self.unravel_u(u)
        q: QTree | None = self.unravel_q(q)
        params: PyTree = {
            k: self.get_param(k, q) for k in self.material.required_params
        }
        diag: Float[jax.Array, "V 3"] = self.material.hess_diag(u, params, self.aux)
        if self.clip:
            diag = jnp.clip(diag, min=0.0)
        return diag

    @override
    def _hess_quad(self, u: U, p: U, q: Q | None = None) -> Scalar:
        u: UTree = self.unravel_u(u)
        p: UTree = self.unravel_u(p)
        q: QTree | None = self.unravel_q(q)
        params: PyTree = {
            k: self.get_param(k, q) for k in self.material.required_params
        }
        pHp: Float[jax.Array, " V"] = self.material.hess_quad(u, p, params, self.aux)
        if self.clip:
            pHp = jnp.clip(pHp, min=0.0)
        return jnp.sum(pHp)
