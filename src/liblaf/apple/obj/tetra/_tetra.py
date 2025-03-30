from typing import override

import attrs
import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Float, Integer, PyTree

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
class ObjectTetra(apple.AbstractObject):
    clip: bool = True
    material: apple.material.tetra.MaterialTetra
    mesh: pv.UnstructuredGrid

    @property
    def cells(self) -> Integer[jax.Array, "C 4"]:
        return jnp.asarray(self.mesh.cells_dict[pv.CellType.TETRA])

    @property
    def n_points(self) -> int:
        return self.mesh.n_points

    @override
    def unravel_u(self, u: U) -> UTree:
        return u.reshape(self.n_points, 3)

    @override
    def _fun(self, u: U, q: Q | None = None) -> Float[jax.Array, ""]:
        u: UTree = self.unravel_u(u)
        q: QTree | None = self.unravel_q(q)
        params: PyTree = {
            k: self.get_param(k, q) for k in self.material.required_params
        }
        return self.material.fun(u[self.cells], params, self.aux)

    @override
    def _jac(self, u: U, q: Q | None = None) -> UFlat:
        u: UTree = self.unravel_u(u)
        q: QTree | None = self.unravel_q(q)
        params: PyTree = {
            k: self.get_param(k, q) for k in self.material.required_params
        }
        jac: Float[jax.Array, "C 4 3"] = self.material.jac(
            u[self.cells], params, self.aux
        )
        jac: Float[jax.Array, "V 3"] = apple.elem.tetra.segment_sum(
            jac, self.cells, self.n_points
        )
        return jac

    @override
    def _hessp(self, u: U, p: U, q: Q | None = None) -> UFlat:
        u: UTree = self.unravel_u(u)
        p: UTree = self.unravel_u(p)
        q: QTree | None = self.unravel_q(q)
        params: PyTree = {
            k: self.get_param(k, q) for k in self.material.required_params
        }
        hessp: Float[jax.Array, "C 4 3"] = self.material.hessp(
            u[self.cells], p[self.cells], params, self.aux
        )
        hessp: Float[jax.Array, "V 3"] = apple.elem.tetra.segment_sum(
            hessp, self.cells, self.n_points
        )
        return hessp

    @override
    def _hess_diag(self, u: U, q: Q | None = None) -> UFlat:
        u: UTree = self.unravel_u(u)
        q: QTree | None = self.unravel_q(q)
        params: PyTree = {
            k: self.get_param(k, q) for k in self.material.required_params
        }
        diag: Float[jax.Array, "C 4 3"] = self.material.hess_diag(
            u[self.cells], params, self.aux
        )
        if self.clip:
            diag = jnp.clip(diag, min=0.0)
        diag: Float[jax.Array, "V 3"] = apple.elem.tetra.segment_sum(
            diag, self.cells, self.n_points
        )
        return diag

    @override
    def _hess_quad(self, u: U, p: U, q: Q | None = None) -> Scalar:
        u: UTree = self.unravel_u(u)
        p: UTree = self.unravel_u(p)
        q: QTree | None = self.unravel_q(q)
        params: PyTree = {
            k: self.get_param(k, q) for k in self.material.required_params
        }
        pHp: Float[jax.Array, " C"] = self.material.hess_quad(
            u[self.cells], p[self.cells], params, self.aux
        )
        if self.clip:
            pHp = jnp.clip(pHp, min=0.0)
        return jnp.sum(pHp)
