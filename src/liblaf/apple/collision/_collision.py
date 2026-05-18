import logging

import attrs
import ipctk
import jax.experimental
import jax.numpy as jnp
import numpy as np
import scipy.sparse
from jaxtyping import Array, Float, Integer

from liblaf import jarp

logger: logging.Logger = logging.getLogger(__name__)

type Full = Array[Float, "points dim"]
type Scalar = Array[Float, ""]
type VDim = Array[Float, "V dim"]


@jarp.define
class Collision:
    @jarp.define
    class State:
        marker: Scalar = jarp.array()

    collision_mesh: ipctk.CollisionMesh = jarp.static()
    indices: Integer[Array, " V"] = jarp.array()
    potential: ipctk.BarrierPotential = jarp.static()

    broad_phase: ipctk.BroadPhase = jarp.static(factory=ipctk.LBVH)
    candidates: ipctk.Candidates = jarp.static(factory=ipctk.Candidates)
    collisions: ipctk.NormalCollisions = jarp.static(factory=ipctk.NormalCollisions)
    narrow_phase_ccd: ipctk.NarrowPhaseCCD = jarp.static(
        factory=ipctk.TightInclusionCCD
    )

    def _default_inflation_radius(self) -> Scalar:
        return jnp.asarray(0.5 * self.potential.dhat)

    dmin: Scalar = jarp.array(default=jnp.zeros(()))
    inflation_radius: Scalar = jarp.array(
        default=attrs.Factory(_default_inflation_radius, takes_self=True)
    )
    min_distance: Scalar = jarp.array(default=jnp.zeros(()))

    def _default_vertices(self) -> VDim:
        return jnp.asarray(self.collision_mesh.rest_positions)

    vertices: Float[Array, "V dim"] = jarp.array(
        default=attrs.Factory(_default_vertices, takes_self=True)
    )

    def max_step_size(self, u: Full, p: Full) -> Scalar:
        vertices_t0: Full = self.vertices + u[self.indices]
        vertices_t1: Full = vertices_t0 + p[self.indices]
        alpha: Scalar = jax.pure_callback(
            self._compute_collision_free_stepsize,
            jnp.zeros(()),
            vertices_t0,
            vertices_t1,
        )
        return alpha

    def fun(self, u: Full) -> Scalar:
        vertices: VDim = self.vertices + u[self.indices]
        output = jax.pure_callback(self._fun, jnp.zeros(()), vertices=vertices)
        return output

    def grad(self, u: Full) -> Full:
        vertices: VDim = self.vertices + u[self.indices]
        grad: Float[Array, " V*dim"] = jax.pure_callback(
            self._gradient,
            jax.ShapeDtypeStruct((vertices.size,), float),
            vertices=vertices,
        )
        grad: VDim = grad.reshape(vertices.shape)
        grad_full: Full = jnp.zeros_like(u)
        grad_full = grad_full.at[self.indices].set(grad)
        return grad_full

    def hess_diag(self, u: Full) -> Full:
        vertices: VDim = self.vertices + u[self.indices]
        H_diag: Float[Array, " V*dim"] = jax.pure_callback(
            self._gauss_newton_hessian_diagonal,
            jax.ShapeDtypeStruct((vertices.size,), float),
            vertices=vertices,
        )
        H_diag: VDim = H_diag.reshape(vertices.shape)
        H_diag_full: Full = jnp.zeros_like(u)
        H_diag_full = H_diag_full.at[self.indices].set(H_diag)
        return H_diag_full

    def hess_prod(self, u: Full, p: Full) -> Full:
        vertices: VDim = self.vertices + u[self.indices]
        Hp: VDim = jax.pure_callback(
            self._hessian_prod,
            jax.ShapeDtypeStruct((vertices.size,), float),
            vertices=vertices,
            p=p[self.indices],
        )
        Hp_full: Full = jnp.zeros_like(u)
        Hp_full = Hp_full.at[self.indices].set(Hp)
        return Hp_full

    def hess_quad(self, u: Full, p: Full) -> Scalar:
        vertices: VDim = self.vertices + u[self.indices]
        pHp: Scalar = jax.pure_callback(
            self._gauss_newton_hessian_quadratic_form,
            jnp.zeros(()),
            vertices=vertices,
            p=p[self.indices],
        )
        return pHp

    def _compute_collision_free_stepsize(
        self, vertices_t0: VDim, vertices_t1: VDim
    ) -> Scalar:
        vertices_t0: Float[np.ndarray, "V dim"] = np.asarray(vertices_t0)
        vertices_t1: Float[np.ndarray, "V dim"] = np.asarray(vertices_t1)
        self.candidates.clear()
        self.candidates.build(
            mesh=self.collision_mesh,
            vertices_t0=vertices_t0,
            vertices_t1=vertices_t1,
            inflation_radius=self.inflation_radius,
            broad_phase=self.broad_phase,
        )
        alpha: float = self.candidates.compute_collision_free_stepsize(
            mesh=self.collision_mesh,
            vertices_t0=vertices_t0,
            vertices_t1=vertices_t1,
            min_distance=self.min_distance,
            narrow_phase_ccd=self.narrow_phase_ccd,
        )
        logger.debug("compute_collision_free_stepsize(): %f", alpha)
        return jnp.asarray(alpha)

    def _fun(self, vertices: VDim) -> Scalar:
        vertices = np.asarray(vertices)
        # self.candidates.clear()
        # self.candidates.build(
        #     mesh=self.collision_mesh,
        #     vertices=vertices,
        #     inflation_radius=self.inflation_radius,
        #     broad_phase=self.broad_phase,
        # )
        self.collisions.clear()
        self.collisions.build(
            candidates=self.candidates,
            mesh=self.collision_mesh,
            vertices=vertices,
            dhat=self.potential.dhat,
            dmin=self.dmin,
        )
        logger.debug("fun(): %d collisions", len(self.collisions))
        fun: float = self.potential(
            collisions=self.collisions, mesh=self.collision_mesh, X=vertices
        )
        return jnp.asarray(fun)

    def _gauss_newton_hessian_diagonal(self, vertices: VDim) -> VDim:
        vertices = np.asarray(vertices)
        # self.candidates.clear()
        # self.candidates.build(
        #     mesh=self.collision_mesh,
        #     vertices=vertices,
        #     inflation_radius=self.inflation_radius,
        #     broad_phase=self.broad_phase,
        # )
        self.collisions.clear()
        self.collisions.build(
            candidates=self.candidates,
            mesh=self.collision_mesh,
            vertices=vertices,
            dhat=self.potential.dhat,
            dmin=self.dmin,
        )
        logger.debug("hess_diag(): %d collisions", len(self.collisions))
        H_diag: Float[np.ndarray, " V*dim"] = (
            self.potential.gauss_newton_hessian_diagonal(
                collisions=self.collisions, mesh=self.collision_mesh, vertices=vertices
            )
        )
        return jnp.asarray(H_diag)

    def _gauss_newton_hessian_quadratic_form(self, vertices: VDim, p: VDim) -> Scalar:
        vertices = np.asarray(vertices)
        p = np.asarray(p).reshape(-1)
        # self.candidates.clear()
        # self.candidates.build(
        #     mesh=self.collision_mesh,
        #     vertices=vertices,
        #     inflation_radius=self.inflation_radius,
        #     broad_phase=self.broad_phase,
        # )
        self.collisions.clear()
        self.collisions.build(
            candidates=self.candidates,
            mesh=self.collision_mesh,
            vertices=vertices,
            dhat=self.potential.dhat,
            dmin=self.dmin,
        )
        logger.debug("hess_quad(): %d collisions", len(self.collisions))
        pHp: float = self.potential.gauss_newton_hessian_quadratic_form(
            collisions=self.collisions, mesh=self.collision_mesh, vertices=vertices, p=p
        )
        return jnp.asarray(pHp)

    def _gradient(self, vertices: VDim) -> Float[Array, " V*dim"]:
        vertices: Float[np.ndarray, " V*dim"] = np.asarray(vertices)
        # self.candidates.clear()
        # self.candidates.build(
        #     mesh=self.collision_mesh,
        #     vertices=vertices,
        #     inflation_radius=self.inflation_radius,
        #     broad_phase=self.broad_phase,
        # )
        self.collisions.clear()
        self.collisions.build(
            candidates=self.candidates,
            mesh=self.collision_mesh,
            vertices=vertices,
            dhat=self.potential.dhat,
            dmin=self.dmin,
        )
        logger.debug("grad(): %d collisions", len(self.collisions))
        grad: Float[np.ndarray, " V*dim"] = self.potential.gradient(
            collisions=self.collisions, mesh=self.collision_mesh, X=vertices
        )
        return jnp.asarray(grad)

    def _hessian_prod(self, vertices: VDim, p: VDim) -> Float[Array, " V*dim"]:
        vertices: Float[np.ndarray, "V dim"] = np.asarray(vertices)
        p: Float[np.ndarray, "V dim"] = np.asarray(p)
        # self.candidates.clear()
        # self.candidates.build(
        #     mesh=self.collision_mesh,
        #     vertices=vertices,
        #     inflation_radius=self.inflation_radius,
        #     broad_phase=self.broad_phase,
        # )
        self.collisions.clear()
        self.collisions.build(
            candidates=self.candidates,
            mesh=self.collision_mesh,
            vertices=vertices,
            dhat=self.potential.dhat,
            dmin=self.dmin,
        )
        logger.debug("hess_prod(): %d collisions", len(self.collisions))
        H: scipy.sparse.csc_matrix = self.potential.hessian(
            collisions=self.collisions, mesh=self.collision_mesh, X=vertices
        )
        Hp: Float[np.ndarray, " V*dim"] = H @ p.flat
        return jnp.asarray(Hp)
