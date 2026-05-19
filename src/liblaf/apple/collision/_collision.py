import logging

import attrs
import ipctk
import numpy as np
import scipy.sparse
import torch
from jaxtyping import Float, Integer
from torch import Tensor

logger: logging.Logger = logging.getLogger(__name__)

type Full = Float[Tensor, "points dim"]
type Scalar = Float[Tensor, ""]
type VDim = Float[Tensor, "V dim"]


@attrs.define
class Collision:
    @attrs.define
    class State:
        candidates: ipctk.Candidates = attrs.field(factory=ipctk.Candidates)
        collisions: ipctk.NormalCollisions = attrs.field(factory=ipctk.NormalCollisions)
        hess: scipy.sparse.csc_matrix | None = None

    @staticmethod
    def _default_broad_phase() -> ipctk.BroadPhase:
        if torch.cuda.is_available() and hasattr(ipctk, "SweepAndTiniestQueue"):
            logger.debug("broad phase: SweepAndTiniestQueue")
            return ipctk.SweepAndTiniestQueue()
        logger.debug("broad phase: LBVH")
        return ipctk.LBVH()

    def _default_inflation_radius(self) -> float:
        return 0.5 * self.potential.dhat

    def _default_vertices(self) -> VDim:
        return torch.as_tensor(self.collision_mesh.rest_positions)

    collision_mesh: ipctk.CollisionMesh
    indices: Integer[Tensor, " V"]
    potential: ipctk.BarrierPotential

    broad_phase: ipctk.BroadPhase = attrs.field(factory=_default_broad_phase)
    narrow_phase_ccd: ipctk.NarrowPhaseCCD = attrs.field(
        factory=lambda: ipctk.TightInclusionCCD(tolerance=1e-3, max_iterations=10**3)
    )

    dmin: float = attrs.field(default=0.0)
    inflation_radius: float = attrs.field(
        default=attrs.Factory(_default_inflation_radius, takes_self=True)
    )
    min_distance: float = attrs.field(default=0.0)
    use_physical_barrier: bool = attrs.field(default=True)

    vertices: Float[Tensor, "V dim"] = attrs.field(
        default=attrs.Factory(_default_vertices, takes_self=True)
    )

    def init(self) -> State:
        state: Collision.State = Collision.State()
        if self.use_physical_barrier:
            state.collisions.use_area_weighting = True
            state.collisions.collision_set_type = (
                ipctk.NormalCollisions.CollisionSetType.IMPROVED_MAX_APPROX
            )
        vertices: Float[np.ndarray, "V dim"] = self.vertices.numpy(force=True)
        state.candidates.build(
            mesh=self.collision_mesh,
            vertices=vertices,
            inflation_radius=self.inflation_radius,
            broad_phase=self.broad_phase,
        )
        state.collisions.build(
            candidates=state.candidates,
            mesh=self.collision_mesh,
            vertices=vertices,
            dhat=self.potential.dhat,
            dmin=self.dmin,
        )
        return state

    def max_step_size(self, state: State, u: Full, p: Full) -> Scalar:
        vertices_t0: Full = self.vertices + u[self.indices]
        vertices_t1: Full = vertices_t0 + p[self.indices]
        vertices_t0: Float[np.ndarray, "V dim"] = vertices_t0.numpy(force=True)
        vertices_t1: Float[np.ndarray, "V dim"] = vertices_t1.numpy(force=True)
        state.candidates.clear()
        state.candidates.build(
            mesh=self.collision_mesh,
            vertices_t0=vertices_t0,
            vertices_t1=vertices_t1,
            inflation_radius=self.inflation_radius,
            broad_phase=self.broad_phase,
        )
        # if state.candidates.is_step_collision_free(
        #     mesh=self.collision_mesh,
        #     vertices_t0=vertices_t0,
        #     vertices_t1=vertices_t1,
        #     min_distance=self.min_distance,
        #     narrow_phase_ccd=self.narrow_phase_ccd,
        # ):
        #     print("step is collision free")
        #     return torch.ones((), dtype=u.dtype)
        alpha: float = state.candidates.compute_collision_free_stepsize(
            mesh=self.collision_mesh,
            vertices_t0=vertices_t0,
            vertices_t1=vertices_t1,
            min_distance=self.min_distance,
            narrow_phase_ccd=self.narrow_phase_ccd,
        )
        print("CCD alpha:", alpha)
        return torch.as_tensor(alpha)

    def update(self, state: State, u: Full) -> None:
        vertices: VDim = self.vertices + u[self.indices]
        vertices: Float[np.ndarray, "V dim"] = vertices.numpy(force=True)
        state.candidates.clear()
        state.candidates.build(
            mesh=self.collision_mesh,
            vertices=vertices,
            inflation_radius=self.inflation_radius,
            broad_phase=self.broad_phase,
        )
        state.collisions.clear()
        state.collisions.build(
            candidates=state.candidates,
            mesh=self.collision_mesh,
            vertices=vertices,
            dhat=self.potential.dhat,
            dmin=self.dmin,
        )
        state.hess = None

    def fun(self, state: State, u: Full) -> Scalar:
        vertices: VDim = self.vertices + u[self.indices]
        vertices: Float[np.ndarray, "V dim"] = vertices.numpy(force=True)
        fun: float = self.potential(
            collisions=state.collisions, mesh=self.collision_mesh, X=vertices
        )
        return torch.as_tensor(fun)

    def grad(self, state: State, u: Full, output: Full) -> None:
        vertices: VDim = self.vertices + u[self.indices]
        vertices: Float[np.ndarray, "V dim"] = vertices.numpy(force=True)
        grad: Float[np.ndarray, " V*dim"] = self.potential.gradient(
            state.collisions, mesh=self.collision_mesh, X=vertices
        )
        grad: Float[Tensor, " V*dim"] = torch.as_tensor(grad)
        grad: VDim = grad.reshape(vertices.shape)
        output.index_add_(0, self.indices, grad)

    def hess_diag(self, state: State, u: Full, output: Full) -> None:
        vertices: VDim = self.vertices + u[self.indices]
        vertices: Float[np.ndarray, "V dim"] = vertices.numpy(force=True)
        H_diag: Float[np.ndarray, " V*dim"] = (
            self.potential.gauss_newton_hessian_diagonal(
                collisions=state.collisions, mesh=self.collision_mesh, vertices=vertices
            )
        )
        H_diag: Float[Tensor, " V*dim"] = torch.as_tensor(H_diag)
        H_diag: VDim = H_diag.reshape(vertices.shape)
        output.index_add_(0, self.indices, H_diag)

    def hess_prod(self, state: State, u: Full, p: Full, output: Full) -> None:
        if state.hess is None:
            vertices: VDim = self.vertices + u[self.indices]
            vertices: Float[np.ndarray, "V dim"] = vertices.numpy(force=True)
            state.hess = self.potential.hessian(
                collisions=state.collisions, mesh=self.collision_mesh, X=vertices
            )
        p: Float[Tensor, " V*dim"] = p[self.indices].flatten()
        p: Float[np.ndarray, " V*dim"] = p.numpy(force=True)
        Hp: Float[np.ndarray, " V*dim"] = state.hess @ p
        Hp: VDim = torch.as_tensor(Hp).reshape(self.vertices.shape)
        output.index_add_(0, self.indices, Hp)

    def hess_quad(self, state: State, u: Full, p: Full) -> Scalar:
        vertices: VDim = self.vertices + u[self.indices]
        vertices: Float[np.ndarray, "V dim"] = vertices.numpy(force=True)
        p: VDim = p[self.indices]
        p: Float[Tensor, " V*dim"] = p.flatten()
        p: Float[np.ndarray, " V*dim"] = p.numpy(force=True)
        pHp: float = self.potential.gauss_newton_hessian_quadratic_form(
            collisions=state.collisions,
            mesh=self.collision_mesh,
            vertices=vertices,
            p=p,
        )
        return torch.as_tensor(pHp)
