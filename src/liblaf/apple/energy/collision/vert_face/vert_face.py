from typing import Self, override

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Integer

from liblaf.apple import sim, struct, utils

from .kernel import (
    collision_detect_vert_face_kernel,
    collision_energy_vert_face_fun_callable,
    collision_energy_vert_face_hess_diag_callable,
    collision_energy_vert_face_hess_quad_callable,
)


@struct.pytree
class CollisionCandidatesVertFace(struct.PyTreeMixin):
    closest: Float[Array, " points 3"] = struct.array(default=None)
    collision_to_vertex: Integer[Array, " points"] = struct.array(default=None)
    count: Integer[Array, ""] = struct.array(default=None)
    distance: Float[Array, " points"] = struct.array(default=None)
    face_id: Integer[Array, " points"] = struct.array(default=None)
    face_normal: Float[Array, "points 3"] = struct.array(default=None)
    uv: Float[Array, "points 2"] = struct.array(default=None)
    vertex_to_collision: Integer[Array, " points"] = struct.array(default=None)


@struct.pytree
class CollisionVertFace(sim.Energy):
    rigid: sim.Actor = struct.data(default=None)
    soft: sim.Actor = struct.data(default=None)

    stiffness: Float[Array, ""] = struct.array(default=1e3)
    rest_length: Float[Array, ""] = struct.array(default=1e-3)
    max_dist: Float[Array, ""] = struct.array(default=1e-2)
    epsilon: Float[Array, ""] = struct.array(default=1e-3)

    candidates: CollisionCandidatesVertFace = struct.data(
        factory=CollisionCandidatesVertFace
    )

    @classmethod
    def from_actors(
        cls,
        rigid: sim.Actor,
        soft: sim.Actor,
        *,
        stiffness: float = 1e3,
        rest_length: float = 1e-3,
        max_dist: float | None = None,
        epsilon: float = 1e-3,
    ) -> Self:
        if max_dist is None:
            max_dist = 2.0 * rest_length
        return cls(
            rigid=rigid,
            soft=soft,
            stiffness=jnp.asarray(stiffness),
            rest_length=jnp.asarray(rest_length),
            max_dist=jnp.asarray(max_dist),
            epsilon=jnp.asarray(epsilon),
        )

    @property
    @override
    def actors(self) -> struct.NodeContainer[sim.Actor]:
        return struct.NodeContainer([self.rigid, self.soft])

    @override
    def with_actors(self, actors: struct.NodeContainer[sim.Actor]) -> Self:
        return self.evolve(rigid=actors[self.rigid.id], soft=actors[self.soft.id])

    @override
    def pre_optim_iter(self, params: sim.GlobalParams) -> Self:
        candidates: CollisionCandidatesVertFace = self.collide()
        return self.evolve(candidates=candidates)

    @override
    @utils.jit_method(inline=True)
    def fun(self, x: struct.ArrayDict, /, params: sim.GlobalParams) -> Float[Array, ""]:
        points: Float[Array, "points dim"] = self.soft.points + x[self.soft.id]
        energy: Float[Array, " 1"]
        (energy,) = collision_energy_vert_face_fun_callable(
            points,
            self.candidates.closest,
            self.candidates.collision_to_vertex,
            self.candidates.distance,
            self.candidates.count.reshape((1,)),
            self.rest_length.reshape((1,)),
            self.stiffness.reshape((1,)),
            output_dims={"energy": (1,)},
        )
        return energy.squeeze()

    @override
    @utils.jit_method(inline=True)
    def jac(self, x: struct.ArrayDict, /, params: sim.GlobalParams) -> struct.ArrayDict:
        points: Float[Array, "points dim"] = self.soft.points + x[self.soft.id]
        jac_soft: Float[Array, " points dim"]
        (jac_soft,) = collision_energy_vert_face_fun_callable(
            points,
            self.candidates.closest,
            self.candidates.collision_to_vertex,
            self.candidates.distance,
            self.candidates.count.reshape((1,)),
            self.rest_length.reshape((1,)),
            self.stiffness.reshape((1,)),
            output_dims={"jac": (self.soft.n_points,)},
        )
        return struct.ArrayDict(
            {self.soft.id: jac_soft, self.rigid.id: jnp.zeros_like(x[self.rigid.id])}
        )

    @override
    @utils.jit_method(inline=True)
    def hess_diag(
        self, x: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> struct.ArrayDict:
        points: Float[Array, "points dim"] = self.soft.points + x[self.soft.id]
        hess_diag: Float[Array, " points dim"]
        (hess_diag,) = collision_energy_vert_face_hess_diag_callable(
            points,
            self.candidates.closest,
            self.candidates.collision_to_vertex,
            self.candidates.distance,
            self.candidates.count.reshape((1,)),
            self.rest_length.reshape((1,)),
            self.stiffness.reshape((1,)),
            output_dims={"hess_diag": (self.soft.n_points,)},
        )
        return struct.ArrayDict(
            {self.soft.id: hess_diag, self.rigid.id: jnp.zeros_like(x[self.rigid.id])}
        )

    @override
    @utils.jit_method(inline=True)
    def hess_quad(
        self, x: struct.ArrayDict, p: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> Float[Array, ""]:
        points: Float[Array, "points dim"] = self.soft.points + x[self.soft.id]
        hess_quad: Float[Array, " 1"]
        (hess_quad,) = collision_energy_vert_face_hess_quad_callable(
            points,
            p[self.soft.id],
            self.candidates.closest,
            self.candidates.collision_to_vertex,
            self.candidates.distance,
            self.candidates.count.reshape((1,)),
            self.rest_length.reshape((1,)),
            self.stiffness.reshape((1,)),
            output_dims={"hess_quad": (1,)},
        )
        return hess_quad.squeeze()

    def collide(self) -> CollisionCandidatesVertFace:
        (
            closest,
            collision_to_vertex,
            count,
            distance,
            face_id,
            face_normal,
            uv,
            vertex_to_collision,
        ) = collision_detect_vert_face_kernel(
            self.soft.points,
            np.uint64(self.rigid.collision_mesh.id),
            self.rest_length.reshape((1,)),
            self.max_dist.reshape((1,)),
            self.epsilon.reshape((1,)),
            output_dims={
                "closest": (self.soft.n_points,),
                "collision_to_vertex": (self.soft.n_points,),
                "count": (1,),
                "distance": (self.soft.n_points,),
                "face_id": (self.soft.n_points,),
                "face_normal": (self.soft.n_points,),
                "uv": (self.soft.n_points,),
                "vertex_to_collision": (self.soft.n_points,),
            },
            launch_dims=(self.soft.n_points,),
        )
        return CollisionCandidatesVertFace(
            closest=closest,
            collision_to_vertex=collision_to_vertex,
            count=count.squeeze(),
            distance=distance,
            face_id=face_id,
            face_normal=face_normal,
            uv=uv,
            vertex_to_collision=vertex_to_collision,
        )
