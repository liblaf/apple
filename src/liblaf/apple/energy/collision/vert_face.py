from typing import Self, no_type_check, override

import einops
import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from jaxtyping import Array, Bool, Float, Integer

from liblaf.apple import sim, struct, utils


@struct.pytree
class CandidatesVertFace(struct.PyTreeMixin):
    collide: Bool[Array, " points"] = struct.array(default=None, kw_only=True)
    sign: Float[Array, " points"] = struct.array(default=None, kw_only=True)
    face: Integer[Array, " points"] = struct.array(default=None, kw_only=True)
    face_normal: Float[Array, "points dim"] = struct.array(default=None, kw_only=True)
    target: Float[Array, "points dim"] = struct.array(default=None, kw_only=True)
    uv: Float[Array, "points 2"] = struct.array(default=None, kw_only=True)


@struct.pytree
class CollisionVertFace(sim.Energy):
    rigid_id: str = struct.static(kw_only=True)
    soft_id: str = struct.static(kw_only=True)

    max_dist: Float[Array, ""] = struct.array(default=1e-2)
    rest_length: Float[Array, ""] = struct.array(default=1e-3)
    stiffness: Float[Array, ""] = struct.array(default=1e3)

    candidates: CandidatesVertFace = struct.data(factory=CandidatesVertFace)

    @classmethod
    def from_actors(
        cls,
        soft: sim.Actor,
        rigid: sim.Actor,
        *,
        max_dist: float | None = None,
        rest_length: float = 1e-3,
        stiffness: float = 1e6,
    ) -> Self:
        if max_dist is None:
            max_dist = rest_length * 10
        return cls(
            actors=struct.NodeContainer([soft, rigid]),
            soft_id=soft.id,
            rigid_id=rigid.id,
            max_dist=jnp.asarray(max_dist),
            rest_length=jnp.asarray(rest_length),
            stiffness=jnp.asarray(stiffness),
        )

    @property
    def soft(self) -> sim.Actor:
        return self.actors[self.soft_id]

    @property
    def rigid(self) -> sim.Actor:
        return self.actors[self.rigid_id]

    @override
    def pre_optim_iter(self, params: sim.GlobalParams) -> Self:
        collide, face, face_normal, sign, target, uv = detect_vert_face(
            self.soft.positions,
            np.uint64(self.rigid.collision_mesh.id),
            jnp.reshape(self.max_dist, (1,)),
            jnp.reshape(self.rest_length, (1,)),
        )
        candidates = CandidatesVertFace(
            collide=collide,
            sign=sign,
            face=face,
            face_normal=face_normal,
            target=target,
            uv=uv,
        )
        return self.evolve(candidates=candidates)

    @override
    @utils.jit_method(inline=True)
    def fun(self, x: struct.ArrayDict, /, params: sim.GlobalParams) -> Float[Array, ""]:
        mask: Bool[Array, " points"] = self.make_mask(x)
        distance: Float[Array, " points"] = self.make_distance(x)
        rest_length: Float[Array, ""] = self.candidates.sign * self.rest_length
        Psi: Float[Array, " points"] = (
            0.5 * self.stiffness * (distance - rest_length) ** 2
        )
        Psi = jnp.where(mask, Psi, 0.0)
        return jnp.sum(Psi)

    @override
    @utils.jit_method(inline=True)
    def jac(self, x: struct.ArrayDict, /, params: sim.GlobalParams) -> struct.ArrayDict:
        mask: Bool[Array, " points"] = self.make_mask(x)
        t: Float[Array, " points dim"] = self.make_t(x)
        soft_jac: Float[Array, " points dim"] = jax.vmap(self.compute_jac_element)(
            t, self.candidates.sign
        )
        soft_jac = jnp.where(mask[:, None], soft_jac, 0.0)
        jac: struct.ArrayDict = struct.ArrayDict(
            {
                self.soft.id: soft_jac,
                self.rigid.id: jnp.zeros_like(x[self.rigid.id]),
            }
        )
        return jac

    @override
    @utils.jit_method(inline=True)
    def hess_diag(
        self, x: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> struct.ArrayDict:
        mask: Bool[Array, " points"] = self.make_mask(x)
        hess: Float[Array, "points dim dim"] = self.compute_hess(x, params)
        hess_diag_values: Float[Array, " points dim"] = jnp.diagonal(
            hess, axis1=-2, axis2=-1
        )
        hess_diag_values = jnp.where(mask[:, None], hess_diag_values, 0.0)
        hess_diag: struct.ArrayDict = struct.ArrayDict(
            {
                self.soft.id: hess_diag_values,
                self.rigid.id: jnp.zeros_like(x[self.rigid.id]),
            }
        )
        return hess_diag

    @override
    @utils.jit_method(inline=True)
    def hess_quad(
        self, x: struct.ArrayDict, p: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> Float[Array, ""]:
        mask: Bool[Array, " points"] = self.make_mask(x)
        hess: Float[Array, "points dim dim"] = self.compute_hess(x, params)
        p_soft: Float[Array, "points dim"] = p[self.soft.id]
        hess_quad_soft: Float[Array, " points"] = einops.einsum(
            p_soft, hess, p_soft, "p I, p I J, p J -> p"
        )
        hess_quad_soft = jnp.where(mask, hess_quad_soft, 0.0)
        return jnp.sum(hess_quad_soft)

    def make_t(self, x: struct.ArrayDict, /) -> Float[Array, " points dim"]:
        positions: Float[Array, "p dim"] = self.soft.points + x[self.soft.id]
        t: Float[Array, "p dim"] = positions - self.candidates.target
        return t

    def make_distance(self, x: struct.ArrayDict, /) -> Float[Array, " points"]:
        t: Float[Array, "p dim"] = self.make_t(x)
        distance: Float[Array, " p"] = jnp.linalg.norm(t, axis=-1)
        distance *= self.candidates.sign
        return distance

    def make_mask(self, x: struct.ArrayDict, /) -> Bool[Array, " points"]:
        return self.candidates.collide

    def compute_hess(
        self, x: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> Float[Array, "DOF DOF"]:
        t: Float[Array, "points dim"] = self.make_t(x)
        hess: Float[Array, "points dim dim"] = jax.vmap(self.compute_hess_element)(
            t, self.candidates.sign
        )
        return hess

    def compute_jac_element(
        self, t: Float[Array, " dim"], sign: Float[Array, ""]
    ) -> Float[Array, " dim"]:
        t_norm: Float[Array, ""] = jnp.linalg.norm(t)
        rest_length: Float[Array, ""] = sign * self.rest_length
        return self.stiffness * (t_norm - rest_length) * (t / t_norm)

    def compute_hess_element(
        self, t: Float[Array, " dim"], sign: Float[Array, ""]
    ) -> Float[Array, "dim dim"]:
        t_norm: Float[Array, ""] = jnp.linalg.norm(t)
        tTt: Float[Array, ""] = jnp.vdot(t, t)
        rest_length: Float[Array, ""] = sign * self.rest_length
        return self.stiffness * (
            (1 / tTt - ((t_norm - rest_length) / tTt**1.5)) * jnp.outer(t, t)
            + (t_norm - rest_length) / t_norm * jnp.identity(t.shape[0])
        )


@no_type_check
@utils.jax_kernel(num_outputs=6)
def detect_vert_face(
    points: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    max_dist: wp.array(dtype=wp.float32, shape=(1,)),
    rest_length: wp.array(dtype=wp.float32, shape=(1,)),
    # outputs
    collide: wp.array(dtype=bool),
    face: wp.array(dtype=wp.int32),
    face_normal: wp.array(dtype=wp.vec3),
    sign: wp.array(dtype=wp.float32),
    target: wp.array(dtype=wp.vec3),
    uv: wp.array(dtype=wp.vec2),
) -> None:
    tid = wp.tid()
    point = points[tid]
    query = wp.mesh_query_point_sign_normal(mesh_id, point, max_dist=max_dist[0])
    if query.result:
        sign[tid] = query.sign
        face[tid] = query.face
        face_normal[tid] = wp.mesh_eval_face_normal(mesh_id, query.face)
        target[tid] = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        uv[tid] = wp.vec2(query.u, query.v)
        t = target[tid] - point
        if sign[tid] > 0 and wp.length(t) > rest_length[0]:
            collide[tid] = False
        else:
            collide[tid] = True
    else:
        collide[tid] = False
