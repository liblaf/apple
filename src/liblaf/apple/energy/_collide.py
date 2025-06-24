from typing import Self, no_type_check, override

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from jaxtyping import Bool, Float, Integer

from liblaf.apple import sim, struct, utils


class CollisionVertFace(sim.Energy):
    obj: sim.Actor = struct.data(default=None)
    solid: sim.Actor = struct.data(default=None)
    mesh_wp: wp.Mesh = struct.static(default=None)
    max_dist: np.float32 = struct.static(default=np.float32(0.1))
    threshold: np.float32 = struct.static(default=np.float32(0.01))

    collision: Bool[jax.Array, " points"] = struct.array(default=None)
    face: Integer[jax.Array, " points"] = struct.array(default=None)
    uv: Float[jax.Array, " points 2"] = struct.array(default=None)
    target: Float[jax.Array, " points 3"] = struct.array(default=None)
    k: Float[jax.Array, ""] = struct.array(default=1.0)

    @property
    def deps(self) -> struct.FrozenDict:
        return struct.FrozenDict((self.obj, self.solid))

    @override
    def prepare(self) -> Self:
        self.mesh_wp.refit()
        collision, face, uv, target = detect_collision(
            self.obj.positions,
            np.uint64(self.mesh_wp.id),
            self.max_dist,
            self.threshold,
        )
        return self.evolve(
            collision=collision,
            face=face,
            uv=uv,
            target=target,
        )

    def fun(
        self, x: struct.DictArray, /, params: sim.GlobalParams
    ) -> Float[jax.Array, ""]:
        x: Float[jax.Array, "points dim"] = x[self.obj.id]
        t = x[self.collision] - self.target[self.collision]
        t_norm = jnp.linalg.norm(t, axis=-1)
        return 0.5 * self.k * jnp.sum((t_norm - self.threshold) ** 2)


@no_type_check
@utils.jax_kernel(num_outputs=4)
def detect_collision(
    points: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    max_dist: wp.float32,
    threshold: wp.float32,
    collision: wp.array(dtype=bool),
    face: wp.array(dtype=wp.int32),
    uv: wp.array(dtype=wp.vec2),
    target: wp.array(dtype=wp.vec3),
) -> None:
    tid = wp.tid()
    point = points[tid]
    query = wp.mesh_query_point_sign_normal(
        mesh_id, point, max_dist=max_dist, epsilon=1e-3
    )
    if query.result:
        target = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        dist = wp.length(target - point) * query.sign
        if dist < threshold:
            target_normal = wp.mesh_eval_face_normal(mesh_id, query.face)
            face[tid] = query.face
            uv[tid] = wp.vec2(query.u, query.v)
            target[tid] = target + threshold * target_normal
        else:
            collision[tid] = False
    else:
        collision[tid] = False
