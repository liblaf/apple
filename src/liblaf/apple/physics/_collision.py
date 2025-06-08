from typing import no_type_check

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple import utils


class CollisionRigidSoft(flax.struct.PyTreeNode):
    mesh_wp: wp.Mesh = flax.struct.field(pytree_node=False)
    max_dist: Float[jax.Array, ""] = flax.struct.field(
        default_factory=lambda: np.float64(1e-2), pytree_node=False
    )
    threshold: Float[jax.Array, ""] = flax.struct.field(
        default_factory=lambda: np.float64(1e-3), pytree_node=False
    )

    @utils.jit
    def resolve(
        self, points: Float[ArrayLike, "points 3"]
    ) -> Float[jax.Array, "points 3"]:
        points = jnp.asarray(points)
        return _resolve_collisions_warp(
            points,
            np.uint64(self.mesh_wp.id),
            self.max_dist,
            self.threshold,
        )[0]


@no_type_check
@utils.jit(static_argnums=(1, 2, 3))
@utils.jax_kernel
def _resolve_collisions_warp(
    points: wp.array(dtype=wp.vec3),
    solid: wp.uint64,
    max_dist: wp.float32,
    threshold: wp.float32,
    displacements: wp.array(dtype=wp.vec3),
) -> None:
    tid = wp.tid()
    point = points[tid]
    query = wp.mesh_query_point_sign_normal(solid, point, max_dist=max_dist)
    if query.result:
        target = wp.mesh_eval_position(solid, query.face, query.u, query.v)
        dist = wp.length(target - point) * query.sign
        if dist < threshold:
            target_normal = wp.mesh_eval_face_normal(solid, query.face)
            displacements[tid] = target - point
            displacements[tid] += threshold * target_normal
        else:
            displacements[tid] = wp.vec3()
    else:
        displacements[tid] = wp.vec3()
