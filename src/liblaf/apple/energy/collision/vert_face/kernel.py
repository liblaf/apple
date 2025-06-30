from typing import no_type_check

import warp as wp

from liblaf.apple import utils


@utils.jax_kernel
@no_type_check
def detect_vert_face(
    points: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    max_dist: wp.float32,
    epsilon: wp.float32,
    rest_length: wp.float32,
    # outputs
    collide: wp.array(dtype=bool),
    collision_counter: wp.array(dtype=wp.int32),
    collision_id: wp.array(dtype=wp.int32),
    distance: wp.array(dtype=wp.float32),
    face_normal: wp.array(dtype=wp.vec3),
) -> None:
    tid = wp.tid()
    point = points[tid]
    query = wp.mesh_query_point_sign_normal(
        mesh_id, point, max_dist=max_dist, epsilon=epsilon
    )
    if query.result:
        target = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        dist = wp.length(target - point)
        if dist * query.sign < rest_length:
            collide[tid] = True
            distance[tid] = dist * query.sign
            face_normal[tid] = wp.mesh_eval_face_normal(mesh_id, query.face)
        else:
            collide[tid] = False
    else:
        collide[tid] = False
