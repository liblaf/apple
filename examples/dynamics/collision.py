from typing import no_type_check

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import rich
import warp as wp
from jaxtyping import Float

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import melon


def main() -> None:
    ground: pv.PolyData = gen_ground()
    obj: pv.PolyData = gen_object()
    ground_wp: wp.Mesh = as_warp_mesh(ground)
    rich.inspect(ground_wp)
    obj.translate((0, -0.01, 0), inplace=True)
    points: Float[jax.Array, "points 3"] = jnp.asarray(obj.points)
    displacements: Float[jax.Array, "points 3"]
    (displacements,) = project_collision(points, np.uint64(ground_wp.id))
    obj.point_data["displacement"] = np.asarray(displacements)
    obj.warp_by_vector("displacement", inplace=True)
    melon.save("data/examples/dynamics/collision.vtp", obj)


def as_warp_mesh(mesh_pv: pv.PolyData) -> wp.Mesh:
    return wp.Mesh(
        wp.from_numpy(mesh_pv.points, dtype=wp.vec3),
        wp.from_numpy(mesh_pv.regular_faces.ravel(), dtype=wp.int32),
    )


def gen_ground() -> pv.PolyData:
    return pv.Icosphere()


def gen_object() -> pv.PolyData:
    return pv.Box(bounds=(-1, 1, 1, 3, -1, 1), level=100, quads=False)


@no_type_check
@apple.jax_kernel
def project_collision(
    points: wp.array(dtype=wp.vec3),
    ground_id: wp.uint64,
    displacements: wp.array(dtype=wp.vec3),
) -> None:
    tid = wp.tid()
    point = points[tid]
    query = wp.mesh_query_point_sign_normal(ground_id, point, 1.0)
    # wp.printf("result: %d, sign: %f\n", query.result, query.sign)
    # if query.result:
    # displacements[tid] = wp.vec3(1.0, 0.0, 0.0)
    if query.result and query.sign < 0:
        wp.printf("result: %d, sign: %f\n", query.result, query.sign)
        target_point = wp.mesh_eval_position(ground_id, query.face, query.u, query.v)
        displacements[tid] = target_point - point


if __name__ == "__main__":
    main()
