from typing import no_type_check

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import rich
import warp as wp
from jaxtyping import Bool, Float

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import cherries, grapes, melon


class Config(cherries.BaseConfig):
    n_frames: int = 150
    total_disp: float = 0.5

    @property
    def disp_step(self) -> float:
        return self.total_disp / self.n_frames


def main(cfg: Config) -> None:
    ground: pv.PolyData = gen_ground()
    ic(ground)
    obj: pv.PolyData = gen_object()
    ic(obj)
    obj_init: pv.PolyData = obj.copy()
    ground_wp: wp.Mesh = as_warp_mesh(ground)
    rich.inspect(ground_wp)

    melon.save("data/examples/dynamics/collision-ground.vtp", ground)
    writer = melon.SeriesWriter("data/examples/dynamics/collision.vtp.series")

    for it in grapes.track(range(cfg.n_frames), description="Frames"):
        obj.translate((0, -cfg.disp_step, 0), inplace=True)
        points: Float[jax.Array, "points 3"] = jnp.asarray(obj.points)
        displacements: Float[jax.Array, "points 3"]
        displacements: Bool[jax.Array, " points"]
        (displacements, collision) = project_collision(points, np.uint64(ground_wp.id))
        obj.point_data["displacement"] = np.asarray(displacements)
        obj.warp_by_vector("displacement", inplace=True)
        obj.point_data["collision"] = np.asarray(collision)
        obj.point_data["displacement"] = obj.points - np.asarray(obj_init.points)
        writer.append(obj)


def as_warp_mesh(mesh_pv: pv.PolyData) -> wp.Mesh:
    return wp.Mesh(
        wp.from_numpy(mesh_pv.points, dtype=wp.vec3),
        wp.from_numpy(mesh_pv.regular_faces.ravel(), dtype=wp.int32),
    )


def gen_ground() -> pv.PolyData:
    return pv.Icosphere(nsub=5)


def gen_object() -> pv.PolyData:
    return pv.Box(bounds=(-1, 1, 1, 3, -1, 1), level=100, quads=False)


@no_type_check
@apple.jax_kernel(num_outputs=2)
def project_collision(
    points: wp.array(dtype=wp.vec3),
    ground_id: wp.uint64,
    displacements: wp.array(dtype=wp.vec3),
    collision: wp.array(dtype=wp.bool),
) -> None:
    tid = wp.tid()
    point = points[tid]
    query = wp.mesh_query_point_sign_normal(ground_id, point, 0.1)
    if query.result:
        target = wp.mesh_eval_position(ground_id, query.face, query.u, query.v)
        dist = wp.length(target - point) * query.sign
        if dist < 0.01:
            target_normal = wp.mesh_eval_face_normal(ground_id, query.face)
            displacements[tid] = target - point
            displacements[tid] += 0.01 * target_normal
            collision[tid] = True
            # wp.printf(
            #     "result: %d, sign: %f, disp: %f\n",
            #     query.result,
            #     query.sign,
            #     wp.length(displacements[tid]),
            # )
        else:
            displacements[tid] = wp.vec3()
            collision[tid] = False
    else:
        displacements[tid] = wp.vec3()
        collision[tid] = False


if __name__ == "__main__":
    cherries.run(main, play=True)
