from collections.abc import Mapping
from pathlib import Path

import einops
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Array, Bool, Float
from loguru import logger

from liblaf import cherries, melon
from liblaf.apple import energy, helper, optim, sim, struct, utils


class Config(cherries.BaseConfig):
    output_dir: Path = utils.data("animation")
    mesh: Path = utils.data("head.vtu")
    duration: float = 1.0
    fps: float = 120.0

    density: float = 1
    lambda_: float = 3 * 1e3
    mu: float = 1 * 1e3

    d_hat: float = 0.05
    collision_stiffness: float = 500.0

    @property
    def n_frames(self) -> int:
        return int(self.duration * self.fps)

    @property
    def time_step(self) -> float:
        return 1 / self.fps


def gen_scene(cfg: Config) -> sim.SceneBuilder:
    head_pv: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.mesh)
    head_pv.cell_data["density"] = cfg.density
    head_pv.cell_data["lambda"] = cfg.lambda_
    head_pv.cell_data["mu"] = cfg.mu
    head: sim.Actor = sim.Actor.from_pyvista(head_pv, grad=True, id_="head")
    head = helper.add_point_mass(head)
    is_skull: Bool[Array, " P"] = head_pv.point_data["is-skull"]
    head = head.with_dirichlet(
        sim.Dirichlet.from_mask(
            einops.repeat(is_skull, "P -> P D", D=3),
            values=jnp.zeros((head.n_points, 3), dtype=jnp.float32),
        )
    )

    builder = sim.SceneBuilder()
    head = builder.assign_dofs(head)
    builder.add_energy(energy.PhacePassive.from_actor(head))

    head_pv.point_data["point-id"] = np.arange(head_pv.n_points)
    surface_pv: pv.PolyData = head_pv.extract_surface()
    lip_top_pv: pv.PolyData = surface_pv.extract_points(
        surface_pv.point_data["is-lip-top"], adjacent_cells=False
    ).extract_surface()
    # lip_bottom_pv: pv.PolyData = surface_pv.extract_points(
    #     surface_pv.point_data["is-lip-bottom"], adjacent_cells=False
    # ).extract_surface()
    lip_top: sim.Actor = sim.Actor.from_pyvista(
        lip_top_pv, collision=True, id_="lip-top"
    )
    lip_top.dofs_global = sim.DOFs.from_index(
        head.dofs_global.index.reshape(head.dofs_global.shape)[
            lip_top.point_data["point-id"]
        ]
    )
    # ic(lip_top)
    # lip_bottom: sim.Actor = sim.Actor.from_pyvista(
    #     lip_bottom_pv, collision=True, id_="lip-bottom"
    # )
    # lip_bottom.dofs_global = sim.DOFs.from_index(
    #     head.dofs_global.index.reshape(head.dofs_global.shape)[
    #         lip_bottom.point_data["point-id"]
    #     ]
    # )
    # builder.add_actor(lip_top)
    # builder.add_actor(lip_bottom)
    # builder.add_energy(
    #     energy.CollisionVertFace.from_actors(
    #         lip_top,
    #         lip_bottom,
    #         stiffness=cfg.collision_stiffness,
    #         rest_length=cfg.d_hat,
    #     )
    # )
    # builder.add_energy(
    #     energy.CollisionVertFace.from_actors(
    #         lip_bottom,
    #         lip_top,
    #         stiffness=cfg.collision_stiffness,
    #         rest_length=cfg.d_hat,
    #     )
    # )

    builder.integrator = sim.TimeIntegratorStatic()

    return builder


def update_dirichlet(
    builder: sim.SceneBuilder, scene: sim.Scene, rotate_rad: float
) -> sim.Scene:
    actors: struct.NodeContainer[sim.Actor] = scene.actors
    for actor in scene.actors.values():
        if actor.id != "head":
            continue
        skull_mask: Bool[Array, " P"] = actor.point_data["is-skull"]
        mandible_mask: Bool[Array, " P"] = actor.point_data["is-mandible"]
        points: Float[Array, "P 3"] = actor.points
        matrix: Float[np.ndarray, "4 4"] = tm.transformations.rotation_matrix(
            rotate_rad, [1.0, 0.0, 0.0], point=[0.69505, 29.141, 0.8457]
        )  # pyright: ignore[reportAssignmentType]
        points = points.at[mandible_mask].set(
            tm.transform_points(points[mandible_mask], matrix)
        )
        disp: Float[Array, "P 3"] = points - actor.points
        actor_new: sim.Actor = actor.with_dirichlet(
            sim.Dirichlet.from_mask(
                mask=einops.repeat(skull_mask, "points -> points dim", dim=3),
                values=disp,
            )
        )
        actors.add(actor_new)
    builder.actors = actors
    scene = scene.replace(actors=actors, dirichlet=builder.dirichlet)
    return scene


def main(cfg: Config) -> None:
    builder: sim.SceneBuilder = gen_scene(cfg)
    scene: sim.Scene = builder.finish()
    optimizer = optim.PNCG(d_hat=cfg.d_hat, maxiter=500, rtol=1e-5)

    jaw_rotate_total: float = np.deg2rad(-10.0)
    jaw_rotate: np.ndarray = np.linspace(0.0, jaw_rotate_total, num=cfg.n_frames + 1)

    writers: Mapping[str, melon.SeriesWriter] = {
        name: melon.SeriesWriter(cfg.output_dir / f"{name}.vtu.series", fps=cfg.fps)
        for name in scene.actors
    }
    meshes: Mapping[str, pv.DataSet] = helper.dump_all_pyvista(scene=scene)
    for name, writer in writers.items():
        writer.append(meshes[name], time=0.0)
    for t in range(1, cfg.n_frames + 1):
        scene = update_dirichlet(builder=builder, scene=scene, rotate_rad=jaw_rotate[t])
        result: optim.OptimizeResult
        scene, result = scene.solve(optimizer=optimizer)
        if not result["success"]:
            logger.error("{}", result)
        scene = scene.step(result["x"])
        meshes: Mapping[str, pv.DataSet] = helper.dump_all_pyvista(
            scene=scene, result=result
        )
        for name, writer in writers.items():
            writer.append(meshes[name], time=t * cfg.time_step)


if __name__ == "__main__":
    cherries.main(main, profile="playground")
