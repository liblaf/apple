from collections.abc import Mapping
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from loguru import logger

from liblaf import cherries, melon
from liblaf.apple import energy, helper, optim, sim, utils


class Config(cherries.BaseConfig):
    output_dir: Path = utils.data("")
    duration: float = 1.0
    fps: float = 120.0

    d_hat: float = 2e-3
    density: float = 50.0
    lambda_: float = 3 * 1e3
    mu: float = 1 * 1e3
    collision_stiffness: float = 200.0

    @property
    def n_frames(self) -> int:
        return int(self.duration * self.fps)

    @property
    def time_step(self) -> float:
        return 1 / self.fps


def gen_box(
    cfg: Config, builder: sim.SceneBuilder, surface_pv: pv.PolyData, name: str
) -> sim.SceneBuilder:
    tetmesh_pv: pv.UnstructuredGrid = melon.tetwild(surface_pv)
    tetmesh_pv.point_data["point-id"] = np.arange(tetmesh_pv.n_points)
    tetmesh_pv.cell_data["density"] = cfg.density
    tetmesh_pv.cell_data["lambda"] = cfg.lambda_
    tetmesh_pv.cell_data["mu"] = cfg.mu
    tetmesh: sim.Actor = sim.Actor.from_pyvista(mesh=tetmesh_pv, grad=True, id_=name)
    tetmesh = helper.add_point_mass(tetmesh)
    tetmesh = helper.add_gravity(tetmesh)
    tetmesh = builder.assign_dofs(tetmesh)
    builder.add_actor(tetmesh)
    builder.add_energy(energy.PhacePassive.from_actor(tetmesh))
    ic(tetmesh_pv)
    ic(tetmesh.dofs_global)

    surface_pv = tetmesh_pv.extract_surface()
    ic(surface_pv)
    surface: sim.Actor = sim.Actor.from_pyvista(
        mesh=surface_pv, collision=True, id_=f"{name}_surface"
    )
    surface.dofs_global = sim.DOFs.from_index(
        tetmesh.dofs_global.index.reshape(tetmesh.points.shape)[
            surface_pv.point_data["point-id"]
        ]
    )
    ic(surface.dofs_global)
    builder.add_actor(surface)
    return builder


def gen_scene(cfg: Config) -> sim.SceneBuilder:
    builder = sim.SceneBuilder()

    ground_pv: pv.PolyData = pv.Box(
        bounds=(-0.2, 0.2, -0.2, 0.0, -0.2, 0.2), quads=False
    )
    ground_pv.point_data["mass"] = 1.0
    ground: sim.Actor = sim.Actor.from_pyvista(
        mesh=ground_pv, collision=True, id_="ground"
    )
    ground = ground.with_dirichlet(
        sim.Dirichlet.from_mask(
            jnp.ones((ground.n_dofs,), dtype=bool), jnp.zeros((ground.n_dofs,))
        )
    )
    ground = builder.assign_dofs(ground)
    builder.add_actor(ground)

    box_prototype: pv.PolyData = pv.Box(
        bounds=(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1), quads=False
    )
    box_down_pv: pv.PolyData = box_prototype.translate((0, 0.1 + 0.02, 0))
    box_up_pv: pv.PolyData = box_down_pv.translate((0, 0.2 + 0.02, 0))
    gen_box(cfg, builder, box_down_pv, "box-down")
    gen_box(cfg, builder, box_up_pv, "box-up")

    box_up_surface: sim.Actor = builder.actors["box-up_surface"]
    box_down_surface: sim.Actor = builder.actors["box-down_surface"]
    builder.add_energy(
        energy.CollisionVertFace.from_actors(
            box_up_surface,
            box_down_surface,
            stiffness=cfg.collision_stiffness,
            rest_length=cfg.d_hat,
        )
    )
    builder.add_energy(
        energy.CollisionVertFace.from_actors(
            box_down_surface,
            box_up_surface,
            stiffness=cfg.collision_stiffness,
            rest_length=cfg.d_hat,
        )
    )
    builder.add_energy(
        energy.CollisionVertFace.from_actors(
            ground,
            box_up_surface,
            stiffness=cfg.collision_stiffness,
            rest_length=cfg.d_hat,
        )
    )
    builder.add_energy(
        energy.CollisionVertFace.from_actors(
            ground,
            box_down_surface,
            stiffness=cfg.collision_stiffness,
            rest_length=cfg.d_hat,
        )
    )

    return builder


def main(cfg: Config) -> None:
    builder: sim.SceneBuilder = gen_scene(cfg)
    scene: sim.Scene = builder.finish()
    optimizer: optim.Optimizer = optim.PNCG(d_hat=cfg.d_hat, maxiter=500, rtol=1e-5)

    writers: Mapping[str, melon.SeriesWriter] = {
        name: melon.SeriesWriter(cfg.output_dir / f"{name}.vtu.series", fps=cfg.fps)
        for name in scene.actors
    }
    meshes: Mapping[str, pv.DataSet] = helper.dump_all_pyvista(scene=scene)
    for name, writer in writers.items():
        writer.append(meshes[name], time=0.0)
    for t in range(1, cfg.n_frames + 1):
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
    cherries.run(main, profile="playground")
