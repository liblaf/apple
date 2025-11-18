from pathlib import Path
from typing import cast

import jax.numpy as jnp
import pyvista as pv
import warp as wp

from liblaf import cherries, grapes, melon
from liblaf.apple import energy, helper, optim, sim, utils


class Config(cherries.BaseConfig):
    output_dir: Path = utils.data("")
    duration: float = 10.0
    fps: float = 30.0

    d_hat: float = 1e-3
    density: float = 1e3
    lambda_: float = 3 * 1e3
    mu: float = 1 * 1e3

    @property
    def n_frames(self) -> int:
        return int(self.duration * self.fps)

    @property
    def time_step(self) -> float:
        return 1 / self.fps


def main(cfg: Config) -> None:
    grapes.logging.init()
    wp.init()
    soft: sim.Actor = gen_actor(cfg)
    ground: sim.Actor = gen_rigid(cfg)
    builder: sim.SceneBuilder = gen_scene(cfg, soft, ground)
    builder.params = builder.params.replace(time_step=cfg.time_step)
    soft = builder.actors_concrete[soft.id]
    scene: sim.Scene = builder.finish()
    optimizer = optim.PNCG(d_hat=cfg.d_hat, maxiter=10**3, rtol=5e-5)

    writer = melon.SeriesWriter(cfg.output_dir / "animation.vtu.series", fps=cfg.fps)
    melon.save(cfg.output_dir / "ground.vtp", ground.to_pyvista())
    soft = scene.export_actor(soft)
    mesh: pv.UnstructuredGrid = soft.to_pyvista()
    writer.append(mesh, time=0.0)
    for t in range(1, cfg.n_frames + 1):
        result: optim.OptimizeResult
        scene, result = scene.solve(optimizer=optimizer)
        # if not result["success"]:
        #     ic(result)
        ic(result)
        scene = scene.step(result["x"])
        soft = scene.export_actor(soft)
        soft = helper.dump_optim_result(scene, soft, result)
        collision: energy.CollisionVertFace = scene.energies["CollisionVertFace-000"]  # pyright: ignore[reportAssignmentType]
        soft = helper.dump_collision(soft, collision)
        mesh: pv.UnstructuredGrid = soft.to_pyvista()
        writer.append(mesh, time=t * cfg.time_step)
    writer.end()


def gen_pyvista(cfg: Config) -> pv.UnstructuredGrid:
    surface: pv.PolyData = cast("pv.PolyData", pv.examples.download_bunny())
    mesh: pv.UnstructuredGrid = melon.tetwild(surface)
    y_min: float
    _, _, y_min, _, _, _ = mesh.bounds
    mesh.translate((0, 0.1 - y_min, 0), inplace=True)
    mesh.cell_data["density"] = cfg.density
    mesh.cell_data["lambda"] = cfg.lambda_
    mesh.cell_data["mu"] = cfg.mu
    return mesh


def gen_actor(cfg: Config) -> sim.Actor:
    mesh: pv.UnstructuredGrid = gen_pyvista(cfg)
    actor: sim.Actor = sim.Actor.from_pyvista(mesh, grad=True)
    actor = helper.add_point_mass(actor)
    actor = helper.add_gravity(actor)
    return actor


def gen_rigid(_cfg: Config) -> sim.Actor:
    # surface: pv.PolyData = pv.Box((-0.2, 0.2, -0.2, 0, -0.2, 0.2), quads=False)
    surface: pv.PolyData = cast("pv.PolyData", pv.examples.download_bunny())
    surface = melon.mesh_fix(surface, check=False)
    surface.compute_normals(auto_orient_normals=True, inplace=True)
    y_max: float
    _, _, _, y_max, _, _ = surface.bounds
    surface.translate((0, 0.0 - y_max, 0), inplace=True)
    actor: sim.Actor = sim.Actor.from_pyvista(surface, collision=True)
    actor = actor.with_collision_mesh()
    actor = actor.set_point_data("mass", jnp.ones((actor.n_points,)))
    actor = actor.with_dirichlet(
        sim.Dirichlet.from_mask(
            jnp.ones((actor.n_dofs,), dtype=bool), jnp.zeros((actor.n_dofs,))
        )
    )
    return actor


def gen_scene(cfg: Config, soft: sim.Actor, rigid: sim.Actor) -> sim.SceneBuilder:
    builder = sim.SceneBuilder()
    soft = builder.assign_dofs(soft)
    rigid = builder.assign_dofs(rigid)
    builder.add_energy(energy.PhacePassive.from_actor(soft))
    builder.add_energy(
        energy.CollisionVertFace.from_actors(rigid, soft, rest_length=cfg.d_hat)
    )
    return builder


if __name__ == "__main__":
    cherries.main(main, profile="playground")
