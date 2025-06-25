from typing import cast

import einops
import numpy as np
import pyvista as pv
from jaxtyping import Bool

from liblaf import cherries, melon
from liblaf.apple import energy, helper, optim, sim


class Config(cherries.BaseConfig):
    dirichlet_thickness: float = 0.05
    duration: float = 3.0
    fps: float = 30.0

    mu: float = 1e-5
    density: float = 1e3

    @property
    def n_frames(self) -> int:
        return int(self.duration * self.fps)

    @property
    def time_step(self) -> float:
        return 1 / self.fps


def main(cfg: Config) -> None:
    actor: sim.Actor = gen_actor(cfg)
    builder: sim.SceneBuilder = gen_scene(cfg, actor)
    actor = builder.actors_concrete[actor.id]
    scene: sim.Scene = builder.finish()
    optimizer = optim.PNCG()

    writer = melon.SeriesWriter("data/examples/dynamics/arap.vtu.series", fps=cfg.fps)
    actor = scene.export_actor(actor)
    mesh: pv.UnstructuredGrid = actor.to_pyvista()
    writer.append(mesh, time=0.0)
    for t in range(cfg.n_frames):
        result: optim.OptimizeResult
        scene, result = scene.solve(optimizer=optimizer)
        if not result["success"]:
            ic(result)
        scene = scene.step(result["x"])
        actor = scene.export_actor(actor)
        for key in ["hess_diag", "jac", "p", "P"]:
            actor = actor.set_point_data(key, actor.dofs.get(result[key]))
        mesh: pv.UnstructuredGrid = actor.to_pyvista()
        writer.append(mesh, time=t * cfg.time_step)
    writer.end()


def gen_pyvista(_cfg: Config) -> pv.UnstructuredGrid:
    surface: pv.PolyData = cast("pv.PolyData", pv.examples.download_bunny())
    mesh: pv.UnstructuredGrid = melon.tetwild(surface)
    return mesh


def gen_actor(cfg: Config) -> sim.Actor:
    mesh: pv.UnstructuredGrid = gen_pyvista(cfg)
    mesh.cell_data["density"] = cfg.density
    mesh.cell_data["mu"] = cfg.mu
    y_min: float
    y_max: float
    _, _, y_min, y_max, _, _ = mesh.bounds
    y_length: float = y_max - y_min
    dirichlet_mask: Bool[np.ndarray, " points"] = (
        mesh.points[:, 1] < y_min + cfg.dirichlet_thickness * y_length
    )
    mesh.point_data["dirichlet-mask"] = dirichlet_mask
    mesh.point_data["dirichlet-values"] = np.zeros((mesh.n_points, 3))
    actor: sim.Actor = sim.Actor.from_pyvista(mesh)
    actor = actor.set_dirichlet(
        sim.Dirichlet.from_mask(
            einops.repeat(dirichlet_mask, "points -> points dim", dim=3),
            mesh.point_data["dirichlet-values"],
        )
    )
    actor = helper.add_point_mass(actor)
    actor = helper.add_gravity(actor)
    return actor


def gen_scene(_cfg: Config, actor: sim.Actor) -> sim.SceneBuilder:
    builder = sim.SceneBuilder()
    actor = builder.assign_dofs(actor)
    builder.add_energy(energy.ARAP.from_actor(actor))
    return builder


if __name__ == "__main__":
    cherries.run(main, play=True)
