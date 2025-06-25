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

    density: float = 1e5
    lambda_: float = 3.0
    mu: float = 1.0

    @property
    def n_frames(self) -> int:
        return int(self.duration * self.fps)

    @property
    def time_step(self) -> float:
        return 1 / self.fps


def main(cfg: Config) -> None:
    actor: sim.Actor = gen_actor(cfg)
    builder: sim.SceneBuilder = gen_scene(cfg, actor)
    builder.params = builder.params.evolve(time_step=cfg.time_step)
    actor = builder.actors_concrete[actor.id]
    scene: sim.Scene = builder.finish()
    optimizer = optim.PNCG(maxiter=10**3)

    writer = melon.SeriesWriter(
        "data/examples/dynamics/gravity.vtu.series", fps=cfg.fps
    )
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
        actor = helper.dump_optim_result(scene, actor, result)
        ic(actor.point_data["jac"] / actor.point_data["mass"][:, None])
        mesh: pv.UnstructuredGrid = actor.to_pyvista()
        writer.append(mesh, time=t * cfg.time_step)
    writer.end()


def gen_pyvista(cfg: Config) -> pv.UnstructuredGrid:
    surface: pv.PolyData = cast("pv.PolyData", pv.examples.download_bunny())
    mesh: pv.UnstructuredGrid = melon.tetwild(surface)
    # mesh = pv.examples.cells.Tetrahedron()
    # mesh = cast("pv.UnstructuredGrid", pv.examples.download_tetrahedron())
    mesh.cell_data["density"] = cfg.density
    mesh.cell_data["lambda"] = cfg.lambda_
    mesh.cell_data["mu"] = cfg.mu
    return mesh


def gen_dirichlet(cfg: Config, mesh: pv.UnstructuredGrid) -> sim.Dirichlet:
    y_min: float
    y_max: float
    _, _, y_min, y_max, _, _ = mesh.bounds
    y_length: float = y_max - y_min
    dirichlet_mask: Bool[np.ndarray, " points"] = (
        mesh.points[:, 1] < y_min + cfg.dirichlet_thickness * y_length
    )
    dirichlet_values: np.ndarray = np.zeros((mesh.n_points, 3))
    return sim.Dirichlet.from_mask(
        einops.repeat(dirichlet_mask, "points -> points dim", dim=3),
        dirichlet_values,
    )


def gen_actor(cfg: Config) -> sim.Actor:
    mesh: pv.UnstructuredGrid = gen_pyvista(cfg)
    actor: sim.Actor = sim.Actor.from_pyvista(mesh)
    # actor = actor.set_dirichlet(gen_dirichlet(cfg, mesh))
    actor = helper.add_point_mass(actor)
    actor = helper.add_gravity(actor)
    return actor


def gen_scene(_cfg: Config, actor: sim.Actor) -> sim.SceneBuilder:
    builder = sim.SceneBuilder()
    actor = builder.assign_dofs(actor)
    builder.add_energy(energy.EnergyZero.from_actor(actor))
    return builder


if __name__ == "__main__":
    cherries.run(main, play=True)
