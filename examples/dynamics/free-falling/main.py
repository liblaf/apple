from pathlib import Path
from typing import cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from jaxtyping import ArrayLike, Float

from liblaf import cherries, melon
from liblaf.apple import energy, helper, optim, sim, utils


class Config(cherries.BaseConfig):
    output_dir: Path = utils.data("")

    dirichlet_thickness: float = 0.05
    duration: float = 5.0
    fps: float = 30.0

    density: float = 1e3
    lambda_: float = 3.0 * 1e4
    mu: float = 1.0 * 1e4

    @property
    def n_frames(self) -> int:
        return int(self.duration * self.fps)

    @property
    def time_step(self) -> float:
        return 1 / self.fps


def main(cfg: Config) -> None:
    plt.rc("figure", dpi=300)
    actor: sim.Actor = gen_actor(cfg)
    builder: sim.SceneBuilder = gen_scene(cfg, actor)
    builder.params = builder.params.evolve(time_step=cfg.time_step)
    actor = builder.actors_concrete[actor.id]
    scene: sim.Scene = builder.finish()
    optimizer = optim.PNCG(maxiter=10**3, rtol=0.0, atol=1e-7)

    timestamps: Float[np.ndarray, " frames"] = np.zeros((cfg.n_frames + 1,))
    displacement: Float[np.ndarray, " frames"] = np.zeros((cfg.n_frames + 1,))
    velocity: Float[np.ndarray, " frames"] = np.zeros((cfg.n_frames + 1,))

    writer = melon.SeriesWriter(cfg.output_dir / "animation.vtu.series", fps=cfg.fps)
    actor = scene.export_actor(actor)
    mesh: pv.UnstructuredGrid = actor.to_pyvista()
    writer.append(mesh, time=0.0)
    timestamps[0] = 0.0
    displacement[0] = 0.0
    velocity[0] = 0.0
    for t in range(1, cfg.n_frames + 1):
        result: optim.OptimizeResult
        scene, result = scene.solve(optimizer=optimizer)
        if not result["success"]:
            ic(result)
        scene = scene.step(result["x"])
        actor = scene.export_actor(actor)
        actor = helper.dump_optim_result(scene, actor, result)
        mesh: pv.UnstructuredGrid = actor.to_pyvista()
        timestamps[t] = t * cfg.time_step
        displacement[t] = jnp.linalg.norm(helper.center_of_mass_displacement(actor))
        velocity[t] = jnp.linalg.norm(helper.center_of_mass_velocity(actor))
        writer.append(mesh, time=t * cfg.time_step)
    writer.end()

    gravity: float = jnp.linalg.norm(helper.DEFAULT_GRAVITY)
    displacement_expected: Float[np.ndarray, " frames"] = 0.5 * gravity * timestamps**2
    velocity_expected: Float[np.ndarray, " frames"] = gravity * timestamps
    plot_time_series(
        timestamps=timestamps,
        actual=displacement,
        expected=displacement_expected,
        ylabel="Displacement (m)",
    )
    plt.savefig(cfg.output_dir / "displacement.png")
    plot_time_series(
        timestamps=timestamps,
        actual=velocity,
        expected=velocity_expected,
        ylabel="Velocity (m/s)",
    )
    plt.savefig(cfg.output_dir / "velocity.png")


def gen_pyvista(cfg: Config) -> pv.UnstructuredGrid:
    surface: pv.PolyData = cast("pv.PolyData", pv.examples.download_bunny())
    mesh: pv.UnstructuredGrid = melon.tetwild(surface)
    # mesh.scale(20.0 / mesh.length, inplace=True)
    # mesh: pv.UnstructuredGrid = pv.examples.cells.Tetrahedron()
    # mesh: pv.UnstructuredGrid = cast(
    #     "pv.UnstructuredGrid", pv.examples.download_tetrahedron()
    # )
    # mesh.scale(0.2 / mesh.length, inplace=True)
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


def gen_scene(_cfg: Config, actor: sim.Actor) -> sim.SceneBuilder:
    builder = sim.SceneBuilder()
    actor = builder.assign_dofs(actor)
    builder.add_energy(energy.ARAP.from_actor(actor))
    return builder


def plot_time_series(
    timestamps: Float[ArrayLike, " frames"],
    actual: Float[ArrayLike, " frames"],
    expected: Float[ArrayLike, " frames"],
    ylabel: str,
) -> None:
    plt.figure()
    plt.plot(timestamps, actual, label="Actual")
    plt.plot(timestamps, expected, label="Expected")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.tight_layout()


if __name__ == "__main__":
    cherries.run(main, play=True)
