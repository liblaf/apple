import collections
from pathlib import Path
from typing import cast

import attrs
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from jaxtyping import Array, ArrayLike, Float

from liblaf import cherries, melon
from liblaf.apple import energy, helper, optim, sim, struct, utils


class Config(cherries.BaseConfig):
    output_dir: Path = utils.data("")

    dirichlet_thickness: float = 0.05
    duration: float = 10.0
    fps: float = 30.0

    density: float = 1e3
    lambda_: float = 3.0 * 1e3
    mu: float = 1.0 * 1e3

    @property
    def n_frames(self) -> int:
        return int(self.duration * self.fps)

    @property
    def time_step(self) -> float:
        return 1 / self.fps


@attrs.define
class Callback:
    fun: dict[str, list[Float[Array, ""]]] = attrs.field(
        factory=lambda: collections.defaultdict(list)
    )

    def first(self, scene: sim.Scene) -> None:
        result = optim.OptimizeResult({"x": scene.x0})
        self.callback(result, scene)

    def callback(self, result: optim.OptimizeResult, scene: sim.Scene) -> None:
        x: Float[Array, " DOF"] = result["x"]
        fields: struct.ArrayDict = scene.scatter(x)
        for e in scene.energies.values():
            self.fun[e.id].append(e.fun(fields, scene.params))
        self.fun[scene.integrator.name].append(
            scene.integrator.fun(x, scene.state, scene.params)
        )
        self.fun["Energy"].append(scene.fun(x))


def main(cfg: Config) -> None:
    plt.rc("figure", dpi=300)
    actor: sim.Actor = gen_actor(cfg)
    builder: sim.SceneBuilder = gen_scene(cfg, actor)
    builder.params = builder.params.replace(time_step=cfg.time_step)
    actor = builder.actors_concrete[actor.id]
    scene: sim.Scene = builder.finish()
    optimizer = optim.PNCG(maxiter=500, rtol=1e-10)

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
        callback = Callback()
        scene = scene.pre_time_step()
        callback.first(scene.pre_optim_iter())
        scene, result = scene.solve(optimizer=optimizer, callback=callback.callback)
        if not result["success"]:
            ic(result)
        scene = scene.step(result["x"])
        actor = scene.export_actor(actor)
        actor = helper.dump_optim_result(scene, actor, result)
        mesh: pv.UnstructuredGrid = actor.to_pyvista()
        mesh.field_data["ratio"] = result["Delta_E"] / result["Delta_E0"]
        mesh.field_data.update(callback.fun)

        fields: struct.ArrayDict = scene.scatter(result["x"])
        for e in scene.energies.values():
            jac_dict: struct.ArrayDict = e.jac(fields, scene.params)
            jac_flat: Float[Array, " DOF"] = scene.gather(jac_dict)
            mesh.point_data[f"{e.id}.jac"] = actor.dofs_global.get(jac_flat)
        mesh.point_data[f"{scene.integrator.name}.jac"] = actor.dofs_global.get(
            scene.integrator.jac(result["x"], scene.state, scene.params)
        )

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
    print(mesh.cell_data["Volume"])
    ic(np.count_nonzero(mesh.cell_data["Volume"] < 0.0))
    ic(np.count_nonzero(mesh.cell_data["Volume"] > 0.0))
    # mesh.scale(20.0 / mesh.length, inplace=True)
    # mesh: pv.UnstructuredGrid = pv.examples.cells.Tetrahedron()
    # mesh = cast("pv.UnstructuredGrid", pv.examples.download_tetrahedron())
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
    cherries.run(main, profile="playground")
