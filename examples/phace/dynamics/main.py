from collections.abc import Mapping
from pathlib import Path

import einops
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Bool, Float
from loguru import logger

from liblaf import cherries, melon
from liblaf.apple import energy, helper, optim, sim, utils

# wp.config.verify_fp = True
# wp.config.mode = "debug"
# wp.config.verbose_warnings = True

NUMERIC_SCALE: float = 1.0


class Config(cherries.BaseConfig):
    output_dir: Path = utils.data("animation")
    mesh: Path = utils.data("head.vtu")
    duration: float = 2.0
    fps: float = 120.0

    d_hat: float = 0.1
    density: float = 1.0 * NUMERIC_SCALE
    lambda_: float = 3 * 1e3 * NUMERIC_SCALE
    mu: float = 1 * 1e3 * NUMERIC_SCALE

    amplitude: float = 10.0
    frequency: float = 1.0

    @property
    def n_frames(self) -> int:
        return int(self.duration * self.fps)

    @property
    def time_step(self) -> float:
        return 1 / self.fps


def gen_scene(cfg: Config) -> sim.SceneBuilder:
    mesh_pv: pv.UnstructuredGrid = pv.read(cfg.mesh)
    # surface: pv.PolyData = pv.Box()
    # mesh_pv: pv.UnstructuredGrid = melon.tetwild(surface)
    mesh_pv.cell_data["density"] = cfg.density
    mesh_pv.cell_data["lambda"] = cfg.lambda_
    mesh_pv.cell_data["mu"] = cfg.mu

    builder = sim.SceneBuilder(
        integrator=sim.ImplicitEuler(),
        params=sim.GlobalParams(time_step=jnp.asarray(cfg.time_step)),
    )

    actor: sim.Actor = sim.Actor.from_pyvista(mesh_pv, grad=True, id_="head")
    is_skull: Bool[Array, " P"] = actor.point_data["is-skull"]
    actor = actor.with_dirichlet(
        sim.Dirichlet.from_mask(
            einops.repeat(is_skull, "points -> points dim", dim=3),
            values=jnp.zeros((actor.n_points, 3), dtype=float),
        )
    )

    actor = helper.add_point_mass(actor)
    actor = builder.assign_dofs(actor)
    builder.add_energy(energy.PhacePassive.from_actor(actor))

    return builder


def update_dirichlet(
    builder: sim.SceneBuilder, scene: sim.Scene, displacement: float
) -> sim.Scene:
    actor: sim.Actor = scene.actors["head"]
    is_skull: Bool[Array, " P"] = actor.point_data["is-skull"]
    disp: Float[Array, " P D"] = einops.repeat(
        jnp.asarray([0.0, displacement, 0.0]),
        "dim -> points dim",
        points=actor.n_points,
    )
    actor = actor.with_dirichlet(
        sim.Dirichlet.from_mask(
            einops.repeat(is_skull, "points -> points dim", dim=3), values=disp
        )
    )
    builder.actors.add(actor)
    scene.actors.add(actor)
    scene = scene.replace(dirichlet=builder.dirichlet)
    return scene


def main(cfg: Config) -> None:
    builder: sim.SceneBuilder = gen_scene(cfg)
    scene: sim.Scene = builder.finish()
    optimizer: optim.Optimizer = optim.PNCG(maxiter=500, rtol=1e-8)

    displacements: Float[np.ndarray, " frames"] = cfg.amplitude * (
        1.0
        - np.cos(
            2
            * np.pi
            * cfg.frequency
            * np.linspace(0, cfg.frequency * cfg.duration, num=cfg.n_frames)
        )
    )

    writers: Mapping[str, melon.SeriesWriter] = {
        name: melon.SeriesWriter(cfg.output_dir / f"{name}.vtu.series", fps=cfg.fps)
        for name in scene.actors
    }
    meshes: Mapping[str, pv.UnstructuredGrid] = helper.dump_all_pyvista(scene)
    for name, mesh in meshes.items():
        writers[name].append(mesh, time=0.0)
    for frame in range(1, cfg.n_frames + 1):
        disp: float = displacements[frame - 1]
        scene = update_dirichlet(builder=builder, scene=scene, displacement=disp)
        result: optim.OptimizeResult
        scene, result = scene.solve(optimizer=optimizer)
        if not result["success"]:
            logger.error("{}", result)
        scene = scene.step(result["x"])
        meshes = helper.dump_all_pyvista(scene, result)
        for name, mesh in meshes.items():
            mesh.field_data["displacement"] = disp
            writers[name].append(mesh, time=frame * cfg.time_step)


if __name__ == "__main__":
    cherries.run(main, profile="playground")
