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


class Config(cherries.BaseConfig):
    output_dir: Path = utils.data("animation")
    mesh: Path = utils.data("head.vtu")
    duration: float = 1.0
    fps: float = 120.0

    d_hat: float = 0.1
    density: float = 1
    lambda_: float = 3 * 1e3
    mu: float = 1 * 1e3

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
        integrator=sim.TimeIntegratorStatic(),
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
    actor.cell_data["muscle-orientation"] = actor.cell_data[
        "muscle-orientation"
    ].reshape(actor.n_cells, 3, 3)

    actor = helper.add_point_mass(actor)
    actor = builder.assign_dofs(actor)
    builder.add_energy(energy.PhaceActive.from_actor(actor))

    return builder


def update_activation(scene: sim.Scene, activation_scalar: float) -> sim.Scene:
    actor: sim.Actor = scene.actors["head"]
    activation: Float[Array, "cells 3 3"] = einops.repeat(
        jnp.diagflat(
            jnp.asarray(
                [
                    activation_scalar,
                    activation_scalar**-0.3,
                    activation_scalar**-0.3,
                ]
            )
        ),
        "i j -> cells i j",
        cells=actor.n_cells,
    )
    orientation: Float[Array, "cells 3 3"] = actor.cell_data["muscle-orientation"]
    activation = einops.einsum(
        orientation,
        activation,
        orientation,
        "cells i j, cells j k, cells l k -> cells i l",
    )
    actor.cell_data["activation"] = activation
    scene.actors.add(actor)
    return scene


def main(cfg: Config) -> None:
    builder: sim.SceneBuilder = gen_scene(cfg)
    scene: sim.Scene = builder.finish()
    optimizer: optim.Optimizer = optim.PNCG(maxiter=500, rtol=1e-5)

    activation_scalars: Float[np.ndarray, " frames"] = np.geomspace(
        1.0, 1e-3, num=cfg.n_frames
    )

    writers: Mapping[str, melon.SeriesWriter] = {
        name: melon.SeriesWriter(cfg.output_dir / f"{name}.vtu.series", fps=cfg.fps)
        for name in scene.actors
    }
    meshes: Mapping[str, pv.UnstructuredGrid] = helper.dump_all_pyvista(scene)
    for name, mesh in meshes.items():
        writers[name].append(mesh, time=0.0)
    for frame in range(1, cfg.n_frames + 1):
        activation_scalar: float = activation_scalars[frame - 1]
        scene = update_activation(scene, activation_scalar=activation_scalar)
        result: optim.OptimizeResult
        scene, result = scene.solve(optimizer=optimizer)
        if not result["success"]:
            logger.error("{}", result)
        scene = scene.step(result["x"])
        meshes = helper.dump_all_pyvista(scene, result)
        for name, mesh in meshes.items():
            mesh.field_data["activation"] = activation_scalar
            writers[name].append(mesh, time=frame * cfg.time_step)


if __name__ == "__main__":
    cherries.run(main, profile="playground")
