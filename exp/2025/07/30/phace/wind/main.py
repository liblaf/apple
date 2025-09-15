from collections.abc import Mapping
from pathlib import Path

import einops
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Bool, Float
from loguru import logger

from liblaf import cherries, melon
from liblaf.apple import energy, helper, optim, sim, struct, utils


class Config(cherries.BaseConfig):
    output_dir: Path = utils.data("animation")
    mesh: Path = utils.data("head.vtu")
    cranium: Path = utils.data("cranium.ply")

    duration: float = 1.0
    fps: float = 120.0

    density: float = 1
    lambda_: float = 3 * 1e3
    mu: float = 1 * 1e3

    collision_stiffness: float = 500
    d_hat: float = 0.05

    wind: float = 1e3

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

    cranium_pv: pv.PolyData = melon.load_poly_data(cfg.cranium)
    cranium_pv.point_data["mass"] = 1.0
    cranium: sim.Actor = sim.Actor.from_pyvista(
        cranium_pv, collision=True, id_="cranium"
    )
    cranium = cranium.with_dirichlet(
        sim.Dirichlet.from_mask(
            jnp.ones((cranium.n_points, 3), dtype=jnp.bool_),
            jnp.zeros((cranium.n_points, 3)),
        )
    )
    cranium = builder.assign_dofs(cranium)
    builder.add_actor(cranium)

    head_pv.point_data["point-id"] = np.arange(head_pv.n_points)
    surface_pv: pv.PolyData = head_pv.extract_surface()
    face_pv: pv.PolyData = surface_pv.extract_points(
        surface_pv.point_data["is-face"], adjacent_cells=False
    ).extract_surface()
    face: sim.Actor = sim.Actor.from_pyvista(face_pv, id_="face")
    face.dofs_global = sim.DOFs.from_index(
        head.dofs_global.index.reshape(head.dofs_global.shape)[
            face.point_data["point-id"]
        ]
    )
    builder.add_actor(face)

    builder.add_energy(
        energy.CollisionVertFace.from_actors(
            cranium, face, stiffness=cfg.collision_stiffness, rest_length=cfg.d_hat
        )
    )

    builder.integrator = sim.ImplicitEuler()

    return builder


def update_external_force(
    builder: sim.SceneBuilder, scene: sim.Scene, wind: float
) -> sim.Scene:
    actors: struct.NodeContainer[sim.Actor] = scene.actors
    for actor in scene.actors.values():
        if actor.id != "head":
            continue
        face_mask: Bool[Array, " P"] = actor.point_data["is-face"]
        force: Float[Array, "P 3"] = jnp.zeros((actor.n_points, 3))
        force = force.at[face_mask].set([0.0, wind, -wind])
        force *= actor.mass[:, jnp.newaxis]
        actor.point_data["force-ext"] = force
        actors.add(actor)
        force_global: Float[Array, " DOF"] = jnp.zeros((scene.n_dofs,))
        force_global = actor.dofs_global.set(force_global, force)
        scene.state["force-ext"] = force_global
    builder.actors = actors
    scene = scene.replace(actors=actors)
    return scene


def main(cfg: Config) -> None:
    builder: sim.SceneBuilder = gen_scene(cfg)
    scene: sim.Scene = builder.finish()
    optimizer = optim.PNCG(d_hat=cfg.d_hat, maxiter=500, rtol=1e-5)

    wind: Float[Array, " frames"] = (
        cfg.wind * (1.0 - jnp.cos(jnp.linspace(0, 4 * jnp.pi, cfg.n_frames))) / 2.0
    )

    writers: Mapping[str, melon.SeriesWriter] = {
        name: melon.SeriesWriter(cfg.output_dir / f"{name}.vtu.series", fps=cfg.fps)
        for name in scene.actors
    }
    meshes: Mapping[str, pv.DataSet] = helper.dump_all_pyvista(scene=scene)
    for name, writer in writers.items():
        writer.append(meshes[name], time=0.0)
    for t in range(1, cfg.n_frames + 1):
        scene = update_external_force(
            builder=builder, scene=scene, wind=wind[t - 1].item()
        )
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
