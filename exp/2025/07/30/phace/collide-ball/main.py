from collections.abc import Mapping
from pathlib import Path

import einops
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Bool, Float, Integer
from loguru import logger

from liblaf import cherries, melon
from liblaf.apple import energy, helper, optim, sim, struct, utils


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


def gen_scene(cfg: Config) -> sim.Scene:
    head_pv: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.mesh)
    head_pv.cell_data["density"] = cfg.density
    head_pv.cell_data["lambda"] = cfg.lambda_
    head_pv.cell_data["mu"] = cfg.mu
    head: sim.Actor = sim.Actor.from_pyvista(head_pv, grad=True, id_="head")
    head = helper.add_point_mass(head)
    is_skull: Bool[Array, " P"] = head_pv.point_data["is-cranium"]
    head = head.with_dirichlet(
        sim.Dirichlet.from_mask(
            einops.repeat(is_skull, "P -> P D", D=3),
            values=jnp.zeros((head.n_points, 3), dtype=jnp.float32),
        )
    )

    center: np.ndarray = np.asarray(head_pv.center)
    center += np.asarray(
        [0.20 * head_pv.length, -0.05 * head_pv.length, 0.20 * head_pv.length]
    )
    ball_pv: pv.PolyData = pv.Icosphere(radius=0.1 * head_pv.length, center=center)
    ball_pv.point_data["mass"] = np.ones((ball_pv.n_points,))
    ball: sim.Actor = sim.Actor.from_pyvista(ball_pv, collision=True, id_="ball")
    ball = ball.with_dirichlet(
        sim.Dirichlet.from_mask(
            np.ones((ball.n_points, 3), dtype=bool), values=np.zeros((ball.n_points, 3))
        )
    )

    builder = sim.SceneBuilder(integrator=sim.TimeIntegratorStatic())
    ball = builder.assign_dofs(ball)
    head = builder.assign_dofs(head)
    builder.add_energy(energy.PhacePassive.from_actor(head))
    builder.add_energy(
        energy.CollisionVertFace.from_actors(
            rigid=ball, soft=head, rest_length=cfg.d_hat, stiffness=1e4
        )
    )

    return builder.finish()


def update_dirichlet(scene: sim.Scene, displacement: Float[Array, "3"]) -> sim.Scene:
    mask: Bool[Array, " DOF"] = jnp.zeros((scene.n_dofs,), dtype=bool)
    values: Float[Array, " DOF"] = jnp.zeros((scene.n_dofs,))
    actors: struct.NodeContainer[sim.Actor] = scene.actors
    for actor in scene.actors.values():
        actor: sim.Actor
        if actor.id == "ball":
            actor = actor.with_dirichlet(  # noqa: PLW2901
                sim.Dirichlet.from_mask(
                    mask=jnp.ones((actor.n_points, 3), dtype=bool),
                    values=einops.repeat(displacement, "D -> P D", P=actor.n_points),
                )
            )
        if actor.dirichlet_local is None or actor.dirichlet_local.dofs is None:
            continue
        actors.add(actor)
        dofs: Integer[Array, " DOF"] = jnp.asarray(actor.dofs_global)
        idx: Integer[Array, " dirichlet"] = actor.dirichlet_local.dofs.get(dofs).ravel()
        mask = mask.at[idx].set(True)
        values = values.at[idx].set(actor.dirichlet_local.values.ravel())
    return scene.replace(actors=actors, dirichlet=sim.Dirichlet.from_mask(mask, values))


def main(cfg: Config) -> None:
    scene: sim.Scene = gen_scene(cfg)
    optimizer = optim.PNCG(atol=7e-8, d_hat=cfg.d_hat, maxiter=10**3, rtol=1e-5)
    ball: sim.Actor = scene.actors["ball"]
    head: sim.Actor = scene.actors["head"]
    head_pv: pv.UnstructuredGrid = head.to_pyvista()

    ball_disp_total: float = 0.05 * head_pv.length
    ball_disp: np.ndarray = np.linspace(0.0, ball_disp_total, num=cfg.n_frames + 1)

    writers: Mapping[str, melon.SeriesWriter] = {
        ball.id: melon.SeriesWriter(
            cfg.output_dir / f"{ball.id}.vtp.series", fps=cfg.fps
        ),
        head.id: melon.SeriesWriter(
            cfg.output_dir / f"{head.id}.vtu.series", fps=cfg.fps
        ),
    }
    actors: struct.NodeContainer[sim.Actor] = helper.dump_actors(scene)
    meshes: Mapping[str, pv.DataSet] = helper.actors_to_pyvista(actors)
    for id_, writer in writers.items():
        writer.append(meshes[id_], time=0.0)
    for t in range(1, cfg.n_frames + 1):
        scene = update_dirichlet(
            scene, displacement=jnp.asarray([-ball_disp[t], 0.0, 0.0])
        )
        result: optim.OptimizeResult
        scene, result = scene.solve(optimizer=optimizer)
        scene = scene.step(result["x"])
        if not result["success"]:
            logger.error("{}", result)
        scene = scene.step(result["x"])
        actors: struct.NodeContainer[sim.Actor] = helper.dump_actors(
            scene, result=result
        )
        meshes: Mapping[str, pv.DataSet] = helper.actors_to_pyvista(actors)
        for id_, writer in writers.items():
            writer.append(meshes[id_], time=t * cfg.time_step)


if __name__ == "__main__":
    cherries.main(main, profile="playground")
