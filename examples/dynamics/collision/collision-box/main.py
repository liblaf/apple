import collections
from pathlib import Path
from typing import cast

import attrs
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import warp as wp
from jaxtyping import Array, Float

from liblaf import cherries, grapes, melon
from liblaf.apple import energy, helper, optim, sim, struct, utils


class Config(cherries.BaseConfig):
    output_dir: Path = utils.data("")
    duration: float = 5.0
    fps: float = 120.0

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
    grapes.logging.init()
    wp.init()
    soft: sim.Actor = gen_actor(cfg)
    ground: sim.Actor = gen_rigid(cfg)
    builder: sim.SceneBuilder = gen_scene(cfg, soft, ground)
    builder.params = builder.params.replace(time_step=cfg.time_step)
    soft = builder.actors_concrete[soft.id]
    scene: sim.Scene = builder.finish()
    optimizer = optim.PNCG(d_hat=cfg.d_hat, maxiter=500, rtol=1e-5)

    writer = melon.SeriesWriter(cfg.output_dir / "animation.vtu.series", fps=cfg.fps)
    melon.save(cfg.output_dir / "ground.vtp", ground.to_pyvista())
    soft = scene.export_actor(soft)
    mesh: pv.UnstructuredGrid = soft.to_pyvista()
    writer.append(mesh, time=0.0)

    for t in range(1, cfg.n_frames + 1):
        result: optim.OptimizeResult
        callback = Callback()
        scene = scene.pre_time_step()
        callback.first(scene.pre_optim_iter())
        scene, result = scene.solve(optimizer=optimizer, callback=callback.callback)
        # if not result["success"]:
        #     ic(result)
        ic(result)
        scene = scene.step(result["x"])
        soft = scene.export_actor(soft)
        soft = helper.dump_optim_result(scene, soft, result)
        collision: energy.CollisionVertFace = scene.energies["CollisionVertFace-000"]  # pyright: ignore[reportAssignmentType]
        soft = helper.dump_collision(soft, collision)
        mesh: pv.UnstructuredGrid = soft.to_pyvista()
        mesh.field_data["ratio"] = result["Delta_E"] / result["Delta_E0"]
        for k, v in callback.fun.items():
            mesh.field_data[k] = np.maximum(v, np.finfo(float).eps)
        writer.append(mesh, time=t * cfg.time_step)
    writer.end()


def gen_pyvista(cfg: Config) -> pv.UnstructuredGrid:
    # surface: pv.PolyData = cast("pv.PolyData", pv.examples.download_bunny())
    # mesh: pv.UnstructuredGrid = melon.tetwild(surface)
    # mesh: pv.UnstructuredGrid = pv.examples.cells.Tetrahedron()
    # mesh.scale(0.1, inplace=True)
    mesh: pv.UnstructuredGrid = cast(
        "pv.UnstructuredGrid", pv.examples.download_tetrahedron()
    )
    mesh.translate(-np.asarray(mesh.center), inplace=True)
    mesh.scale(0.2 / mesh.length, inplace=True)
    y_min: float
    _, _, y_min, _, _, _ = mesh.bounds
    mesh.translate((0, 0.05 - y_min, 0), inplace=True)
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
    surface: pv.PolyData = pv.Box((-0.2, 0.2, -0.2, 0, -0.2, 0.2), quads=False)
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
    builder.add_energy(energy.PhaceStatic.from_actor(soft))
    builder.add_energy(
        energy.CollisionVertFace.from_actors(
            rigid, soft, stiffness=1e4, rest_length=cfg.d_hat
        )
    )
    return builder


if __name__ == "__main__":
    cherries.run(main, profile="playground")
