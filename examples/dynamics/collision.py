
import jax.numpy as jnp
import pyvista as pv
import warp as wp

from liblaf import cherries, grapes, melon
from liblaf.apple import energy, helper, optim, sim


class Config(cherries.BaseConfig):
    duration: float = 3.0
    fps: float = 30.0

    d_hat: float = 1e-3
    density: float = 1e3
    lambda_: float = 3e4
    mu: float = 1e4

    @property
    def n_frames(self) -> int:
        return int(self.duration * self.fps)

    @property
    def time_step(self) -> float:
        return 1 / self.fps


def main(cfg: Config) -> None:
    grapes.init_logging()
    wp.init()
    soft: sim.Actor = gen_actor(cfg)
    ground: sim.Actor = gen_rigid(cfg)
    builder: sim.SceneBuilder = gen_scene(cfg, soft, ground)
    builder.params = builder.params.evolve(time_step=cfg.time_step)
    soft = builder.actors_concrete[soft.id]
    scene: sim.Scene = builder.finish()
    optimizer = optim.PNCG(maxiter=10**3, d_hat=cfg.d_hat)

    writer = melon.SeriesWriter(
        "data/examples/dynamics/collision.vtu.series", fps=cfg.fps
    )
    melon.save("data/examples/dynamics/collision-ground.vtp", ground.to_pyvista())
    soft = scene.export_actor(soft)
    mesh: pv.UnstructuredGrid = soft.to_pyvista()
    writer.append(mesh, time=0.0)
    for t in range(cfg.n_frames):
        result: optim.OptimizeResult
        scene, result = scene.solve(optimizer=optimizer)
        # if not result["success"]:
        #     ic(result)
        ic(result)
        scene = scene.step(result["x"])
        soft = scene.export_actor(soft)
        soft = helper.dump_optim_result(scene, soft, result)
        collision_energy: energy.CollisionVertFace = scene.energies[
            "CollisionVertFace-000"
        ]  # pyright: ignore[reportAssignmentType]
        soft = soft.set_point_data("collide", collision_energy.candidates.collide)
        soft = soft.set_point_data("sign", collision_energy.candidates.sign)
        mesh: pv.UnstructuredGrid = soft.to_pyvista()
        writer.append(mesh, time=t * cfg.time_step)
    writer.end()


def gen_pyvista(cfg: Config) -> pv.UnstructuredGrid:
    # surface: pv.PolyData = cast("pv.PolyData", pv.examples.download_bunny())
    # mesh: pv.UnstructuredGrid = melon.tetwild(surface)
    mesh: pv.UnstructuredGrid = pv.examples.cells.Tetrahedron()
    # mesh = cast("pv.UnstructuredGrid", pv.examples.download_tetrahedron())
    y_min: float
    _, _, y_min, _, _, _ = mesh.bounds
    mesh.translate((0, 0.2 - y_min, 0), inplace=True)
    mesh.cell_data["density"] = cfg.density
    mesh.cell_data["lambda"] = cfg.lambda_
    mesh.cell_data["mu"] = cfg.mu
    return mesh


def gen_actor(cfg: Config) -> sim.Actor:
    mesh: pv.UnstructuredGrid = gen_pyvista(cfg)
    actor: sim.Actor = sim.Actor.from_pyvista(mesh)
    actor = helper.add_point_mass(actor)
    actor = helper.add_gravity(actor)
    return actor


def gen_rigid(_cfg: Config) -> sim.Actor:
    surface: pv.PolyData = pv.Box((-1, 1, -1, 0, -1, 1), quads=False)
    actor: sim.Actor = sim.Actor.from_pyvista(surface)
    actor = actor.with_collision_mesh()
    actor = actor.set_point_data("mass", jnp.ones((actor.n_points,)))
    actor = actor.set_dirichlet(
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
        energy.CollisionVertFace.from_actors(soft, rigid, rest_length=cfg.d_hat)
    )
    return builder


if __name__ == "__main__":
    cherries.run(main, play=True)
