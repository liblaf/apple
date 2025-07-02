from pathlib import Path

import einops
import jax
import numpy as np
import pyvista as pv
import pyvista.examples
from jaxtyping import Bool, Float

from liblaf import cherries, melon
from liblaf.apple import energy, helper, optim, sim, utils


class Config(cherries.BaseConfig):
    output_dir: Path = utils.data("")


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = gen_pyvista()
    actor: sim.Actor = gen_actor(mesh)
    builder: sim.SceneBuilder = gen_scene(actor)
    builder.integrator = sim.TimeIntegratorStatic()
    actor = builder.actors_concrete[actor.id]
    scene: sim.Scene = builder.finish()
    optimizer = optim.PNCG(maxiter=10**3, rtol=1e-10)

    x0: Float[jax.Array, " DOF"] = gen_init(scene, mesh.length)
    scene = scene.pre_optim_iter(x0)

    writer = melon.SeriesWriter(cfg.output_dir / "animation.vtu.series")
    actor = scene.export_actor(actor)
    mesh: pv.UnstructuredGrid = actor.to_pyvista()
    writer.append(mesh)

    def callback(result: optim.OptimizeResult, scene: sim.Scene) -> None:
        nonlocal actor
        # if result["n_iter"] % 10 != 0:
        #     return
        ic(result)
        actor = scene.export_actor(actor)
        actor = helper.dump_optim_result(scene, actor, result)
        mesh: pv.UnstructuredGrid = actor.to_pyvista()
        writer.append(mesh)

    result: optim.OptimizeResult
    scene, result = scene.solve(optimizer=optimizer, callback=callback)
    ic(result)


def gen_pyvista(lr: float = 0.05) -> pv.UnstructuredGrid:
    surface: pv.PolyData = pyvista.examples.download_bunny(load=True)
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=lr)
    # mesh: pv.UnstructuredGrid = pv.examples.cells.Tetrahedron()
    mesh.cell_data["density"] = 1.0
    mesh.cell_data["lambda"] = 3.0
    mesh.cell_data["mu"] = 1.0
    return mesh


def gen_dirichlet(mesh: pv.UnstructuredGrid) -> sim.Dirichlet:
    dirichlet_mask: Bool[np.ndarray, " points"] = np.zeros((mesh.n_points,), dtype=bool)
    dirichlet_values: Float[np.ndarray, " points 3"] = np.zeros(
        (mesh.n_points, 3), dtype=float
    )
    y_min: float
    y_max: float
    _x_min, _x_max, y_min, y_max, _z_min, _z_max = mesh.bounds
    y_length: float = y_max - y_min
    dirichlet_mask: Bool[np.ndarray, " points"] = (
        mesh.points[:, 1] < y_min + 0.05 * y_length
    )
    dirichlet_values[dirichlet_mask] = np.asarray([0.0, 0.0, 0.0])
    mesh.point_data["dirichlet-mask"] = dirichlet_mask
    mesh.point_data["dirichlet-values"] = dirichlet_values
    dirichlet_mask: Bool[np.ndarray, "points 3"] = einops.repeat(
        dirichlet_mask, " points -> points 3"
    )
    return sim.Dirichlet.from_mask(dirichlet_mask, dirichlet_values)


def gen_actor(mesh: pv.UnstructuredGrid) -> sim.Actor:
    actor: sim.Actor = sim.Actor.from_pyvista(mesh, grad=True)
    actor = actor.set_dirichlet(gen_dirichlet(mesh))
    actor = helper.add_point_mass(actor)
    return actor


def gen_scene(actor: sim.Actor) -> sim.SceneBuilder:
    builder = sim.SceneBuilder()
    actor = builder.assign_dofs(actor)
    builder.add_energy(energy.ARAP.from_actor(actor))
    return builder


def gen_init(scene: sim.Scene, length: float) -> Float[jax.Array, " free"]:
    random = utils.Random()
    u0: Float[jax.Array, " free"] = random.uniform(
        (scene.n_dofs,), minval=-0.5 * length, maxval=0.5 * length
    )
    u0 = scene.dirichlet.apply(u0)
    return u0


if __name__ == "__main__":
    cherries.run(main, play=True)
