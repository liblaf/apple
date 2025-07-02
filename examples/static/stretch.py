import einops
import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float

from liblaf import grapes, melon
from liblaf.apple import energy, helper, optim, sim


def main() -> None:
    grapes.init_logging()
    mesh: pv.UnstructuredGrid = gen_pyvista()
    actor: sim.Actor = gen_actor(mesh)
    builder: sim.SceneBuilder = gen_scene(actor)
    builder.integrator = sim.TimeIntegratorStatic()
    actor = builder.actors_concrete[actor.id]
    scene: sim.Scene = builder.finish()

    optimizer = optim.PNCG(maxiter=10**3, rtol=1e-6)

    writer = melon.SeriesWriter("data/examples/static/stretch.vtu.series")
    actor = scene.export_actor(actor)
    mesh: pv.UnstructuredGrid = actor.to_pyvista()
    writer.append(mesh)

    def callback(result: optim.OptimizeResult, scene: sim.Scene) -> None:
        nonlocal actor
        actor = scene.export_actor(actor)
        actor = helper.dump_optim_result(scene, actor, result)
        mesh: pv.UnstructuredGrid = actor.to_pyvista()
        writer.append(mesh)

    result: optim.OptimizeResult
    scene, result = scene.solve(optimizer=optimizer, callback=callback)
    ic(result)


def gen_pyvista(lr: float = 0.05) -> pv.UnstructuredGrid:
    surface: pv.PolyData = pv.Cylinder(direction=(1, 0, 0))
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=lr)
    mesh.cell_data["density"] = 1.0
    mesh.cell_data["lambda"] = 3.0
    mesh.cell_data["mu"] = 1.0
    return mesh


def gen_dirichlet(
    mesh: pv.UnstructuredGrid,
) -> sim.Dirichlet:
    dirichlet_mask: Bool[np.ndarray, " points"] = np.zeros((mesh.n_points,), dtype=bool)
    dirichlet_values: Float[np.ndarray, " points 3"] = np.zeros(
        (mesh.n_points, 3), dtype=float
    )
    x_min: float
    x_max: float
    x_min, x_max, _y_min, _y_max, _z_min, _z_max = mesh.bounds
    x_length: float = x_max - x_min
    left_side_mask: Bool[np.ndarray, " points"] = (
        mesh.points[:, 0] < x_min + 0.01 * x_length
    )
    right_side_mask: Bool[np.ndarray, " points"] = (
        mesh.points[:, 0] > x_max - 0.01 * x_length
    )
    dirichlet_values[left_side_mask] = np.asarray([-0.5 * x_length, 0.0, 0.0])
    dirichlet_values[right_side_mask] = np.asarray([0.5 * x_length, 0.0, 0.0])
    dirichlet_mask: Bool[np.ndarray, " points"] = left_side_mask | right_side_mask
    mesh.point_data["dirichlet-mask"] = dirichlet_mask
    mesh.point_data["dirichlet-values"] = dirichlet_values
    dirichlet_mask: Bool[np.ndarray, "points 3"] = einops.repeat(
        dirichlet_mask, " points -> points 3"
    )
    return sim.Dirichlet.from_mask(dirichlet_mask, dirichlet_values)


def gen_actor(mesh: pv.UnstructuredGrid) -> sim.Actor:
    actor: sim.Actor = sim.Actor.from_pyvista(mesh, grad=True)
    dirichlet: sim.Dirichlet = gen_dirichlet(mesh)
    actor = actor.set_dirichlet(dirichlet)
    actor = helper.add_point_mass(actor)
    return actor


def gen_scene(actor: sim.Actor) -> sim.SceneBuilder:
    builder = sim.SceneBuilder()
    actor = builder.assign_dofs(actor)
    builder.add_energy(energy.PhaceStatic.from_actor(actor))
    return builder


if __name__ == "__main__":
    main()
