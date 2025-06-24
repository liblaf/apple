import einops
import jax
import numpy as np
import pyvista as pv
import pyvista.examples
from jaxtyping import Bool, Float, Integer

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import grapes, melon
from liblaf.apple import sim, struct, utils


def main() -> None:
    grapes.init_logging()
    geometry: sim.Geometry = gen_geometry()
    scene: sim.Scene = gen_scene(geometry)
    ic(scene)
    x0: Float[jax.Array, " free"] = gen_init(scene, geometry.structure.length)
    scene = scene.update(x0)
    optimizer = apple.PNCG(maxiter=1000, tol=1e-18)

    writer = melon.SeriesWriter("data/examples/static/random.vtu.series")

    def callback(result: apple.OptimizeResult) -> None:
        nonlocal scene
        if result["n_iter"] % 100 != 0:
            return
        if "Delta_E" in result:
            ic(result["Delta_E"] / result["Delta_E0"])
        scene = scene.update(result["x"])
        writer.append(scene.objects["Object-000"].geometry.to_pyvista())

    solution: apple.OptimizeResult = scene.solve(optimizer=optimizer)
    scene = scene.update(solution["x"])
    # ic(solution)
    obj = scene.objects["Object-000"]
    melon.save("data/examples/static/random-solution.vtu", obj.geometry.to_pyvista())


def gen_geometry(lr: float = 0.05) -> sim.Geometry:
    surface: pv.PolyData = pyvista.examples.download_bunny(load=True)
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=lr)
    mesh.cell_data["density"] = 1.0
    mesh.cell_data["mu"] = 1.0
    return sim.GeometryTetra.from_pyvista(mesh)


def gen_dirichlet(
    geometry: sim.Geometry,
) -> tuple[Integer[np.ndarray, " dirichlet"], Float[np.ndarray, " dirichlet"]]:
    mesh: pv.UnstructuredGrid = geometry.structure
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
        dirichlet_mask, " points -> (points 3)"
    )
    dirichlet_index: Integer[np.ndarray, " dirichlet"]
    (dirichlet_index,) = np.nonzero(dirichlet_mask)
    return dirichlet_index, dirichlet_values.ravel()[dirichlet_index]


def gen_scene(geometry: sim.Geometry) -> sim.Scene:
    builder = sim.SceneBuilder()
    region: sim.Region = sim.Region.from_geometry(geometry)
    obj: sim.Actor = sim.Actor.from_region(region)
    obj = builder.assign_dof(obj)
    dirichlet_index: Integer[np.ndarray, " dirichlet"]
    dirichlet_values: Float[np.ndarray, " dirichlet"]
    dirichlet_index, dirichlet_values = gen_dirichlet(geometry)
    obj = obj.evolve(
        dirichlet=sim.Dirichlet(struct.DofMapInteger(dirichlet_index), dirichlet_values)
    )
    energy = apple.energy.elastic.ARAP(obj)
    builder.add_energy(energy)
    return builder.build()


def gen_init(scene: sim.Scene, length: float) -> Float[jax.Array, " free"]:
    random = utils.Random()
    u0: Float[jax.Array, " free"] = random.uniform(
        (scene.n_dof,), minval=-0.5 * length, maxval=0.5 * length
    )
    u0 = scene.dirichlet_apply(u0)
    return u0


if __name__ == "__main__":
    main()
