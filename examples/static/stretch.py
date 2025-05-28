import einops
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float, Integer

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import grapes, melon


def main() -> None:
    grapes.init_logging()
    geometry: apple.Geometry = gen_geometry()
    scene: apple.Scene = gen_scene(geometry)
    writer = melon.SeriesWriter("data/examples/static/stretch.vtu.series")

    def callback(result: apple.OptimizeResult) -> None:
        if "Delta_E" in result:
            ic(result["Delta_E"] / result["Delta_E0"])
        geometries: dict[str, apple.Geometry] = scene.make_geometries(result["x"])
        writer.append(geometries["box"].mesh)

    solution: apple.OptimizeResult = apple.minimize(
        scene.fun,
        x0=jnp.zeros((scene.n_free,)),
        jac=scene.jac,
        jac_and_hess_diag=scene.jac_and_hess_diag,
        hess_quad=scene.hess_quad,
        method=apple.PNCG(maxiter=150, tol=1e-5),
        callback=callback,
    )
    ic(solution)

    geometries: dict[str, apple.Geometry] = scene.make_geometries(solution["x"])
    melon.save("data/examples/static/stretch-solution.vtu", geometries["box"].mesh)


def gen_geometry(lr: float = 0.05) -> apple.Geometry:
    surface: pv.PolyData = pv.Box()
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=lr)
    geometry: apple.Geometry = apple.Geometry(mesh=mesh, id="box")
    return geometry


def gen_dirichlet(
    geometry: apple.Geometry,
) -> tuple[Integer[np.ndarray, " dirichlet"], Float[np.ndarray, " dirichlet"]]:
    mesh: pv.UnstructuredGrid = geometry.mesh
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
        dirichlet_mask, " points -> (points 3)"
    )
    dirichlet_index: Integer[np.ndarray, " dirichlet"]
    (dirichlet_index,) = np.nonzero(dirichlet_mask)
    return dirichlet_index, dirichlet_values.ravel()[dirichlet_index]


def gen_scene(geometry: apple.Geometry) -> apple.Scene:
    dirichlet_index: Integer[np.ndarray, " dirichlet"]
    dirichlet_values: Float[np.ndarray, " dirichlet"]
    domain: apple.Domain = apple.Domain.from_geometry(geometry)
    dirichlet_index, dirichlet_values = gen_dirichlet(geometry)
    field: apple.Field = apple.Field.from_domain(
        domain=domain, id="displacement"
    ).with_dirichlet(dirichlet_index=dirichlet_index, dirichlet_values=dirichlet_values)
    energy = apple.energy.elastic.PhaceStatic(field_id=field.id)
    scene = apple.Scene()
    scene.add_field(field)
    scene.add_energy(energy)
    return scene


if __name__ == "__main__":
    main()
