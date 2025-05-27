import einops
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float, Integer

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import grapes, melon


def main() -> None:
    grapes.init_logging()
    geometry: pv.UnstructuredGrid = gen_geometry()
    scene: apple.Scene = gen_scene(geometry)
    writer = melon.SeriesWriter("data/examples/static/stretch.vtu.series")

    def warp_result(solution: apple.OptimizeResult) -> pv.UnstructuredGrid:
        fields: dict[str, apple.Field] = scene.make_fields(solution.x)
        result: pv.UnstructuredGrid = geometry.copy()
        result.point_data["solution"] = fields["displacement"].values
        result.warp_by_vector("solution", inplace=True)
        return result

    def callback(intermediate_result: apple.OptimizeResult) -> None:
        result: pv.UnstructuredGrid = warp_result(intermediate_result)
        writer.append(result)

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


def gen_geometry(lr: float = 0.05) -> pv.UnstructuredGrid:
    surface: pv.PolyData = pv.Box()
    geometry: pv.UnstructuredGrid = melon.tetwild(surface, lr=lr)
    return geometry


def gen_dirichlet(
    geometry: pv.UnstructuredGrid,
) -> tuple[Integer[np.ndarray, " dirichlet"], Float[np.ndarray, " dirichlet"]]:
    dirichlet_mask: Bool[np.ndarray, " points"] = np.zeros(
        (geometry.n_points,), dtype=bool
    )
    dirichlet_values: Float[np.ndarray, " points 3"] = np.zeros(
        (geometry.n_points, 3), dtype=float
    )
    x_min: float
    x_max: float
    x_min, x_max, _y_min, _y_max, _z_min, _z_max = geometry.bounds
    x_length: float = x_max - x_min
    left_side_mask: Bool[np.ndarray, " points"] = (
        geometry.points[:, 0] < x_min + 0.01 * x_length
    )
    right_side_mask: Bool[np.ndarray, " points"] = (
        geometry.points[:, 0] > x_max - 0.01 * x_length
    )
    dirichlet_values[left_side_mask] = np.asarray([-0.5 * x_length, 0.0, 0.0])
    dirichlet_values[right_side_mask] = np.asarray([0.5 * x_length, 0.0, 0.0])
    dirichlet_mask: Bool[np.ndarray, " points"] = left_side_mask | right_side_mask
    geometry.point_data["dirichlet-mask"] = dirichlet_mask
    geometry.point_data["dirichlet-values"] = dirichlet_values
    dirichlet_mask: Bool[np.ndarray, "points 3"] = einops.repeat(
        dirichlet_mask, " points -> (points 3)"
    )
    dirichlet_index: Integer[np.ndarray, " dirichlet"]
    (dirichlet_index,) = np.nonzero(dirichlet_mask)
    return dirichlet_index, dirichlet_values.ravel()[dirichlet_index]


def gen_scene(geometry: pv.UnstructuredGrid) -> apple.Scene:
    dirichlet_index: Integer[np.ndarray, " dirichlet"]
    dirichlet_values: Float[np.ndarray, " dirichlet"]
    dirichlet_index, dirichlet_values = gen_dirichlet(geometry)
    domain: apple.Domain = apple.Domain.from_geometry(geometry)
    field_spec: apple.FieldSpec = apple.FieldSpec.from_domain(
        domain=domain,
        dirichlet_index=dirichlet_index,
        dirichlet_values=dirichlet_values,
        id="displacement",
    )
    energy = apple.energy.elastic.PhaceStatic(field_id=field_spec.id)
    scene = apple.Scene()
    scene.add_field(field_spec)
    scene.add_energy(energy)
    return scene


if __name__ == "__main__":
    main()
