import einops
import jax
import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float, Integer

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import grapes, melon
from liblaf.apple import utils


def main() -> None:
    grapes.init_logging()
    geometry: pv.UnstructuredGrid = gen_geometry()
    scene: apple.Scene = gen_scene(geometry)
    x0: Float[jax.Array, " free"] = gen_init(scene, geometry.length)
    writer = melon.SeriesWriter("data/examples/static/random.vtu.series")

    def warp_result(solution: apple.OptimizeResult) -> pv.UnstructuredGrid:
        fields: dict[str, apple.Field] = scene.make_fields(solution.x)
        result: pv.UnstructuredGrid = geometry.copy()
        result.point_data["solution"] = fields["displacement"].values
        result.warp_by_vector("solution", inplace=True)
        return result

    def callback(intermediate_result: apple.OptimizeResult) -> None:
        if intermediate_result["n_iter"] % 100 != 0:
            return
        result: pv.UnstructuredGrid = warp_result(intermediate_result)
        writer.append(result)

    writer.append(warp_result(apple.OptimizeResult({"x": x0})))

    solution: apple.OptimizeResult = apple.minimize(
        scene.fun,
        x0=x0,
        jac=scene.jac,
        jac_and_hess_diag=scene.jac_and_hess_diag,
        hess_quad=scene.hess_quad,
        method=apple.PNCG(maxiter=10**4, tol=1e-15),
        callback=callback,
    )
    ic(solution)

    result: pv.UnstructuredGrid = warp_result(solution)
    melon.save("data/examples/static/random-solution.vtu", result)


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
    y_min: float
    y_max: float
    _x_min, _x_max, y_min, y_max, _z_min, _z_max = geometry.bounds
    y_length: float = y_max - y_min
    dirichlet_mask: Bool[np.ndarray, " points"] = (
        geometry.points[:, 1] < y_min + 0.01 * y_length
    )
    dirichlet_values[dirichlet_mask] = np.asarray([0.0, 0.0, 0.0])
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
    energy = apple.energy.elastic.ARAP(field_id=field_spec.id)
    scene = apple.Scene()
    scene.add_field(field_spec)
    scene.add_energy(energy)
    return scene


def gen_init(scene: apple.Scene, length: float) -> Float[jax.Array, " free"]:
    random = utils.Random()
    u0: Float[jax.Array, " free"] = random.uniform(
        (scene.n_free,), minval=-0.5 * length, maxval=0.5 * length
    )
    return u0


if __name__ == "__main__":
    main()
