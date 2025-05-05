from pathlib import Path

import einops
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Bool, Float

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import cherries, melon

DIRECTION: Float[np.ndarray, " 3"] = np.asarray([1.0, 0.0, 0.0])
PIVOT: Float[np.ndarray, " 3"] = np.asarray([0.0, 29.141, 0.8457])


class Config(cherries.BaseConfig):
    input: Path = Path("data/input.vtu")
    output: Path = Path("data/")


def load_problem(cfg: Config) -> apple.PhysicsProblem:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    dirichlet_mask: Bool[jax.Array, " N"] = einops.repeat(
        jnp.asarray(mesh.point_data["is-skull"]), "V -> (V 3)"
    )
    dirichlet_values: Float[jax.Array, " N"] = jnp.zeros((mesh.n_points * 3,))
    face = apple.obj.tetra.ObjectTetra(
        name="face",
        params={
            "lambda": jnp.full((mesh.n_cells,), 532879.8185941039),
            "mu": jnp.full((mesh.n_cells,), 34013.60544217687),
        },
        material=apple.material.tetra.Corotated(),
        mesh=mesh,
    )
    problem = apple.PhysicsProblem(
        objects=[face], dirichlet_mask=dirichlet_mask, dirichlet_values=dirichlet_values
    )
    return problem


def update_dirichlet(
    problem: apple.PhysicsProblem, face: apple.obj.ObjectTetra, angle: float = 0.0
) -> apple.PhysicsProblem:
    mandible_mask: Bool[np.ndarray, " V"] = face.mesh.point_data["is-mandible"]
    rest_pos: Float[np.ndarray, "V 3"] = face.mesh.points[mandible_mask]
    transform_matrix: Float[np.ndarray, "4 4"] = tm.transformations.rotation_matrix(
        angle, DIRECTION, PIVOT
    )  # pyright: ignore[reportAssignmentType]
    transformed: Float[np.ndarray, "V 3"] = tm.transform_points(
        rest_pos, transform_matrix
    )
    displacement: Float[np.ndarray, "V 3"] = transformed - rest_pos
    dirichlet_values: Float[np.ndarray, "V 3"] = np.zeros((face.n_points, 3))
    dirichlet_values[mandible_mask] = displacement
    problem.dirichlet_values = jnp.asarray(dirichlet_values).ravel()
    return problem


def forward(
    problem: apple.PhysicsProblem,
    face: apple.obj.ObjectTetra,
    u0: Float[jax.Array, " DoF"] | None = None,
) -> tuple[apple.MinimizeResult, pv.UnstructuredGrid]:
    problem.prepare()
    result: apple.MinimizeResult = problem.solve(
        u0=u0, algo=apple.MinimizePNCG(eps=1e-10, iter_max=150)
    )
    solution: Float[jax.Array, " DoF"] = result["x"]
    solution: Float[jax.Array, " F"] = problem.fill(solution)
    solution: Float[jax.Array, " N"] = face.select_dof(solution)
    solution: Float[jax.Array, "V 3"] = face.unravel_u(solution)
    output: pv.UnstructuredGrid = face.mesh.copy()
    output.point_data["solution"] = np.asarray(solution)
    jac: Float[jax.Array, " DoF"] = result["jac"]  # pyright: ignore[reportAssignmentType]
    jac: Float[jax.Array, " F"] = problem.fill_zeros(jac)
    jac: Float[jax.Array, " N"] = face.select_dof(jac)
    jac: Float[jax.Array, "V 3"] = face.unravel_u(jac)
    output.point_data["jac"] = np.asarray(jac)
    output.warp_by_vector("solution", inplace=True)
    return result, output


def main(cfg: Config) -> None:
    problem: apple.PhysicsProblem = load_problem(cfg)
    face: apple.obj.ObjectTetra = problem.objects[0]  # pyright: ignore[reportAssignmentType]
    writer_face = melon.SeriesWriter(cfg.output / "face.vtp.series")
    writer_mandible = melon.SeriesWriter(cfg.output / "mandible.vtp.series")
    writer_skull = melon.SeriesWriter(cfg.output / "skull.vtp.series")
    writer_solution = melon.SeriesWriter(cfg.output / "solution.vtu.series")
    u0: Float[jax.Array, " DoF"] | None = None
    for angle in np.linspace(np.deg2rad(0.0), np.deg2rad(30.0), 31):
        ic(np.rad2deg(angle))
        problem = update_dirichlet(problem, face, angle)
        result: apple.MinimizeResult
        output: pv.UnstructuredGrid
        result, output = forward(problem, face, u0=u0)
        u0 = result["x"]
        writer_solution.append(output)
        surface: pv.PolyData = output.extract_surface()  # pyright: ignore[reportAssignmentType]
        writer_face.append(
            melon.triangle.extract_points(surface, surface.point_data["is-face"])
        )
        writer_mandible.append(
            melon.triangle.extract_points(surface, surface.point_data["is-mandible"])
        )
        writer_skull.append(
            melon.triangle.extract_points(surface, surface.point_data["is-skull"])
        )


if __name__ == "__main__":
    cherries.run(main)
