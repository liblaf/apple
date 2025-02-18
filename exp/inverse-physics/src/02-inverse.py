from collections.abc import Mapping

import attrs
import einops
import felupe
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import scipy.optimize
from jaxtyping import Bool, Float

import liblaf.apple as apple  # noqa: PLR0402
import liblaf.grapes as grapes  # noqa: PLR0402


def load_forward_physics() -> apple.Fixed:
    mesh: pv.UnstructuredGrid = pv.read("data/input.vtu")
    mesh_felupe = felupe.Mesh(mesh.points, mesh.cells_dict[pv.CellType.TETRA], "tetra")
    fixed_mask: Bool[np.ndarray, " D"] = einops.repeat(
        mesh.point_data["fixed_mask"], "P -> (P 3)"
    )
    fixed_values: Float[jax.Array, " D"] = jnp.asarray(
        mesh.point_data["fixed_disp"].flatten()
    )
    problem: apple.PhysicsProblem = apple.Fixed(
        problem=apple.Corotated(
            mesh=mesh_felupe, p={"lambda": jnp.asarray(3e3), "mu": jnp.asarray(1e3)}
        ),
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )
    return problem


@attrs.define
class InverseProblem(apple.InversePhysicsProblem):
    forward_problem: apple.Fixed  # pyright: ignore[reportIncompatibleVariableOverride]
    target: pv.UnstructuredGrid

    def _fun(
        self, u: Float[jax.Array, " DoF"], p: Mapping[str, Float[jax.Array, "..."]]
    ) -> Float[jax.Array, ""]:
        u: Float[jax.Array, "P 3"] = self.forward_problem.fill(u).reshape(
            self.target.n_points, 3
        )
        surface_point_mask: Bool[np.ndarray, " P"] = self.target.point_data[
            "is_surface"
        ]
        loss: Float[jax.Array, ""] = jnp.sum(
            (
                u[surface_point_mask]
                - self.target.point_data["solution"][surface_point_mask]
            )
            ** 2
        )
        regular: Float[jax.Array, ""] = jnp.sum(p["lambda"] ** 2)
        return loss


def main() -> None:
    grapes.init_logging()
    target: pv.UnstructuredGrid = pv.read("data/solution.vtu")
    forward: apple.Fixed = load_forward_physics()
    inverse = InverseProblem(forward_problem=forward, target=target)

    def callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
        ic(intermediate_result)

    ic(inverse.fun({"lambda": jnp.asarray(1.0)}))
    ic(inverse.jac({"lambda": jnp.asarray(1.0)}))
    result: scipy.optimize.OptimizeResult = apple.minimize(
        x0=jnp.asarray(1.0),
        fun=lambda q: inverse.fun({"lambda": q}),
        jac=lambda q: inverse.jac({"lambda": q})["lambda"],
        algo=apple.opt.MinimizeScipy(method="L-BFGS-B"),
        callback=callback,
    )
    ic(result)


if __name__ == "__main__":
    main()
