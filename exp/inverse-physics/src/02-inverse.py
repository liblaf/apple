import attrs
import einops
import felupe
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import scipy.optimize
from jaxtyping import Bool, Float, PyTree

import liblaf.apple as apple  # noqa: PLR0402
import liblaf.grapes as grapes  # noqa: PLR0402


def load_forward_physics() -> apple.Fixed:
    mesh: pv.UnstructuredGrid = pv.read("data/input.vtu")  # pyright: ignore[reportAssignmentType]
    mesh_felupe = felupe.Mesh(mesh.points, mesh.cells_dict[pv.CellType.TETRA], "tetra")
    fixed_mask: Bool[np.ndarray, " D"] = einops.repeat(
        mesh.point_data["fixed_mask"], "P -> (P 3)"
    )
    fixed_values: Float[jax.Array, " D"] = jnp.asarray(
        mesh.point_data["fixed_disp"].flatten()
    )
    problem: apple.AbstractPhysicsProblem = apple.Fixed(
        problem=apple.Corotated(
            mesh=mesh_felupe,
            params={"lambda": jnp.asarray(3e3), "mu": jnp.asarray(1e3)},
        ),
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )
    return problem


@apple.register_dataclass()
@attrs.define(kw_only=True)
class InverseProblem(apple.InversePhysicsProblem):
    forward_problem: apple.Fixed  # pyright: ignore[reportIncompatibleVariableOverride]
    target: pv.UnstructuredGrid = attrs.field(metadata={"static": True})

    @apple.jit()
    def objective(self, u: PyTree, q: PyTree) -> Float[jax.Array, ""]:
        u: Float[jax.Array, "P 3"] = self.forward_problem.fill(u)
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
        regular: Float[jax.Array, ""] = jnp.sum(q["lambda"] ** 2)
        return loss


def main() -> None:
    grapes.init_logging()
    undeformed: pv.UnstructuredGrid = pv.read("data/input.vtu")  # pyright: ignore[reportAssignmentType]
    target: pv.UnstructuredGrid = pv.read("data/target.vtu")  # pyright: ignore[reportAssignmentType]
    forward: apple.Fixed = load_forward_physics()
    forward.prepare()
    inverse = InverseProblem(forward_problem=forward, target=target)
    q0: PyTree = {"lambda": jnp.asarray(0.0)}

    def callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
        ic(intermediate_result)

    result: scipy.optimize.OptimizeResult = apple.minimize(
        x0=inverse.ravel_q(q0),
        fun=inverse.fun_flat,
        jac=inverse.jac_flat,
        algo=apple.opt.MinimizeScipy(method="L-BFGS-B"),
        callback=callback,
    )
    ic(result)
    q: PyTree = inverse.unravel_q(result["x"])
    ic(q)
    u: PyTree = inverse.forward(q)
    undeformed.point_data["solution"] = forward.fill(u)
    undeformed.warp_by_vector("solution", inplace=True)
    undeformed.save("data/inverse.vtu")


if __name__ == "__main__":
    main()
