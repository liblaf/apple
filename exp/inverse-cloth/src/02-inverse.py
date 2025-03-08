import attrs
import einops
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import scipy.optimize
from jaxtyping import Bool, Float, PyTree

import liblaf.apple as apple  # noqa: PLR0402
import liblaf.grapes as grapes  # noqa: PLR0402


def load_forward_physics() -> apple.Fixed:
    mesh: pv.PolyData = pv.read("data/input.vtp")  # pyright: ignore[reportAssignmentType]
    fixed_mask: Bool[np.ndarray, " D"] = einops.repeat(
        mesh.point_data["fixed_mask"], "P -> (P 3)"
    )
    fixed_values: Float[jax.Array, " D"] = jnp.asarray(
        mesh.point_data["fixed_disp"].flatten()
    )
    problem: apple.Fixed = apple.Fixed(
        problem=apple.Sum(
            problems=[
                apple.Koiter(
                    mesh=mesh, lmbda=mesh.cell_data["lmbda"], mu=mesh.cell_data["mu"]
                ),
                apple.Gravity(
                    mass=apple.elem.triangle.mass(mesh), n_points=mesh.n_points
                ),
            ]
        ),
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )
    return problem


def regularization(q: Float[jax.Array, " N"]) -> Float[jax.Array, ""]:
    return jnp.var(q)


@apple.register_dataclass()
@attrs.define(kw_only=True, on_setattr=attrs.setters.convert)
class InverseProblem(apple.InversePhysicsProblem):
    forward_problem: apple.Fixed  # pyright: ignore[reportIncompatibleVariableOverride]
    target: pv.PolyData = attrs.field(metadata={"static": True})

    @apple.jit()
    def objective(self, u: PyTree, q: PyTree) -> Float[jax.Array, ""]:
        u: Float[jax.Array, "P 3"] = self.forward_problem.fill(u)
        loss: Float[jax.Array, ""] = jnp.sum(
            (u - self.target.point_data["solution"]) ** 2
        )
        return loss


def main() -> None:
    grapes.init_logging()
    undeformed: pv.PolyData = pv.read("data/input.vtp")  # pyright: ignore[reportAssignmentType]
    target: pv.PolyData = pv.read("data/target.vtp")  # pyright: ignore[reportAssignmentType]
    forward: apple.Fixed = load_forward_physics()
    inverse = InverseProblem(forward_problem=forward, target=target)
    q0: PyTree = {
        "koiter": {
            "lmbda": jnp.full((undeformed.n_cells,), 3e3),
            "mu": jnp.full((undeformed.n_cells,), 1e3),
        }
    }
    # q0: PyTree = {
    #     "koiter": {
    #         "lmbda": jnp.asarray(3e3),
    #         "mu": jnp.asarray(1e3),
    #     }
    # }
    forward.prepare(q0)
    u: Float[jax.Array, " DoF"] = inverse.forward(q0)
    undeformed.point_data["solution"] = forward.fill(u)
    initial: pv.PolyData = undeformed.warp_by_vector("solution")  # pyright: ignore[reportAssignmentType]
    initial.save("data/initial.vtp")
    q0_flat: Float[jax.Array, " Q"] = inverse.ravel_q(q0)
    x0_flat: Float[jax.Array, " Q"] = jnp.log(q0_flat)

    def composite(x_flat: Float[jax.Array, " Q"]) -> Float[jax.Array, " Q"]:
        return jnp.exp(x_flat)

    def fun(x_flat: Float[jax.Array, " Q"]) -> Float[jax.Array, ""]:
        x_flat = jnp.asarray(x_flat)
        q_flat: Float[jax.Array, " Q"] = composite(x_flat)
        return inverse.fun_flat(q_flat)

    def jac(x_flat: Float[jax.Array, " Q"]) -> Float[jax.Array, " Q"]:
        x_flat = jnp.asarray(x_flat)
        q_flat: Float[jax.Array, " Q"] = composite(x_flat)
        dJ_dq: Float[jax.Array, " Q"] = inverse.jac_flat(q_flat)
        dJ_dx: Float[jax.Array, " Q"] = apple.math.vjp_fun(composite, x_flat)(dJ_dq)
        return dJ_dx

    def callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
        ic(intermediate_result)

    result: scipy.optimize.OptimizeResult = apple.minimize(
        x0=x0_flat,
        fun=fun,
        jac=jac,
        algo=apple.opt.MinimizeScipy(method="L-BFGS-B", options={"disp": True}),
        callback=callback,
    )
    ic(result)
    x_flat: Float[jax.Array, " Q"] = result["x"]
    q_flat: Float[jax.Array, " Q"] = composite(x_flat)
    q: PyTree = inverse.unravel_q(q_flat)
    ic(q)
    u: PyTree = inverse.forward(q)
    undeformed.point_data["solution"] = forward.fill(u)
    undeformed.cell_data["lmbda"] = q["koiter"]["lmbda"]
    undeformed.cell_data["mu"] = q["koiter"]["mu"]
    undeformed.cell_data["E"], undeformed.cell_data["nu"] = (
        apple.constitution.lame_to_E_nu(q["koiter"]["lmbda"], q["koiter"]["mu"])
    )
    undeformed.warp_by_vector("solution", inplace=True)
    undeformed.save("data/inverse.vtp")


if __name__ == "__main__":
    main()
