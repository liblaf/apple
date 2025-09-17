import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pyvista as pv
import warp as wp
from jaxtyping import Array, Float
from loguru import logger

import liblaf.apple.warp.sim as sim_wp
import liblaf.apple.warp.utils as wp_utils
from liblaf import cherries, grapes, melon
from liblaf.apple import sim
from liblaf.apple.jax import optim, tree
from liblaf.apple.jax.typing import Scalar, Vector
from liblaf.apple.warp.typing import vec6


@lx.is_positive_semidefinite.register(lx.DiagonalLinearOperator)
def _(operator: lx.DiagonalLinearOperator) -> bool:  # noqa: ARG001
    return True


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"


class Config(cherries.BaseConfig):
    input: Path = cherries.input("10-input.vtu")
    target: Path = cherries.input("20-target.vtu")

    output: Path = cherries.output("30-inverse.vtu.series")


@tree.pytree
class Params:
    activation: Float[Array, "c 6"] = tree.array()


@tree.pytree
class Forward:
    model: sim.Model
    optimizer: optim.Minimizer = tree.field(
        factory=lambda: optim.MinimizerScipy(method="trust-constr", tol=1e-5)
    )

    def solve(self, x0: Vector) -> Vector:
        x0 = jnp.zeros((self.model.n_free,))
        solution: optim.Solution = self.optimizer.minimize(
            x0=x0,
            fun=sim.fun,
            jac=sim.jac,
            hessp=sim.hess_prod,
            hess_diag=sim.hess_diag,
            fun_and_jac=sim.fun_and_jac,
            jac_and_hess_diag=sim.jac_and_hess_diag,
            args=(self.model,),
        )
        return self.model.to_full(solution["x"])


@tree.pytree
class InversePhysics:
    forward: Forward
    solution: Vector = tree.array()
    target: pv.UnstructuredGrid = tree.field()

    @property
    def model(self) -> sim.Model:
        return self.forward.model

    @property
    def energy(self) -> sim_wp.ArapActive:
        return self.model.model_warp.energies["elastic"]  # pyright: ignore[reportReturnType]

    def fun(self, params: Params) -> Scalar:
        wp.copy(
            self.energy.params.activation, wp_utils.to_warp(params.activation, vec6)
        )
        self.solution = self.forward.solve(self.solution)
        return self.loss(self.solution, params)

    def fun_and_jac(self, params: Params) -> tuple[Scalar, Params]:
        wp.copy(
            self.energy.params.activation, wp_utils.to_warp(params.activation, vec6)
        )
        u: Vector = self.forward.solve(self.solution)
        self.solution = u
        L: Scalar
        dLdu: Vector
        L, dLdu = jax.value_and_grad(self.loss)(u, params)
        jac: Params = eqx.filter_grad(lambda params: self.loss(u, params))(params)
        preconditioner: Vector = jnp.reciprocal(self.model.hess_diag(u))
        with grapes.timer(name="linear solve"):
            solution: lx.Solution = lx.linear_solve(
                lx.FunctionLinearOperator(
                    lambda p: self.model.hess_prod(u, p),
                    jax.ShapeDtypeStruct(u.shape, u.dtype),
                    [lx.symmetric_tag, lx.positive_semidefinite_tag],
                ),
                -dLdu,
                # lx.GMRES(rtol=1e-5, atol=1e-5, stagnation_iters=50, restart=50),
                lx.NormalCG(rtol=1e-3, atol=1e-3),
                options={"preconditioner": lx.DiagonalLinearOperator(preconditioner)},
                throw=False,
            )
        logger.info(lx.RESULTS[solution.result])
        logger.info(solution.stats)
        p: Vector = solution.value
        outputs: dict[str, dict[str, Array]] = self.model.mixed_derivative_prod(u, p)
        jac.activation += outputs[self.energy.id]["activation"]
        with grapes.config.pretty.overrides(short_arrays=False):
            ic(L, jac)
        return L, jac

    def loss(self, u: Vector, params: Params) -> Scalar:
        diff: Vector = u - self.target.point_data["solution"]
        diff = diff[self.target.point_data["is-surface"]]
        objective: Scalar = 0.5 * jnp.sum(diff**2)
        reg_weight = 1.0
        activation_mean: Float[Array, "c 6"] = jnp.mean(params.activation, axis=0)
        regularization: Float[Array, "c 6"] = reg_weight * jnp.sum(
            (params.activation - activation_mean) ** 2
        )
        loss: Scalar = objective + regularization
        return loss


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    target: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.target)
    activation_gt: Float[Array, "c 6"] = jnp.asarray(target.cell_data["activation"])

    builder = sim.ModelBuilder()
    mesh = builder.assign_dofs(mesh)
    builder.add_dirichlet(mesh)
    builder.add_energy(
        sim_wp.ArapActive.from_pyvista(
            mesh, id="elastic", requires_grad=("activation",)
        )
    )
    model: sim.Model = builder.finish()

    surface: pv.PolyData = target.extract_surface()  # pyright: ignore[reportAssignmentType]
    target.point_data["is-surface"] = np.zeros((target.n_points,), dtype=np.bool_)
    target.point_data["is-surface"][surface.point_data["point-id"]] = True
    ic(target, surface)

    forward = Forward(model=model)
    inverse: InversePhysics = InversePhysics(
        forward=forward,
        target=target,
        solution=jnp.zeros_like(model.points),
    )
    writer = melon.SeriesWriter(cfg.output)

    def callback(intermediate_result: optim.Solution) -> None:
        params: Params = intermediate_result["x"]
        activation: Float[Array, "c 6"] = params.activation
        with grapes.config.pretty.overrides(short_arrays=False):
            ic(activation[0])
        activation_residual: Float[Array, "c 6"] = activation - activation_gt
        mesh.cell_data["activation"] = np.asarray(activation)
        mesh.cell_data["activation-residual"] = np.asarray(activation_residual)
        mesh.point_data["solution"] = np.asarray(
            inverse.solution[mesh.point_data["point-id"]]
        )
        mesh.point_data["point-residual"] = np.asarray(
            inverse.solution - inverse.target.point_data["solution"]
        )[mesh.point_data["point-id"]]
        result: pv.UnstructuredGrid = mesh.warp_by_vector("solution")  # pyright: ignore[reportAssignmentType]
        writer.append(result)

    init_params: Params = Params(activation=jnp.zeros((mesh.n_cells, 6)))
    callback(optim.Solution({"x": init_params}))
    optimizer = optim.MinimizerScipy(jit=False, method="L-BFGS-B", tol=1e-4, options={})
    solution: optim.Solution = optimizer.minimize(
        x0=init_params, fun_and_jac=inverse.fun_and_jac, callback=callback
    )
    callback(solution)
    ic(solution)


if __name__ == "__main__":
    cherries.run(main)
