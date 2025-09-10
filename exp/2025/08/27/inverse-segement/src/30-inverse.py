import functools
from pathlib import Path

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pyvista as pv
from jaxtyping import Array, Bool, Float
from loguru import logger

from liblaf import cherries, grapes, melon
from liblaf.apple import sim
from liblaf.apple.jax import optim, tree
from liblaf.apple.jax import sim as sim_jax
from liblaf.apple.jax.typing import Scalar, Vector


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
        factory=lambda: optim.MinimizerScipy(
            timer=False, method="trust-constr", tol=1e-5, options={"verbose": 2}
        )
    )

    def solve(self, x0: Vector) -> Vector:
        x0 = jnp.zeros((self.model.n_free,))
        solution: optim.Solution = self.optimizer.minimize(
            x0=x0,
            fun=sim.Model.static_fun,
            jac=sim.Model.static_jac,
            hessp=sim.Model.static_hess_prod,
            fun_and_jac=sim.Model.static_fun_and_jac,
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
    def energy(self) -> sim_jax.PhaceActive:
        return self.model.model_jax.energies["elastic"]  # pyright: ignore[reportReturnType]

    @functools.cached_property
    def muscle_0_mask(self) -> Bool[Array, " c"]:
        return jnp.asarray(self.target.cell_data["muscle-0-mask"], dtype=jnp.bool_)

    @functools.cached_property
    def muscle_1_mask(self) -> Bool[Array, " c"]:
        return jnp.asarray(self.target.cell_data["muscle-1-mask"], dtype=jnp.bool_)

    def fun(self, params: Params) -> Scalar:
        self.energy.activation = params.activation
        self.solution = self.forward.solve(self.solution)
        return self.loss(self.solution, params)

    def fun_and_jac(self, params: Params) -> tuple[Scalar, Params]:
        self.energy.activation = params.activation
        u: Vector = self.forward.solve(self.solution)
        self.solution = u
        L: Scalar
        dLdu: Vector
        L, dLdu = jax.value_and_grad(self.loss)(u, params)
        jac: Params = eqx.filter_grad(lambda params: self.loss(u, params))(params)
        solution: lx.Solution = lx.linear_solve(
            lx.FunctionLinearOperator(
                eqx.filter_jit(lambda p: self.model.hess_prod(u, p)),
                jax.ShapeDtypeStruct(u.shape, u.dtype),
                [lx.symmetric_tag, lx.positive_semidefinite_tag],
            ),
            -dLdu,
            lx.NormalCG(rtol=1e-5, atol=1e-5),
        )
        logger.info(lx.RESULTS[solution.result])
        p: Vector = solution.value
        outputs: dict[str, dict[str, Array]] = self.model.mixed_derivative_prod(u, p)
        jac.activation += outputs[self.energy.id]["activation"]
        return L, jac

    def loss(self, u: Vector, params: Params) -> Scalar:
        diff: Vector = u - self.target.point_data["solution"]
        diff = diff[self.target.point_data["surface-mask"]]
        objective: Scalar = 0.5 * jnp.sum(diff**2)
        regularization: Scalar = self.regularization(params)
        loss: Scalar = objective + regularization
        return loss

    def regularization(self, params: Params) -> Scalar:
        reg_weight = 1e-3
        regularization: Scalar = reg_weight * (
            self.regularization_muscle(params.activation[self.muscle_0_mask])
            + self.regularization_muscle(params.activation[self.muscle_1_mask])
        )
        return regularization

    def regularization_muscle(self, a: Float[Array, "c 6"]) -> Scalar:
        return jnp.sum((a - jnp.mean(a, axis=0)) ** 2)


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    target: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.target)
    activation_gt: Float[Array, "c 6"] = jnp.asarray(target.cell_data["activation"])

    builder = sim.ModelBuilder()
    mesh = builder.assign_dofs(mesh)
    builder.add_dirichlet(mesh)
    builder.add_energy(
        sim_jax.PhaceActive.from_pyvista(
            mesh, id="elastic", requires_grad=("activation",)
        )
    )
    model: sim.Model = builder.finish()

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

    init_params: Params = Params(
        activation=einops.repeat(
            jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), "i -> c i", c=mesh.n_cells
        )
    )
    callback(optim.Solution({"x": init_params}))
    optimizer = optim.MinimizerScipy(jit=False, method="L-BFGS-B", tol=1e-4, options={})
    solution: optim.Solution = optimizer.minimize(
        x0=init_params, fun_and_jac=inverse.fun_and_jac, callback=callback
    )
    callback(solution)
    ic(solution)


if __name__ == "__main__":
    cherries.run(main)
