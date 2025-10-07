import os
from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pyvista as pv
import warp as wp
from jaxtyping import Array, Bool, Float
from loguru import logger

import liblaf.apple.jax.sim as sim_jax
import liblaf.apple.warp.sim as sim_wp
import liblaf.apple.warp.utils as wp_utils
from liblaf import cherries, grapes, melon
from liblaf.apple import sim
from liblaf.apple.jax import optim, tree
from liblaf.apple.jax.sim.energy.elastic import utils
from liblaf.apple.jax.typing import Scalar, Vector
from liblaf.apple.warp.typing import vec6

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"


# ! dirty hack to make NormalCG work with DiagonalLinearOperator
@lx.is_positive_semidefinite.register(lx.DiagonalLinearOperator)
def _(operator: lx.DiagonalLinearOperator) -> bool:  # noqa: ARG001
    return True


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
        # factory=lambda: optim.MinimizerScipy(
        #     method="trust-constr", tol=1e-5, options={"verbose": 3}
        # )
        factory=lambda: optim.MinimizerPNCG(rtol=1e-5, maxiter=500)
    )

    def solve(self, x0: Vector | None = None) -> Vector:
        if x0 is None:
            x0 = jnp.zeros((self.model.n_free,))
        solution: optim.Solution = self.optimizer.minimize(
            x0=x0,
            fun=sim.fun,
            jac=sim.jac,
            hessp=sim.hess_prod,
            hess_diag=sim.hess_diag,
            hess_quad=sim.hess_quad,
            fun_and_jac=sim.fun_and_jac,
            jac_and_hess_diag=sim.jac_and_hess_diag,
            args=(self.model,),
        )
        logger.info(solution)
        return self.model.to_full(solution["x"])


@tree.pytree
class Inverse:
    forward: Forward
    input: pv.UnstructuredGrid
    solution: Vector = tree.array()
    target: pv.UnstructuredGrid

    @property
    def n_active_cells(self) -> int:
        return jnp.count_nonzero(self.active_mask)  # pyright: ignore[reportReturnType]

    @property
    def n_muscles(self) -> int:
        return jnp.max(self.muscle_id) + 1  # pyright: ignore[reportReturnType]

    @property
    def active_fraction(self) -> Float[Array, " c"]:
        return jnp.asarray(self.input.cell_data["active-fraction"])

    @property
    def active_mask(self) -> Bool[Array, " c"]:
        return self.active_fraction > 1e-3

    @property
    def energy(self) -> sim_wp.Phace:
        return self.model.model_warp.energies["elastic"]  # pyright: ignore[reportReturnType]

    @property
    def model(self) -> sim.Model:
        return self.forward.model

    @property
    def muscle_id(self) -> Array:
        return jnp.asarray(self.input.cell_data["muscle-id"])

    def make_params(self, q: Float[Array, "ca 6"]) -> Params:
        activation: Float[Array, "c 6"] = sim_jax.rest_activation(self.input.n_cells)
        activation = activation.at[self.active_mask].set(q)
        return Params(activation=activation)

    def set_params(self, params: Params) -> None:
        wp.copy(
            self.energy.params.activation, wp_utils.to_warp(params.activation, vec6)
        )

    def fun_and_jac(self, q: Array) -> tuple[Scalar, Array]:
        params: Params
        vjp: Callable[[Params], Array]
        params, vjp = jax.vjp(self.make_params, q)
        self.set_params(params)
        u: Vector = self.forward.solve()
        self.solution = u
        L: Scalar
        dLdu: Vector
        L, dLdu = jax.value_and_grad(self.loss)(u, params)
        jac: Params = eqx.filter_grad(lambda params: self.loss(u, params))(params)
        preconditioner: Vector = jnp.reciprocal(self.model.hess_diag(u))
        with grapes.timer(name="linear solve"):
            linear_solver: lx.AbstractLinearSolver = lx.NormalCG(rtol=1e-2, atol=1e-2)
            solution: lx.Solution = lx.linear_solve(
                lx.FunctionLinearOperator(
                    lambda p: self.model.hess_prod(u, p),
                    jax.ShapeDtypeStruct(u.shape, u.dtype),
                    [lx.symmetric_tag],
                ),
                -dLdu,
                linear_solver,
                options={"preconditioner": lx.DiagonalLinearOperator(preconditioner)},
            )
        logger.info(lx.RESULTS[solution.result])
        logger.info(solution.stats)
        p: Vector = solution.value
        # p: Vector = preconditioner * -dLdu
        with grapes.config.pretty.overrides(short_arrays=False):
            ic(p)
        outputs: dict[str, dict[str, Array]] = self.model.mixed_derivative_prod(u, p)
        jac.activation += outputs[self.energy.id]["activation"]
        jac_q: Array
        (jac_q,) = vjp(jac)
        with grapes.config.pretty.overrides(short_arrays=False):
            ic(L, jac_q)
        return L, jac_q

    def loss(self, x: Vector, params: Params) -> Scalar:
        result: Scalar = self.loss_surface(x) + 1e3 * self.regularize_mean(params)
        return result

    def loss_surface(self, u: Vector) -> Scalar:
        face_mask: Bool[Array, " p"] = jnp.asarray(self.target.point_data["is-face"])
        target: Float[Array, "p 3"] = jnp.asarray(self.target.point_data["solution"])
        diff: Vector = u[face_mask] - target[face_mask]
        objective: Scalar = 0.5 * jnp.sum(diff**2)
        return objective

    def regularize_mean(self, params: Params) -> Scalar:
        regularization: Float[Array, ""] = jnp.zeros(())
        for muscle_id in range(self.n_muscles):
            muscle_mask = self.muscle_id == muscle_id
            activation: Float[Array, "c 6"] = params.activation[muscle_mask]
            active_volume: Float[Array, " c"] = self.active_fraction[muscle_mask]
            activation_mean: Float[Array, " 6"] = jnp.mean(activation, axis=0)
            regularization += jnp.dot(
                active_volume,
                jnp.sum((activation - activation_mean[jnp.newaxis, ...]) ** 2, axis=-1),
            )
        return regularization


@tree.pytree
class InversePhysics:
    forward: Forward
    input: pv.UnstructuredGrid
    solution: Vector = tree.array()
    target: pv.UnstructuredGrid

    @property
    def model(self) -> sim.Model:
        return self.forward.model

    @property
    def energy(self) -> sim_wp.Phace:
        return self.model.model_warp.energies["elastic"]  # pyright: ignore[reportReturnType]

    @property
    def active_volume(self) -> Float[Array, " c"]:
        if "Volume" not in self.input.cell_data:
            self.input = self.input.compute_cell_sizes()  # pyright: ignore[reportAttributeAccessIssue]
        active_fraction: Float[Array, " c"] = jnp.asarray(
            self.input.cell_data["active-fraction"]
        )
        volume: Float[Array, " c"] = jnp.asarray(self.input.cell_data["Volume"])
        return active_fraction * volume

    @property
    def active_mask(self) -> Bool[Array, " c"]:
        return jnp.asarray(self.input.cell_data["active-fraction"]) > 1e-3

    @property
    def n_active_cells(self) -> int:
        return int(jnp.count_nonzero(self.active_mask))

    def make_params(self, q: Float[Array, "ca 6"]) -> Params:
        activation: Float[Array, "c 6"] = jnp.zeros((self.input.n_cells, 6))
        activation = activation.at[self.active_mask].set(q)
        return Params(activation=activation)

    def fun(self, params: Params) -> Scalar:
        wp.copy(
            self.energy.params.activation, wp_utils.to_warp(params.activation, vec6)
        )
        self.solution = self.forward.solve(self.solution)
        return self.loss(self.solution, params)

    def fun_and_jac(self, q: Array) -> tuple[Scalar, Array]:
        params: Params = self.make_params(q)
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
                # lx.NormalCG(rtol=1e-5, atol=1e-5),
                lx.NormalCG(rtol=1e-5, atol=1e-5),
                options={"preconditioner": lx.DiagonalLinearOperator(preconditioner)},
            )
        logger.info(lx.RESULTS[solution.result])
        logger.info(solution.stats)
        p: Vector = solution.value
        with grapes.config.pretty.overrides(short_arrays=False):
            ic(p)
        outputs: dict[str, dict[str, Array]] = self.model.mixed_derivative_prod(u, p)
        jac.activation += outputs[self.energy.id]["activation"]
        ic(L)
        return L, jac.activation[self.active_mask]

    def loss(self, u: Vector, params: Params) -> Scalar:
        diff: Vector = u - self.target.point_data["solution"]
        diff = diff[self.target.point_data["is-surface"]]
        objective: Scalar = 0.5 * jnp.sum(diff**2)

        regularization: Float[Array, ""] = (
            1e3 * self.regularize_mean(params)
            + 1e3 * self.regularize_shear(params)
            + 1e3 * self.regularize_volume(params)
        )

        loss: Scalar = objective + regularization
        return loss

    def reg_fat(self, params: Params) -> Scalar:
        regularization: Float[Array, ""] = jnp.zeros(())
        fat_mask = self.target.cell_data["muscle-ids"] == -1
        activation: Float[Array, " c 6"] = params.activation[fat_mask]
        target_activation: Float[Array, " 6"] = jnp.asarray(
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        )
        regularization += jnp.dot(
            self.target.cell_data["Volume"][fat_mask],
            jnp.sum((activation - target_activation[jnp.newaxis, :]) ** 2, axis=-1),
        )
        return regularization

    def regularize_mean(self, params: Params) -> Scalar:
        regularization: Float[Array, ""] = jnp.zeros(())
        for muscle_id in range(2):
            muscle_mask = self.target.cell_data["muscle-ids"] == muscle_id
            activation: Float[Array, " c 6"] = params.activation[muscle_mask]
            active_volume: Float[Array, " c"] = self.active_volume[muscle_mask]
            activation_mean: Float[Array, " 6"] = jnp.mean(activation, axis=0)
            regularization += jnp.dot(
                active_volume,
                jnp.sum((activation - activation_mean[jnp.newaxis, ...]) ** 2, axis=-1),
            )
        return regularization

    def reg_act(self, params: Params) -> Scalar:
        # direction: Float[Array, "c 3"] = jnp.asarray(
        #     self.target.cell_data["muscle-direction"]
        # )
        regularization: Float[Array, ""] = jnp.zeros(())
        for muscle_id in range(2):
            muscle_mask = self.target.cell_data["muscle-ids"] == muscle_id
            activation: Float[Array, " c 6"] = params.activation[muscle_mask]
            active_volume: Float[Array, " c"] = self.active_volume[muscle_mask]
            Q = jnp.identity(3)[jnp.newaxis, ...]
            gamma = jnp.reciprocal(jnp.mean(activation[:, 0]))
            gamma = jax.lax.stop_gradient(gamma)
            regularization += jnp.dot(
                active_volume,
                jnp.sum(
                    jnp.square(
                        Q.mT
                        @ jnp.diagflat(
                            jnp.asarray(
                                [
                                    jnp.reciprocal(gamma),
                                    jnp.sqrt(gamma),
                                    jnp.sqrt(gamma),
                                ]
                            )
                        )[jnp.newaxis, ...]
                        @ Q
                        - utils.make_activation(activation)
                    ),
                    axis=(-2, -1),
                ),
            )
        return regularization

    def regularize_shear(self, params: Params) -> Scalar:
        regularization: Float[Array, ""] = jnp.zeros(())
        for muscle_id in range(2):
            muscle_mask = self.target.cell_data["muscle-ids"] == muscle_id
            activation: Float[Array, " c 6"] = params.activation[muscle_mask]
            active_volume: Float[Array, " c"] = self.active_volume[muscle_mask]
            regularization += jnp.dot(
                active_volume, jnp.sum(activation[:, 3:] ** 2, axis=-1)
            )
        return regularization

    def regularize_volume(self, params: Params) -> Scalar:
        regularization: Float[Array, ""] = jnp.zeros(())
        for muscle_id in range(2):
            muscle_mask = self.target.cell_data["muscle-ids"] == muscle_id
            activation: Float[Array, " c 6"] = params.activation[muscle_mask]
            active_volume: Float[Array, " c"] = self.active_volume[muscle_mask]
            regularization += jnp.dot(
                active_volume,
                jnp.square(
                    activation[:, 0] * activation[:, 1] * activation[:, 2] - 1.0
                ),
            )
        return regularization


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    target: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.target)
    activation_gt: Float[Array, "c 6"] = jnp.asarray(target.cell_data["activation"])

    builder = sim.ModelBuilder()
    mesh = builder.assign_dofs(mesh)
    builder.add_dirichlet(mesh)
    builder.add_energy(
        sim_wp.Phace.from_pyvista(mesh, id="elastic", requires_grad=("activation",))
    )
    model: sim.Model = builder.finish()

    forward = Forward(model=model)
    inverse: Inverse = Inverse(
        forward=forward,
        target=target,
        input=mesh,
        solution=jnp.zeros_like(model.points),
    )
    writer = melon.SeriesWriter(cfg.output)

    def callback(intermediate_result: optim.Solution) -> None:
        logger.info(intermediate_result)
        q: Array = intermediate_result["x"]
        params: Params = inverse.make_params(q)
        activation: Float[Array, "c 6"] = params.activation
        with grapes.config.pretty.overrides(short_arrays=False):
            ic(activation[0])
        activation_residual: Float[Array, "c 6"] = activation - activation_gt
        mesh.cell_data["activation"] = np.asarray(activation)
        mesh.cell_data["activation-residual"] = np.asarray(activation_residual)
        mesh.point_data["solution"] = np.asarray(
            inverse.solution[mesh.point_data["point-ids"]]
        )
        mesh.point_data["point-residual"] = np.asarray(
            inverse.solution - inverse.target.point_data["solution"]
        )[mesh.point_data["point-ids"]]
        # result: pv.UnstructuredGrid = mesh.warp_by_vector("solution")  # pyright: ignore[reportAssignmentType]
        writer.append(mesh)

    q_init: Float[Array, "ca 6"] = sim_jax.rest_activation(inverse.n_active_cells)
    callback(optim.Solution({"x": q_init}))
    optimizer = optim.MinimizerScipy(jit=False, method="L-BFGS-B", tol=1e-5, options={})
    solution: optim.Solution = optimizer.minimize(
        x0=q_init, fun_and_jac=inverse.fun_and_jac, callback=callback
    )
    callback(solution)
    ic(solution)


if __name__ == "__main__":
    cherries.run(main)
