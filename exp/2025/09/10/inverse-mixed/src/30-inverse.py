import functools
import os
from pathlib import Path

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import liblaf.apple.warp.sim as sim_wp
import liblaf.apple.warp.utils as wp_utils
import lineax as lx
import numpy as np
import pyvista as pv
import warp as wp
from jaxtyping import Array, Bool, Float
from liblaf.apple.jax.sim.energy.elastic import utils
from liblaf.apple.jax.typing import Scalar, Vector
from liblaf.apple.warp.typing import vec6
from loguru import logger

from liblaf import cherries, grapes, melon
from liblaf.apple import sim
from liblaf.apple.jax import optim, tree


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
        # factory=lambda: optim.MinimizerScipy(
        #     method="trust-constr", tol=1e-5, options={"verbose": 3}
        # )
        factory=lambda: optim.MinimizerPNCG(rtol=1e-5, maxiter=1000)
    )

    def solve(self, x0: Vector) -> Vector:
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
        return self.model.to_full(solution["x"])


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

    @functools.cached_property
    def active_mask(self) -> Bool[Array, " c"]:
        return jnp.asarray(self.input.cell_data["active-fraction"]) > 1e-3

    @property
    def n_active_cells(self) -> int:
        return int(jnp.count_nonzero(self.active_mask))

    def make_params(self, q: Array) -> Params:
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

    surface: pv.PolyData = target.extract_surface()  # pyright: ignore[reportAssignmentType]
    target.point_data["is-surface"] = np.zeros((target.n_points,), dtype=np.bool_)
    target.point_data["is-surface"][surface.point_data["point-ids"]] = True
    ic(target, surface)

    forward = Forward(model=model)
    inverse: InversePhysics = InversePhysics(
        forward=forward,
        target=target,
        input=mesh,
        solution=jnp.zeros_like(model.points),
    )
    writer = melon.SeriesWriter(cfg.output)

    def callback(intermediate_result: optim.Solution) -> None:
        q: Array = intermediate_result["x"]
        params: Params = inverse.make_params(q)
        fat_mask = inverse.target.cell_data["muscle-ids"] == -1
        activation: Float[Array, "c 6"] = params.activation
        activation = activation.at[fat_mask].set(
            jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        )
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

    init_q: Array = einops.repeat(
        jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        "i -> c i",
        c=inverse.n_active_cells,
    )
    callback(optim.Solution({"x": init_q}))
    optimizer = optim.MinimizerScipy(jit=False, method="L-BFGS-B", tol=1e-5, options={})
    solution: optim.Solution = optimizer.minimize(
        x0=init_q, fun_and_jac=inverse.fun_and_jac, callback=callback
    )
    callback(solution)
    ic(solution)


if __name__ == "__main__":
    cherries.main(main)
