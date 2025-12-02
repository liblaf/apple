import os
from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pyvista as pv
import scipy.optimize
import warp as wp
from jaxtyping import Array, Bool, Float
from loguru import logger

import liblaf.apple.jax.sim as sim_jax
import liblaf.apple.warp.sim as sim_wp
import liblaf.apple.warp.utils as wp_utils
from liblaf import cherries, grapes, melon
from liblaf.apple import sim
from liblaf.apple.jax import optim, tree
from liblaf.apple.jax.typing import Scalar, Vector
from liblaf.apple.warp.typing import vec6

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"


# ! dirty hack to make NormalCG work with DiagonalLinearOperator
@lx.is_positive_semidefinite.register(lx.DiagonalLinearOperator)
def _(operator: lx.DiagonalLinearOperator) -> bool:  # noqa: ARG001
    return True


class Config(cherries.BaseConfig):
    input: Path = cherries.input("11-input.vtu")
    target: Path = cherries.input("21-target.vtu")

    output: Path = cherries.output("33-inverse.vtu.series")


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
        factory=lambda: optim.MinimizerPNCG(atol=1e-16, maxiter=1000, rtol=1e-8)
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
class PNCGLinearSolver:
    hess_diag: Callable[[], Vector]
    hess_prod: Callable[[Vector], Vector]
    hess_quad: Callable[[Vector], Scalar]
    b: Vector
    optimizer: optim.Minimizer = tree.field(
        factory=lambda: optim.MinimizerPNCG(maxiter=10000, rtol=1e-5)
    )

    def fun(self, x: Vector) -> Scalar:
        return 0.5 * self.hess_quad(x) - jnp.vdot(x, self.b)

    def jac(self, x: Vector) -> Vector:
        return self.hess_prod(x) - self.b

    def jac_and_hess_diag(self, x: Vector) -> tuple[Vector, Vector]:
        return self.jac(x), self.hess_diag()

    def solve(self) -> Vector:
        def callback(intermediate_result: optim.Solution) -> None:
            n_iter: int = intermediate_result["n_iter"]
            if n_iter % 50 != 0:
                return
            x: Vector = intermediate_result["x"]
            logger.info(intermediate_result)
            residual = self.hess_prod(x) - self.b
            rel_residual: float = jnp.linalg.norm(residual) / jnp.linalg.norm(self.b)
            logger.info(
                "linear solve callback > relative residual #{}: {}",
                n_iter,
                rel_residual,
            )
            logger.info("linear solve callback > fun: {}", self.fun(x))

        solution = self.optimizer.minimize(
            x0=jnp.zeros_like(self.b),
            fun=self.fun,
            jac=self.jac,
            hess_diag=lambda _u: self.hess_diag(),
            hess_quad=lambda _u, p: self.hess_quad(p),
            jac_and_hess_diag=self.jac_and_hess_diag,
            callback=callback,
        )
        return solution["x"]


@tree.pytree
class InverseLossAux:
    loss_surface: Scalar
    reg_mean: Scalar
    reg_shear: Scalar
    reg_volume: Scalar


@tree.pytree
class Inverse:
    forward: Forward
    input: pv.UnstructuredGrid
    solution: Vector = tree.array()
    target: pv.UnstructuredGrid
    linear_solver: lx.AbstractLinearSolver = tree.field(
        factory=lambda: lx.NormalCG(rtol=1e-1, atol=1e-3)
    )

    reg_mean_weight: float = 1e3
    reg_shear_weight: float = 1e3
    reg_volume_weight: float = 1e3
    step: int = 0

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
    def active_volume(self) -> Float[Array, " c"]:
        if "Volume" not in self.input.cell_data:
            self.input = self.input.compute_cell_sizes()  # pyright: ignore[reportAttributeAccessIssue]
        volume: Float[Array, " c"] = jnp.asarray(self.input.cell_data["Volume"])
        return self.active_fraction * volume

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

    @property
    def muscle_orientation(self) -> Float[Array, "c 3 3"]:
        return jnp.asarray(self.input.cell_data["muscle-orientation"]).reshape(-1, 3, 3)

    def make_params(self, q: Float[Array, "ca 6"]) -> Params:
        activation: Float[Array, "c 6"] = sim_jax.rest_activation(self.input.n_cells)
        # q = jnp.exp(q)
        # q = q.at[:, :3].set(jnp.exp(q[:, :3]))
        for muscle_id in range(self.n_muscles):
            mask: Bool[Array, " c"] = self.muscle_id == muscle_id
            activation = activation.at[mask, 0].set(jnp.reciprocal(q[muscle_id]))
            activation = activation.at[mask, 1].set(jnp.sqrt(q[muscle_id]))
            activation = activation.at[mask, 2].set(jnp.sqrt(q[muscle_id]))
        activation = sim_jax.transform_activation(activation, self.muscle_orientation)
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
        aux: InverseLossAux
        (L, aux), dLdu = eqx.filter_value_and_grad(self.loss, has_aux=True)(u, params)
        cherries.log_metrics(
            {
                "loss": {
                    "total": L,
                    "surface": aux.loss_surface,
                    "reg": {
                        "mean": aux.reg_mean,
                        "shear": aux.reg_shear,
                        "volume": aux.reg_volume,
                    },
                }
            },
            step=self.step,
        )
        jac: Params
        jac, _ = eqx.filter_grad(lambda params: self.loss(u, params), has_aux=True)(
            params
        )
        preconditioner: Vector = jnp.reciprocal(self.model.hess_diag(u))
        u_free: Vector = self.model.dirichlet.get_free(u)
        dLdu_free: Vector = self.model.dirichlet.get_free(dLdu)
        P_free: Vector = self.model.dirichlet.get_free(preconditioner)

        def relative_residual(p_free: Vector) -> float:
            return jnp.linalg.norm(
                self.model.hess_prod(u_free, p_free) + dLdu_free
            ) / jnp.linalg.norm(dLdu_free)

        with grapes.timer(name="CG"):
            # solution: lx.Solution = lx.linear_solve(
            #     lx.FunctionLinearOperator(
            #         lambda p_free: self.model.hess_prod(u_free, p_free),
            #         jax.ShapeDtypeStruct(u_free.shape, u_free.dtype),
            #         [lx.symmetric_tag, lx.positive_semidefinite_tag],
            #     ),
            #     -dLdu_free,
            #     self.linear_solver,
            #     options={
            #         "preconditioner": lx.DiagonalLinearOperator(
            #             self.model.dirichlet.get_free(preconditioner)
            #         )
            #     },
            #     throw=False,
            # )
            # logger.info(lx.RESULTS[solution.result])
            # logger.info(solution.stats)
            # p_free = solution.value

            p_free, info = jax.scipy.sparse.linalg.cg(
                lambda p_free: self.model.hess_prod(u_free, p_free),
                -dLdu_free,
                tol=1e-5,
                atol=1e-15,
                maxiter=ic(u_free.size // 10),
                M=lambda x: P_free * x,
            )
        logger.info("CG > info: {}", info)
        rel_res: float = relative_residual(p_free)
        logger.info("CG > relative residual: {}", rel_res)
        if rel_res > 0.5:
            logger.warning("CG failed to converge, switching to NormalCG")
            with grapes.timer(name="NormalCG"):
                solver = lx.NormalCG(
                    rtol=1e-5, atol=1e-15, max_steps=(u_free.size // 10)
                )
                solution: lx.Solution = lx.linear_solve(
                    lx.FunctionLinearOperator(
                        lambda p_free: self.model.hess_prod(u_free, p_free),
                        jax.ShapeDtypeStruct(u_free.shape, u_free.dtype),
                        [lx.symmetric_tag, lx.positive_semidefinite_tag],
                    ),
                    -dLdu_free,
                    solver,
                    options={"preconditioner": lx.DiagonalLinearOperator(P_free)},
                    throw=False,
                )
            p_free: Vector = solution.value
            logger.info("NormalCG > results: {}", lx.RESULTS[solution.result])
            rel_res: float = relative_residual(p_free)
            logger.info("NormalCG > relative residual: {}", rel_res)

        p: Vector = self.model.to_full(p_free, zero=True)

        # solver = PNCGLinearSolver(
        #     hess_diag=lambda: self.model.hess_diag(u),
        #     hess_prod=lambda p: self.model.hess_prod(u, p),
        #     hess_quad=lambda p: self.model.hess_quad(u, p),
        #     b=-dLdu,
        # )
        # p: Vector = solver.solve()
        # relative error
        cherries.log_metric(
            "linear_solve/relative_residual_free", rel_res, step=self.step
        )
        rel_residual: float = jnp.linalg.norm(
            self.model.hess_prod(u, p) + dLdu
        ) / jnp.linalg.norm(dLdu)
        logger.info("linear solve > relative residual (all): {}", rel_residual)
        with grapes.config.pretty.overrides(short_arrays=False):
            ic(p)
        outputs: dict[str, dict[str, Array]] = self.model.mixed_derivative_prod(u, p)
        jac.activation += outputs[self.energy.id]["activation"]
        jac_q: Array
        (jac_q,) = vjp(jac)
        with grapes.config.pretty.overrides(short_arrays=False):
            ic(L, jac_q)

        self.step += 1

        return L, jac_q

    def loss(self, x: Vector, params: Params) -> tuple[Scalar, InverseLossAux]:
        loss_surface: Scalar = self.loss_surface(x)
        reg_mean: Scalar = self.reg_mean_weight * self.regularize_mean(params)
        reg_shear: Scalar = self.reg_shear_weight * self.regularize_shear(params)
        reg_volume: Scalar = self.reg_volume_weight * self.regularize_volume(params)
        # jax.debug.print("loss_surface = {}", loss_surface)
        # jax.debug.print("reg_mean = {}", reg_mean)
        # jax.debug.print("reg_shear = {}", reg_shear)
        # jax.debug.print("reg_volume = {}", reg_volume)
        result: Scalar = loss_surface + reg_mean + reg_shear + reg_volume
        return result, InverseLossAux(
            loss_surface=loss_surface,
            reg_mean=reg_mean,
            reg_shear=reg_shear,
            reg_volume=reg_volume,
        )

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
            active_volume: Float[Array, " c"] = self.active_volume[muscle_mask]
            activation_mean: Float[Array, " 6"] = jnp.mean(activation, axis=0)
            regularization += jnp.dot(
                active_volume,
                jnp.sum((activation - activation_mean[jnp.newaxis, ...]) ** 2, axis=-1),
            )
        return regularization

    def regularize_shear(self, params: Params) -> Scalar:
        regularization: Float[Array, ""] = jnp.zeros(())
        for muscle_id in range(self.n_muscles):
            muscle_mask = self.muscle_id == muscle_id
            activation: Float[Array, "ca 6"] = params.activation[muscle_mask]
            orientation: Float[Array, "ca 3 3"] = self.muscle_orientation[muscle_mask]
            activation = sim_jax.transform_activation(
                activation, orientation, inverse=True
            )
            active_volume: Float[Array, " ca"] = self.active_volume[muscle_mask]
            regularization += jnp.dot(
                active_volume, jnp.sum(activation[:, 3:] ** 2, axis=-1)
            )
        return regularization

    def regularize_volume(self, params: Params) -> Scalar:
        regularization: Float[Array, ""] = jnp.zeros(())
        for muscle_id in range(self.n_muscles):
            muscle_mask = self.target.cell_data["muscle-id"] == muscle_id
            activation: Float[Array, " c 6"] = params.activation[muscle_mask]
            active_volume: Float[Array, " c"] = self.active_volume[muscle_mask]
            orientation: Float[Array, " c 3 3"] = self.muscle_orientation[muscle_mask]
            activation = sim_jax.transform_activation(
                activation, orientation, inverse=True
            )
            regularization += jnp.dot(
                active_volume, jnp.square(jnp.prod(activation[:, :3], axis=-1) - 1.0)
            )
        return regularization


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    target: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.target)

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
    activation_gt: Float[Array, "ca 6"] = sim_jax.transform_activation(
        jnp.asarray(target.cell_data["activation"])[inverse.active_mask],
        inverse.muscle_orientation[inverse.active_mask],
        inverse=True,
    )

    writer = melon.SeriesWriter(cfg.output)

    def callback(intermediate_result: optim.Solution) -> None:
        logger.info(intermediate_result)
        q: Array = intermediate_result["x"]
        params: Params = inverse.make_params(q)
        muscle_mask: Bool[Array, " c"] = inverse.active_mask
        activation: Float[Array, "ca 6"] = sim_jax.transform_activation(
            params.activation[muscle_mask],
            inverse.muscle_orientation[muscle_mask],
            inverse=True,
        )
        with grapes.config.pretty.overrides(short_arrays=False):
            ic(activation[0])
        activation_residual: Float[Array, "ca 6"] = activation - activation_gt
        mesh.cell_data["activation"] = np.asarray(params.activation)
        mesh.cell_data["activation-residual"] = np.zeros((mesh.n_cells, 6))
        mesh.cell_data["activation-residual"][np.asarray(muscle_mask)] = np.asarray(
            activation_residual
        )
        mesh.point_data["solution"] = np.asarray(
            inverse.solution[mesh.point_data["point-ids"]]
        )
        mesh.point_data["point-residual"] = np.asarray(
            inverse.solution - inverse.target.point_data["solution"]
        )[mesh.point_data["point-ids"]]
        # result: pv.UnstructuredGrid = mesh.warp_by_vector("solution")  # pyright: ignore[reportAssignmentType]
        writer.append(mesh)

    q_init: Float[Array, " ca"] = jnp.ones((inverse.n_muscles,))
    inverse.reg_mean_weight = 0.0
    inverse.reg_shear_weight = 0.0
    inverse.reg_volume_weight = 0.0
    callback(optim.Solution({"x": q_init}))
    optimizer = optim.MinimizerScipy(
        jit=False, method="L-BFGS-B", tol=1e-15, options={}
    )
    # inverse.linear_solver = lx.CG(rtol=1e-5, atol=1e-15, max_steps=model.n_free // 10)
    solution: optim.Solution = optimizer.minimize(
        x0=q_init,
        fun_and_jac=inverse.fun_and_jac,
        callback=callback,
        bounds=scipy.optimize.Bounds(lb=0.1, ub=100.0, keep_feasible=True),
    )
    callback(solution)
    ic(solution)


if __name__ == "__main__":
    cherries.main(main)
