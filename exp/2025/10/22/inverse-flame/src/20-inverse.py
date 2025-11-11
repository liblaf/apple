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
from liblaf.peach import optim, tree
from loguru import logger

import liblaf.apple.jax.sim as sim_jax
import liblaf.apple.warp.sim as sim_wp
import liblaf.apple.warp.utils as wp_utils
from liblaf import cherries, grapes, melon
from liblaf.apple import sim
from liblaf.apple.jax.typing import Scalar, Vector
from liblaf.apple.warp.typing import vec6

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"


# ! dirty hack to make NormalCG work with DiagonalLinearOperator
@lx.is_positive_semidefinite.register(lx.DiagonalLinearOperator)
def _(operator: lx.DiagonalLinearOperator) -> bool:  # noqa: ARG001
    return True


class Config(cherries.BaseConfig):
    input: Path = cherries.input("10-input.vtu")
    target: Path = cherries.input("10-target.vtu")

    output: Path = cherries.temporary("20-inverse.vtu.series")


@tree.define
class Params:
    activation: Float[Array, "c 6"] = tree.array()


@tree.define
class Forward:
    model: sim.Model
    optimizer: optim.Optimizer = tree.field(
        # factory=lambda: optim.MinimizerScipy(
        #     method="trust-constr", tol=1e-5, options={"verbose": 3}
        # )
        factory=lambda: optim.PNCG(atol=1e-16, max_steps=1000, rtol=1e-8)
    )

    def solve(self, x0: Vector | None = None) -> Vector:
        if x0 is None:
            x0 = jnp.zeros((self.model.n_free,))
        solution: optim.OptimizeSolution = self.optimizer.minimize(
            objective=optim.Objective(
                fun=sim.fun,
                grad=sim.jac,
                hess_prod=sim.hess_prod,
                hess_diag=sim.hess_diag,
                hess_quad=sim.hess_quad,
                value_and_grad=sim.fun_and_jac,
                grad_and_hess_diag=sim.jac_and_hess_diag,
            ).partial(self.model),
            params=x0,
        )
        logger.info(solution)
        return self.model.to_full(solution.params)


@tree.define
class PNCGLinearSolver:
    hess_diag: Callable[[], Vector]
    hess_prod: Callable[[Vector], Vector]
    hess_quad: Callable[[Vector], Scalar]
    b: Vector
    optimizer: optim.Optimizer = tree.field(
        factory=lambda: optim.PNCG(max_steps=10000, rtol=1e-5)
    )

    def fun(self, x: Vector) -> Scalar:
        return 0.5 * self.hess_quad(x) - jnp.vdot(x, self.b)

    def jac(self, x: Vector) -> Vector:
        return self.hess_prod(x) - self.b

    def jac_and_hess_diag(self, x: Vector) -> tuple[Vector, Vector]:
        return self.jac(x), self.hess_diag()

    def solve(self) -> Vector:
        def callback(state: optim.PNCGState, stats: optim.PNCGStats) -> None:
            n_iter: int = stats.n_steps
            if n_iter % 50 != 0:
                return
            x: Vector = state.params
            logger.info(state)
            residual = self.hess_prod(x) - self.b
            rel_residual: float = jnp.linalg.norm(residual) / jnp.linalg.norm(self.b)
            logger.info(
                "linear solve callback > relative residual #{}: {}",
                n_iter,
                rel_residual,
            )
            logger.info("linear solve callback > fun: {}", self.fun(x))

        solution = self.optimizer.minimize(
            objective=optim.Objective(
                fun=self.fun,
                grad=self.jac,
                hess_diag=lambda _u: self.hess_diag(),
                hess_quad=lambda _u, p: self.hess_quad(p),
                grad_and_hess_diag=self.jac_and_hess_diag,
            ),
            params=jnp.zeros_like(self.b),
            callback=callback,
        )
        return solution.params


@tree.define
class InverseLossAux:
    loss_surface: Scalar
    reg_act: Scalar
    reg_mean: Scalar
    reg_shear: Scalar
    reg_sparse: Scalar
    reg_volume: Scalar


@tree.define
class Inverse:
    forward: Forward
    # input: pv.UnstructuredGrid
    solution: Vector = tree.array()
    # target: pv.UnstructuredGrid
    linear_solver: lx.AbstractLinearSolver = tree.field(
        factory=lambda: lx.NormalCG(rtol=1e-1, atol=1e-3)
    )

    active_fraction: Float[Array, " c"] = tree.field(kw_only=True)
    active_volume: Float[Array, " c"] = tree.field(kw_only=True)
    muscle_id: Array = tree.field(kw_only=True, metadata={"static": True})
    muscle_orientation: Float[Array, "c 3 3"] = tree.field(kw_only=True)
    n_cells: int = tree.field(kw_only=True)
    is_face: Bool[Array, " p"] = tree.field(kw_only=True)
    target_solution: Float[Array, "p 3"] = tree.field(kw_only=True)
    n_muscles: int = tree.field(kw_only=True)

    reg_act: float = 0.0
    reg_mean_weight: float = 0.0
    reg_shear_weight: float = 0.0
    reg_sparse: float = 1e-3
    reg_volume_weight: float = 0.0
    step: int = 0

    @property
    def n_active_cells(self) -> int:
        return jnp.count_nonzero(self.active_mask)  # pyright: ignore[reportReturnType]

    # @property
    # def n_muscles(self) -> int:
    #     return jnp.max(self.muscle_id) + 1  # pyright: ignore[reportReturnType]

    # @property
    # def active_fraction(self) -> Float[Array, " c"]:
    #     return jnp.asarray(self.input.cell_data["active-fraction"])

    # @property
    # def active_volume(self) -> Float[Array, " c"]:
    #     if "Volume" not in self.input.cell_data:
    #         self.input = self.input.compute_cell_sizes()  # pyright: ignore[reportAttributeAccessIssue]
    #     volume: Float[Array, " c"] = jnp.asarray(self.input.cell_data["Volume"])
    #     return self.active_fraction * volume

    @property
    def active_mask(self) -> Bool[Array, " c"]:
        return self.active_fraction > 1e-3

    @property
    def energy(self) -> sim_wp.Phace:
        return self.model.model_warp.energies["elastic"]  # pyright: ignore[reportReturnType]

    @property
    def model(self) -> sim.Model:
        return self.forward.model

    # @property
    # def muscle_id(self) -> Array:
    #     return jnp.asarray(self.input.cell_data["muscle-id"])

    # @property
    # def muscle_orientation(self) -> Float[Array, "c 3 3"]:
    #     return jnp.asarray(self.input.cell_data["muscle-orientation"]).reshape(-1, 3, 3)

    def make_params(self, q: Float[Array, "ca 6"]) -> Params:
        activation: Float[Array, "c 6"] = sim_jax.rest_activation(self.n_cells)
        # q = q.at[:, :3].set(jnp.exp(q[:, :3]))
        activation = activation.at[self.active_mask].set(q)
        # activation = sim_jax.transform_activation(activation, self.muscle_orientation)
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
        (L, aux), dLdu = eqx.filter_jit(
            eqx.filter_value_and_grad(self.loss, has_aux=True)
        )(u, params)
        cherries.log_metrics(
            {
                "loss": {
                    "total": L,
                    "surface": aux.loss_surface,
                    "reg": {
                        "act": aux.reg_act,
                        "mean": aux.reg_mean,
                        "shear": aux.reg_shear,
                        "sparse": aux.reg_sparse,
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

        with grapes.timer(label="CG"):
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
            logger.warning("CG failed to converge, switching to GMRES")
            with grapes.timer(label="GMRES"):
                solver = lx.GMRES(
                    rtol=1e-5,
                    atol=1e-15,
                    max_steps=(u_free.size // 10),
                    # restart=(u_free.size // 1000),
                    # stagnation_iters=(u_free.size // 1000),
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
            logger.info("GMRES > results: {}", lx.RESULTS[solution.result])
            rel_res: float = relative_residual(p_free)
            logger.info("GMRES > relative residual: {}", rel_res)

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

    @eqx.filter_jit
    def loss(self, x: Vector, params: Params) -> tuple[Scalar, InverseLossAux]:
        loss_surface: Scalar = self.loss_surface(x)
        # reg_act: Scalar = self.reg_act * self.regularize_act(params)
        reg_act: Scalar = jnp.zeros(())
        # reg_mean: Scalar = self.reg_mean_weight * self.regularize_mean(params)
        reg_mean: Scalar = jnp.zeros(())
        # reg_shear: Scalar = self.reg_shear_weight * self.regularize_shear(params)
        reg_shear: Scalar = jnp.zeros(())
        reg_sparse: Scalar = self.reg_sparse * self.regularize_sparse(params)
        # reg_volume: Scalar = self.reg_volume_weight * self.regularize_volume(params)
        reg_volume: Scalar = jnp.zeros(())

        # jax.debug.print("loss_surface = {}", loss_surface)
        # jax.debug.print("reg_mean = {}", reg_mean)
        # jax.debug.print("reg_shear = {}", reg_shear)
        # jax.debug.print("reg_volume = {}", reg_volume)
        result: Scalar = (
            loss_surface + reg_act + reg_mean + reg_shear + reg_sparse + reg_volume
        )
        return result, InverseLossAux(
            loss_surface=loss_surface,
            reg_act=reg_act,
            reg_mean=reg_mean,
            reg_shear=reg_shear,
            reg_sparse=reg_sparse,
            reg_volume=reg_volume,
        )

    def loss_surface(self, u: Vector) -> Scalar:
        face_mask: Bool[Array, " p"] = jnp.asarray(self.is_face)
        target: Float[Array, "p 3"] = jnp.asarray(self.target_solution)
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
                jnp.sum(jnp.square(activation - activation_mean), axis=-1),
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
                active_volume, jnp.sum(jnp.square(activation[:, 3:]), axis=-1)
            )
        return regularization

    def regularize_volume(self, params: Params) -> Scalar:
        regularization: Float[Array, ""] = jnp.zeros(())
        for muscle_id in range(self.n_muscles):
            muscle_mask = self.muscle_id == muscle_id
            activation: Float[Array, " c 6"] = params.activation[muscle_mask]
            active_volume: Float[Array, " c"] = self.active_volume[muscle_mask]
            orientation: Float[Array, " c 3 3"] = self.muscle_orientation[muscle_mask]
            activation = sim_jax.transform_activation(
                activation, orientation, inverse=True
            )
            # residual: Float[Array, " c"] = jnp.abs(
            #     jnp.prod(activation[:, :3], axis=-1) - 1.0
            # )
            gamma: Float[Array, " c"] = jnp.prod(activation[:, [1, 2]], axis=-1)
            gamma = jax.lax.stop_gradient(gamma)
            gamma_inv: Float[Array, " c"] = jnp.nan_to_num(jnp.reciprocal(gamma))
            residual: Float[Array, " c"] = (
                jnp.square(activation[:, 0] - gamma_inv)
                # + jnp.square(activation[:, 1] - jnp.sqrt(gamma))
                # + jnp.square(activation[:, 2] - jnp.sqrt(gamma))
            )
            # residual: Float[Array, " c"] = jnp.square(
            #     jnp.cbrt(jnp.prod(activation[:, :3], axis=-1)) - 1.0
            # )
            # residual: Float[Array, " c"] = jnp.sum(jnp.log(activation[:, :3]), axis=-1)
            regularization += jnp.dot(active_volume, residual)
        return regularization

    def regularize_sparse(self, params: Params) -> Scalar:
        regularization: Float[Array, ""] = jnp.zeros(())
        for muscle_id in range(self.n_muscles):
            muscle_mask = self.muscle_id == muscle_id
            activation: Float[Array, "c 6"] = params.activation[muscle_mask]
            active_volume: Float[Array, " c"] = self.active_volume[muscle_mask]
            rest_act: Float[Array, " c 6"] = sim_jax.rest_activation(1)
            regularization += jnp.dot(
                active_volume,
                jnp.sum(jnp.square(activation - rest_act), axis=-1),
            )
        return regularization

    def regularize_act(self, params: Params) -> Scalar:
        regularization: Float[Array, ""] = jnp.zeros(())
        for muscle_id in range(self.n_muscles):
            muscle_mask = self.muscle_id == muscle_id
            activation: Float[Array, "c 6"] = params.activation[muscle_mask]
            orientation: Float[Array, "c 3 3"] = self.muscle_orientation[muscle_mask]
            act_oriented = sim_jax.transform_activation(
                activation, orientation, inverse=True
            )
            active_volume: Float[Array, " c"] = self.active_volume[muscle_mask]
            a0: Scalar = jax.lax.stop_gradient(
                jnp.nan_to_num(
                    jnp.reciprocal(jnp.mean(act_oriented[:, 1] * act_oriented[:, 2])),
                    nan=1.0,
                )
            )
            # a1: Scalar = jax.lax.stop_gradient(
            #     jnp.nan_to_num(
            #         jnp.reciprocal(jnp.mean(activation[:, 0] * activation[:, 2])),
            #         nan=1.0,
            #     )
            # )
            a1: Scalar = jnp.mean(act_oriented[:, 2])
            # a2: Scalar = jax.lax.stop_gradient(
            #     jnp.nan_to_num(
            #         jnp.reciprocal(jnp.mean(activation[:, 0] * activation[:, 1])),
            #         nan=1.0,
            #     )
            # )
            a2: Scalar = jnp.mean(act_oriented[:, 1])
            target_act: Float[Array, " 6"] = jnp.asarray([[a0, a1, a2, 0.0, 0.0, 0.0]])
            jax.debug.print("muscle {}, target act: {}", muscle_id, target_act)
            target_act = sim_jax.transform_activation(target_act, orientation)
            regularization += jnp.dot(
                active_volume, jnp.sum(jnp.square(activation - target_act), axis=-1)
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
    active_fraction: Float[Array, " c"] = jnp.asarray(mesh.cell_data["active-fraction"])
    active_volume: Float[Array, " c"]
    if "Volume" not in mesh.cell_data:
        mesh = mesh.compute_cell_sizes()  # pyright: ignore[reportAssignmentType]
    volume: Float[Array, " c"] = jnp.asarray(mesh.cell_data["Volume"])
    active_volume = active_fraction * volume
    inverse: Inverse = Inverse(
        forward=forward,
        # target=target,
        # input=mesh,
        solution=jnp.zeros_like(model.points),
        active_fraction=active_fraction,
        active_volume=active_volume,
        muscle_id=np.asarray(mesh.cell_data["muscle-id"]),
        muscle_orientation=jnp.asarray(mesh.cell_data["muscle-orientation"]).reshape(
            mesh.n_cells, 3, 3
        ),
        target_solution=jnp.asarray(target.point_data["solution"]),
        is_face=jnp.nonzero(jnp.asarray(target.point_data["is-face"]))[0],
        n_cells=mesh.n_cells,
        n_muscles=int(jnp.max(jnp.asarray(mesh.cell_data["muscle-id"]))) + 1,
    )
    activation_gt: Float[Array, "ca 6"] = sim_jax.rest_activation(
        inverse.n_active_cells
    )

    writer = melon.SeriesWriter(cfg.output)

    def callback(intermediate_result: optim.OptimizeSolution) -> None:
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
            inverse.solution - inverse.target_solution
        )[mesh.point_data["point-ids"]]
        # result: pv.UnstructuredGrid = mesh.warp_by_vector("solution")  # pyright: ignore[reportAssignmentType]
        writer.append(mesh)

    q_init: Float[Array, "ca 6"] = sim_jax.rest_activation(inverse.n_active_cells)
    # q_init = q_init.at[:, :3].set(jnp.log(q_init[:, :3]))
    callback(optim.OptimizeSolution({"x": q_init}))
    optimizer = optim.ScipyOptimizer(jit=False, method="L-BFGS-B", tol=1e-6, options={})
    solution: optim.OptimizeSolution = optimizer.minimize(
        objective=optim.Objective(
            value_and_grad=inverse.fun_and_jac
        )
        x0=q_init, fun_and_jac=inverse.fun_and_jac, callback=callback
    )
    callback(solution)
    ic(solution)

    # inverse.reg_mean_weight = 1e2
    # inverse.reg_shear_weight = 1e2
    # inverse.reg_volume_weight = 1e2
    # optimizer.tol = 1e-5
    # q_init = solution["x"]
    # solution = optimizer.minimize(
    #     x0=q_init, fun_and_jac=inverse.fun_and_jac, callback=callback
    # )
    # callback(solution)
    # ic(solution)

    # inverse.reg_mean_weight = 1e1
    # inverse.reg_shear_weight = 1e1
    # inverse.reg_volume_weight = 1e1
    # q_init = solution["x"]
    # optimizer.tol = 1e-6
    # solution = optimizer.minimize(
    #     x0=q_init, fun_and_jac=inverse.fun_and_jac, callback=callback
    # )
    # callback(solution)
    # ic(solution)


if __name__ == "__main__":
    cherries.run(main)
