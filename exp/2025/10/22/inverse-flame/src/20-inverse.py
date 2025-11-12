from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import warp as wp
from jaxtyping import Array, Bool, Float, Integer
from liblaf.peach import linalg, optim, tree
from loguru import logger

from liblaf import cherries, melon
from liblaf.apple import sim
from liblaf.apple.jax import sim as sim_jax
from liblaf.apple.warp import sim as sim_wp
from liblaf.apple.warp import utils as wp_utils
from liblaf.apple.warp.typing import vec6


class Config(cherries.BaseConfig):
    input: Path = cherries.input("10-input.vtu")
    target: Path = cherries.input("10-target.vtu")

    output: Path = cherries.output("20-inverse.vtu")


@tree.define
class Params:
    activation: Float[Array, "a 6"]


class Aux(TypedDict):
    reconstruction: Float[Array, ""]
    sparse: Float[Array, ""]


@tree.define
class Inverse:
    active_index: Integer[Array, " a"]
    active_indices: dict[int, Integer[Array, " m"]]
    active_volume: dict[int, Float[Array, " m"]]
    face_index: Integer[Array, " f"]
    model: sim.Model
    n_cells: int
    target: Float[Array, "face 3"]
    optimizer: optim.Optimizer = tree.field(
        factory=lambda: optim.PNCG(rtol=1e-5, max_steps=500)
    )
    step: int = tree.array(default=1)
    u: Float[Array, "p 3"] = tree.array(default=None)

    weight_sparse: float = 1e-2

    @property
    def energy(self) -> sim_wp.Phace:
        return self.model.model_warp.energies["elastic"]  # pyright: ignore[reportReturnType]

    def value_and_grad(self, params: Params) -> tuple[Float[Array, ""], Params]:
        act: Float[Array, "c 6"]
        act_vjp: Callable[[Float[Array, "c 6"]], Params]
        act, act_vjp = jax.vjp(self.make_activation, params)
        self.set_activation(act)
        u: Float[Array, "p 3"] = self.forward()
        self.u = u
        (loss, aux), (dLdu, dLdq) = self.loss_and_grad(u, act)
        ic(loss, aux, dLdu, dLdq)
        cherries.log_metrics(aux, step=self.step)
        self.step += 1
        p: Float[Array, "p 3"] = self.adjoint(u, dLdu)
        outputs: dict[str, dict[str, Array]] = self.model.mixed_derivative_prod(u, p)
        act_grad: Float[Array, "c 6"] = dLdq + outputs[self.energy.id]["activation"]
        grad: Params = act_vjp(act_grad)
        return loss, grad

    def make_activation(self, params: Params) -> Float[Array, "c 6"]:
        activation: Float[Array, "c 6"] = sim_jax.rest_activation(self.n_cells)
        activation = activation.at[self.active_index].set(params.activation)
        return activation

    def set_activation(self, activation: Float[Array, "c 6"]) -> None:
        wp.copy(self.energy.params.activation, wp_utils.to_warp(activation, dtype=vec6))

    def forward(self) -> Float[Array, "p 3"]:
        objective = optim.Objective(
            fun=self.model.fun,
            grad=self.model.jac,
            hess_diag=self.model.hess_diag,
            hess_prod=self.model.hess_prod,
            hess_quad=self.model.hess_quad,
            value_and_grad=self.model.fun_and_jac,
            grad_and_hess_diag=self.model.jac_and_hess_diag,
        )
        u_free: Float[Array, " free"] = jnp.zeros((self.model.n_free,))
        solution: optim.OptimizeSolution[optim.PNCGState, optim.PNCGStats] = (
            self.optimizer.minimize(objective=objective, params=u_free)
        )
        u_free = solution.params
        logger.info("Forward time: {}", solution.stats.time)
        u_full: Float[Array, "p 3"] = self.model.to_full(u_free)
        return u_full

    def adjoint(
        self, u: Float[Array, "p 3"], dLdu: Float[Array, "p 3"]
    ) -> Float[Array, "p 3"]:
        u_free: Float[Array, " free"] = self.model.dirichlet.get_free(u)
        preconditioner: Float[Array, " free"] = jnp.reciprocal(
            self.model.dirichlet.get_free(self.model.hess_diag(u))
        )
        solver = linalg.jax.JaxCompositeSolver(
            solvers=[
                linalg.JaxCG(max_steps=self.model.n_free // 10),
                linalg.JaxGMRES(max_steps=self.model.n_free // 10),
            ]
        )
        op = linalg.LinearOperator(
            lambda p_free: self.model.hess_prod(u_free, p_free),
            preconditioner=lambda p_free: preconditioner * p_free,
        )
        b: Float[Array, " free"] = -self.model.dirichlet.get_free(dLdu)
        solution: linalg.LinearSolution[linalg.JaxState, linalg.JaxStats] = (
            solver.solve(op, b, params=jnp.zeros((self.model.n_free,)))
        )
        logger.info("Adjoint time: {}", solution.stats.time)
        assert solution.success
        return self.model.to_full(solution.params, zero=True)

    @eqx.filter_jit
    def loss_and_grad(
        self, u: Float[Array, "p 3"], act: Float[Array, "c 6"]
    ) -> tuple[
        tuple[Float[Array, ""], Aux], tuple[Float[Array, "p 3"], Float[Array, "c 6"]]
    ]:
        loss: Float[Array, ""]
        (loss, aux), (dLdu, dLdq) = jax.jit(
            jax.value_and_grad(self.loss, argnums=(0, 1), has_aux=True)
        )(u, act)
        return (loss, aux), (dLdu, dLdq)

    def loss(
        self, u: Float[Array, "p 3"], act: Float[Array, "c 6"]
    ) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
        loss_recon: Float[Array, ""] = self.loss_reconstruction(u)
        reg_sparse: Float[Array, ""] = self.weight_sparse * self.reg_sparse(act)
        loss: Float[Array, ""] = loss_recon + reg_sparse
        return loss, {"reconstruction": loss_recon, "sparse": reg_sparse}

    def loss_reconstruction(self, u: Float[Array, "p 3"]) -> Float[Array, ""]:
        residual: Float[Array, "f 3"] = u[self.face_index] - self.target
        return 0.5 * jnp.sum(jnp.square(residual))

    def reg_sparse(self, act: Float[Array, "c 6"]) -> Float[Array, ""]:
        reg: Float[Array, ""] = jnp.zeros(())
        for i, indices in self.active_indices.items():
            act_i: Float[Array, " m 6"] = act[indices]
            vol_i: Float[Array, " m"] = self.active_volume[i]
            mag: Float[Array, " m"] = jnp.sum(
                jnp.square(act_i - sim_jax.rest_activation(act_i.shape[0])), axis=-1
            )
            reg += jnp.vdot(vol_i, mag)
        return reg


def prepare(mesh: pv.UnstructuredGrid) -> sim.Model:
    builder = sim.ModelBuilder()
    mesh = builder.assign_dofs(mesh)
    builder.add_dirichlet(mesh)
    energy: sim_wp.Phace = sim_wp.Phace.from_pyvista(
        mesh, id="elastic", requires_grad=("activation",)
    )
    builder.add_energy(energy)
    model: sim.Model = builder.finish()
    return model


def inverse(
    model: sim.Model,
    mesh: pv.UnstructuredGrid,
    target: pv.UnstructuredGrid,
    idx: str = "00",
) -> pv.UnstructuredGrid:
    active_index: Integer[Array, " a"] = jnp.flatnonzero(
        mesh.cell_data["active-fraction"] > 1e-3
    )
    face_index: Integer[Array, " f"] = jnp.flatnonzero(mesh.point_data["is-face"])
    active_indices: dict[int, Integer[Array, " m"]] = {}
    active_volume: dict[int, Float[Array, " m"]] = {}
    for i in range(np.max(mesh.cell_data["muscle-id"]) + 1):
        mask: Bool[Array, " c"] = (mesh.cell_data["active-fraction"] > 1e-3) & (
            mesh.cell_data["muscle-id"] == i
        )
        indices: Integer[Array, " m"] = jnp.flatnonzero(mask)
        volume: Float[Array, " m"] = jnp.asarray(mesh.cell_data["Volume"][indices])
        volume *= jnp.asarray(mesh.cell_data["active-fraction"][indices])
        active_indices[i] = indices
        active_volume[i] = volume
    inverse = Inverse(
        active_indices=active_indices,
        active_volume=active_volume,
        active_index=active_index,
        face_index=face_index,
        model=model,
        n_cells=mesh.n_cells,
        target=jnp.asarray(target.point_data[f"expression-{idx}"])[face_index],
    )
    params: Params = Params(activation=sim_jax.rest_activation(active_index.shape[0]))
    optimizer = optim.ScipyOptimizer(method="L-BFGS-B", tol=1e-3, timer=True)

    writer = melon.SeriesWriter(f"tmp/20-inverse-{idx}.vtu.series")

    def callback(state: optim.ScipyState, stats: optim.ScipyStats) -> None:
        ic(state, stats)
        activation: Float[Array, "c 6"] = inverse.make_activation(state.params)
        mesh.point_data[f"displacement-{idx}"] = np.asarray(inverse.u)
        mesh.point_data[f"residual-{idx}"] = np.zeros((mesh.n_points, 3))
        mesh.point_data[f"residual-{idx}"][np.asarray(inverse.face_index)] = np.asarray(
            inverse.u[inverse.face_index] - inverse.target
        )
        mesh.cell_data[f"activation-{idx}"] = np.asarray(activation)
        mesh.cell_data[f"activation-{idx}-mag"] = np.asarray(
            activation - sim_jax.rest_activation(mesh.n_cells)
        )
        writer.append(mesh)

    solution: optim.OptimizeSolution[optim.ScipyState, optim.ScipyStats] = (
        optimizer.minimize(
            objective=optim.Objective(value_and_grad=inverse.value_and_grad),
            params=params,
            callback=callback,
        )
    )
    ic(solution)
    params: Params = solution.params
    activation: Float[Array, "c 6"] = sim_jax.rest_activation(mesh.n_cells)
    activation = activation.at[active_index].set(params.activation)
    mesh.point_data[f"displacement-{idx}"] = np.asarray(inverse.u)
    mesh.cell_data[f"activation-{idx}"] = np.asarray(activation)
    mesh.cell_data[f"activation-{idx}-mag"] = np.asarray(
        activation - sim_jax.rest_activation(mesh.n_cells)
    )
    return mesh


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    target: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.target)
    model = prepare(mesh)
    mesh = inverse(model, mesh, target, idx="00")
    mesh = inverse(model, mesh, target, idx="01")
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.run(main)
