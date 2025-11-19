import logging
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
from liblaf.peach import tree
from liblaf.peach.linalg import (
    JaxCG,
    JaxCompositeSolver,
    JaxGMRES,
    LinearSolution,
    LinearSystem,
)
from liblaf.peach.optim import (
    PNCG,
    Objective,
    Optimizer,
    OptimizeSolution,
    ScipyOptimizer,
)

from liblaf import cherries, melon
from liblaf.apple import sim
from liblaf.apple.warp import sim as sim_wp
from liblaf.apple.warp import utils as wp_utils
from liblaf.apple.warp.typing import vec6

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    input: Path = cherries.input("10-input.vtu")
    target: Path = cherries.input("10-target.vtu")

    output: Path = cherries.output("20-inverse.vtu")


@tree.define
class Params:
    activation: Float[Array, "a 6"]


class Aux(TypedDict):
    recon: Float[Array, ""]
    sparse: Float[Array, ""]


@tree.define
class Inverse:
    face_idx: Integer[Array, " f"]
    model: sim.Model
    muscle_id_to_idx: dict[int, Integer[Array, " m"]]
    muscle_id_to_volume: dict[int, Float[Array, " m"]]
    muscle_idx: Integer[Array, " a"]
    n_cells: int
    target: Float[Array, "face 3"]
    optimizer: Optimizer = tree.field(factory=lambda: PNCG(rtol=1e-5, max_steps=500))
    step: int = tree.array(default=1)
    u: Float[Array, "p 3"] = tree.array(default=None)

    weight_sparse: float = 1e-2

    @property
    def energy(self) -> sim_wp.Phace:
        return self.model.model_warp.energies["elastic"]  # pyright: ignore[reportReturnType]

    def value_and_grad(self, params: Params) -> tuple[Float[Array, ""], Params]:
        act: Float[Array, "c 6"]
        act_vjp: Callable[[Float[Array, "c 6"]], Params]
        act, act_vjp = jax.vjp(self.make_activations, params)
        self.set_activation(act)
        u: Float[Array, "p 3"] = self.forward()
        self.u = u
        (loss, aux), (dLdu, dLdq) = self.loss_and_grad(u, act)
        cherries.log_metrics(aux, step=self.step)
        self.step += 1
        p: Float[Array, "p 3"] = self.adjoint(u, dLdu)
        outputs: dict[str, dict[str, Array]] = self.model.mixed_derivative_prod(u, p)
        act_grad: Float[Array, "c 6"] = dLdq + outputs[self.energy.id]["activations"]
        grad: Params = act_vjp(act_grad)
        return loss, grad

    def make_activations(self, params: Params) -> Float[Array, "c 6"]:
        activations: Float[Array, "c 6"] = jnp.zeros((self.n_cells, 6))
        activations = activations.at[self.muscle_idx].set(params.activation)
        return activations

    def set_activation(self, activations: Float[Array, "c 6"]) -> None:
        wp.copy(
            self.energy.params.activations, wp_utils.to_warp(activations, dtype=vec6)
        )

    def forward(self) -> Float[Array, "p 3"]:
        objective = Objective(
            fun=self.model.fun,
            grad=self.model.jac,
            hess_diag=self.model.hess_diag,
            hess_prod=self.model.hess_prod,
            hess_quad=self.model.hess_quad,
            value_and_grad=self.model.fun_and_jac,
            grad_and_hess_diag=self.model.jac_and_hess_diag,
        )
        u_free: Float[Array, " free"] = jnp.zeros((self.model.n_free,))
        solution: OptimizeSolution[PNCG.State, PNCG.Stats] = self.optimizer.minimize(
            objective=objective, params=u_free
        )
        u_free = solution.params
        logger.info("Forward time: %g", solution.stats.time)
        assert solution.success
        u_full: Float[Array, "p 3"] = self.model.to_full(u_free)
        return u_full

    def adjoint(
        self, u: Float[Array, "p 3"], dLdu: Float[Array, "p 3"]
    ) -> Float[Array, "p 3"]:
        u_free: Float[Array, " free"] = self.model.dirichlet.get_free(u)
        preconditioner: Float[Array, " free"] = jnp.reciprocal(
            self.model.dirichlet.get_free(self.model.hess_diag(u))
        )
        solver = JaxCompositeSolver(
            solvers=[
                JaxCG(max_steps=self.model.n_free // 10),
                JaxGMRES(max_steps=self.model.n_free // 10),
            ]
        )
        system = LinearSystem(
            lambda p_free: self.model.hess_prod(u_free, p_free),
            b=-self.model.dirichlet.get_free(dLdu),
            preconditioner=lambda p_free: preconditioner * p_free,
        )
        solution: LinearSolution[JaxCompositeSolver.State, JaxCompositeSolver.Stats] = (
            solver.solve(system, jnp.zeros((self.model.n_free,)))
        )
        logger.info("Adjoint time: %g", solution.stats.time)
        assert solution.success
        return self.model.to_full(solution.params, zero=True)

    @eqx.filter_jit
    def loss_and_grad(
        self, u: Float[Array, "p 3"], act: Float[Array, "c 6"]
    ) -> tuple[
        tuple[Float[Array, ""], Aux], tuple[Float[Array, "p 3"], Float[Array, "c 6"]]
    ]:
        loss: Float[Array, ""]
        (loss, aux), (dLdu, dLdq) = jax.value_and_grad(
            self.loss, argnums=(0, 1), has_aux=True
        )(u, act)
        return (loss, aux), (dLdu, dLdq)

    def loss(
        self, u: Float[Array, "p 3"], act: Float[Array, "c 6"]
    ) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
        loss_recon: Float[Array, ""] = self.loss_recon(u)
        reg_sparse: Float[Array, ""] = self.weight_sparse * self.reg_sparse(act)
        loss: Float[Array, ""] = loss_recon + reg_sparse
        return loss, {"total": loss, "recon": loss_recon, "sparse": reg_sparse}

    def loss_recon(self, u: Float[Array, "p 3"]) -> Float[Array, ""]:
        residual: Float[Array, "f 3"] = u[self.face_idx] - self.target
        return 0.5 * jnp.sum(jnp.square(residual))

    def reg_sparse(self, act: Float[Array, "c 6"]) -> Float[Array, ""]:
        reg: Float[Array, ""] = jnp.zeros(())
        for i, indices in self.muscle_id_to_idx.items():
            act_i: Float[Array, " m 6"] = act[indices]
            vol_i: Float[Array, " m"] = self.muscle_id_to_volume[i]
            mag: Float[Array, " m"] = jnp.sum(jnp.square(act_i), axis=-1)
            reg += jnp.vdot(vol_i, mag)
        return reg


def prepare(mesh: pv.UnstructuredGrid) -> sim.Model:
    builder = sim.ModelBuilder()
    mesh = builder.assign_dofs(mesh)
    builder.add_dirichlet(mesh)
    energy: sim_wp.Phace = sim_wp.Phace.from_pyvista(
        mesh, id="elastic", requires_grad=("activations",)
    )
    builder.add_energy(energy)
    model: sim.Model = builder.finish()
    return model


def inverse(
    model: sim.Model,
    mesh: pv.UnstructuredGrid,
    target: pv.UnstructuredGrid,
    idx: str = "000",
) -> pv.UnstructuredGrid:
    mesh = mesh.compute_cell_sizes(length=False, area=False, volume=True)  # pyright: ignore[reportAssignmentType]
    muscle_fractions: Float[Array, " c"] = jnp.asarray(
        mesh.cell_data["MuscleFractions"]
    )
    muscle_idx: Integer[Array, " a"] = jnp.flatnonzero(muscle_fractions > 1e-3)
    face_idx: Integer[Array, " f"] = jnp.flatnonzero(mesh.point_data["IsFace"])
    muscle_id_to_idx: dict[int, Integer[Array, " m"]] = {}
    muscle_id_to_volume: dict[int, Float[Array, " m"]] = {}
    for i in range(np.max(mesh.cell_data["MuscleIds"]) + 1):
        mask: Bool[Array, " c"] = (muscle_fractions > 1e-3) & (
            mesh.cell_data["MuscleIds"] == i
        )
        indices: Integer[Array, " m"] = jnp.flatnonzero(mask)
        volume: Float[Array, " m"] = jnp.asarray(mesh.cell_data["Volume"][indices])
        volume *= jnp.asarray(muscle_fractions[indices])
        muscle_id_to_idx[i] = indices
        muscle_id_to_volume[i] = volume
    inverse = Inverse(
        muscle_id_to_idx=muscle_id_to_idx,
        muscle_id_to_volume=muscle_id_to_volume,
        muscle_idx=muscle_idx,
        face_idx=face_idx,
        model=model,
        n_cells=mesh.n_cells,
        target=jnp.asarray(target.point_data[f"Expression{idx}"])[face_idx],
    )
    params: Params = Params(activation=jnp.zeros((muscle_idx.shape[0], 6)))
    optimizer = ScipyOptimizer(method="L-BFGS-B", tol=1e-3, timer=True)

    with melon.SeriesWriter(cherries.temp(f"20-inverse-{idx}.vtu.series")) as writer:

        def callback(state: ScipyOptimizer.State, _stats: ScipyOptimizer.Stats) -> None:
            activation: Float[Array, "c 6"] = inverse.make_activations(state.params)
            mesh.point_data[f"Displacement{idx}"] = np.asarray(inverse.u)
            mesh.point_data[f"Residual{idx}"] = np.zeros((mesh.n_points, 3))
            mesh.point_data[f"Residual{idx}"][np.asarray(inverse.face_idx)] = (
                np.asarray(inverse.u[inverse.face_idx] - inverse.target)
            )
            mesh.cell_data[f"Activations{idx}"] = np.asarray(activation)
            writer.append(mesh)

        solution: OptimizeSolution[ScipyOptimizer.State, ScipyOptimizer.Stats] = (
            optimizer.minimize(
                objective=Objective(value_and_grad=inverse.value_and_grad),
                params=params,
                callback=callback,
            )
        )
    ic(solution)
    params: Params = solution.params
    activations: Float[Array, "c 6"] = jnp.zeros((mesh.n_cells, 6))
    activations = activations.at[muscle_idx].set(params.activation)
    mesh.point_data[f"Displacement{idx}"] = np.asarray(inverse.u)
    mesh.cell_data[f"Activations{idx}"] = np.asarray(activations)
    return mesh


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    target: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.target)
    model = prepare(mesh)
    mesh = inverse(model, mesh, target, idx="000")
    mesh = inverse(model, mesh, target, idx="001")
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.main(main)
