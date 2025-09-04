from pathlib import Path

import einops
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Float
from loguru import logger

from liblaf import cherries, grapes, melon
from liblaf.apple.jax import optim, sim, tree
from liblaf.apple.jax.typing import Scalar, Vector

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


class Config(cherries.BaseConfig):
    input: Path = cherries.input("10-input.vtu")
    target: Path = cherries.input("20-target.vtu")

    output: Path = cherries.output("30-inverse.vtu.series")


type Params = Float[Array, "c J J"]


@tree.pytree
class Forward:
    model: sim.Model
    optimizer: optim.Minimizer = tree.field(
        factory=lambda: optim.MinimizerScipy(
            timer=False, method="trust-constr", options={"verbose": 0}
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

    def fun(self, params: Params) -> Scalar:
        energy: sim.PhaceActive = self.model.energies[0]  # pyright: ignore[reportAssignmentType]
        energy.activation = params
        self.solution = self.forward.solve(self.solution)
        return self.loss(self.solution, params)

    def fun_and_jac(self, params: Params) -> tuple[Scalar, Params]:
        energy: sim.PhaceActive = self.model.energies[0]  # pyright: ignore[reportAssignmentType]
        energy.activation = params
        u: Vector = self.forward.solve(self.solution)
        assert not jnp.any(jnp.isnan(u))
        self.solution = u
        loss: Scalar = self.loss(u, params)
        jac: Params = self._jac(u, params)
        with grapes.config.pretty.overrides(short_arrays=False):
            ic(loss, jac[0])
        return loss, jac

    def loss(self, u: Vector, params: Params) -> Scalar:
        diff: Vector = u - self.target.point_data["solution"]
        diff = diff[self.target.point_data["is-surface"]]
        objective: Scalar = 0.5 * jnp.sum(diff**2)
        reg_weight = 1.0
        activation_mean = jnp.mean(params, axis=0)
        regularization = reg_weight * jnp.sum((params - activation_mean) ** 2)
        loss: Scalar = objective + regularization
        return loss

    def _jac(self, u: Vector, params: Params) -> Params:
        energy: sim.PhaceActive = self.model.energies[0]  # pyright: ignore[reportAssignmentType]
        L, dLdu = jax.value_and_grad(self.loss)(u, params)
        dLdq = jax.grad(self.loss, argnums=1)(u, params)
        p, info = jax.scipy.sparse.linalg.gmres(
            lambda p: self.model.hess_prod(u, p), -dLdu, x0=jnp.zeros_like(u)
        )
        # p = jnp.nan_to_num(p)
        logger.info(info)
        assert not jnp.any(jnp.isnan(u))
        assert not jnp.any(jnp.isnan(p))
        jac = energy.mixed_derivative_prod(u, p)
        return jac + dLdq


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    target: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.target)
    activation_gt: Float[Array, "c J J"] = jnp.reshape(
        target.cell_data["activation"], (-1, 3, 3)
    )

    builder = sim.ModelBuilder()
    mesh = builder.assign_dofs(mesh)
    builder.add_dirichlet(mesh)
    builder.add_energy(sim.PhaceActive.from_pyvista(mesh))
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
        activation: Params = jnp.reshape(intermediate_result["x"], (-1, 3, 3))
        with grapes.config.pretty.overrides(short_arrays=False):
            ic(activation[0])
        activation_residual: Params = activation - activation_gt
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

    init_activation: Float[Array, "c J J"] = einops.repeat(
        jnp.identity(3), "i j -> c i j", c=mesh.n_cells
    )
    init_params: Params = init_activation

    callback(optim.Solution({"x": init_params.flatten()}))

    optimizer = optim.MinimizerScipy(jit=False, method="L-BFGS-B", tol=1e-4, options={})
    solution: optim.Solution = optimizer.minimize(
        x0=init_params, fun_and_jac=inverse.fun_and_jac, callback=callback
    )
    callback(solution)
    ic(solution)


if __name__ == "__main__":
    cherries.run(main)
