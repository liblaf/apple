from pathlib import Path

import einops
import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Float
from loguru import logger

from liblaf import cherries, melon
from liblaf.apple.jax import math, optim, sim, tree
from liblaf.apple.jax.typing._types import Scalar, Vector


class Config(cherries.BaseConfig):
    input: Path = cherries.input("10-input.vtu")
    target: Path = cherries.input("20-target.vtu")


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
        x0 = self.model.dirichlet.get_free(x0)
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
    target: Vector = tree.array()

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
        self.solution = u
        loss: Scalar = self.loss(u, params)
        jac: Params = self._jac(u, params)
        return loss, jac

    def loss(self, u: Vector, params: Params) -> Scalar:
        diff: Vector = u - self.target
        objective: Scalar = 0.5 * jnp.sum(diff**2)
        reg_weight = 1e-3
        activation_mean = jnp.mean(params)
        regularization = reg_weight * jnp.sum((params - activation_mean) ** 2)
        loss: Scalar = objective + regularization
        return loss

    def _jac(self, u: Vector, params: Params) -> Params:
        energy: sim.PhaceActive = self.model.energies[0]  # pyright: ignore[reportAssignmentType]
        L, dLdu = jax.value_and_grad(self.loss)(u, params)
        dLdq = jax.grad(self.loss, argnums=1)(u, params)
        p, info = jax.scipy.sparse.linalg.cg(
            lambda p: energy.hess_prod(u, p), -dLdu, x0=jnp.zeros_like(u)
        )
        logger.info(info)
        jac = energy.mixed_derivative_prod(u, p)
        return jac + dLdq


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    target: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.target)

    builder = sim.ModelBuilder()
    mesh = builder.assign_dofs(mesh)
    builder.add_dirichlet(mesh)
    builder.add_energy(sim.PhaceActive.from_pyvista(mesh))
    model: sim.Model = builder.finish()

    forward = Forward(model=model)
    inverse: InversePhysics = InversePhysics(
        forward=forward,
        target=math.asarray(target.point_data["solution"]),
        solution=jnp.zeros((model.n_dofs,)),
    )

    def callback(intermediate_result: optim.Solution) -> None:
        ic(intermediate_result)

    init_activation: Float[Array, "c J J"] = einops.repeat(
        jnp.identity(3), "i j -> c i j", c=mesh.n_cells
    )
    init_params: Params = init_activation
    optimizer = optim.MinimizerScipy(jit=False, method="L-BFGS-B", options={})
    solution: optim.Solution = optimizer.minimize(
        x0=init_params, fun_and_jac=inverse.fun_and_jac, callback=callback
    )
    ic(solution)


if __name__ == "__main__":
    cherries.run(main)
