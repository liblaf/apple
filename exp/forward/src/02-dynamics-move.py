from pathlib import Path

import attrs
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import rich.traceback
from jaxtyping import Bool, Float

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import cherries, grapes, melon


class Config(cherries.BaseConfig):
    input: Path = grapes.find_project_dir() / "data/bunny/input.vtu"
    method: str = "pncg"
    output_animation: Path = (
        grapes.find_project_dir() / "data/bunny/dynamic/animation.pvd"
    )
    output: Path = grapes.find_project_dir() / "data/bunny/dynamic/output.vtu"
    time_step: float = 1.0 / 30


def main(cfg: Config) -> None:
    rich.traceback.install(show_locals=True)
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    ic(mesh)
    ic(mesh.length)
    mesh.point_data["mass"] = np.asarray(apple.utils.point_mass(mesh))
    dirichlet_mask: Bool[jax.Array, " N"] = jnp.asarray(
        mesh.point_data["dirichlet-mask"]
    ).ravel()
    dirichlet_values: Float[jax.Array, " N"] = jnp.asarray(
        mesh.point_data["dirichlet-values"]
    ).ravel()
    box = apple.obj.tetra.ObjectTetra(
        name="box",
        params={
            "lambda": jnp.asarray(mesh.cell_data["lambda"]),
            "mu": jnp.asarray(mesh.cell_data["mu"]),
        },
        material=apple.material.tetra.AsRigidAsPossibleFilter(),
        mesh=mesh,
    )
    u0: Float[jax.Array, " F"] = jnp.zeros_like(mesh.point_data["initial"]).ravel()
    # u0: Float[jax.Array, " DoF"] = u0[~dirichlet_mask]
    inertia = apple.obj.ObjectPoint(
        name="inertia",
        data=mesh.cast_to_pointset(),
        material=apple.material.point.MaterialPointInertia(),
        params={"mass": jnp.asarray(mesh.point_data["mass"])},
    )
    problem = apple.PhysicsProblem(
        objects=[box, inertia],
        dirichlet_mask=dirichlet_mask,
        dirichlet_values=dirichlet_values,
    )
    problem.prepare()

    def warp_result(result: apple.MinimizeResult) -> pv.UnstructuredGrid:
        solution: Float[jax.Array, " DoF"] = result["x"]
        solution: Float[jax.Array, " F"] = problem.fill(solution)
        solution: Float[jax.Array, " N"] = box.select_dof(solution)
        solution: Float[jax.Array, "V 3"] = box.unravel_u(solution)
        output: pv.UnstructuredGrid = mesh.copy()
        output.point_data["solution"] = np.asarray(solution)
        if (jac := result.get("jac")) is not None:  # pyright: ignore[reportAssignmentType]
            jac: Float[jax.Array, " F"] = problem.fill_zeros(jac)
            jac: Float[jax.Array, " N"] = box.select_dof(jac)
            jac: Float[jax.Array, "V 3"] = box.unravel_u(jac)
            output.point_data["jac"] = np.asarray(jac)
        output.warp_by_vector("solution", inplace=True)
        return output

    writer = melon.io.PVDWriter(cfg.output_animation, fps=1.0 / cfg.time_step)
    if cfg.method == "pncg":
        algo = apple.MinimizePNCG(eps=1e-3, iter_max=25)
    elif cfg.method == "scipy":
        algo = apple.MinimizeScipy(
            method="trust-constr", options={"disp": True, "verbose": 3}
        )
    else:
        raise NotImplementedError(f"Unknown method: {cfg.method}")  # noqa: EM102

    dynamics = Dynamics(
        problem=problem, inertia=inertia, algo=algo, u=u0, time_step=cfg.time_step
    )
    writer.append(warp_result(apple.MinimizeResult({"x": u0[~dirichlet_mask]})))  # pyright: ignore[reportAssignmentType]
    for t in range(int(3 / cfg.time_step)):
        ic(t)
        result: apple.MinimizeResult = dynamics.step()
        output: pv.UnstructuredGrid = warp_result(result)
        writer.append(output)
    writer.end()
    melon.save(cfg.output, output)  # pyright: ignore[reportPossiblyUnboundVariable]


@attrs.define
class Dynamics:
    def _default_u(self) -> Float[jax.Array, " F"]:
        return jnp.zeros(self.n_dof_full)

    def _default_v(self) -> Float[jax.Array, " F"]:
        return jnp.zeros(self.n_dof_full)

    problem: apple.PhysicsProblem
    inertia: apple.obj.ObjectPoint
    time_step: float = 0.1
    time: float = 0.0
    algo: apple.MinimizeAlgorithm | None = None
    u: Float[jax.Array, " F"] = attrs.field(
        default=attrs.Factory(_default_u, takes_self=True)
    )
    v: Float[jax.Array, " F"] = attrs.field(
        default=attrs.Factory(_default_v, takes_self=True)
    )

    @property
    def free_mask(self) -> Bool[np.ndarray, " N"]:
        return self.problem.free_mask

    @property
    def n_dof_full(self) -> int:
        return self.problem.n_dof_full

    def step(self) -> apple.MinimizeResult:
        self.update_dirichlet()
        result: apple.MinimizeResult = self.problem.solve(
            q={
                "inertia": {
                    "velocity": self.inertia.unravel_u(self.v),
                    "displacement": self.inertia.unravel_u(self.u),
                    "time-step": jnp.full(self.inertia.n_points, self.time_step),
                }
            },
            u0=self.u[self.free_mask],
            algo=self.algo,
        )
        u_next: Float[jax.Array, " DoF"] = result["x"]
        u_next: Float[jax.Array, " F"] = self.problem.fill(u_next)
        self.v = (u_next - self.u) / self.time_step
        self.u = u_next
        self.time += self.time_step
        return result

    def update_dirichlet(self) -> None:
        x_mask: Bool[jax.Array, " N"] = jnp.zeros(self.problem.n_dof_full, dtype=bool)
        x_mask = x_mask.reshape(-1, 3)
        x_mask = x_mask.at[:, 0].set(True)
        x_mask = x_mask.ravel()
        dirichlet_value: Float[jax.Array, ""] = 0.05 * jnp.sin(
            self.time * (2 * jnp.pi / 1)
        )
        ic(dirichlet_value)
        self.problem.dirichlet_values = self.problem.dirichlet_values.at[
            x_mask & self.problem.dirichlet_mask
        ].set(dirichlet_value)


if __name__ == "__main__":
    cherries.run(main)
