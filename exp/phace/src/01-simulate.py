from pathlib import Path
from typing import override

import attrs
import einops
import felupe
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float, Integer, PyTree

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    input: Path = Path("./data/01-input.vtu")
    output: Path = Path("./data/02-output.vtu")


@apple.register_dataclass()
@attrs.define(kw_only=True)
class Problem(apple.AbstractPhysicsProblem):
    # data fields
    # data fields > auxiliaries
    dh_dX: Float[jax.Array, "C 4 3"] = attrs.field(default=None, converter=jnp.asarray)
    dV: Float[jax.Array, " C"] = attrs.field(default=None, converter=jnp.asarray)
    # data fields > parameters
    activation: Float[jax.Array, "C 3 3"] = attrs.field(
        converter=jnp.asarray, factory=lambda: jnp.expand_dims(jnp.identity(3), 0)
    )
    fixed_values: Float[jax.Array, " DoF"] = attrs.field(converter=jnp.asarray)
    lmbda: Float[jax.Array, " C"] = attrs.field(
        converter=jnp.asarray, factory=lambda: jnp.asarray(3.0)
    )
    mu: Float[jax.Array, " C"] = attrs.field(
        converter=jnp.asarray, factory=lambda: jnp.asarray(1.0)
    )
    muscle_fraction: Float[jax.Array, " C"] = attrs.field(
        converter=jnp.asarray, default=jnp.zeros((1,))
    )
    # meta fields
    mesh: pv.UnstructuredGrid = attrs.field(metadata={"static": True})
    fixed_mask: Bool[np.ndarray, " DoF"] = attrs.field(
        metadata={"static": True}, converter=np.asarray
    )

    @property
    def free_mask(self) -> Bool[np.ndarray, " D"]:
        return ~self.fixed_mask

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells

    @property
    @override
    def n_dof(self) -> int:
        return self.n_points * 3 - self.n_fixed

    @property
    def n_fixed(self) -> int:
        return jnp.count_nonzero(self.fixed_mask)  # pyright: ignore[reportReturnType]

    @property
    def n_points(self) -> int:
        return self.mesh.n_points

    @property
    def cell_points(self) -> Float[jax.Array, "C 4 3"]:
        return self.points[self.cells]

    @property
    def cells(self) -> Integer[jax.Array, "C 4"]:
        return jnp.asarray(self.mesh.cells_dict[pv.CellType.TETRA])

    @property
    def points(self) -> Float[jax.Array, "P 3"]:
        return jnp.asarray(self.mesh.points)

    def fill(self, u: PyTree, q: PyTree | None = None) -> PyTree:
        u_flat: Float[jax.Array, " DoF"] = self.ravel_u(u)
        u_flat: Float[jax.Array, " D"] = self.fill_flat(u_flat, q)
        u: PyTree = u_flat.reshape((self.n_points, 3))
        return u

    def fill_flat(
        self, u_flat: Float[jax.Array, " D"], q: PyTree | None = None
    ) -> Float[jax.Array, " D"]:
        fixed_values: Float[jax.Array, " D"] = self.get_param("fixed_values", q)
        u_filled: Float[jax.Array, " D"] = fixed_values.at[self.free_mask].set(u_flat)
        return u_filled

    @override
    @apple.jit()
    def fun(self, u: PyTree, q: PyTree | None = None) -> Float[jax.Array, ""]:
        u = self.fill(u, q)
        u: Float[jax.Array, "C 4 3"] = u[self.cells]
        activation: Float[jax.Array, " ..."] = self.get_param("activation", q)
        lmbda: Float[jax.Array, " ..."] = self.get_param("lmbda", q)
        mu: Float[jax.Array, " ..."] = self.get_param("mu", q)
        activation: Float[jax.Array, "C 3 3"] = jnp.broadcast_to(
            activation, (self.n_cells, 3, 3)
        )
        lmbda: Float[jax.Array, " C"] = jnp.broadcast_to(lmbda, (self.n_cells,))
        mu: Float[jax.Array, " C"] = jnp.broadcast_to(mu, (self.n_cells,))
        F: Float[jax.Array, "C 3 3"] = apple.elem.tetra.deformation_gradient(
            u, self.dh_dX
        )
        muscle_fraction: Float[jax.Array, " C"] = self.get_param("muscle_fraction", q)
        Psi: Float[jax.Array, " C"] = jax.vmap(corotated_passive)(F, lmbda, mu)
        Psi_active: Float[jax.Array, " C"] = jax.vmap(corotated)(
            F, lmbda, mu, activation
        )
        return jnp.sum(
            (1.0 - muscle_fraction) * Psi * self.dV
            + muscle_fraction * Psi_active * self.dV
        )

    @override
    def prepare(self, q: PyTree | None = None) -> None:
        super().prepare(q)
        self.lmbda = self.get_param("lmbda", q)
        self.mu = self.get_param("mu", q)
        self.dh_dX = apple.elem.tetra.dh_dX(self.cell_points)
        self.dV = apple.elem.tetra.dV(self.cell_points)

    @override
    def unravel_u(self, u_flat: Float[jax.Array, " DoF"]) -> Float[jax.Array, "P 3"]:
        return u_flat


def corotated_passive(
    F: Float[jax.Array, "3 3"], lmbda: Float[jax.Array, ""], mu: Float[jax.Array, ""]
) -> Float[jax.Array, ""]:
    return corotated(F, lmbda, mu, jnp.identity(3))


def corotated(
    F: Float[jax.Array, "3 3"],
    lmbda: Float[jax.Array, ""],
    mu: Float[jax.Array, ""],
    activation: Float[jax.Array, "3 3"],
) -> Float[jax.Array, ""]:
    R: Float[jax.Array, "3 3"]
    R, _S = apple.polar_rv(F)
    R = jax.lax.stop_gradient(R)  # TODO: support gradient of `polar_rv()`
    Psi: Float[jax.Array, ""] = (
        mu * jnp.sum((F - R @ activation) ** 2)
        + lmbda * (jnp.linalg.det(F) - jnp.linalg.det(activation)) ** 2
    )
    return Psi


def load_problem(fpath: Path) -> Problem:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(fpath)
    # mesh = fix_winding(mesh)
    activation: Float[jax.Array, "C 3 3"] = jnp.asarray(
        mesh.cell_data["activation"]
    ).reshape(mesh.n_cells, 3, 3)
    fixed_mask: Bool[jax.Array, " P"] = jnp.asarray(
        mesh.point_data["is-cranium"], bool
    ) | jnp.asarray(mesh.point_data["is-mandible"], bool)
    fixed_mask: Bool[jax.Array, "P 3"] = einops.repeat(fixed_mask, "P -> P dim", dim=3)
    fixed_mask: Bool[jax.Array, " D"] = fixed_mask.ravel()
    fixed_values: Float[jax.Array, " D"] = jnp.zeros(fixed_mask.shape)
    muscle_fraction: Float[jax.Array, "C M"] = jnp.asarray(
        mesh.cell_data["muscle-fraction"]
    )
    muscle_fraction: Float[jax.Array, " C"] = einops.reduce(
        muscle_fraction, "C M -> C", "sum"
    )
    return Problem(
        activation=activation,
        muscle_fraction=muscle_fraction,
        mesh=mesh,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )


def fix_winding(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    mesh = mesh.compute_cell_sizes(length=False, area=False, volume=True)  # pyright: ignore[reportAssignmentType]
    mesh_felupe = felupe.Mesh(mesh.points, mesh.cells_dict[pv.CellType.TETRA], "tetra")
    mesh_felupe = mesh_felupe.flip(mesh.cell_data["Volume"] < 0)
    return pv.UnstructuredGrid(
        {pv.CellType.TETRA: mesh_felupe.cells}, mesh_felupe.points
    )


@cherries.main()
def main(cfg: Config) -> None:
    problem: Problem = load_problem(cfg.input)
    problem.prepare()
    mesh: pv.UnstructuredGrid = problem.mesh
    result: apple.MinimizeResult = problem.solve(
        algo=apple.MinimizeScipy(
            method="trust-constr", options={"disp": True, "verbose": 3}
        )
    )
    ic(result)
    mesh.point_data["solution"] = problem.fill(result["x"])
    mesh.warp_by_vector("solution", inplace=True)
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    main(Config())
