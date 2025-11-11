import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jax import Array
from jaxtyping import Integer

from liblaf import grapes, melon
from liblaf.apple import sim
from liblaf.apple.jax import optim, tree
from liblaf.apple.jax import sim as sim_jax
from liblaf.apple.jax.typing import Scalar, Vector


@tree.pytree
class Poisson(sim_jax.Energy):
    region: sim_jax.Region
    neumann: Vector = tree.array()
    point_area: Vector = tree.array()

    dirichlet_indices: Integer[Array, " D"] = tree.array()
    dirichlet_values: Vector = tree.array()
    free_indices: Integer[Array, " F"] = tree.field()

    n_dofs: int

    def fun(self, u: Vector) -> Scalar:
        u_full = jnp.zeros((self.n_dofs,))
        u_full = u_full.at[self.dirichlet_indices].set(self.dirichlet_values)
        u_full = u_full.at[self.free_indices].set(u)
        grad: Vector = self.region.gradient(u_full)
        energy = 0.5 * jnp.vdot(jnp.sum(jnp.square(grad), axis=-1), self.region.dV)
        neumann_energy = jnp.vdot(
            u * self.neumann[self.free_indices], self.point_area[self.free_indices]
        )
        return energy - neumann_energy


def load_input() -> pv.UnstructuredGrid:
    surface: pv.PolyData = melon.load_polydata(
        "/home/liblaf/github/liblaf/melon/exp/2025/04/30/human-head-anatomy/Zygomaticus_major001.0.vtp"
    )
    mesh: pv.UnstructuredGrid = melon.tetwild(surface)
    mesh.point_data["point-id"] = np.arange(mesh.n_points)
    mesh.cell_data["cell-id"] = np.arange(mesh.n_cells)

    surface.point_data["__point_id"] = np.arange(surface.n_points)
    surface.cell_data["__cell_id"] = np.arange(surface.n_cells)
    boundary: pv.PolyData = surface.extract_cells(
        surface.cell_data["is-original"], invert=True
    )  # pyright: ignore[reportAssignmentType]
    bodies: pv.MultiBlock = boundary.split_bodies().as_polydata_blocks()
    assert len(bodies) == 2
    neumann = np.zeros((surface.n_points,))
    in_mask = np.zeros((surface.n_points,), dtype=bool)
    in_mask[bodies[0].point_data["__point_id"]] = True
    out_mask = np.zeros((surface.n_points,), dtype=bool)
    out_mask[bodies[1].point_data["__point_id"]] = True
    surface.point_data["in_mask"] = in_mask
    surface.point_data["out_mask"] = out_mask

    in_mask = np.zeros((surface.n_cells,), dtype=bool)
    in_mask[bodies[0].cell_data["__cell_id"]] = True
    out_mask = np.zeros((surface.n_cells,), dtype=bool)
    out_mask[bodies[1].cell_data["__cell_id"]] = True
    surface.cell_data["in_mask"] = in_mask
    surface.cell_data["out_mask"] = out_mask

    neumann[bodies[0].point_data["__point_id"]] = 1.0
    neumann[bodies[1].point_data["__point_id"]] = -1.0
    surface.point_data["neumann"] = neumann

    mesh_surface: pv.PolyData = mesh.extract_surface()
    mesh_surface.compute_normals(auto_orient_normals=True, inplace=True)
    mesh_surface = melon.tri.transfer_cell_data_to_point(
        surface,
        mesh_surface,
        data=["in_mask", "out_mask"],
        fill={"in_mask": False, "out_mask": False},
    )

    mesh = melon.tetra.transfer_point_data_from_surface(
        mesh_surface,
        mesh,
        data=["in_mask", "out_mask"],
        fill={"in_mask": False, "out_mask": False},
    )
    in_point_idx = np.asarray(
        [
            1,
            0,
            2,
            475,
            18,
            304,
            30,
            45,
            58,
            74,
            88,
            93,
            90,
            419,
            82,
            70,
            56,
            507,
            36,
            469,
            11,
            9,
            4,
        ]
    )
    out_cell_idx = np.asarray([928, 926, 924, 919, 910, 909, 906, 908, 892, 881])
    mesh_surface.cell_data["__cell_id"] = np.arange(mesh_surface.n_cells)
    out_surface = mesh_surface.extract_cells(out_cell_idx)
    mesh_surface.cell_data["out_mask"] = np.zeros((mesh_surface.n_cells,), dtype=bool)
    mesh_surface.cell_data["out_mask"][out_surface.cell_data["__cell_id"]] = True

    mesh_surface.point_data["in_mask"][in_point_idx] = True
    # mesh.point_data["in_mask"][mesh_surface.point_data["point-id"][in_point_idx]] = True
    mesh.point_data["in_mask"][in_point_idx] = True

    melon.save("surface.vtp", mesh_surface)
    melon.save("surface.obj", mesh_surface)
    mesh.point_data["out_mask"] = np.zeros((mesh.n_points,), dtype=bool)
    mesh.point_data["out_mask"][out_surface.point_data["point-id"]] = True
    # mesh.cell_data["in_mask"] = np.zeros((mesh.n_cells,), dtype=bool)
    # mesh.cell_data["in_mask"][out_cell_idx] = True
    mesh.point_data["neumann"] = np.zeros((mesh.n_points,))
    mesh.point_data["neumann"][mesh.point_data["in_mask"]] = -1.0
    mesh.point_data["neumann"][mesh.point_data["out_mask"]] = 1.0

    return mesh


def main() -> None:
    mesh = load_input()

    builder = sim_jax.ModelBuilder()
    mesh = builder.assign_dofs(mesh)
    n_dofs: int = mesh.n_points
    # in_mask = mesh.points[:, 0] < -0.49
    # out_mask = mesh.points[:, 0] > 0.49
    # side_mask = ~(in_mask | out_mask)
    dirichlet_mask = mesh.point_data["out_mask"] | mesh.point_data["in_mask"]
    dirichlet_values = jnp.zeros((mesh.n_points,))
    dirichlet_values = dirichlet_values.at[mesh.point_data["in_mask"]].set(-1.0)
    dirichlet_values = dirichlet_values.at[mesh.point_data["out_mask"]].set(1.0)
    dirichlet_indices = jnp.where(dirichlet_mask)[0]
    dirichlet_values = dirichlet_values[dirichlet_indices]
    free_indices = jnp.where(~dirichlet_mask)[0]
    n_dirichlet: int = dirichlet_indices.shape[0]
    n_free: int = n_dofs - n_dirichlet

    surface = mesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    surface = surface.compute_cell_sizes()
    surface = surface.cell_data_to_point_data()  # pyright: ignore[reportAssignmentType]
    ic(surface, surface.point_data)
    area: Array = jnp.zeros((mesh.n_points,))
    area = area.at[surface.point_data["point-ids"]].set(surface.point_data["Area"])
    mesh.point_data["area"] = np.asarray(area)
    # neumann = jnp.zeros((mesh.n_points,))
    # neumann = neumann.at[mesh.point_data["point-ids"][side_mask]].set(0.0)
    # neumann = neumann.at[mesh.point_data["point-ids"][out_mask]].set(1.0)
    # neumann = neumann.at[mesh.point_data["point-ids"][in_mask]].set(-1.0)
    region = sim_jax.Region.from_pyvista(mesh, grad=True)
    # mesh.point_data["neumann"] = np.asarray(neumann)

    energy = Poisson(
        region=region,
        neumann=jnp.asarray(mesh.point_data["neumann"]),
        point_area=area,
        dirichlet_indices=dirichlet_indices,
        free_indices=free_indices,
        dirichlet_values=dirichlet_values,
        n_dofs=n_dofs,
    )
    builder.add_energy(energy)
    model: sim_jax.Model = builder.finish()
    optimizer = optim.MinimizerScipy()
    x0 = jnp.zeros((n_free,))
    ic(model.fun(x0))
    ic(model.jac(x0))
    ic(model.fun_and_jac(x0))
    ic(model.hess_prod(x0, x0))
    solution: optim.Solution = optimizer.minimize(
        x0=jnp.zeros((n_free,)),
        fun=model.fun,
        jac=model.jac,
        fun_and_jac=model.fun_and_jac,
        hessp=model.hess_prod,
    )
    ic(solution)
    u = solution["x"]
    u_full = jnp.zeros((n_dofs,))
    u_full = u_full.at[dirichlet_indices].set(dirichlet_values)
    u_full = u_full.at[free_indices].set(u)
    mesh.point_data["solution"] = np.asarray(u_full)
    grad = region.gradient(u_full)
    mesh.cell_data["gradient"] = np.asarray(grad)
    melon.save("solution.vtu", mesh)


if __name__ == "__main__":
    grapes.logging.init()
    main()
