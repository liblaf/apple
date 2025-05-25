import beartype
import einops
import jax
import jax.numpy as jnp
import jaxtyping
import pyvista as pv
from jaxtyping import Float, Integer

from liblaf.apple import utils

from ._fem import dV as compute_dV
from ._sum import segment_sum


def mass(mesh: pv.UnstructuredGrid) -> Float[jax.Array, " P"]:
    return mass_points(
        points=jnp.asarray(mesh.points),
        cells=jnp.asarray(mesh.cells_dict[pv.CellType.TETRA]),
        density=jnp.asarray(mesh.cell_data["density"]),
        n_points=mesh.n_points,
    )


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@utils.jit(static_argnames=["n_points"])
def mass_points(
    points: Float[jax.Array, "P 3"],
    cells: Integer[jax.Array, "C 4"],
    density: Float[jax.Array, " C"],
    n_points: int,
) -> Float[jax.Array, " P"]:
    dV: Float[jax.Array, " C"] = compute_dV(points[cells])
    dm: Float[jax.Array, " C"] = density * dV
    dm: Float[jax.Array, "C 4"] = einops.repeat(0.25 * dm, "C -> C 4")
    dm: Float[jax.Array, " P"] = segment_sum(dm, cells=cells, n_points=n_points)
    return dm
