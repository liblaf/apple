import beartype
import einops
import jax
import jax.numpy as jnp
import jaxtyping
import pyvista as pv
from jaxtyping import Float, Integer

from liblaf import apple


def mass(mesh: pv.UnstructuredGrid) -> Float[jax.Array, " P"]:
    return mass_points(
        points=jnp.asarray(mesh.points),
        tetras=jnp.asarray(mesh.cells_dict[pv.CellType.TETRA]),
        density=jnp.asarray(mesh.cell_data["density"]),
        n_points=mesh.n_points,
    )


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@apple.jit(static_argnames=["n_points"])
def mass_points(
    points: Float[jax.Array, "P 3"],
    tetras: Integer[jax.Array, "C 4"],
    density: Float[jax.Array, " C"],
    n_points: int,
) -> Float[jax.Array, " P"]:
    dV: Float[jax.Array, " C"] = apple.elem.tetra.dV(points[tetras])
    dm: Float[jax.Array, " C"] = density * dV
    dm: Float[jax.Array, "C 4"] = einops.repeat(0.25 * dm, "C -> C 4")
    dm: Float[jax.Array, " P"] = apple.elem.tetra.segment_sum(
        dm, tetras=tetras, n_points=n_points
    )
    return dm
