import beartype
import einops
import jax
import jax.numpy as jnp
import jaxtyping
import pyvista as pv
from jaxtyping import Float, Integer

from liblaf import apple

from . import area, first_fundamental_form, segment_sum


def mass(mesh: pv.PolyData) -> Float[jax.Array, " P"]:
    return mass_points(
        points=jnp.asarray(mesh.points),
        cells=jnp.asarray(mesh.regular_faces),
        density=jnp.asarray(mesh.cell_data["density"]),
        thickness=jnp.asarray(mesh.cell_data["thickness"]),
        n_points=mesh.n_points,
    )


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@apple.jit(static_argnames=["n_points"])
def mass_points(
    points: Float[jax.Array, "P 3"],
    cells: Integer[jax.Array, "C 3"],
    density: Float[jax.Array, " C"],
    thickness: Float[jax.Array, " C"],
    n_points: int,
) -> Float[jax.Array, " P"]:
    I_u: Float[jax.Array, " C"] = first_fundamental_form(points[cells])
    dA: Float[jax.Array, " C"] = area(I_u)
    dm: Float[jax.Array, " C"] = density * thickness * dA
    dm: Float[jax.Array, "C 4"] = einops.repeat(dm / 3, "C -> C 3")
    dm: Float[jax.Array, " P"] = segment_sum(dm, cells=cells, n_points=n_points)
    return dm
