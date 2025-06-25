import einops
import jax
from jaxtyping import Float

from liblaf.apple import sim


def add_point_mass[T: sim.Actor](actor: T) -> T:
    if "mass" in actor.point_data:
        return actor
    density: Float[jax.Array, " cells"] = actor.cell_data["density"]
    dV: Float[jax.Array, " cells"] = actor.region.integrate(actor.region.dV)
    mass: Float[jax.Array, " cells"] = density * dV
    mass: Float[jax.Array, "cells a"] = (
        einops.repeat(mass, "c -> c a", a=actor.element.n_points) / 4
    )
    mass: Float[jax.Array, " points"] = actor.region.gather(mass)
    return actor.set_point_data("mass", mass)
