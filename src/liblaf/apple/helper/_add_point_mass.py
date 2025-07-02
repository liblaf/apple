import einops
from jaxtyping import Array, Float

from liblaf.apple import sim


def add_point_mass[T: sim.Actor](actor: T) -> T:
    if "mass" in actor.point_data:
        return actor
    density: Float[Array, " cells"] = actor.cell_data["density"]
    density: Float[Array, "cells q"] = density[:, None]
    cell_mass: Float[Array, " cells"] = actor.region.integrate(density)
    cell_mass: Float[Array, "cells a"] = (
        einops.repeat(cell_mass, "c -> c a", a=actor.element.n_points) / 4
    )
    point_mass: Float[Array, " points"] = actor.region.gather(cell_mass)
    return actor.set_point_data("mass", point_mass)
