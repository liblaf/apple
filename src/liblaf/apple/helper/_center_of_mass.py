import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.apple import sim


def center_of_mass_displacement(actor: sim.Actor) -> Float[Array, "3"]:
    return average_at_center_of_mass(actor, "displacement")


def center_of_mass_velocity(actor: sim.Actor) -> Float[Array, "3"]:
    return average_at_center_of_mass(actor, "velocity")


def average_at_center_of_mass(
    actor: sim.Actor, point_data_name: str
) -> Float[Array, "..."]:
    return jnp.average(actor.point_data[point_data_name], weights=actor.mass, axis=0)
