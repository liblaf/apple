import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float

from liblaf.apple import sim

DEFAULT_GRAVITY: Float[jax.Array, " dim"] = jnp.asarray([0.0, -9.8, 0.0])


def add_gravity[T: sim.Actor](actor: T, gravity: ArrayLike = DEFAULT_GRAVITY) -> T:
    mass: Float[jax.Array, " points"] = actor.point_data["mass"]
    force: Float[jax.Array, "points dim"]
    if "force-ext" in actor.point_data:
        force = actor.point_data["force-ext"]
    else:
        force = jnp.zeros((actor.n_points, actor.dim))
    gravity = jnp.asarray(gravity)
    force += mass[:, jnp.newaxis] * gravity[jnp.newaxis, :]
    actor.point_data["force-ext"] = force
    return actor


def clear_force[T: sim.Actor](actor: T) -> T:
    force: Float[jax.Array, "points dim"] = jnp.zeros((actor.n_points, actor.dim))
    actor.point_data["force-ext"] = force
    return actor
