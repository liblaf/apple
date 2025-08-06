import jax.numpy as jnp

from liblaf.apple import energy, sim


def dump_collision(collision: energy.CollisionVertFace, actor: sim.Actor) -> sim.Actor:
    assert collision.soft.id == actor.id
    candidates: energy.CollisionCandidatesVertFace = collision.candidates
    if candidates.collide is None:
        return actor
    if "collide" in actor.point_data:
        actor.point_data["collide"] |= candidates.collide
    else:
        actor.point_data["collide"] = candidates.collide
    if "distance" in actor.point_data:
        actor.point_data["distance"] = jnp.minimum(
            candidates.distance, actor.point_data["distance"]
        )
    else:
        actor.point_data["distance"] = candidates.distance
    return actor
