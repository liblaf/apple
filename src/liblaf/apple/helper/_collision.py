from liblaf.apple import energy, sim


def dump_collision(actor: sim.Actor, collision: energy.CollisionVertFace) -> sim.Actor:
    candidates: energy.CollisionCandidatesVertFace = collision.candidates
    actor = actor.set_point_data("collide", candidates.collide)
    actor = actor.set_point_data("distance", candidates.distance)
    return actor
