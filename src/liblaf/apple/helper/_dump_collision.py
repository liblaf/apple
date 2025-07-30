from liblaf.apple import energy, sim


def dump_collision(collision: energy.CollisionVertFace, actor: sim.Actor) -> sim.Actor:
    assert collision.soft.id == actor.id
    candidates: energy.CollisionCandidatesVertFace = collision.candidates
    actor.point_data["collide"] = candidates.collide
    actor.point_data["distance"] = candidates.distance
    return actor
