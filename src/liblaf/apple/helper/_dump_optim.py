import jax.numpy as jnp

from liblaf.apple import optim, sim, utils


def dump_optim_result(
    scene: sim.Scene, actor: sim.Actor, result: optim.OptimizeResult
) -> sim.Actor:
    for key, value in result.items():
        if utils.is_array(value) and jnp.size(value) == scene.n_dofs:
            actor = actor.set_point_data(key, actor.dofs.get(value))
        elif utils.is_scalar(value):
            actor = actor.set_field_data(key, value)
    return actor
