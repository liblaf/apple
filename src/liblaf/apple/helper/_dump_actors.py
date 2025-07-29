from collections.abc import Mapping

import pyvista as pv

from liblaf.apple import energy as _energy
from liblaf.apple import optim, sim, struct

from ._dump_optim import dump_optim_result


def dump_actors(
    scene: sim.Scene, result: optim.OptimizeResult | None = None
) -> struct.NodeContainer[sim.Actor]:
    actors: struct.NodeContainer[sim.Actor] = struct.NodeContainer()
    for actor_old in scene.actors.values():
        actor: sim.Actor = actor_old.update(
            displacement=actor_old.dofs_global.get(scene.state.displacement),
            velocity=actor_old.dofs_global.get(scene.state.velocity),
        )
        if result is not None:
            actor = dump_optim_result(scene, actor, result)
        actors.add(actor)
    for energy in scene.energies.values():
        if isinstance(energy, _energy.CollisionVertFace):
            actor: sim.Actor = actors[energy.soft.id]
            if energy.candidates.collide is not None:
                actor = actor.set_point_data("collide", energy.candidates.collide)
            if energy.candidates.distance is not None:
                actor = actor.set_point_data("distance", energy.candidates.distance)
            actors.add(actor)
    return actors


def actors_to_pyvista(
    actors: struct.NodeContainer[sim.Actor],
) -> Mapping[str, pv.DataSet]:
    meshes: dict[str, pv.DataSet] = {}
    for actor in actors.values():
        mesh: pv.DataSet = actor.to_pyvista(attributes=True)
        if "displacement" in mesh.point_data:
            mesh = mesh.warp_by_vector("displacement")
        meshes[actor.id] = mesh
    return meshes
