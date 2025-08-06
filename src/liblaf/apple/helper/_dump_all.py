import pyvista as pv

from liblaf.apple import energy as _energy
from liblaf.apple import optim, sim, struct

from ._dump_collision import dump_collision
from ._dump_optim import dump_optim_result


def dump_all_actors(
    scene: sim.Scene, result: optim.OptimizeResult | None = None
) -> struct.NodeContainer[sim.Actor]:
    actors: struct.NodeContainer[sim.Actor] = scene.actors
    if result is not None:
        for actor in actors.values():
            actor_new: sim.Actor = dump_optim_result(
                scene=scene, actor=actor, result=result
            )
            if actor_new.dofs_global is not None:
                actor_new.point_data["dofs-global"] = (
                    actor_new.dofs_global.index.reshape(actor_new.points.shape)
                )
            if "collide" in actor_new.point_data:
                del actor_new.point_data["collide"]
            if "distance" in actor_new.point_data:
                del actor_new.point_data["distance"]
            actors.add(actor_new)
    for energy in scene.energies.values():
        if isinstance(energy, _energy.CollisionVertFace):
            actor: sim.Actor = actors[energy.soft.id]
            actor: sim.Actor = dump_collision(collision=energy, actor=actor)
            actors.add(actor)
    return actors


def dump_all_pyvista(
    scene: sim.Scene, result: optim.OptimizeResult | None = None
) -> dict[str, pv.DataSet]:
    actors: struct.NodeContainer[sim.Actor] = dump_all_actors(
        scene=scene, result=result
    )
    meshes: dict[str, pv.DataSet] = {
        k: v.to_pyvista(attributes=True) for k, v in actors.items()
    }
    for mesh in meshes.values():
        if "displacement" in mesh.point_data:
            mesh.warp_by_vector("displacement", inplace=True)
    return meshes
