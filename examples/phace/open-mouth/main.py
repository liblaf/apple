from collections.abc import Mapping
from pathlib import Path

import einops
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import pyvista as pv
import trimesh as tm
import wadler_lindig as wl
import wrapt
from jaxtyping import Array, Bool, Float, Integer
from loguru import logger

from liblaf import cherries, grapes, melon
from liblaf.apple import energy, helper, optim, sim, struct, utils


def pdoc_wrapper(wrapper: wrapt.FunctionWrapper) -> wl.AbstractDoc:
    return wl.bracketed(
        wl.TextDoc("wrapt.FunctionWrapper") + wl.TextDoc("("),
        docs=wl.named_objs(
            [
                ("__wrapped__", wl.pdoc(wrapper.__wrapped__)),
                ("_self_wrapper", wl.pdoc(wrapper._self_wrapper)),
                ("_self_enabled", wl.pdoc(wrapper._self_enabled)),
            ]
        ),
        sep=wl.comma,
        end=wl.TextDoc(")"),
        indent=2,
    )


@grapes.logging.ic_arg_to_string_function.register(wrapt.FunctionWrapper)
def pretty_wrapper(wrapper: wrapt.FunctionWrapper) -> str:
    return wl.pformat(pdoc_wrapper(wrapper))


def flatten_func(func: wrapt.FunctionWrapper) -> tuple:
    return (
        func.__wrapped__,
        # func._self_instance,
        func._self_wrapper,
        func._self_enabled,
        # func._self_binding,
        # func._self_parent,
        # func._self_owner,
    ), ()


def unflatten_func(aux_data: tuple, children: tuple) -> wrapt.FunctionWrapper:
    # wrapped, instance, wrapper, enabled, binding, parent, owner = children
    return wrapt.FunctionWrapper(*children)


jax.tree_util.register_pytree_node(
    wrapt.FunctionWrapper, flatten_func=flatten_func, unflatten_func=unflatten_func
)


class Config(cherries.BaseConfig):
    output_dir: Path = utils.data("animation")
    mesh: Path = utils.data("head.vtu")
    duration: float = 1.0
    fps: float = 120.0

    d_hat: float = 0.1
    density: float = 1
    lambda_: float = 3 * 1e3
    mu: float = 1 * 1e3

    @property
    def n_frames(self) -> int:
        return int(self.duration * self.fps)

    @property
    def time_step(self) -> float:
        return 1 / self.fps


def gen_scene(cfg: Config) -> sim.Scene:
    head_pv: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.mesh)
    head_pv.cell_data["density"] = cfg.density
    head_pv.cell_data["lambda"] = cfg.lambda_
    head_pv.cell_data["mu"] = cfg.mu
    head: sim.Actor = sim.Actor.from_pyvista(head_pv, grad=True, id_="head")
    head = helper.add_point_mass(head)
    is_skull: Bool[Array, " P"] = head_pv.point_data["is-skull"]
    head = head.with_dirichlet(
        sim.Dirichlet.from_mask(
            einops.repeat(is_skull, "P -> P D", D=3),
            values=jnp.zeros((head.n_points, 3), dtype=jnp.float32),
        )
    )

    builder = sim.SceneBuilder()
    head = builder.assign_dofs(head)
    builder.add_energy(energy.PhaceStatic.from_actor(head))

    return builder.finish()


def update_dirichlet(scene: sim.Scene, rotate_rad: float) -> sim.Scene:
    mask: Bool[Array, " DOF"] = jnp.zeros((scene.n_dofs,), dtype=bool)
    values: Float[Array, " DOF"] = jnp.zeros((scene.n_dofs,))
    actors: struct.NodeContainer[sim.Actor] = scene.actors
    for actor in scene.actors.values():
        actor: sim.Actor
        if actor.id == "head":
            skull_mask: Bool[Array, " P"] = actor.point_data["is-skull"]
            mandible_mask: Bool[Array, " P"] = actor.point_data["is-mandible"]
            points: Float[Array, "P 3"] = actor.points
            matrix: Float[np.ndarray, "4 4"] = tm.transformations.rotation_matrix(
                rotate_rad, [1.0, 0.0, 0.0], point=[0.69505, 29.141, 0.8457]
            )  # pyright: ignore[reportAssignmentType]
            points = points.at[mandible_mask].set(
                tm.transform_points(points[mandible_mask], matrix)
            )
            disp: Float[Array, "P 3"] = points - actor.points
            actor = actor.with_dirichlet(  # noqa: PLW2901
                sim.Dirichlet.from_mask(
                    mask=einops.repeat(skull_mask, "P -> P D", D=3), values=disp
                )
            )
        actors.add(actor)
        dofs: Integer[Array, " DOF"] = jnp.asarray(actor.dofs_global)
        idx: Integer[Array, " dirichlet"] = actor.dirichlet_local.dofs.get(dofs).ravel()
        mask = mask.at[idx].set(True)
        values = values.at[idx].set(actor.dirichlet_local.values.ravel())
    return scene.replace(actors=actors, dirichlet=sim.Dirichlet.from_mask(mask, values))


def main(cfg: Config) -> None:
    scene: sim.Scene = gen_scene(cfg)
    optimizer = optim.PNCG(d_hat=cfg.d_hat, maxiter=10**3, rtol=1e-5)
    head: sim.Actor = scene.actors["head"]

    jaw_rotate_total: float = np.deg2rad(30.0)
    jaw_rotate: np.ndarray = np.linspace(0.0, jaw_rotate_total, num=cfg.n_frames + 1)

    writers: Mapping[str, melon.SeriesWriter] = {
        head.id: melon.SeriesWriter(
            cfg.output_dir / f"{head.id}.vtu.series", fps=cfg.fps
        ),
    }
    actors: struct.NodeContainer[sim.Actor] = helper.dump_actors(scene)
    meshes: Mapping[str, pv.DataSet] = helper.actors_to_pyvista(actors)
    for id_, writer in writers.items():
        writer.append(meshes[id_], time=0.0)
    for t in range(1, cfg.n_frames + 1):
        scene = update_dirichlet(scene, rotate_rad=jaw_rotate[t])
        result: optim.OptimizeResult
        scene, result = scene.solve(optimizer=optimizer)
        scene = scene.step(result["x"])
        if not result["success"]:
            logger.error("{}", result)
        scene = scene.step(result["x"])
        actors: struct.NodeContainer[sim.Actor] = helper.dump_actors(
            scene, result=result
        )
        meshes: Mapping[str, pv.DataSet] = helper.actors_to_pyvista(actors)
        for id_, writer in writers.items():
            writer.append(meshes[id_], time=t * cfg.time_step)


if __name__ == "__main__":
    cherries.run(main, profile="playground")
