from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from environs import env
from jaxtyping import Array, Bool, Float, Integer

from liblaf import cherries, melon
from liblaf.apple.consts import ACTIVATION, POINT_ID
from liblaf.apple.model import Forward, Model, ModelBuilder
from liblaf.apple.warp import Phace


class Config(cherries.BaseConfig):
    suffix: str = env.str("SUFFIX", default="-3152k")

    muscles: Path = cherries.input(
        "/home/liblaf/github/liblaf/melon/exp/2025/04/30/human-head-anatomy/data/30-muscles.vtm"
    )


def compute_muscle_orientation(cfg: Config) -> dict[str, pv.PolyData]:
    muscles: pv.MultiBlock = pv.read(cfg.muscles)  # pyright: ignore[reportAssignmentType]
    for muscle in muscles:
        muscle: pv.PolyData
        components: Float[np.ndarray, " 3"] = melon.as_trimesh(
            muscle
        ).principal_inertia_components
        vectors: Float[np.ndarray, "3 3"] = melon.as_trimesh(
            muscle
        ).principal_inertia_vectors
        index: Integer[np.ndarray, " 3"] = np.argsort(components)
        muscle.field_data["MomentInertia"] = melon.as_trimesh(muscle).moment_inertia
        muscle.field_data["PrincipalInertiaVectors"] = vectors[index]
        muscle.field_data["PrincipalInertiaComponents"] = components[index]
    return {muscle.field_data["MuscleName"].item(): muscle for muscle in muscles}


def gen_activation(
    mesh: pv.UnstructuredGrid, muscles: dict[str, pv.PolyData]
) -> pv.UnstructuredGrid:
    muscle_names: list[str] = mesh.field_data["MuscleName"].tolist()
    # -
    # for muscle_name, gamma in [
    #     ("Levator_labii_superioris_alaeque_nasi001_00", 20.0),
    #     ("Levator_labii_superioris_alaeque_nasi001_01", 20.0),
    #     ("Levator_labii_superioris001_00", 30.0),
    #     ("Levator_labii_superioris001_01", 30.0),
    #     ("Risorius001_00", 10.0),
    #     ("Risorius001_01", 10.0),
    #     ("Zygomaticus_major001_00", 5.0),
    #     ("Zygomaticus_major001_01", 5.0),
    #     ("Zygomaticus_minor001_00", 10.0),
    #     ("Zygomaticus_minor001_01", 10.0),
    # ]:
    for muscle_name, gamma in [
        ("Levator_labii_superioris_alaeque_nasi001_00", 20.0),
        ("Levator_labii_superioris_alaeque_nasi001_01", 20.0),
        ("Levator_labii_superioris001_00", 30.0),
        ("Levator_labii_superioris001_01", 30.0),
        ("Risorius001_00", 10.0),
        ("Risorius001_01", 10.0),
        ("Zygomaticus_major001_00", 5.0),
        ("Zygomaticus_major001_01", 5.0),
        ("Zygomaticus_minor001_00", 10.0),
        ("Zygomaticus_minor001_01", 10.0),
    ]:
        activation_local: Float[Array, "3 3"] = jnp.diagflat(
            jnp.asarray([gamma, gamma**-0.5, gamma**-0.5])
        )
        muscle_id: int = muscle_names.index(muscle_name)
        muscle: pv.PolyData = muscles[muscle_name]
        muscle_orientation: Float[np.ndarray, "3 3"] = muscle.field_data[
            "PrincipalInertiaVectors"
        ]
        activation_global: Float[np.ndarray, "3 3"] = (
            muscle_orientation.mT @ activation_local @ muscle_orientation
        )
        activation_params: Float[Array, " 6"] = jnp.asarray(
            [
                activation_global[0, 0] - 1.0,
                activation_global[1, 1] - 1.0,
                activation_global[2, 2] - 1.0,
                activation_global[0, 1],
                activation_global[0, 2],
                activation_global[1, 2],
            ]
        )
        cell_mask: Bool[Array, " cells"] = mesh.cell_data["MuscleId"] == muscle_id
        mesh.cell_data[ACTIVATION][cell_mask] = jnp.broadcast_to(  # pyright: ignore[reportArgumentType]
            activation_params[jnp.newaxis, :], (jnp.count_nonzero(cell_mask), 6)
        )
    return mesh


def forward(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    builder = ModelBuilder()
    mesh = builder.assign_global_ids(mesh)
    builder.add_dirichlet(mesh)
    elastic = Phace.from_pyvista(mesh)
    builder.add_energy(elastic)
    model: Model = builder.finalize()
    forward = Forward(model)
    forward.step()
    mesh.point_data["Expression004"] = model.u_full[mesh.point_data[POINT_ID]]  # pyright: ignore[reportArgumentType]
    return mesh


def main(cfg: Config) -> None:
    muscles: dict[str, pv.PolyData] = compute_muscle_orientation(cfg)
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(
        cherries.input(f"10-input{cfg.suffix}.vtu")
    )
    mesh = gen_activation(mesh, muscles)
    mesh = forward(mesh)
    melon.save(cherries.output(f"10-input-manual{cfg.suffix}.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main)
