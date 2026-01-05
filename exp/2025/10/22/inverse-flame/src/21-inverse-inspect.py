from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Bool, Float

from liblaf import cherries, melon
from liblaf.apple.constants import ACTIVATION
from liblaf.apple.jax import make_activation


class Config(cherries.BaseConfig):
    muscle_names: Sequence[str] = [
        "Levator_labii_superioris001_00",
        "Orbicularis_oris001",
        "Zygomaticus_major001_00",
        "Zygomaticus_minor001_00",
    ]


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(
        cherries.temp(
            "20-inverse-adam-Expression002-515k.vtu.d/20-inverse-adam-Expression002-515k_001041.vtu"
        )
    )
    muscle_names: list[str] = mesh.field_data["MuscleName"].tolist()
    mask: Bool[Array, " c"] = jnp.zeros((mesh.n_cells,), dtype=bool)
    for name in cfg.muscle_names:
        idx = muscle_names.index(name)
        mask |= mesh.cell_data["MuscleId"] == idx
    mask &= mesh.cell_data["MuscleFraction"] > 1e-3
    mesh = melon.tet.extract_cells(mesh, mask)  # pyright: ignore[reportArgumentType]
    activations: Float[Array, "c 6"] = jnp.asarray(mesh.cell_data[ACTIVATION])
    muscle_mask: Bool[Array, " c"] = jnp.asarray(
        mesh.cell_data["MuscleFraction"] > 1e-3
    )
    activations: Float[Array, "m 6"] = activations[muscle_mask]
    activations: Float[Array, "m 3 3"] = make_activation(activations)
    eigenvalues: Float[Array, "m 3"]
    eigenvectors: Float[Array, "m 3 3"]
    eigenvalues, eigenvectors = jnp.linalg.eigh(activations)
    eigenvectors = jnp.take_along_axis(
        eigenvectors, jnp.argsort(eigenvalues, axis=-1)[:, None, :], axis=-1
    )
    mesh.cell_data["ActivationDirection0"] = np.asarray(
        jnp.zeros((mesh.n_cells, 3)).at[muscle_mask].set(eigenvectors[:, :, 0])
    )
    mesh.cell_data["ActivationDirection1"] = np.asarray(
        jnp.zeros((mesh.n_cells, 3)).at[muscle_mask].set(eigenvectors[:, :, 1])
    )
    mesh.cell_data["ActivationDirection2"] = np.asarray(
        jnp.zeros((mesh.n_cells, 3)).at[muscle_mask].set(eigenvectors[:, :, 2])
    )
    mesh.cell_data["ActivationMagnitude"] = np.asarray(
        jnp.zeros((mesh.n_cells, 3))
        .at[muscle_mask]
        .set(jnp.linalg.norm(activations, axis=-1))
    )
    mesh.cell_data["ActivationEigenvalue0"] = np.asarray(
        jnp.zeros((mesh.n_cells,)).at[muscle_mask].set(eigenvalues[:, 0])
    )
    mesh.cell_data["ActivationEigenvalue1"] = np.asarray(
        jnp.zeros((mesh.n_cells,)).at[muscle_mask].set(eigenvalues[:, 1])
    )
    mesh.cell_data["ActivationEigenvalue2"] = np.asarray(
        jnp.zeros((mesh.n_cells,)).at[muscle_mask].set(eigenvalues[:, 2])
    )

    melon.save(cherries.temp("21-inverse-inspect.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
