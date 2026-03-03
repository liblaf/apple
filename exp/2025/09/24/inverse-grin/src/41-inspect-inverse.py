import jax.numpy as jnp
import liblaf.apple.jax.sim as sim_jax
import numpy as np
import pyvista as pv
from jaxtyping import Array, Float

from liblaf import melon


def main() -> None:
    target: pv.UnstructuredGrid = melon.load_unstructured_grid("data/11-input.vtu")
    solution: pv.UnstructuredGrid = melon.load_unstructured_grid(
        "data/36-inverse.vtu.d/36-inverse_000036.vtu"
    )
    orientation: Float[Array, "c 3 3"] = jnp.asarray(
        target.cell_data["muscle-orientation"]
    ).reshape(-1, 3, 3)
    for mid in range(6):
        muscle_mask = target.cell_data["muscle-id"] == mid
        muscle_activation_actual = jnp.mean(
            sim_jax.transform_activation(
                jnp.asarray(solution.cell_data["activation"][muscle_mask]),
                orientation[muscle_mask],
                inverse=True,
            ),
            axis=0,
        )
        muscle_activation_desired = jnp.mean(
            sim_jax.transform_activation(
                jnp.asarray(target.cell_data["activation"][muscle_mask]),
                orientation[muscle_mask],
                inverse=True,
            ),
            axis=0,
        )
        act = np.asarray([muscle_activation_actual, muscle_activation_desired])
        print(f"muscle {mid}:\n{act}")


if __name__ == "__main__":
    main()
