from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Float

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    input: Path = cherries.input("10-input.vtu")

    output: Path = cherries.output("31-animation.vtp.series")


def main(cfg: Config) -> None:
    start_idx: str = "00"
    end_idx: str = "01"

    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    surface: pv.PolyData = tetmesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    surface = surface.extract_points(surface.point_data["is-face"]).extract_surface()
    start_disp: Float[np.ndarray, "p 3"] = surface.point_data[f"expression-{start_idx}"]
    end_disp: Float[np.ndarray, "p 3"] = surface.point_data[f"expression-{end_idx}"]

    with melon.SeriesWriter(cfg.output) as writer:
        for t in jnp.linspace(0.0, 1.0, num=30):
            disp: Float[np.ndarray, "p 3"] = (1.0 - t) * start_disp + t * end_disp
            surface.point_data["displacement"] = np.asarray(disp)
            writer.append(surface)


if __name__ == "__main__":
    cherries.main(main)
