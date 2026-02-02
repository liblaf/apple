from pathlib import Path

import numpy as np
import pyvista as pv
from environs import env
from jaxtyping import Bool
from liblaf.peach.optim import ScipyOptimizer

from liblaf import cherries, melon
from liblaf.apple.consts import (
    DIRICHLET_MASK,
    DIRICHLET_VALUE,
    GLOBAL_POINT_ID,
    MUSCLE_FRACTION,
    STIFFNESS,
)
from liblaf.apple.jax import JaxMassSpring
from liblaf.apple.model import Forward, Model, ModelBuilder


class Config(cherries.BaseConfig):
    activation: float = env.float("ACTIVATION", 2.0)
    input: Path = cherries.input("10-subface.vtp")


def load_mesh(cfg: Config) -> pv.PolyData:
    mesh: pv.PolyData = melon.load_polydata(cfg.input)
    mesh.cell_data[STIFFNESS] = np.full((mesh.n_cells,), 1.0)
    mesh = melon.tri.cell_data_to_point_data(mesh, [MUSCLE_FRACTION])
    muscle_mask: Bool[np.ndarray, " points"] = mesh.point_data[MUSCLE_FRACTION] > 1e-2
    mesh.point_data[DIRICHLET_MASK] |= muscle_mask[:, np.newaxis]
    mesh.point_data[DIRICHLET_VALUE][muscle_mask, 0] = (  # pyright: ignore[reportArgumentType]
        1.0 / cfg.activation - 1.0
    ) * mesh.points[muscle_mask, 0]
    return mesh


def build_model(mesh: pv.PolyData) -> Model:
    builder = ModelBuilder()
    # mesh = builder.add_points(mesh)
    builder.add_dirichlet(mesh)
    edges: pv.PolyData = mesh.extract_all_edges()  # pyright: ignore[reportAssignmentType]
    elastic: JaxMassSpring = JaxMassSpring.from_pyvista(edges)
    builder.add_energy(elastic)
    model: Model = builder.finalize()
    return model


def main(cfg: Config) -> None:
    mesh: pv.PolyData = load_mesh(cfg)
    model: Model = build_model(mesh)
    forward: Forward = Forward(model, optimizer=ScipyOptimizer(method="L-BFGS-B"))
    forward.step()
    mesh.point_data["Solution"] = np.asarray(
        forward.u_full[mesh.point_data[GLOBAL_POINT_ID]]
    )
    suffix: str = f"-act{cfg.activation:.0f}"
    melon.save(cherries.output(f"20-forward-subface-base{suffix}.vtp"), mesh)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
