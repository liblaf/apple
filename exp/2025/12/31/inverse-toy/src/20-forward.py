import jax
import numpy as np
import optax
import pyvista as pv
from environs import env
from liblaf.peach.optim import PNCG, Optax, Optimizer, ScipyOptimizer

from liblaf import cherries, melon
from liblaf.apple.consts import ACTIVATION, GLOBAL_POINT_ID, LAMBDA
from liblaf.apple.model import Forward, Model, ModelBuilder
from liblaf.apple.warp import Phace


class Config(cherries.BaseConfig):
    # nu = lambda / (2 * (lambda + mu))
    # lambda = 2 * mu * nu / (1 - 2 * nu)
    # lambda= 3.0 -> nu=0.375
    # lambda= 9.0 -> nu=0.45
    # lambda=49.0 -> nu=0.49
    lambda_: float = env.float("LAMBDA", 3.0)
    activation: float = env.float("ACTIVATION", 8.0)
    suffix: str = env.str("SUFFIX", "-4k-coarse-conform")
    solver: str = env.str("SOLVER", "pncg")


def get_solver(name: str, model: Model) -> Optimizer:
    match name:
        case "pncg":
            return PNCG(
                max_delta=0.15 * model.edges_length_mean,
                max_steps=5000,
                rtol=1e-5,
                rtol_primary=1e-13,
                stagnation_max_restarts=100,
            )
        case "sgd":
            return Optax(optax.sgd(1e-1))
        case name:
            return ScipyOptimizer(method=name)


def build_model(mesh: pv.UnstructuredGrid, solver: str) -> Forward:
    builder = ModelBuilder()
    mesh = builder.add_points(mesh)
    elastic: Phace = Phace.from_pyvista(mesh)
    builder.add_energy(elastic)
    model: Model = builder.finalize()
    forward = Forward(model, optimizer=get_solver(solver, model))
    return forward


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(
        cherries.input(f"10-input{cfg.suffix}.vtu")
    )
    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data[ACTIVATION][:, 0] = cfg.activation  # pyright: ignore[reportArgumentType]
    mesh.cell_data[ACTIVATION][:, 1] = cfg.activation**-0.5  # pyright: ignore[reportArgumentType]
    mesh.cell_data[ACTIVATION][:, 2] = cfg.activation**-0.5  # pyright: ignore[reportArgumentType]
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), cfg.lambda_)
    forward: Forward = build_model(mesh, cfg.solver)
    forward.step()
    mesh.point_data["Solution"] = forward.u_full[mesh.point_data[GLOBAL_POINT_ID]]  # pyright: ignore[reportArgumentType]

    suffix: str = cfg.suffix
    suffix += f"-act{round(cfg.activation)}"
    suffix += f"-lambda{round(cfg.lambda_)}"
    suffix += "-float64" if jax.config.read("jax_enable_x64") else "-float32"
    suffix += f"-{cfg.solver}"
    melon.save(cherries.output(f"20-forward{suffix}.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
