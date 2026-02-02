import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyvista as pv
import warp as wp
from environs import env
from liblaf.peach.optim import PNCG, Objective, Optax, Optimizer, ScipyOptimizer

from liblaf import cherries, melon
from liblaf.apple.consts import ACTIVATION, GLOBAL_POINT_ID, LAMBDA
from liblaf.apple.model import Forward, Model, ModelBuilder, ModelState
from liblaf.apple.warp.energies.elastic import (
    # WarpArapMuscle,
    WarpPhaceV2,
    # WarpVolumePreservationDeterminant,
)

# wp.config.mode = "debug"
wp.config.print_launches = True
wp.config.verbose = True
wp.config.max_unroll = 0
wp.init()


class Config(cherries.BaseConfig):
    # nu = lambda / (2 * (lambda + mu))
    # lambda = 2 * mu * nu / (1 - 2 * nu)
    # lambda= 3.0 -> nu=0.375
    # lambda= 9.0 -> nu=0.45
    # lambda=49.0 -> nu=0.49
    lambda_: float = env.float("LAMBDA", 3.0)
    activation: float = env.float("ACTIVATION", 2.0)
    suffix: str = env.str("SUFFIX", "-4k-coarse")
    solver: str = env.str("SOLVER", "pncg")


def get_solver(name: str, model: Model) -> Optimizer:
    match name:
        case "pncg":
            return PNCG(
                max_delta=0.15 * model.edges_length_mean,
                max_steps=jnp.asarray(1000),
                rtol=jnp.asarray(1e-5),
                rtol_primary=jnp.asarray(1e-10),
                stagnation_max_restarts=jnp.asarray(100),
                jit=True,
            )
        case "sgd":
            return Optax(optax.sgd(0.1))
        case name:
            return ScipyOptimizer(method=name, options={"verbose": 3})


def build_model(mesh: pv.UnstructuredGrid, solver: str) -> Forward:
    builder = ModelBuilder()
    mesh = builder.add_points(mesh)
    elastic: WarpPhaceV2 = WarpPhaceV2.from_pyvista(mesh)
    builder.add_energy(elastic)
    model: Model = builder.finalize()
    forward = Forward(model, optimizer=get_solver(solver, model))
    return forward


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(
        cherries.input(f"10-input{cfg.suffix}.vtu")
    )
    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data[ACTIVATION][:, 0] = cfg.activation - 1.0  # pyright: ignore[reportArgumentType]
    mesh.cell_data[ACTIVATION][:, 1] = cfg.activation**-0.5 - 1.0  # pyright: ignore[reportArgumentType]
    mesh.cell_data[ACTIVATION][:, 2] = cfg.activation**-0.5 - 1.0  # pyright: ignore[reportArgumentType]
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), cfg.lambda_)
    forward: Forward = build_model(mesh, cfg.solver)

    def callback(
        _objective: Objective,
        _model_state: ModelState,
        opt_state: PNCG.State,
        _opt_stats: PNCG.Stats,
    ) -> None:
        # ic(opt_state.hess_diag.min())
        ic(opt_state)

    # warmup
    # u_wp: wp.array = wp.zeros((forward.model.n_points,), dtype=wp.vec3f)
    # grad_wp: wp.array = wp.zeros((forward.model.n_points,), dtype=wp.vec3f)
    # hess_diag_wp: wp.array = wp.zeros((forward.model.n_points,), dtype=wp.vec3f)
    # forward.model.warp.__wrapped__.grad(forward.state.warp.__wrapped__, u_wp, grad_wp)
    # print(grad_wp)
    # forward.model.warp.__wrapped__.hess_diag(
    #     forward.state.warp.__wrapped__, u_wp, hess_diag_wp
    # )
    # print(hess_diag_wp)
    # forward.model.warp.__wrapped__.grad(forward.state.warp.__wrapped__, u_wp, grad_wp)
    # print(grad_wp)
    # forward.model.warp.__wrapped__.hess_diag(
    #     forward.state.warp.__wrapped__, u_wp, hess_diag_wp
    # )
    # print(hess_diag_wp)
    # print(forward.model.grad(forward.state))
    # print(forward.model.hess_diag(forward.state))
    # print(forward.model.hess_quad(forward.state, forward.state.u))
    # print(forward.model.grad(forward.state))
    # print(forward.model.hess_diag(forward.state))
    # print(forward.model.hess_quad(forward.state, forward.state.u))
    forward.step(callback=callback)
    mesh.point_data["Solution"] = forward.u_full[mesh.point_data[GLOBAL_POINT_ID]]  # pyright: ignore[reportArgumentType]

    suffix: str = cfg.suffix
    suffix += f"-act{round(cfg.activation)}"
    suffix += f"-lambda{round(cfg.lambda_)}"
    suffix += "-float64" if jax.config.read("jax_enable_x64") else "-float32"
    suffix += f"-{cfg.solver}"
    melon.save(cherries.output(f"20-forward{suffix}.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
