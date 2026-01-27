import jax.numpy as jnp
import numpy as np
import pyvista as pv
from environs import env
from liblaf.peach.optim import PNCG

from liblaf import cherries, melon
from liblaf.apple.consts import ACTIVATION, GLOBAL_POINT_ID, LAMBDA, MU, MUSCLE_FRACTION
from liblaf.apple.model import Forward, Model, ModelBuilder
from liblaf.apple.warp import WarpArapMuscle


class Config(cherries.BaseConfig):
    lr: float = env.float("LR", 0.02)
    coarsen: bool = env.bool("COARSEN", True)

    # nu = lambda / (2 * (lambda + mu))
    # lambda = 2 * mu * nu / (1 - 2 * nu)
    # lambda= 3.0 -> nu=0.375
    # lambda= 9.0 -> nu=0.45
    # lambda=49.0 -> nu=0.49
    activation: float = env.float("ACTIVATION", 2.0)
    lambda_: float = env.float("LAMBDA", 3.0)
    volume_preserve: bool = env.bool("VOLUME_PRESERVE", True)


def tetgen(cfg: Config) -> pv.UnstructuredGrid:
    surface: pv.PolyData = pv.Box((0.4, 1.6, 0.2, 0.3, 0.2, 0.8), quads=False)
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=cfg.lr, coarsen=cfg.coarsen)
    ic(mesh, surface)
    return mesh


def build_model(mesh: pv.UnstructuredGrid) -> Forward:
    builder = ModelBuilder()
    mesh = builder.add_points(mesh)
    elastic: WarpArapMuscle = WarpArapMuscle.from_pyvista(mesh, name="elastic")
    builder.add_energy(elastic)
    model: Model = builder.finalize()
    forward = Forward(
        model,
        optimizer=PNCG(
            max_delta=0.15 * model.edges_length_mean,
            max_steps=jnp.asarray(2000),
            atol=jnp.asarray(1e-14),
            rtol=jnp.asarray(1e-6),
            rtol_primary=jnp.asarray(1e-6),
            stagnation_max_restarts=jnp.asarray(100),
            jit=True,
        ),
    )
    return forward


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = tetgen(cfg)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), cfg.lambda_)
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)
    mesh.cell_data[MUSCLE_FRACTION] = np.ones((mesh.n_cells,))
    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data[ACTIVATION][:, 0] = cfg.activation - 1.0  # pyright: ignore[reportArgumentType]
    if cfg.volume_preserve:
        mesh.cell_data[ACTIVATION][:, 1] = np.reciprocal(np.sqrt(cfg.activation)) - 1.0  # pyright: ignore[reportArgumentType]
        mesh.cell_data[ACTIVATION][:, 2] = np.reciprocal(np.sqrt(cfg.activation)) - 1.0  # pyright: ignore[reportArgumentType]
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), cfg.lambda_)
    forward: Forward = build_model(mesh)

    # def callback(
    #     objective, model_state: ModelState, opt_state: PNCG.State, opt_stats
    # ) -> None:
    #     ic(opt_state)
    #     print(opt_state.params)
    #     print(model_state.u)
    #     print(model_state.warp.u)
    #     print(opt_state.grad)
    #     print(opt_state.hess_diag)
    #     ic(opt_state.hess_diag.min())
    #     ic(opt_state.hess_quad)
    #     # print(forward.model.hess_diag(model_state).min())
    for _ in range(100):
        forward.model.u_free = jnp.zeros_like(forward.model.u_free)
        solution = forward.step()
        ic(solution)

    mesh.point_data["Solution"] = forward.u_full[mesh.point_data[GLOBAL_POINT_ID]]  # pyright: ignore[reportArgumentType]

    suffix: str = f"-{round(mesh.n_cells / 1000)}k"
    if cfg.coarsen:
        suffix += "-coarsen"
    suffix += f"-lambda{round(cfg.lambda_)}"
    suffix += f"-act{round(cfg.activation)}"
    suffix += "-volpres" if cfg.volume_preserve else "-novolpres"
    melon.save(cherries.output(f"20-forward-muscle{suffix}.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main)
