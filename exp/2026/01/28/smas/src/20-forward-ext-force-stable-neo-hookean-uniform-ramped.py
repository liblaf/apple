from pathlib import Path
from typing import cast

import attrs
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import warp as wp
from environs import env

from liblaf import cherries, melon
from liblaf.apple.consts import (
    ACTIVATION,
    DIRICHLET_MASK,
    GLOBAL_POINT_ID,
    LAMBDA,
    MU,
    MUSCLE_FRACTION,
    SMAS_FRACTION,
)
from liblaf.apple.jax import JaxPointForce
from liblaf.apple.model import (
    AdaptiveContinuationPlan,
    Forward,
    MaterialReference,
    Model,
    ModelBuilder,
    StageState,
)
from liblaf.apple.optim import PNCG
from liblaf.apple.warp import WarpStableNeoHookean, WarpStableNeoHookeanMuscle

SUFFIX: str = "-smas46-muscle46"


class Config(cherries.BaseConfig):
    activation: float = env.float("ACTIVATION", 2.0)
    force_scale: float = env.float("FORCE_SCALE", 5.0e-2)
    initial_progress_step: float = env.float("INITIAL_PROGRESS_STEP", 5.0e-2)
    minimum_progress_step: float = env.float("MINIMUM_PROGRESS_STEP", 1.0e-2)
    maximum_progress_step: float = env.float("MAXIMUM_PROGRESS_STEP", 2.0e-1)
    progress_growth_factor: float = env.float("PROGRESS_GROWTH_FACTOR", 1.5)
    progress_shrink_factor: float = env.float("PROGRESS_SHRINK_FACTOR", 0.5)
    maximum_stage_attempts: int = env.int("MAXIMUM_STAGE_ATTEMPTS", 8)
    input: Path = cherries.input(f"10-input{SUFFIX}.vtu")


@attrs.frozen
class ForceRampProgram:
    final_force: jnp.ndarray

    def state_at(self, *, progress: float, forward: Forward) -> StageState:
        del forward
        scale = jnp.asarray(progress, dtype=self.final_force.dtype)
        return StageState(
            material_values={
                MaterialReference("force", "force"): scale * self.final_force,
            }
        )


def load_mesh(cfg: Config) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    return mesh


def format_force_scale(force_scale: float) -> str:
    mantissa, exponent = f"{force_scale:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def apply_bottom_uniform_ext_force(
    mesh: pv.UnstructuredGrid, force_scale: float
) -> pv.UnstructuredGrid:
    bottom_mask: np.ndarray = mesh.points[:, 1] < 1e-3
    mesh.point_data[DIRICHLET_MASK][bottom_mask, :] = False

    surface: pv.PolyData = mesh.extract_surface(algorithm=None)
    surface_bottom_mask: np.ndarray = surface.points[:, 1] < 1e-3
    bottom_indices: np.ndarray = surface.point_data[GLOBAL_POINT_ID][
        surface_bottom_mask
    ]
    mesh.point_data["Force"] = np.zeros_like(mesh.points)
    mesh.point_data["Force"][bottom_indices, 1] = force_scale
    return mesh


def build_phace_v3(
    mesh: pv.UnstructuredGrid, activation: float, force_scale: float
) -> Model:
    builder = ModelBuilder()
    mesh = builder.add_points(mesh)
    mesh = apply_bottom_uniform_ext_force(mesh, force_scale)

    muscle_frac: np.ndarray = mesh.cell_data[MUSCLE_FRACTION]
    smas_frac: np.ndarray = mesh.cell_data[SMAS_FRACTION]
    aponeurosis_frac: np.ndarray = smas_frac - muscle_frac
    fat_frac: np.ndarray = 1.0 - smas_frac

    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data[ACTIVATION][smas_frac > 1e-3] = np.asarray(
        [activation - 1.0, 0.25 - 1.0, activation - 1.0, 0.0, 0.0, 0.0]
    )
    builder.add_dirichlet(mesh)

    mesh.cell_data["Fraction"] = fat_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0)
    builder.add_energy(WarpStableNeoHookean.from_pyvista(mesh))

    mesh.cell_data["Fraction"] = aponeurosis_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0e2)
    builder.add_energy(WarpStableNeoHookeanMuscle.from_pyvista(mesh))

    mesh.cell_data["Fraction"] = muscle_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0e2)
    builder.add_energy(
        WarpStableNeoHookeanMuscle.from_pyvista(
            mesh, requires_grad=("activation",), name="muscle"
        )
    )

    ext_force = JaxPointForce.from_pyvista(mesh)
    ext_force.name = "force"
    builder.add_energy(ext_force)
    return builder.finalize()


def build_force_ramp_plan(forward: Forward, cfg: Config) -> AdaptiveContinuationPlan:
    force_ref = MaterialReference("force", "force")
    final_force = forward.read_material_values()[force_ref]
    return AdaptiveContinuationPlan(
        state_program=ForceRampProgram(final_force=final_force),
        initial_progress_step=cfg.initial_progress_step,
        minimum_progress_step=cfg.minimum_progress_step,
        maximum_progress_step=cfg.maximum_progress_step,
        progress_growth_factor=cfg.progress_growth_factor,
        progress_shrink_factor=cfg.progress_shrink_factor,
        maximum_stage_attempts=cfg.maximum_stage_attempts,
    )


def main(cfg: Config) -> None:
    wp.init()
    mesh: pv.UnstructuredGrid = load_mesh(cfg)
    ic(mesh)
    model: Model = build_phace_v3(mesh, cfg.activation, cfg.force_scale)
    forward: Forward = Forward(model)
    optimizer = cast("PNCG", forward.optimizer)
    optimizer.convergence = attrs.evolve(
        optimizer.convergence,
        acceptable_relative_gradient_norm=jnp.asarray(1e-3),
        target_relative_gradient_norm=jnp.asarray(1e-5),
    )

    result = forward.solve(plan=build_force_ramp_plan(forward, cfg))
    ic(result)
    mesh.point_data["Solution"] = np.asarray(
        forward.u_full[mesh.point_data[GLOBAL_POINT_ID]]
    )
    melon.save(
        cherries.output(
            "20-forward"
            f"{SUFFIX}-prestrain-ext-force-stable-neo-hookean-uniform-ramped-"
            f"{format_force_scale(cfg.force_scale)}.vtu"
        ),
        mesh,
    )


if __name__ == "__main__":
    cherries.main(main, profile="debug")
