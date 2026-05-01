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
    Forward,
    ForwardStage,
    MaterialReference,
    Model,
    ModelBuilder,
    StageState,
)
from liblaf.apple.optim import PNCG
from liblaf.apple.warp import WarpStableNeoHookean, WarpStableNeoHookeanMuscle

SUFFIX: str = "-smas46-muscle46-conform"


class Config(cherries.BaseConfig):
    activation: float = env.float("ACTIVATION", 2.0)
    force_scale: float = env.float("FORCE_SCALE", 1.0)
    lambda_value: float = env.float("LAMBDA_VALUE", 3.0)
    num_steps: int = env.int("NUM_STEPS", 20)
    input: Path = cherries.input(f"10-input{SUFFIX}.vtu")


@attrs.frozen
class ForceRampProgram:
    unit_force: jnp.ndarray

    def state_at(self, *, progress: float, forward: Forward) -> StageState:
        del forward
        scale = jnp.asarray(progress, dtype=self.unit_force.dtype)
        return StageState(
            material_values={
                MaterialReference("force", "force"): scale * self.unit_force,
            }
        )


def load_mesh(cfg: Config) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    return mesh


def format_force_scale(force_scale: float) -> str:
    mantissa, exponent = f"{force_scale:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def format_lambda_value(lambda_value: float) -> str:
    if np.isclose(lambda_value, round(lambda_value)):
        return str(int(round(lambda_value)))
    mantissa, exponent = f"{lambda_value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def make_output_stem(cfg: Config) -> str:
    return (
        "20-forward"
        f"{SUFFIX}-prestrain-ext-force-stable-neo-hookean-ramped-lambda-"
        f"{format_lambda_value(cfg.lambda_value)}-force-"
        f"{format_force_scale(cfg.force_scale)}-steps-{cfg.num_steps}"
    )


def make_force_scales(cfg: Config) -> np.ndarray:
    if cfg.num_steps < 1:
        raise ValueError(f"NUM_STEPS must be at least 1, got {cfg.num_steps}")
    return np.linspace(0.0, cfg.force_scale, cfg.num_steps + 1)


def apply_bottom_ext_force(
    mesh: pv.UnstructuredGrid, force_scale: float
) -> pv.UnstructuredGrid:
    bottom_mask: np.ndarray = (mesh.points[:, 1] < 1e-2) & ~mesh.point_data[
        DIRICHLET_MASK
    ][:, 0]
    mesh.point_data[DIRICHLET_MASK][bottom_mask, :] = False

    surface: pv.PolyData = mesh.extract_surface(algorithm=None)
    surface = melon.tri.compute_point_area(surface)
    surface_bottom_mask: np.ndarray = (
        surface.points[:, 1] < 1e-2
    ) & ~surface.point_data[DIRICHLET_MASK][:, 0]
    bottom_indices: np.ndarray = surface.point_data[GLOBAL_POINT_ID][
        surface_bottom_mask
    ]
    mesh.point_data["Force"] = np.zeros_like(mesh.points)
    mesh.point_data["Force"][bottom_indices, 1] = (
        force_scale * surface.point_data["Area"][surface_bottom_mask]
    )
    return mesh


def build_phace_v3(
    mesh: pv.UnstructuredGrid,
    activation: float,
    lambda_value: float,
) -> Model:
    builder = ModelBuilder()
    mesh = builder.add_points(mesh)
    mesh = apply_bottom_ext_force(mesh, 1.0)

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
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), lambda_value)
    builder.add_energy(WarpStableNeoHookean.from_pyvista(mesh))

    mesh.cell_data["Fraction"] = aponeurosis_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), lambda_value * 1.0e2)
    builder.add_energy(WarpStableNeoHookeanMuscle.from_pyvista(mesh))

    mesh.cell_data["Fraction"] = muscle_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), lambda_value * 1.0e2)
    builder.add_energy(
        WarpStableNeoHookeanMuscle.from_pyvista(
            mesh, requires_grad=("activation",), name="muscle"
        )
    )

    ext_force = JaxPointForce.from_pyvista(mesh)
    ext_force.name = "force"
    builder.add_energy(ext_force)
    return builder.finalize()


def update_mesh_solution(
    mesh: pv.UnstructuredGrid,
    forward: Forward,
    unit_force: np.ndarray,
    *,
    stage_index: int,
    force_scale: float,
) -> pv.UnstructuredGrid:
    mesh.point_data["Force"] = force_scale * unit_force
    mesh.point_data["Solution"] = np.asarray(
        forward.u_full[mesh.point_data[GLOBAL_POINT_ID]]
    )
    mesh.field_data["ForceScale"] = np.asarray([force_scale], dtype=float)
    mesh.field_data["StageIndex"] = np.asarray([stage_index], dtype=int)
    return mesh


def main(cfg: Config) -> None:
    wp.init()
    mesh: pv.UnstructuredGrid = load_mesh(cfg)
    ic(mesh)
    model: Model = build_phace_v3(mesh, cfg.activation, cfg.lambda_value)
    forward: Forward = Forward(model)
    optimizer = cast("PNCG", forward.optimizer)
    optimizer.convergence = attrs.evolve(
        optimizer.convergence,
        acceptable_relative_gradient_norm=jnp.asarray(1e-3),
        target_relative_gradient_norm=jnp.asarray(1e-5),
    )

    force_ref = MaterialReference("force", "force")
    unit_force = np.asarray(forward.read_material_values()[force_ref])
    force_program = ForceRampProgram(unit_force=jnp.asarray(unit_force))
    force_scales = make_force_scales(cfg)

    with melon.io.SeriesWriter(
        cherries.output(f"{make_output_stem(cfg)}.vtu.series")
    ) as writer:
        for stage_index, force_scale in enumerate(force_scales):
            stage = ForwardStage(
                name=f"force-{format_force_scale(force_scale)}",
                progress=float(force_scale),
                initial_guess="zero" if stage_index == 0 else "last_successful",
            )
            stage_result = forward.solve_stage(stage, state_program=force_program)
            ic(stage_result)
            if not stage_result.solver_result.success:
                raise RuntimeError(
                    "forward stage failed at "
                    f"force scale {force_scale:g} with status "
                    f"{stage_result.solver_result.status}"
                )
            cherries.set_step(stage_index)
            mesh = update_mesh_solution(
                mesh,
                forward,
                unit_force,
                stage_index=stage_index,
                force_scale=float(force_scale),
            )
            writer.append(mesh, time=float(force_scale))

    melon.save(cherries.output(f"{make_output_stem(cfg)}.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
