from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
import warp as wp

from liblaf import cherries, melon

OUTPUT_STEM = "40-activation-affine-check-no-collision"
TARGET_STRETCH = "TargetStretch"


class Config(cherries.BaseConfig):
    output_input: Path = cherries.output(f"{OUTPUT_STEM}-input.vtu")
    output_target: Path = cherries.output(f"{OUTPUT_STEM}-target.vtu")
    output: Path = cherries.output(f"{OUTPUT_STEM}.vtu")

    E: float = 1.0
    nu: float = 0.49
    lr: float = 0.2
    coarsen: bool = False
    target_stretch: tuple[float, float, float] = (1.25, 0.8, 1.0)
    target_energy_tol: float = 1.0e-8
    target_grad_tol: float = 1.0e-7
    final_error_tol: float = 1.0e-5


def configure_runtime() -> None:
    if not torch.cuda.is_available():
        msg = "This experiment uses Warp kernels through Torch and needs CUDA."
        raise RuntimeError(msg)
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cuda")
    wp.config.mode = "release"
    wp.init()


def make_box_mesh(cfg: Config) -> pv.UnstructuredGrid:
    surface = pv.Box((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), quads=False)
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=cfg.lr, coarsen=cfg.coarsen)
    return melon.tet.fix_winding(mesh)


def activation_inv_from_stretch(stretch: tuple[float, float, float]) -> np.ndarray:
    stretch_array = np.asarray(stretch, dtype=np.float64)
    if np.any(stretch_array <= 0.0):
        msg = f"target_stretch must be positive, got {stretch}"
        raise ValueError(msg)
    activation_inv = np.zeros(6, dtype=np.float64)
    activation_inv[:3] = (1.0 / stretch_array) - 1.0
    return activation_inv


def target_displacement(
    points: np.ndarray, stretch: tuple[float, float, float]
) -> np.ndarray:
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    stretch_array = np.asarray(stretch, dtype=np.float64)
    return (points - center) * (stretch_array - 1.0)


def boundary_point_ids(mesh: pv.UnstructuredGrid) -> np.ndarray:
    surface: pv.PolyData = mesh.extract_surface(algorithm=None)
    return np.unique(np.asarray(surface.point_data["vtkOriginalPointIds"]))


def add_boundary_conditions(
    mesh: pv.UnstructuredGrid, displacement: np.ndarray
) -> None:
    from liblaf.apple.common import FIXED_MASK, FIXED_VALUE

    boundary = boundary_point_ids(mesh)
    fixed_mask = np.zeros((mesh.n_points, 3), dtype=bool)
    fixed_value = np.zeros((mesh.n_points, 3), dtype=np.float64)
    fixed_mask[boundary, :] = True
    fixed_value[boundary, :] = displacement[boundary]

    mesh.point_data[FIXED_MASK.vtk] = fixed_mask
    mesh.point_data[FIXED_VALUE.vtk] = fixed_value
    mesh.point_data["BoundaryFixed"] = fixed_mask.any(axis=1).astype(np.int8)


def add_material(mesh: pv.UnstructuredGrid, cfg: Config) -> np.ndarray:
    from liblaf.apple.common import ACTIVATION_INV, LAMBDA, MU, NU, lame_converter
    from liblaf.apple.common import E as YOUNG_MODULUS

    lambda_, mu = lame_converter(cfg.E, cfg.nu)
    activation_inv = activation_inv_from_stretch(cfg.target_stretch)

    mesh.cell_data[YOUNG_MODULUS.vtk] = np.full(mesh.n_cells, cfg.E, dtype=np.float64)
    mesh.cell_data[NU.vtk] = np.full(mesh.n_cells, cfg.nu, dtype=np.float64)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, float(lambda_), dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, float(mu), dtype=np.float64)
    mesh.cell_data[ACTIVATION_INV.vtk] = np.repeat(
        activation_inv[np.newaxis, :], mesh.n_cells, axis=0
    )
    mesh.cell_data[TARGET_STRETCH] = np.repeat(
        np.asarray(cfg.target_stretch, dtype=np.float64)[np.newaxis, :],
        mesh.n_cells,
        axis=0,
    )
    return activation_inv


def add_reference_fields(
    mesh: pv.UnstructuredGrid,
    cfg: Config,
    displacement: np.ndarray,
    activation_inv: np.ndarray,
) -> None:
    mesh.point_data["TargetDisplacement"] = displacement
    mesh.point_data["TargetPoint"] = mesh.points + displacement
    mesh.field_data["E"] = np.asarray([cfg.E])
    mesh.field_data["Nu"] = np.asarray([cfg.nu])
    mesh.field_data["TargetStretch"] = np.asarray([cfg.target_stretch])
    mesh.field_data["ActivationInv"] = activation_inv[np.newaxis, :]
    mesh.field_data["WithCollision"] = np.asarray([0])


def build_model(mesh: pv.UnstructuredGrid):
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import StableNeoHookeanActive

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="active"))
    return Forward(builder.finalize())


def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def tensor_scalar(value: Any) -> float:
    return float(to_numpy(value).reshape(-1)[0])


def full_state_like(forward: Any, displacement: np.ndarray):
    u = torch.as_tensor(
        displacement, dtype=forward.state.u.dtype, device=forward.state.u.device
    )
    return forward.model.State(u=u)


def state_metrics(forward: Any, state: Any, prefix: str) -> dict[str, float]:
    energy = tensor_scalar(forward.problem.fun(state))
    free_grad = forward.problem.grad(state)
    full_grad = forward.model.grad(state)
    return {
        f"{prefix}_energy": energy,
        f"{prefix}_free_grad_norm": float(np.linalg.norm(to_numpy(free_grad))),
        f"{prefix}_full_grad_norm": float(np.linalg.norm(to_numpy(full_grad))),
    }


def result_mesh(
    mesh: pv.UnstructuredGrid,
    displacement: np.ndarray,
    target: np.ndarray,
    metrics: dict[str, float | str],
) -> pv.UnstructuredGrid:
    result = mesh.copy(deep=True)
    target_error = displacement - target
    result.point_data["Displacement"] = displacement
    result.point_data["TargetDisplacement"] = target
    result.point_data["TargetError"] = target_error
    result.point_data["DeformedPoint"] = result.points + displacement
    result.point_data["TargetErrorNorm"] = np.linalg.norm(target_error, axis=1)
    for name, value in metrics.items():
        result.field_data[name] = np.asarray([value])
    return result


def solve(
    mesh: pv.UnstructuredGrid, cfg: Config, target: np.ndarray
) -> tuple[pv.UnstructuredGrid, pv.UnstructuredGrid, dict[str, float | str]]:
    forward = build_model(mesh)

    rest_state = full_state_like(forward, np.zeros_like(target))
    target_state = full_state_like(forward, target)

    metrics: dict[str, float | str] = {}
    metrics.update(state_metrics(forward, rest_state, "rest"))
    metrics.update(state_metrics(forward, target_state, "target"))
    metrics.update(state_metrics(forward, forward.state, "initial"))

    solution = forward.step()
    from liblaf.apple.common import GLOBAL_POINT_ID

    final = to_numpy(forward.state.u)[mesh.point_data[GLOBAL_POINT_ID.vtk]]
    target_error = final - target
    metrics.update(state_metrics(forward, forward.state, "final"))
    metrics["max_target_error"] = float(np.linalg.norm(target_error, axis=1).max())
    metrics["rms_target_error"] = float(np.sqrt(np.mean(np.square(target_error))))
    metrics["optimizer_steps"] = float(solution.state.step)
    metrics["optimizer_result"] = solution.result.name
    metrics["activation_check_pass"] = float(
        metrics["target_energy"] <= cfg.target_energy_tol
        and metrics["target_free_grad_norm"] <= cfg.target_grad_tol
        and metrics["max_target_error"] <= cfg.final_error_tol
    )
    metrics["n_points"] = float(mesh.n_points)
    metrics["n_cells"] = float(mesh.n_cells)
    metrics["n_fixed"] = float(forward.model.n_fixed)
    metrics["n_free"] = float(forward.model.n_free)

    target_result = result_mesh(mesh, target, target, metrics)
    final_result = result_mesh(mesh, final, target, metrics)
    return target_result, final_result, metrics


def main(cfg: Config) -> None:
    configure_runtime()

    mesh = make_box_mesh(cfg)
    target = target_displacement(mesh.points, cfg.target_stretch)
    activation_inv = add_material(mesh, cfg)
    add_boundary_conditions(mesh, target)
    add_reference_fields(mesh, cfg, target, activation_inv)

    melon.save(cfg.output_input, mesh)
    target_result, final_result, metrics = solve(mesh, cfg, target)
    melon.save(cfg.output_target, target_result)
    melon.save(cfg.output, final_result)

    cherries.log_metrics(
        {name: value for name, value in metrics.items() if not isinstance(value, str)}
    )
    print(metrics)
    print(f"saved: {cfg.output}")


if __name__ == "__main__":
    cherries.main(main, profile="debug")
