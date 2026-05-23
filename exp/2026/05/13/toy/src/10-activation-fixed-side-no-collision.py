from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
import warp as wp

from liblaf import cherries, melon

OUTPUT_STEM = "10-activation-fixed-side-no-collision"


class Config(cherries.BaseConfig):
    output_input: Path = cherries.output(f"{OUTPUT_STEM}-input.vtu")
    output: Path = cherries.output(f"{OUTPUT_STEM}.vtu")

    E: float = 1.0
    nu: float = 0.49
    lr: float = 0.2
    coarsen: bool = False
    y_stretch: float = 1.25
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


def activation_from_y_stretch(y_stretch: float) -> np.ndarray:
    if y_stretch <= 0.0:
        msg = f"y_stretch must be positive, got {y_stretch}"
        raise ValueError(msg)
    activation = np.zeros(6, dtype=np.float64)
    activation[1] = y_stretch - 1.0
    return activation


def activation_inv_from_activation(activation: np.ndarray) -> np.ndarray:
    stretch = 1.0 + activation[:3]
    activation_inv = np.zeros(6, dtype=np.float64)
    activation_inv[:3] = (1.0 / stretch) - 1.0
    return activation_inv


def target_displacement(points: np.ndarray, y_stretch: float) -> np.ndarray:
    y0 = points[:, 1].min()
    displacement = np.zeros_like(points)
    displacement[:, 1] = (points[:, 1] - y0) * (y_stretch - 1.0)
    return displacement


def add_boundary_conditions(mesh: pv.UnstructuredGrid) -> None:
    from liblaf.apple.common import FIXED_MASK, FIXED_VALUE

    y = mesh.points[:, 1]
    fixed = np.isclose(y, y.min())

    fixed_mask = np.zeros((mesh.n_points, 3), dtype=bool)
    fixed_value = np.zeros((mesh.n_points, 3), dtype=np.float64)
    fixed_mask[fixed, :] = True

    mesh.point_data[FIXED_MASK.vtk] = fixed_mask
    mesh.point_data[FIXED_VALUE.vtk] = fixed_value
    mesh.point_data["FixedYMin"] = fixed.astype(np.int8)


def add_material(mesh: pv.UnstructuredGrid, cfg: Config) -> np.ndarray:
    from liblaf.apple.common import ACTIVATION, LAMBDA, MU, NU, lame_converter
    from liblaf.apple.common import E as YOUNG_MODULUS

    lambda_, mu = lame_converter(cfg.E, cfg.nu)
    activation = activation_from_y_stretch(cfg.y_stretch)
    activation_inv = activation_inv_from_activation(activation)

    mesh.cell_data[YOUNG_MODULUS.vtk] = np.full(mesh.n_cells, cfg.E, dtype=np.float64)
    mesh.cell_data[NU.vtk] = np.full(mesh.n_cells, cfg.nu, dtype=np.float64)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, float(lambda_), dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, float(mu), dtype=np.float64)
    mesh.cell_data[ACTIVATION.vtk] = np.repeat(
        activation[np.newaxis, :], mesh.n_cells, axis=0
    )
    mesh.cell_data["ExpectedActivationInv"] = np.repeat(
        activation_inv[np.newaxis, :], mesh.n_cells, axis=0
    )
    mesh.cell_data["ExpectedYStretch"] = np.full(
        mesh.n_cells, cfg.y_stretch, dtype=np.float64
    )
    return activation


def add_reference_fields(
    mesh: pv.UnstructuredGrid,
    cfg: Config,
    target: np.ndarray,
    activation: np.ndarray,
) -> None:
    mesh.point_data["ExpectedDisplacement"] = target
    mesh.point_data["ExpectedPoint"] = mesh.points + target
    mesh.field_data["E"] = np.asarray([cfg.E])
    mesh.field_data["Nu"] = np.asarray([cfg.nu])
    mesh.field_data["ExpectedYStretch"] = np.asarray([cfg.y_stretch])
    mesh.field_data["Activation"] = activation[np.newaxis, :]
    mesh.field_data["ExpectedActivationInv"] = activation_inv_from_activation(
        activation
    )[np.newaxis, :]
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
    expected: np.ndarray,
    metrics: dict[str, float | str],
) -> pv.UnstructuredGrid:
    result = mesh.copy(deep=True)
    error = displacement - expected
    result.point_data["Displacement"] = displacement
    result.point_data["ExpectedDisplacement"] = expected
    result.point_data["ActivationError"] = error
    result.point_data["DeformedPoint"] = result.points + displacement
    result.point_data["ExpectedPoint"] = result.points + expected
    result.point_data["ActivationErrorNorm"] = np.linalg.norm(error, axis=1)
    for name, value in metrics.items():
        if isinstance(value, str):
            continue
        result.field_data[name] = np.asarray([value])
    return result


def solve(
    mesh: pv.UnstructuredGrid, cfg: Config, expected: np.ndarray
) -> tuple[pv.UnstructuredGrid, dict[str, float | str]]:
    from liblaf.apple.common import GLOBAL_POINT_ID

    forward = build_model(mesh)

    rest_state = full_state_like(forward, np.zeros_like(expected))
    expected_state = full_state_like(forward, expected)

    metrics: dict[str, float | str] = {}
    metrics.update(state_metrics(forward, rest_state, "rest"))
    metrics.update(state_metrics(forward, expected_state, "expected"))
    metrics.update(state_metrics(forward, forward.state, "initial"))

    solution = forward.step()
    displacement = to_numpy(forward.state.u)[mesh.point_data[GLOBAL_POINT_ID.vtk]]
    error = displacement - expected

    metrics.update(state_metrics(forward, forward.state, "final"))
    metrics["max_activation_error"] = float(np.linalg.norm(error, axis=1).max())
    metrics["rms_activation_error"] = float(np.sqrt(np.mean(np.square(error))))
    metrics["optimizer_steps"] = float(solution.state.step)
    metrics["optimizer_result"] = solution.result.name
    metrics["activation_check_pass"] = float(
        metrics["expected_energy"] <= cfg.target_energy_tol
        and metrics["expected_free_grad_norm"] <= cfg.target_grad_tol
        and metrics["max_activation_error"] <= cfg.final_error_tol
    )
    metrics["n_points"] = float(mesh.n_points)
    metrics["n_cells"] = float(mesh.n_cells)
    metrics["n_fixed"] = float(forward.model.n_fixed)
    metrics["n_free"] = float(forward.model.n_free)

    return result_mesh(mesh, displacement, expected, metrics), metrics


def main(cfg: Config) -> None:
    configure_runtime()

    mesh = make_box_mesh(cfg)
    expected = target_displacement(mesh.points, cfg.y_stretch)
    activation = add_material(mesh, cfg)
    add_boundary_conditions(mesh)
    add_reference_fields(mesh, cfg, expected, activation)

    melon.save(cfg.output_input, mesh)
    result, metrics = solve(mesh, cfg, expected)
    melon.save(cfg.output, result)

    cherries.log_metrics(
        {name: value for name, value in metrics.items() if not isinstance(value, str)}
    )
    print(metrics)
    print(f"saved: {cfg.output}")


if __name__ == "__main__":
    cherries.main(main, profile="debug")
