import json
import logging
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
import warp as wp

from liblaf import cherries, melon

logger = logging.getLogger(__name__)

PREP_STEM = "10-forward-face-3152k-expr001-smas100"
OUTPUT_STEM = "20-forward-face-3152k-expr001-smas100"
BACKGROUND_FRACTION = "BackgroundFraction"
ACTIVE_FRACTION = "ActiveFraction"
SMAS_STIFFNESS_FRACTION = "SmasStiffnessFraction"
TARGET_SURFACE_MASK = "TargetSurfaceMask"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input: Path = cherries.input(f"{PREP_STEM}-input.vtu")
    target: Path = cherries.input(f"{PREP_STEM}-target.vtu")
    output_input: Path = cherries.output(f"{OUTPUT_STEM}-input.vtu")
    output: Path = cherries.output(f"{OUTPUT_STEM}.vtu")
    output_snapshot: Path = cherries.output(f"{OUTPUT_STEM}.png")
    output_summary: Path = cherries.output(f"{OUTPUT_STEM}-summary.json")
    output_stem: str = OUTPUT_STEM

    E: float = 1.0
    nu: float = 0.49
    smas_stiffness_ratio: float = 1.0e2
    use_smas: bool = True

    activation_local: tuple[float, float, float, float, float, float] = (
        -0.87,
        0.65,
        0.65,
        0.0,
        0.0,
        0.0,
    )

    forward_rtol: float = 5.0e-4
    forward_atol: float = 0.0
    forward_max_steps: int = 10000


def configure_runtime() -> None:
    if not torch.cuda.is_available():
        msg = "This experiment uses Warp kernels through Torch and needs CUDA."
        raise RuntimeError(msg)
    logging.getLogger("liblaf.apple.forward._forward").setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message=r"The \.grad attribute of a Tensor that is not a leaf Tensor.*",
        category=UserWarning,
    )
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cuda")
    wp.config.mode = "release"
    wp.init()


def require_path(path: Path) -> None:
    if path.exists():
        return
    msg = f"missing input: {path}. Run {PREP_STEM}.py first."
    raise FileNotFoundError(msg)


def require_array(obj: pv.DataSet, association: str, name: str) -> np.ndarray:
    data = obj.cell_data if association == "cell" else obj.point_data
    if name not in data:
        msg = f"{association}_data[{name!r}] is missing"
        raise KeyError(msg)
    return np.asarray(data[name])


def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def relative_value(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def load_problem(cfg: Config) -> tuple[pv.UnstructuredGrid, pv.UnstructuredGrid]:
    require_path(cfg.input)
    require_path(cfg.target)
    mesh = pv.read(cfg.input)
    target = pv.read(cfg.target)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    if not isinstance(target, pv.UnstructuredGrid):
        target = target.cast_to_unstructured_grid()
    if mesh.n_points != target.n_points or mesh.n_cells != target.n_cells:
        msg = (
            "input and target topology sizes differ: "
            f"points {mesh.n_points} != {target.n_points}, "
            f"cells {mesh.n_cells} != {target.n_cells}"
        )
        raise ValueError(msg)
    if "Displacement" not in target.point_data:
        msg = f"{cfg.target} has no point_data['Displacement']"
        raise KeyError(msg)
    return mesh, target


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return float(lambda_), float(mu)


def set_material(
    mesh: pv.UnstructuredGrid,
    *,
    E: float,
    nu: float,
    fraction: np.ndarray,
) -> None:
    from liblaf.apple.common import FRACTION, LAMBDA, MU, NU
    from liblaf.apple.common import E as YOUNG_MODULUS

    lambda_, mu = lame_parameters(E, nu)
    mesh.cell_data[YOUNG_MODULUS.vtk] = np.full(mesh.n_cells, E, dtype=np.float64)
    mesh.cell_data[NU.vtk] = np.full(mesh.n_cells, nu, dtype=np.float64)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, lambda_, dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, mu, dtype=np.float64)
    mesh.cell_data[FRACTION.vtk] = np.asarray(fraction, dtype=np.float64)


def unpack_symmetric(
    values: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    xx, yy, zz, xy, xz, yz = values
    return np.asarray(
        [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]],
        dtype=np.float64,
    )


def pack_symmetric(matrices: np.ndarray) -> np.ndarray:
    packed = np.empty((matrices.shape[0], 6), dtype=np.float64)
    packed[:, 0] = matrices[:, 0, 0]
    packed[:, 1] = matrices[:, 1, 1]
    packed[:, 2] = matrices[:, 2, 2]
    packed[:, 3] = matrices[:, 0, 1]
    packed[:, 4] = matrices[:, 0, 2]
    packed[:, 5] = matrices[:, 1, 2]
    return packed


def apply_activation(mesh: pv.UnstructuredGrid, cfg: Config) -> np.ndarray:
    from liblaf.apple.common import ACTIVATION, ACTIVATION_INV

    active = require_array(mesh, "cell", "ActivationMask").astype(bool)
    orientation = require_array(mesh, "cell", "MuscleOrientation").astype(np.float64)
    orientation = orientation.reshape(mesh.n_cells, 3, 3)
    delta_local = unpack_symmetric(cfg.activation_local)
    local_activation = np.eye(3, dtype=np.float64) + delta_local

    activation = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    activation_inv = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    if np.any(active):
        R = orientation[active]
        world_activation = np.einsum("cji,jk,ckl->cil", R, local_activation, R)
        world_delta = world_activation - np.eye(3, dtype=np.float64)
        world_inv_delta = np.linalg.inv(world_activation) - np.eye(3, dtype=np.float64)
        activation[active] = pack_symmetric(world_delta)
        activation_inv[active] = pack_symmetric(world_inv_delta)

    mesh.cell_data[ACTIVATION.vtk] = activation
    mesh.cell_data[ACTIVATION_INV.vtk] = activation_inv
    mesh.cell_data["ActivationNorm"] = np.linalg.norm(activation, axis=1)
    mesh.cell_data["ActivationInvNorm"] = np.linalg.norm(activation_inv, axis=1)
    return activation_inv


def build_forward(mesh: pv.UnstructuredGrid, cfg: Config):
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import StableNeoHookean, StableNeoHookeanActive

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)

    set_material(mesh, E=cfg.E, nu=cfg.nu, fraction=mesh.cell_data[BACKGROUND_FRACTION])
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="background"))

    set_material(
        mesh,
        E=cfg.smas_stiffness_ratio * cfg.E,
        nu=cfg.nu,
        fraction=mesh.cell_data[ACTIVE_FRACTION],
    )
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="muscle"))

    if cfg.use_smas:
        set_material(
            mesh,
            E=cfg.smas_stiffness_ratio * cfg.E,
            nu=cfg.nu,
            fraction=mesh.cell_data[SMAS_STIFFNESS_FRACTION],
        )
        builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="smas"))

    forward = Forward(builder.finalize())
    forward.optimizer = forward.default_optimizer(
        max_steps=cfg.forward_max_steps,
        atol=cfg.forward_atol,
        rtol=cfg.forward_rtol,
    )
    return forward


def forward_solution_metrics(solution: Any) -> dict[str, Any]:
    if solution is None:
        return {
            "forward/result": "missing",
            "forward/success": False,
            "forward/steps": math.nan,
            "forward/grad_norm": math.nan,
            "forward/relative_grad_norm": math.nan,
            "forward/grad_norm_first": math.nan,
            "forward/line_search_ok": False,
            "forward/line_search_steps": math.nan,
            "forward/stagnation_count": math.nan,
        }
    convergence_state = solution.state.convergence_state
    line_search_state = solution.state.line_search_state
    grad_norm = to_float(convergence_state.grad_norm)
    grad_norm_first = to_float(convergence_state.grad_norm_first)
    return {
        "forward/result": str(solution.result),
        "forward/success": bool(solution.success),
        "forward/steps": int(convergence_state.step),
        "forward/grad_norm": grad_norm,
        "forward/relative_grad_norm": relative_value(grad_norm, grad_norm_first),
        "forward/grad_norm_first": grad_norm_first,
        "forward/line_search_ok": bool(line_search_state.ok),
        "forward/line_search_steps": int(line_search_state.step),
        "forward/stagnation_count": int(convergence_state.stagnation_count),
    }


def mask_metrics(
    metrics: dict[str, Any],
    prefix: str,
    mask: np.ndarray,
    displacement: np.ndarray,
    target_displacement: np.ndarray,
) -> None:
    if not np.any(mask):
        return
    disp = displacement[mask]
    target = target_displacement[mask]
    disp_norm = np.linalg.norm(disp, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    metrics[f"{prefix}/n_points"] = int(mask.sum())
    metrics[f"{prefix}/displacement_mean"] = float(disp_norm.mean())
    metrics[f"{prefix}/displacement_rms"] = float(
        np.linalg.norm(disp) / math.sqrt(mask.sum())
    )
    metrics[f"{prefix}/displacement_max"] = float(disp_norm.max())
    metrics[f"{prefix}/target_mean"] = float(target_norm.mean())
    metrics[f"{prefix}/target_rms"] = float(
        np.linalg.norm(target) / math.sqrt(mask.sum())
    )
    metrics[f"{prefix}/target_max"] = float(target_norm.max())
    metrics[f"{prefix}/rms_ratio_to_target"] = relative_value(
        metrics[f"{prefix}/displacement_rms"],
        metrics[f"{prefix}/target_rms"],
    )


def summarize(
    mesh: pv.UnstructuredGrid,
    target: pv.UnstructuredGrid,
    displacement: np.ndarray,
    activation_inv: np.ndarray,
    forward_metrics: dict[str, Any],
    elapsed_s: float,
    cfg: Config,
) -> dict[str, Any]:
    target_displacement = np.asarray(
        target.point_data["Displacement"], dtype=np.float64
    )
    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    volume = np.asarray(mesh.cell_data["Volume"], dtype=np.float64)
    muscle = np.asarray(mesh.cell_data["MuscleFraction"], dtype=np.float64)
    metrics: dict[str, Any] = {
        "mesh/n_points": int(mesh.n_points),
        "mesh/n_cells": int(mesh.n_cells),
        "output/stem": cfg.output_stem,
        "activation/local_xx": float(cfg.activation_local[0]),
        "activation/local_yy": float(cfg.activation_local[1]),
        "activation/local_zz": float(cfg.activation_local[2]),
        "activation/local_xy": float(cfg.activation_local[3]),
        "activation/local_xz": float(cfg.activation_local[4]),
        "activation/local_yz": float(cfg.activation_local[5]),
        "activation/n_tets": int(active.sum()),
        "activation/fraction_volume": float(np.sum(muscle[active] * volume[active])),
        "activation/inv_norm_max": float(np.linalg.norm(activation_inv, axis=1).max()),
        "time/total_s": float(elapsed_s),
        "E": float(cfg.E),
        "nu": float(cfg.nu),
        "smas_stiffness_ratio": float(cfg.smas_stiffness_ratio),
        "smas/enabled": bool(cfg.use_smas),
        **forward_metrics,
    }
    for name in ("IsFace", "IsSkin", "IsLipTop", "IsLipBottom"):
        if name in mesh.point_data:
            mask_metrics(
                metrics,
                name,
                np.asarray(mesh.point_data[name], dtype=bool),
                displacement,
                target_displacement,
            )
    return metrics


def add_metric_fields(
    mesh: pv.UnstructuredGrid, metrics: dict[str, float | int | bool | str]
) -> None:
    for name, value in metrics.items():
        if isinstance(value, str):
            continue
        mesh.field_data[name] = np.asarray([value])


def make_result_mesh(
    mesh: pv.UnstructuredGrid,
    target: pv.UnstructuredGrid,
    displacement: np.ndarray,
    metrics: dict[str, float | int | bool | str],
) -> pv.UnstructuredGrid:
    result = mesh.copy(deep=True)
    target_displacement = np.asarray(
        target.point_data["Displacement"], dtype=np.float64
    )
    error = displacement - target_displacement
    result.point_data["Displacement"] = displacement
    result.point_data["DisplacementNorm"] = np.linalg.norm(displacement, axis=1)
    result.point_data["DeformedPoint"] = result.points + displacement
    result.point_data["TargetDisplacement"] = target_displacement
    result.point_data["TargetDisplacementNorm"] = np.linalg.norm(
        target_displacement, axis=1
    )
    result.point_data["TargetPoint"] = result.points + target_displacement
    result.point_data["DisplacementError"] = error
    result.point_data["DisplacementErrorNorm"] = np.linalg.norm(error, axis=1)
    add_metric_fields(result, metrics)
    return result


def save_snapshot(path: Path, result: pv.UnstructuredGrid) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    deformed = result.copy(deep=True)
    deformed.points = np.asarray(deformed.point_data["DeformedPoint"], dtype=np.float64)
    reference = result.copy(deep=True)
    reference.points = np.asarray(reference.point_data["TargetPoint"], dtype=np.float64)

    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1800, 900))
    plotter.subplot(0, 0)
    plotter.add_mesh(
        deformed.extract_surface(),
        scalars="DisplacementNorm",
        cmap="viridis",
        show_edges=False,
    )
    plotter.add_text("forward", font_size=12)
    plotter.view_xy()
    plotter.camera.zoom(1.25)

    plotter.subplot(0, 1)
    plotter.add_mesh(
        reference.extract_surface(),
        scalars="TargetDisplacementNorm",
        cmap="viridis",
        show_edges=False,
    )
    plotter.add_text("target", font_size=12)
    plotter.view_xy()
    plotter.camera.zoom(1.25)

    plotter.screenshot(path)
    plotter.close()


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def output_paths(cfg: Config) -> tuple[Path, Path, Path, Path]:
    if cfg.output_stem == OUTPUT_STEM:
        return cfg.output_input, cfg.output, cfg.output_snapshot, cfg.output_summary
    base = cfg.output.parent
    return (
        base / f"{cfg.output_stem}-input.vtu",
        base / f"{cfg.output_stem}.vtu",
        base / f"{cfg.output_stem}.png",
        base / f"{cfg.output_stem}-summary.json",
    )


def log_dynamic_outputs(cfg: Config, paths: tuple[Path, Path, Path, Path]) -> None:
    if cfg.output_stem == OUTPUT_STEM:
        return
    for path in paths:
        if path.exists():
            cherries.log_output(path)


def numeric_metrics(data: dict[str, Any]) -> dict[str, float | int | bool]:
    return {
        name: value
        for name, value in data.items()
        if isinstance(value, (bool, int, float))
    }


def main(cfg: Config) -> None:
    configure_runtime()
    output_input, output, output_snapshot, output_summary = output_paths(cfg)
    mesh, target = load_problem(cfg)
    activation_inv = apply_activation(mesh, cfg)
    melon.save(output_input, mesh)

    start = time.perf_counter()
    forward = build_forward(mesh, cfg)
    solution = forward.step()
    elapsed_s = time.perf_counter() - start
    displacement = to_numpy(forward.state.u)
    forward_metrics = forward_solution_metrics(solution)
    summary = summarize(
        mesh,
        target,
        displacement,
        activation_inv,
        forward_metrics,
        elapsed_s,
        cfg,
    )
    result = make_result_mesh(mesh, target, displacement, summary)

    melon.save(output, result)
    try:
        save_snapshot(output_snapshot, result)
    except (OSError, RuntimeError, ValueError):
        logger.warning("failed to save snapshot: %s", output_snapshot, exc_info=True)
    save_json(output_summary, summary)
    log_dynamic_outputs(cfg, (output_input, output, output_snapshot, output_summary))
    cherries.log_metrics(numeric_metrics(summary))
    print(json.dumps(summary, indent=2))
    print(f"saved: {output_input}")
    print(f"saved: {output}")
    print(f"saved: {output_summary}")


if __name__ == "__main__":
    cherries.main(main)
