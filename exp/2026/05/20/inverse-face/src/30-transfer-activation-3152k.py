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
from scipy.spatial import cKDTree

from liblaf import cherries, melon

logger = logging.getLogger(__name__)

HIGH_RES_SOURCE = Path(
    "/home/liblaf/github/liblaf/melon/exp/2025/04/30/"
    "human-head-anatomy/data/41-expression-3152k.vtu"
)
SOLVED_SERIES = "20-inverse-face-smooth-w1-lr003.vtu.series"
OUTPUT_STEM = "30-transfer-activation-3152k"
IN_FACE_CONVEX = "InFaceConvex"
IN_FACE_CONTEXT_TYPO = "InFaceContex"
TARGET_SURFACE_MASK = "TargetSurfaceMask"
BACKGROUND_FRACTION = "BackgroundFraction"
ACTIVE_FRACTION = "ActiveFraction"
SMAS_STIFFNESS_FRACTION = "SmasStiffnessFraction"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    source: Path = cherries.input(HIGH_RES_SOURCE)
    solved: Path = cherries.input(SOLVED_SERIES)
    output_input: Path = cherries.output(f"{OUTPUT_STEM}-input.vtu")
    output: Path = cherries.output(f"{OUTPUT_STEM}.vtu")
    output_series: Path = cherries.output(f"{OUTPUT_STEM}.vtu.series")
    output_summary: Path = cherries.output(f"{OUTPUT_STEM}-summary.json")

    solved_frame_index: int = -1
    transfer_k: int = 4
    transfer_power: float = 2.0
    transfer_chunk_size: int = 200_000

    expression: str = "Expression000"
    target_scale: float = 1.0
    target_point_mask: str = "IsFace"
    fixed_point_mask: str = "IsCranium"
    active_fraction_tol: float = 1.0e-3

    E: float = 1.0
    nu: float = 0.49
    smas_stiffness_ratio: float = 1.0e2
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


def require_path(path: Path) -> None:
    if path.exists():
        return
    msg = f"missing input: {path}"
    raise FileNotFoundError(msg)


def require_array(obj: pv.DataSet, association: str, name: str) -> np.ndarray:
    data = obj.cell_data if association == "cell" else obj.point_data
    if name not in data:
        msg = f"{association}_data[{name!r}] is missing"
        raise KeyError(msg)
    return np.asarray(data[name])


def face_cell_mask(mesh: pv.UnstructuredGrid) -> tuple[np.ndarray, str]:
    if IN_FACE_CONVEX in mesh.cell_data:
        return np.asarray(mesh.cell_data[IN_FACE_CONVEX], dtype=bool), IN_FACE_CONVEX
    if IN_FACE_CONTEXT_TYPO in mesh.cell_data:
        return (
            np.asarray(mesh.cell_data[IN_FACE_CONTEXT_TYPO], dtype=bool),
            IN_FACE_CONTEXT_TYPO,
        )
    msg = f"source mesh has neither {IN_FACE_CONVEX!r} nor {IN_FACE_CONTEXT_TYPO!r}"
    raise KeyError(msg)


def extract_face_mesh(source: pv.UnstructuredGrid) -> tuple[pv.UnstructuredGrid, str]:
    mask, selected_name = face_cell_mask(source)
    if not np.any(mask):
        msg = f"no tetrahedra selected by {selected_name}"
        raise ValueError(msg)
    mesh = source.extract_cells(mask)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    cell_types = set(np.asarray(mesh.celltypes).tolist())
    if cell_types != {int(pv.CellType.TETRA)}:
        msg = f"expected tetra-only face mesh, got cell types {sorted(cell_types)}"
        raise ValueError(msg)
    return mesh, selected_name


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return float(lambda_), float(mu)


def active_mask(mesh: pv.UnstructuredGrid, cfg: Config) -> np.ndarray:
    muscle = require_array(mesh, "cell", "MuscleFraction").astype(np.float64)
    return muscle > cfg.active_fraction_tol


def zero_activation_fields(mesh: pv.UnstructuredGrid) -> None:
    from liblaf.apple.common import ACTIVATION, ACTIVATION_INV

    zero = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    mesh.cell_data[ACTIVATION.vtk] = zero.copy()
    mesh.cell_data[ACTIVATION_INV.vtk] = zero.copy()


def add_material_fields(mesh: pv.UnstructuredGrid, cfg: Config) -> None:
    from liblaf.apple.common import FRACTION, LAMBDA, MU, NU
    from liblaf.apple.common import E as YOUNG_MODULUS

    muscle = require_array(mesh, "cell", "MuscleFraction").astype(np.float64)
    smas = require_array(mesh, "cell", "SmasFraction").astype(np.float64)
    background = np.clip(1.0 - muscle - smas, 0.0, 1.0)
    lambda_, mu = lame_parameters(cfg.E, cfg.nu)

    mesh.cell_data[BACKGROUND_FRACTION] = background
    mesh.cell_data[ACTIVE_FRACTION] = muscle
    mesh.cell_data[SMAS_STIFFNESS_FRACTION] = smas
    mesh.cell_data["ActivationMask"] = active_mask(mesh, cfg).astype(np.int8)
    mesh.cell_data["InverseActiveMask"] = mesh.cell_data["ActivationMask"]
    mesh.cell_data[YOUNG_MODULUS.vtk] = np.full(mesh.n_cells, cfg.E, dtype=np.float64)
    mesh.cell_data[NU.vtk] = np.full(mesh.n_cells, cfg.nu, dtype=np.float64)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, lambda_, dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, mu, dtype=np.float64)
    mesh.cell_data[FRACTION.vtk] = np.ones(mesh.n_cells, dtype=np.float64)
    zero_activation_fields(mesh)


def add_boundary_conditions(mesh: pv.UnstructuredGrid, cfg: Config) -> np.ndarray:
    from liblaf.apple.common import FIXED_MASK, FIXED_VALUE

    fixed = require_array(mesh, "point", cfg.fixed_point_mask).astype(bool)
    fixed_mask = np.zeros((mesh.n_points, 3), dtype=bool)
    fixed_value = np.zeros((mesh.n_points, 3), dtype=np.float64)
    fixed_mask[fixed, :] = True

    mesh.point_data[FIXED_MASK.vtk] = fixed_mask
    mesh.point_data[FIXED_VALUE.vtk] = fixed_value
    mesh.point_data["FixedCranium"] = fixed.astype(np.int8)
    return fixed


def target_displacement(
    mesh: pv.UnstructuredGrid, cfg: Config, fixed: np.ndarray
) -> np.ndarray:
    displacement = require_array(mesh, "point", cfg.expression).astype(np.float64)
    displacement = cfg.target_scale * displacement
    displacement[fixed] = 0.0
    return displacement


def target_point_ids(mesh: pv.UnstructuredGrid, cfg: Config) -> np.ndarray:
    if cfg.target_point_mask in mesh.point_data:
        mask = np.asarray(mesh.point_data[cfg.target_point_mask], dtype=bool)
    elif TARGET_SURFACE_MASK in mesh.point_data:
        mask = np.asarray(mesh.point_data[TARGET_SURFACE_MASK], dtype=bool)
    else:
        msg = (
            f"mesh has neither point_data[{cfg.target_point_mask!r}] nor "
            f"point_data[{TARGET_SURFACE_MASK!r}]"
        )
        raise KeyError(msg)
    ids = np.flatnonzero(mask)
    if ids.size == 0:
        msg = "target point mask selected no points"
        raise ValueError(msg)
    return ids.astype(np.int64)


def add_metadata(
    mesh: pv.UnstructuredGrid, cfg: Config, selected_cell_data: str, solved_frame: Path
) -> None:
    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    fixed = np.asarray(mesh.point_data["FixedCranium"], dtype=bool)
    mesh.field_data["Source"] = np.asarray([str(cfg.source)])
    mesh.field_data["SolvedActivationSource"] = np.asarray([str(cfg.solved)])
    mesh.field_data["SolvedActivationFrame"] = np.asarray([str(solved_frame)])
    mesh.field_data["SelectedCellData"] = np.asarray([selected_cell_data])
    mesh.field_data["E"] = np.asarray([cfg.E])
    mesh.field_data["Nu"] = np.asarray([cfg.nu])
    mesh.field_data["SmasStiffnessRatio"] = np.asarray([cfg.smas_stiffness_ratio])
    mesh.field_data["ActiveFractionTol"] = np.asarray([cfg.active_fraction_tol])
    mesh.field_data["ActiveTetCount"] = np.asarray([int(active.sum())])
    mesh.field_data["FixedPointCount"] = np.asarray([int(fixed.sum())])
    mesh.field_data["NoCollision"] = np.asarray([1])


def resolve_series_frame(path: Path, frame_index: int) -> Path:
    require_path(path)
    if not path.name.endswith(".vtu.series"):
        return path
    series = json.loads(path.read_text(encoding="utf-8"))
    files = series.get("files", [])
    if not files:
        msg = f"series has no frames: {path}"
        raise ValueError(msg)
    frame = files[frame_index]
    return path.parent / frame["name"]


def activation_field(mesh: pv.UnstructuredGrid) -> np.ndarray:
    from liblaf.apple.common import ACTIVATION_INV

    if "RecoveredActivationInv" in mesh.cell_data:
        return np.asarray(mesh.cell_data["RecoveredActivationInv"], dtype=np.float64)
    return np.asarray(mesh.cell_data[ACTIVATION_INV.vtk], dtype=np.float64)


def tet_cell_centers(mesh: pv.UnstructuredGrid) -> np.ndarray:
    cells = np.asarray(mesh.cells, dtype=np.int64).reshape(mesh.n_cells, -1)
    if cells.shape[1] != 5 or np.any(cells[:, 0] != 4):
        msg = "cell-center transfer expects tetrahedral cells"
        raise ValueError(msg)
    return np.asarray(mesh.points, dtype=np.float64)[cells[:, 1:]].mean(axis=1)


def query_tree(
    tree: cKDTree, points: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    try:
        distances, indices = tree.query(points, k=k, workers=-1)
    except TypeError:
        distances, indices = tree.query(points, k=k)
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    return distances, indices


def interpolate_values(
    values: np.ndarray, distances: np.ndarray, indices: np.ndarray, power: float
) -> np.ndarray:
    gathered = values[indices]
    exact = distances <= np.finfo(np.float64).eps
    output = np.empty((distances.shape[0], values.shape[1]), dtype=np.float64)
    exact_rows = np.any(exact, axis=1)
    if np.any(exact_rows):
        first_exact = np.argmax(exact[exact_rows], axis=1)
        output[exact_rows] = gathered[exact_rows, first_exact]
    if np.any(~exact_rows):
        far_distances = distances[~exact_rows]
        far_values = gathered[~exact_rows]
        weights = np.power(far_distances, -power)
        weights /= weights.sum(axis=1, keepdims=True)
        output[~exact_rows] = np.einsum("nk,nkd->nd", weights, far_values)
    return output


def transfer_activation_inv(
    source_mesh: pv.UnstructuredGrid, target_mesh: pv.UnstructuredGrid, cfg: Config
) -> tuple[np.ndarray, dict[str, Any]]:
    source_active = active_mask(source_mesh, cfg)
    target_active = active_mask(target_mesh, cfg)
    source_values = activation_field(source_mesh)
    if source_values.shape != (source_mesh.n_cells, 6):
        msg = (
            f"source activation has shape {source_values.shape}; "
            f"expected {(source_mesh.n_cells, 6)}"
        )
        raise ValueError(msg)
    if not np.any(source_active):
        msg = "source solved mesh has no active cells"
        raise ValueError(msg)
    if not np.any(target_active):
        msg = "target high-res mesh has no active cells"
        raise ValueError(msg)

    source_centers = tet_cell_centers(source_mesh)[source_active]
    target_centers = tet_cell_centers(target_mesh)
    source_active_values = source_values[source_active]
    target_active_ids = np.flatnonzero(target_active)
    k = min(max(1, cfg.transfer_k), source_centers.shape[0])
    tree = cKDTree(source_centers)

    transferred = np.zeros((target_mesh.n_cells, 6), dtype=np.float64)
    nearest_distances: list[float] = []
    start_time = time.perf_counter()
    for start in range(0, target_active_ids.size, cfg.transfer_chunk_size):
        stop = min(start + cfg.transfer_chunk_size, target_active_ids.size)
        chunk_ids = target_active_ids[start:stop]
        distances, indices = query_tree(tree, target_centers[chunk_ids], k)
        transferred[chunk_ids] = interpolate_values(
            source_active_values, distances, indices, cfg.transfer_power
        )
        nearest_distances.append(float(distances[:, 0].max()))
        logger.info(
            "Transferred active cells %d:%d / %d",
            stop,
            target_active_ids.size,
            target_active_ids.size,
        )

    stats = {
        "transfer/source_active_cells": int(source_active.sum()),
        "transfer/target_active_cells": int(target_active.sum()),
        "transfer/k": int(k),
        "transfer/power": float(cfg.transfer_power),
        "transfer/time_s": float(time.perf_counter() - start_time),
        "transfer/nearest_distance_max": float(max(nearest_distances)),
        "transfer/activation_rms": float(
            np.linalg.norm(transferred[target_active])
            / math.sqrt(transferred[target_active].size)
        ),
        "transfer/activation_min": float(transferred[target_active].min()),
        "transfer/activation_max": float(transferred[target_active].max()),
    }
    return transferred, stats


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


def build_forward(mesh: pv.UnstructuredGrid, cfg: Config):
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import StableNeoHookean, StableNeoHookeanActive

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)

    set_material(mesh, E=cfg.E, nu=cfg.nu, fraction=mesh.cell_data[BACKGROUND_FRACTION])
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="background"))

    set_material(mesh, E=cfg.E, nu=cfg.nu, fraction=mesh.cell_data[ACTIVE_FRACTION])
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="muscle"))

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


def material_tree(
    base_materials: dict[str, dict[str, torch.Tensor]], activation_inv: np.ndarray
) -> dict[str, dict[str, torch.Tensor]]:
    materials = {name: dict(values) for name, values in base_materials.items()}
    materials["muscle"]["activation_inv"] = torch.as_tensor(
        activation_inv,
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device(),
    )
    return materials


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
    relative_grad_norm = (
        0.0
        if grad_norm_first == 0.0 and grad_norm == 0.0
        else grad_norm / grad_norm_first
    )
    return {
        "forward/result": str(solution.result),
        "forward/success": bool(solution.success),
        "forward/steps": int(convergence_state.step),
        "forward/grad_norm": grad_norm,
        "forward/relative_grad_norm": float(relative_grad_norm),
        "forward/grad_norm_first": grad_norm_first,
        "forward/line_search_ok": bool(line_search_state.ok),
        "forward/line_search_steps": int(line_search_state.step),
        "forward/stagnation_count": int(convergence_state.stagnation_count),
    }


def add_metric_fields(
    mesh: pv.UnstructuredGrid, metrics: dict[str, float | int | bool | str]
) -> None:
    for name, value in metrics.items():
        if isinstance(value, str):
            continue
        mesh.field_data[name] = np.asarray([value])


def make_result_mesh(
    mesh: pv.UnstructuredGrid,
    displacement: np.ndarray,
    transferred_activation_inv: np.ndarray,
    target: np.ndarray,
    target_ids: np.ndarray,
    metrics: dict[str, float | int | bool | str],
) -> pv.UnstructuredGrid:
    from liblaf.apple.common import ACTIVATION_INV

    result = mesh.copy(deep=True)
    error = displacement - target
    result.point_data["Displacement"] = displacement
    result.point_data["DeformedPoint"] = result.points + displacement
    result.point_data["TargetDisplacement"] = target
    result.point_data["TargetPoint"] = result.points + target
    result.point_data["DisplacementError"] = error
    result.point_data["DisplacementErrorNorm"] = np.linalg.norm(error, axis=1)
    target_mask = np.zeros(result.n_points, dtype=np.int8)
    target_mask[target_ids] = 1
    result.point_data[TARGET_SURFACE_MASK] = target_mask
    result.cell_data[ACTIVATION_INV.vtk] = transferred_activation_inv
    result.cell_data["TransferredActivationInv"] = transferred_activation_inv
    result.cell_data["TransferredActivationInvNorm"] = np.linalg.norm(
        transferred_activation_inv, axis=1
    )
    add_metric_fields(result, metrics)
    return result


def summarize(
    mesh: pv.UnstructuredGrid,
    displacement: np.ndarray,
    transferred_activation_inv: np.ndarray,
    target: np.ndarray,
    target_ids: np.ndarray,
    transfer_metrics: dict[str, Any],
    forward_metrics: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    error = displacement - target
    target_error = error[target_ids]
    target_error_norm = np.linalg.norm(target_error, axis=1)
    all_error_norm = np.linalg.norm(error, axis=1)
    target_norm = np.linalg.norm(target[target_ids], axis=1)
    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    active_activation = transferred_activation_inv[active]
    return {
        "mesh/n_points": int(mesh.n_points),
        "mesh/n_cells": int(mesh.n_cells),
        "target/n_points": int(target_ids.size),
        "activation/n_active_tets": int(active.sum()),
        "activation/n_params": int(active.sum() * 6),
        "time/total_s": float(elapsed_s),
        "target/displacement_mean": float(target_norm.mean()),
        "target/displacement_rms": float(
            np.linalg.norm(target[target_ids]) / math.sqrt(target_ids.size)
        ),
        "target/displacement_max": float(target_norm.max()),
        "target/error_mean": float(target_error_norm.mean()),
        "target/error_rms": float(
            np.linalg.norm(target_error) / math.sqrt(target_ids.size)
        ),
        "target/error_max": float(target_error_norm.max()),
        "all/error_rms": float(np.linalg.norm(error) / math.sqrt(error.shape[0])),
        "all/error_max": float(all_error_norm.max()),
        "active_activation_inv/mean": active_activation.mean(axis=0).tolist(),
        "active_activation_inv/min": active_activation.min(axis=0).tolist(),
        "active_activation_inv/max": active_activation.max(axis=0).tolist(),
        "active_activation_inv/rms": float(
            np.linalg.norm(active_activation) / math.sqrt(active_activation.size)
        ),
        **transfer_metrics,
        **forward_metrics,
    }


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def numeric_metrics(
    summary: dict[str, Any], *, exclude: frozenset[str] = frozenset()
) -> dict[str, int | float]:
    return {
        name: value
        for name, value in summary.items()
        if name not in exclude
        and isinstance(value, int | float)
        and not isinstance(value, bool)
    }


def main(cfg: Config) -> None:
    total_start = time.perf_counter()
    configure_runtime()
    require_path(cfg.source)
    solved_frame = resolve_series_frame(cfg.solved, cfg.solved_frame_index)
    require_path(solved_frame)

    high_source = pv.read(cfg.source)
    if not isinstance(high_source, pv.UnstructuredGrid):
        high_source = high_source.cast_to_unstructured_grid()
    solved_mesh = pv.read(solved_frame)
    if not isinstance(solved_mesh, pv.UnstructuredGrid):
        solved_mesh = solved_mesh.cast_to_unstructured_grid()

    mesh, selected_cell_data = extract_face_mesh(high_source)
    add_material_fields(mesh, cfg)
    fixed = add_boundary_conditions(mesh, cfg)
    target = target_displacement(mesh, cfg, fixed)
    target_ids = target_point_ids(mesh, cfg)
    transferred_activation_inv, transfer_metrics = transfer_activation_inv(
        solved_mesh, mesh, cfg
    )
    from liblaf.apple.common import ACTIVATION_INV

    mesh.cell_data[ACTIVATION_INV.vtk] = transferred_activation_inv
    mesh.cell_data["TransferredActivationInv"] = transferred_activation_inv
    mesh.cell_data["TransferredActivationInvNorm"] = np.linalg.norm(
        transferred_activation_inv, axis=1
    )
    add_metadata(mesh, cfg, selected_cell_data, solved_frame)
    melon.save(cfg.output_input, mesh)
    logger.info("Wrote prepared high-res transfer mesh: %s", cfg.output_input)

    forward_start = time.perf_counter()
    forward = build_forward(mesh, cfg)
    materials = material_tree(forward.model.get_materials(), transferred_activation_inv)
    forward.model.set_materials(materials)
    solution = forward.step()
    displacement = to_numpy(forward.state.u)
    forward_metrics = forward_solution_metrics(solution)
    forward_metrics["time/forward_s"] = float(time.perf_counter() - forward_start)

    elapsed_s = time.perf_counter() - total_start
    summary = summarize(
        mesh,
        displacement,
        transferred_activation_inv,
        target,
        target_ids,
        transfer_metrics,
        forward_metrics,
        elapsed_s,
    )
    result = make_result_mesh(
        mesh,
        displacement,
        transferred_activation_inv,
        target,
        target_ids,
        numeric_metrics(summary),
    )
    melon.save(cfg.output, result)
    with melon.SeriesWriter(cfg.output_series, clear=True) as series_writer:
        series_writer.append(result, time=0.0)
    save_json(cfg.output_summary, summary)
    cherries.log_metrics(numeric_metrics(summary))

    print(
        "transfer forward result:",
        f"forward={summary['forward/result']}",
        f"success={summary['forward/success']}",
        f"steps={summary['forward/steps']}",
        f"target_mean_error={summary['target/error_mean']:.3e}cm",
        f"target_rms_error={summary['target/error_rms']:.3e}cm",
        f"target_max_error={summary['target/error_max']:.3e}cm",
    )
    print(f"saved: {cfg.output_input}")
    print(f"saved: {cfg.output}")
    print(f"saved: {cfg.output_series}")
    print(f"saved: {cfg.output_summary}")


if __name__ == "__main__":
    cherries.main(main)
