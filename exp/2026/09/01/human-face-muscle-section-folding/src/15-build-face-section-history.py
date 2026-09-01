"""Extract every primary-history state for the audited 31-tetra muscle section."""

from __future__ import annotations

# ruff: noqa: C901, PLR0912, PLR0915
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pydantic_settings as ps
import pyvista as pv

from liblaf import cherries

PRIMARY_CASE = "20-human-face-smile-no-skin-lr3"
EXPECTED_STEPS = 201


class Config(cherries.BaseConfig):
    """Inputs are fixed by the current section-selection contract."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    selection_summary: Path = cherries.input("10-face-muscle-section/summary.json")
    source_history: Path = (
        Path(__file__).resolve().parents[4]
        / "06/17/human-face-smile-prestrain-v2/data"
        / f"{PRIMARY_CASE}-steps.vtkhdf"
    )
    output_dir: Path = cherries.output("15-face-muscle-section-history", mkdir=True)
    chunk_cells: int = 100_000


def fail(message: str) -> None:
    raise ValueError(message)


def require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        fail(f"{context} missing {key!r}; present={sorted(mapping)}")
    return mapping[key]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tetrahedra(grid: pv.UnstructuredGrid, context: str) -> np.ndarray:
    if not np.all(grid.celltypes == pv.CellType.TETRA):
        fail(f"{context} must contain tetrahedra only")
    packed = np.asarray(grid.cells)
    if packed.size != grid.n_cells * 5 or not np.all(packed.reshape(-1, 5)[:, 0] == 4):
        fail(f"{context} has non-tetra connectivity")
    return packed.reshape(-1, 5)[:, 1:]


def det_ainv(activation_inv: np.ndarray) -> np.ndarray:
    if activation_inv.ndim != 2 or activation_inv.shape[1] != 6:
        fail(f"ActivationInv must have shape (n, 6), got {activation_inv.shape}")
    matrix = np.zeros((activation_inv.shape[0], 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = 1.0 + activation_inv[:, 0]
    matrix[:, 1, 1] = 1.0 + activation_inv[:, 1]
    matrix[:, 2, 2] = 1.0 + activation_inv[:, 2]
    matrix[:, 0, 1] = matrix[:, 1, 0] = activation_inv[:, 3]
    matrix[:, 1, 2] = matrix[:, 2, 1] = activation_inv[:, 4]
    matrix[:, 0, 2] = matrix[:, 2, 0] = activation_inv[:, 5]
    return np.linalg.det(matrix)


def rest_volumes(reference: np.ndarray, cells: np.ndarray) -> np.ndarray:
    edges = np.stack(
        (
            reference[cells[:, 1]] - reference[cells[:, 0]],
            reference[cells[:, 2]] - reference[cells[:, 0]],
            reference[cells[:, 3]] - reference[cells[:, 0]],
        ),
        axis=2,
    )
    determinant = np.linalg.det(edges)
    if not np.all(np.isfinite(determinant)) or np.any(np.abs(determinant) <= 0.0):
        fail("reference selected tetrahedra are non-finite or degenerate")
    return np.abs(determinant) / 6.0


def det_f(
    deformed: np.ndarray,
    cells: np.ndarray,
    reference_edge_determinant: np.ndarray,
) -> np.ndarray:
    edges = np.stack(
        (
            deformed[cells[:, 1]] - deformed[cells[:, 0]],
            deformed[cells[:, 2]] - deformed[cells[:, 0]],
            deformed[cells[:, 3]] - deformed[cells[:, 0]],
        ),
        axis=2,
    )
    result = np.linalg.det(edges) / reference_edge_determinant
    if not np.all(np.isfinite(result)):
        fail("DetF contains non-finite values")
    return result


def compact_grid(
    deformed: np.ndarray,
    cells: np.ndarray,
    source_ids: np.ndarray,
    origin: np.ndarray,
    axes: np.ndarray,
    fields: dict[str, np.ndarray],
) -> pv.UnstructuredGrid:
    used = np.unique(cells[source_ids].ravel())
    local_cells = np.searchsorted(used, cells[source_ids])
    packed = np.column_stack(
        (np.full(source_ids.size, 4, dtype=np.int64), local_cells)
    ).ravel()
    grid = pv.UnstructuredGrid(
        packed,
        np.full(source_ids.size, pv.CellType.TETRA, dtype=np.uint8),
        (deformed[used] - origin) @ axes,
    )
    grid.cell_data["SourceCellId"] = source_ids.astype(np.int64)
    for name, values in fields.items():
        grid.cell_data[name] = values
    return grid


def source_grid(reader: Any, step: int) -> pv.UnstructuredGrid:
    reader.SetStep(step)
    reader.Update()
    grid = pv.wrap(reader.GetOutput())
    if not isinstance(grid, pv.UnstructuredGrid):
        fail(f"VTKHDF step {step} did not yield an unstructured grid")
    return grid


def main(config: Config) -> None:
    if config.chunk_cells < 1:
        fail("chunk_cells must be positive")
    contract_path = config.selection_summary.resolve()
    history_path = config.source_history.resolve()
    if not contract_path.is_file() or not history_path.is_file():
        fail(
            f"missing selection contract or source history: {contract_path}, {history_path}"
        )
    contract = json.loads(contract_path.read_text())
    selection = require(contract, "selection", "section contract")
    if require(selection, "primary_case", "section contract") != PRIMARY_CASE:
        fail("selection contract primary case is not the canonical no-skin-lr3 case")
    slab = require(selection, "slab", "section contract")
    source_ids = np.asarray(
        require(slab, "source_cell_ids", "section contract"), dtype=np.int64
    )
    if source_ids.shape != (31,) or np.unique(source_ids).size != 31:
        fail("selection contract must contain exactly 31 unique SourceCellIds")
    origin = np.asarray(
        require(selection, "pca_origin", "section contract"), dtype=np.float64
    )
    axes = np.asarray(
        require(selection, "pca_axes_columns", "section contract"), dtype=np.float64
    )
    if origin.shape != (3,) or axes.shape != (3, 3) or not np.all(np.isfinite(axes)):
        fail("selection contract local-coordinate basis is invalid")
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1.0e-10):
        fail("selection contract local-coordinate basis is not orthonormal")

    with h5py.File(history_path, "r") as hdf:
        steps = np.asarray(hdf["VTKHDF/FieldData/inverse_step"], dtype=np.int64)
    if not np.array_equal(steps, np.arange(EXPECTED_STEPS, dtype=np.int64)):
        fail("source VTKHDF inverse_step must be exactly 0..200")

    reader = pv.HDFReader(history_path).reader
    if int(reader.GetNumberOfSteps()) != EXPECTED_STEPS:
        fail(f"source VTKHDF reports {reader.GetNumberOfSteps()} steps, not 201")
    first = source_grid(reader, 0)
    cells = tetrahedra(first, "source VTKHDF step 0")
    if source_ids[-1] >= first.n_cells:
        fail("selection contract SourceCellIds exceed source cell count")
    reference = np.asarray(first.points, dtype=np.float64).copy()
    selected_cells = cells[source_ids]
    reference_edges = np.stack(
        (
            reference[selected_cells[:, 1]] - reference[selected_cells[:, 0]],
            reference[selected_cells[:, 2]] - reference[selected_cells[:, 0]],
            reference[selected_cells[:, 3]] - reference[selected_cells[:, 0]],
        ),
        axis=2,
    )
    reference_edge_determinant = np.linalg.det(reference_edges)
    rest_volume = rest_volumes(reference, selected_cells)
    if not np.all(np.isfinite(rest_volume)):
        fail("RestVolume contains non-finite values")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    frames = config.output_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    series: list[dict[str, Any]] = []
    missing_states = 0
    nonfinite_states = 0
    reference_cells = cells.copy()
    for step in range(EXPECTED_STEPS):
        grid = source_grid(reader, step)
        if grid.n_points != first.n_points or grid.n_cells != first.n_cells:
            fail(f"VTKHDF step {step} mesh cardinality changed")
        current_cells = tetrahedra(grid, f"source VTKHDF step {step}")
        if not np.array_equal(current_cells, reference_cells):
            fail(f"VTKHDF step {step} tetra connectivity changed")
        points = np.asarray(grid.points, dtype=np.float64)
        if not np.array_equal(points, reference):
            fail(f"VTKHDF step {step} reference points changed")
        if (
            "DeformedPoint" not in grid.point_data
            or "ActivationInv" not in grid.cell_data
        ):
            missing_states += 1
            fail(f"VTKHDF step {step} misses DeformedPoint or ActivationInv")
        deformed = np.asarray(grid.point_data["DeformedPoint"], dtype=np.float64)
        activation = np.asarray(grid.cell_data["ActivationInv"], dtype=np.float64)[
            source_ids
        ]
        if deformed.shape != reference.shape or activation.shape != (31, 6):
            missing_states += 1
            fail(f"VTKHDF step {step} has invalid DeformedPoint or ActivationInv shape")
        current_det_f = det_f(deformed, selected_cells, reference_edge_determinant)
        current_det_ainv = det_ainv(activation)
        current_det_g = current_det_f * current_det_ainv
        activation_norm = np.linalg.norm(activation, axis=1)
        arrays = (current_det_f, current_det_ainv, current_det_g, activation_norm)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            nonfinite_states += 1
            fail(f"VTKHDF step {step} produced non-finite section fields")
        fields = {
            "RestVolume": rest_volume,
            "DetF": current_det_f,
            "DetAinv": current_det_ainv,
            "DetG": current_det_g,
            "InvertedDetF": (current_det_f < 0.0).astype(np.uint8),
            "InvertedDetAinv": (current_det_ainv < 0.0).astype(np.uint8),
            "InvertedDetG": (current_det_g < 0.0).astype(np.uint8),
            "DoubleInverted": ((current_det_f < 0.0) & (current_det_ainv < 0.0)).astype(
                np.uint8
            ),
            "ActivationNorm": activation_norm,
        }
        frame = frames / f"step-{step:03d}.vtu"
        compact_grid(deformed, cells, source_ids, origin, axes, fields).save(frame)
        series.append({"name": str(frame.relative_to(config.output_dir)), "time": step})

    best = pv.read(contract_path.parent / f"{PRIMARY_CASE}-section-deformed.vtu")
    frame_194 = pv.read(frames / "step-194.vtu")
    for name in (
        "SourceCellId",
        "RestVolume",
        "DetF",
        "DetAinv",
        "DetG",
        "InvertedDetF",
        "DoubleInverted",
        "ActivationNorm",
    ):
        if name not in best.cell_data or name not in frame_194.cell_data:
            fail(f"frame-194 comparison missing {name}")
        if not np.allclose(
            best.cell_data[name], frame_194.cell_data[name], rtol=0.0, atol=1.0e-12
        ):
            fail(f"frame-194 {name} disagrees with the primary best-section export")
    if not np.array_equal(best.cells, frame_194.cells) or not np.allclose(
        best.points, frame_194.points, rtol=0.0, atol=1.0e-12
    ):
        fail("frame-194 geometry disagrees with the primary best-section export")

    series_path = config.output_dir / "history.vtu.series"
    series_path.write_text(
        json.dumps({"file-series-version": "1.0", "files": series}, indent=2) + "\n"
    )
    frame_hashes = {path.name: sha256(path) for path in sorted(frames.glob("*.vtu"))}
    receipt = {
        "selection_contract": {
            "path": str(contract_path),
            "sha256": sha256(contract_path),
        },
        "source_history": {
            "path": str(history_path),
            "bytes": history_path.stat().st_size,
            "sha256": sha256(history_path),
        },
        "primary_case": PRIMARY_CASE,
        "source_cell_ids": source_ids.tolist(),
        "frame_count": EXPECTED_STEPS,
        "inverse_steps": {
            "first": int(steps[0]),
            "last": int(steps[-1]),
            "exact_consecutive": True,
        },
        "validation": {
            "connectivity_and_reference_points_constant": True,
            "missing_states": missing_states,
            "nonfinite_states": nonfinite_states,
            "frame_194_matches_primary_best_section": True,
        },
        "history_series": {"path": str(series_path), "sha256": sha256(series_path)},
        "frames": frame_hashes,
    }
    receipt_path = config.output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (config.output_dir / "summary.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    cherries.main(main)
