"""Build strict, local-coordinate muscle folding artifacts from six face endpoints."""

from __future__ import annotations

# ruff: noqa: C901, EM102, FBT001, PLR0912, PLR0915, TRY003
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv

from liblaf import cherries

CASE_PRIMARY = "20-human-face-smile-no-skin-lr3"
CASE_COMPARATOR = "20-human-face-smile-skin-estimated-plus-tightening-lr1"
ZYGO_IDS = (63, 64)
SECTION_MUSCLE_ID = 64
REQUIRED_CASES = frozenset(
    {
        CASE_PRIMARY,
        CASE_COMPARATOR,
        "20-human-face-smile-skin-no-prestrain-lr1",
        "20-human-face-smile-no-skin-lr1",
        "20-human-face-smile-skin-estimated-plus-tightening-lr2-cont-lr02-warm-from-best",
        "20-human-face-smile-skin-no-prestrain-lr3-cont-lr03-from-best",
    }
)
REQUIRED_CELL_DATA = (
    "ActivationMask",
    "MuscleFraction",
    "MuscleId",
    "ActivationInv",
)


class Config(cherries.BaseConfig):
    """Inputs and deterministic section-selection policy."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    source_summary: Path = (
        Path(__file__).resolve().parents[4]
        / "06/17/human-face-smile-prestrain-v2/data/23-final-comparison-summary.json"
    )
    output_dir: Path = cherries.output("10-face-muscle-section", mkdir=True)
    chunk_cells: int = 100_000
    slab_half_width_fraction: float = 0.15


def fail(message: str) -> None:
    raise ValueError(message)


def require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        fail(f"{context} missing required key {key!r}; present={sorted(mapping)}")
    return mapping[key]


def finite(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric, got {value!r}") from error
    if not math.isfinite(result):
        fail(f"{context} must be finite, got {result!r}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cells_as_tets(grid: pv.UnstructuredGrid, context: str) -> np.ndarray:
    if not np.all(grid.celltypes == pv.CellType.TETRA):
        fail(f"{context} must contain tetrahedra only")
    packed = np.asarray(grid.cells)
    if packed.size != grid.n_cells * 5 or not np.all(packed.reshape(-1, 5)[:, 0] == 4):
        fail(f"{context} has unexpected tetra connectivity")
    return packed.reshape(-1, 5)[:, 1:]


def activation_det(activation_inv: np.ndarray) -> np.ndarray:
    if activation_inv.ndim != 2 or activation_inv.shape[1] != 6:
        fail(f"ActivationInv must have shape (n, 6), got {activation_inv.shape}")
    result = np.zeros((activation_inv.shape[0], 3, 3), dtype=np.float64)
    result[:, 0, 0] = 1.0 + activation_inv[:, 0]
    result[:, 1, 1] = 1.0 + activation_inv[:, 1]
    result[:, 2, 2] = 1.0 + activation_inv[:, 2]
    result[:, 0, 1] = result[:, 1, 0] = activation_inv[:, 3]
    result[:, 1, 2] = result[:, 2, 1] = activation_inv[:, 4]
    result[:, 0, 2] = result[:, 2, 0] = activation_inv[:, 5]
    return np.linalg.det(result)


def determinants(
    reference: np.ndarray,
    deformed: np.ndarray,
    tets: np.ndarray,
    activation_inv: np.ndarray,
    chunk_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute reference volumes and exact F/Ainv/G determinants in bounded chunks."""
    count = tets.shape[0]
    volume = np.empty(count, dtype=np.float64)
    det_f = np.empty(count, dtype=np.float64)
    det_ainv = np.empty(count, dtype=np.float64)
    for start in range(0, count, chunk_cells):
        stop = min(count, start + chunk_cells)
        ids = tets[start:stop]
        ref_edges = np.stack(
            (
                reference[ids[:, 1]] - reference[ids[:, 0]],
                reference[ids[:, 2]] - reference[ids[:, 0]],
                reference[ids[:, 3]] - reference[ids[:, 0]],
            ),
            axis=2,
        )
        def_edges = np.stack(
            (
                deformed[ids[:, 1]] - deformed[ids[:, 0]],
                deformed[ids[:, 2]] - deformed[ids[:, 0]],
                deformed[ids[:, 3]] - deformed[ids[:, 0]],
            ),
            axis=2,
        )
        ref_det = np.linalg.det(ref_edges)
        if np.any(~np.isfinite(ref_det)) or np.any(
            np.abs(ref_det) <= np.finfo(float).tiny
        ):
            fail(f"reference tetrahedra [{start}, {stop}) are degenerate or non-finite")
        volume[start:stop] = np.abs(ref_det) / 6.0
        det_f[start:stop] = np.linalg.det(def_edges) / ref_det
        det_ainv[start:stop] = activation_det(activation_inv[start:stop])
    det_g = det_f * det_ainv
    if not all(
        np.all(np.isfinite(value)) for value in (volume, det_f, det_ainv, det_g)
    ):
        fail("computed determinant arrays are non-finite")
    return volume, det_f, det_ainv, det_g


def weighted_fraction(mask: np.ndarray, volume: np.ndarray) -> float:
    total = float(volume.sum())
    if total <= 0.0:
        fail("selected cells have non-positive total rest volume")
    return float(volume[mask].sum() / total)


def determinant_sign_metrics(
    volume: np.ndarray,
    det_f: np.ndarray,
    det_ainv: np.ndarray,
    det_g: np.ndarray,
) -> dict[str, Any]:
    """Return explicit strict-sign counts and rest-volume fractions."""
    if volume.size == 0:
        fail("cannot summarize an empty determinant selection")
    signs = {
        "f_negative": det_f < 0.0,
        "ainv_negative": det_ainv < 0.0,
        "g_negative": det_g < 0.0,
        "double_inverted": (det_f < 0.0) & (det_ainv < 0.0),
    }
    output: dict[str, Any] = {
        "cells": int(volume.size),
        "rest_volume": float(volume.sum()),
        "min_det_f": float(det_f.min()),
        "min_det_ainv": float(det_ainv.min()),
        "min_det_g": float(det_g.min()),
    }
    for name, mask in signs.items():
        output[f"{name}_cells"] = int(mask.sum())
        output[f"{name}_rest_volume_fraction"] = weighted_fraction(mask, volume)
    return output


def analysis_row(
    case: dict[str, Any],
    volume: np.ndarray,
    det_f: np.ndarray,
    det_ainv: np.ndarray,
    det_g: np.ndarray,
    active: np.ndarray,
    zygomaticus: np.ndarray,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case": require(case, "case", "case summary"),
        "setup": require(case, "case/setup", "case summary"),
        "best_step": int(
            finite(require(case, "best/step", "case summary"), "best/step")
        ),
        "best_loss_mm2": finite(
            require(case, "best/loss_mm2", "case summary"), "best/loss_mm2"
        ),
        "best_error_rms_mm": finite(
            require(case, "best/error_rms_mm", "case summary"), "best/error_rms_mm"
        ),
        "roughness_displacement_edge_rms": finite(
            require(case, "bumpiness/displacement_edge_rms", "case summary"),
            "bumpiness/displacement_edge_rms",
        ),
        "roughness_displacement_laplacian_rms": finite(
            require(case, "bumpiness/displacement_laplacian_rms", "case summary"),
            "bumpiness/displacement_laplacian_rms",
        ),
        "inverse_converged": bool(require(case, "inverse/converged", "case summary")),
        "forward_failures": int(
            require(case, "inverse/forward_fail_count", "case summary")
        ),
        "adjoint_failures": int(
            require(case, "inverse/adjoint_fail_count", "case summary")
        ),
    }
    for name, mask in (("active_muscle", active), ("zygomaticus_63_64", zygomaticus)):
        if not np.any(mask):
            fail(f"{row['case']} has no selected {name} cells")
        metrics = determinant_sign_metrics(
            volume[mask], det_f[mask], det_ainv[mask], det_g[mask]
        )
        row.update(
            {
                f"{name}_{metric_name}": metric_value
                for metric_name, metric_value in metrics.items()
            }
        )
    return row


def stable_pca(
    points: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.average(points, axis=0, weights=weights)
    centered = points - origin
    covariance = (centered * weights[:, None]).T @ centered / weights.sum()
    eigenvalues, vectors = np.linalg.eigh(covariance)
    axes = vectors[:, np.argsort(eigenvalues)[::-1]]
    for column in range(2):
        axis = axes[:, column]
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0.0:
            axes[:, column] *= -1.0
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    pivot = int(np.argmax(np.abs(axes[:, 2])))
    if axes[pivot, 2] < 0.0:
        axes[:, 2] *= -1.0
        axes[:, 1] *= -1.0
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1.0e-10):
        fail("PCA axes are not orthonormal")
    return origin, axes


def compact_grid(
    reference: np.ndarray,
    deformed: np.ndarray,
    tets: np.ndarray,
    source_ids: np.ndarray,
    origin: np.ndarray,
    axes: np.ndarray,
    fields: dict[str, np.ndarray],
    deformed_coordinates: bool,
) -> pv.UnstructuredGrid:
    used = np.unique(tets[source_ids].ravel())
    local_tets = np.searchsorted(used, tets[source_ids])
    packed = np.column_stack(
        (np.full(source_ids.size, 4, dtype=np.int64), local_tets)
    ).ravel()
    physical = deformed[used] if deformed_coordinates else reference[used]
    grid = pv.UnstructuredGrid(
        packed,
        np.full(source_ids.size, pv.CellType.TETRA, dtype=np.uint8),
        (physical - origin) @ axes,
    )
    grid.cell_data["SourceCellId"] = source_ids.astype(np.int64)
    for name, values in fields.items():
        grid.cell_data[name] = values[source_ids]
    return grid


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(config: Config) -> None:
    if config.chunk_cells < 1 or not 0.0 < config.slab_half_width_fraction <= 1.0:
        fail(
            "chunk_cells must be positive and slab_half_width_fraction must be in (0, 1]"
        )
    source = config.source_summary.resolve()
    if not source.is_file():
        fail(f"canonical source summary does not exist: {source}")
    document = json.loads(source.read_text())
    cases = require(document, "cases", "source summary")
    if not isinstance(cases, list) or len(cases) != 6:
        fail("source summary must contain exactly six cases")
    by_case = {require(case, "case", "source case"): case for case in cases}
    if set(by_case) != REQUIRED_CASES:
        fail(f"canonical case set differs: {sorted(by_case)}")
    if len(by_case) != len(cases):
        fail("canonical source summary contains duplicate case names")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    processed: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    source_root = source.parent
    for case_name in sorted(by_case):
        case = by_case[case_name]
        for key, expected in (
            ("activation/mode", "per-muscle-tet-6dof"),
            ("activation/range_clamping", False),
            ("activation/shared", False),
        ):
            if require(case, key, case_name) != expected:
                fail(
                    f"{case_name} must be materialized unbounded per-muscle-tet 6DoF: {key}={case[key]!r}"
                )
        path = source_root / f"{case_name}.vtu"
        if not path.is_file():
            fail(f"{case_name} materialized final endpoint is absent: {path}")
        grid = pv.read(path)
        if not isinstance(grid, pv.UnstructuredGrid):
            fail(f"{path} is not an unstructured grid")
        tets = cells_as_tets(grid, str(path))
        for key in REQUIRED_CELL_DATA:
            if key not in grid.cell_data:
                fail(f"{path} missing cell data {key!r}")
        if "DeformedPoint" not in grid.point_data:
            fail(f"{path} missing point data 'DeformedPoint'")
        reference = np.asarray(grid.points, dtype=np.float64)
        deformed = np.asarray(grid.point_data["DeformedPoint"], dtype=np.float64)
        activation_inv = np.asarray(grid.cell_data["ActivationInv"], dtype=np.float64)
        active = np.asarray(grid.cell_data["ActivationMask"], dtype=bool)
        fraction = np.asarray(grid.cell_data["MuscleFraction"], dtype=np.float64)
        muscle_id = np.asarray(grid.cell_data["MuscleId"], dtype=np.int64)
        if deformed.shape != reference.shape or not np.all(np.isfinite(deformed)):
            fail(f"{path} DeformedPoint is invalid")
        volume, det_f, det_ainv, det_g = determinants(
            reference, deformed, tets, activation_inv, config.chunk_cells
        )
        zygomaticus = active & np.isin(muscle_id, ZYGO_IDS)
        rows.append(
            analysis_row(case, volume, det_f, det_ainv, det_g, active, zygomaticus)
        )
        processed[case_name] = {
            "case": case,
            "path": path,
            "path_sha256": sha256(path),
            "reference": reference,
            "deformed": deformed,
            "tets": tets,
            "volume": volume,
            "det_f": det_f,
            "det_ainv": det_ainv,
            "det_g": det_g,
            "active": active,
            "fraction": fraction,
            "muscle_id": muscle_id,
            "activation_inv": activation_inv,
        }

    primary = processed[CASE_PRIMARY]
    comparator = processed[CASE_COMPARATOR]
    criterion = (
        primary["active"]
        & (primary["muscle_id"] == SECTION_MUSCLE_ID)
        & (primary["fraction"] >= 0.5)
    )
    if not np.any(criterion):
        fail(
            "primary contains no id64 active material cells with MuscleFraction >= 0.5"
        )
    centers = primary["reference"][primary["tets"][criterion]].mean(axis=1)
    origin, axes = stable_pca(centers, primary["volume"][criterion])
    local_centers = (centers - origin) @ axes
    selected_source = np.flatnonzero(criterion)
    focus_local_index = int(np.argmin(primary["det_f"][criterion]))
    focus_source_id = int(selected_source[focus_local_index])
    focus = local_centers[focus_local_index]
    span = np.quantile(local_centers, 0.95, axis=0) - np.quantile(
        local_centers, 0.05, axis=0
    )
    half_width = config.slab_half_width_fraction * span[1:]
    if np.any(half_width <= 0.0):
        fail("id64 PCA transverse spans are non-positive")
    section_source = selected_source[
        (np.abs(local_centers[:, 1] - focus[1]) <= half_width[0])
        & (np.abs(local_centers[:, 2] - focus[2]) <= half_width[1])
    ]
    if section_source.size == 0 or focus_source_id not in section_source:
        fail("fold-focused longitudinal slab is empty or omits its focus tet")
    if comparator["tets"].shape != primary["tets"].shape or not np.array_equal(
        comparator["tets"], primary["tets"]
    ):
        fail("primary and comparator do not share the same source tetrahedron IDs")

    for case_name in (CASE_PRIMARY, CASE_COMPARATOR):
        item = processed[case_name]
        fields = {
            "RestVolume": item["volume"],
            "MuscleId": item["muscle_id"],
            "MuscleFraction": item["fraction"],
            "DetF": item["det_f"],
            "DetAinv": item["det_ainv"],
            "DetG": item["det_g"],
            "InvertedDetF": (item["det_f"] < 0.0).astype(np.uint8),
            "DoubleInverted": ((item["det_f"] < 0.0) & (item["det_ainv"] < 0.0)).astype(
                np.uint8
            ),
            "ActivationNorm": np.linalg.norm(item["activation_inv"], axis=1),
        }
        all_source = np.flatnonzero(item["active"])
        for label, source_ids in (
            ("all-muscle", all_source),
            ("section", section_source),
        ):
            for state in ("reference", "deformed"):
                output = config.output_dir / f"{case_name}-{label}-{state}.vtu"
                compact_grid(
                    item["reference"],
                    item["deformed"],
                    item["tets"],
                    source_ids,
                    origin,
                    axes,
                    fields,
                    state == "deformed",
                ).save(output)

    rows.sort(key=lambda row: str(row["case"]))
    csv_path = config.output_dir / "cases.csv"
    write_csv(rows, csv_path)
    section_metrics = {
        case_name: determinant_sign_metrics(
            processed[case_name]["volume"][section_source],
            processed[case_name]["det_f"][section_source],
            processed[case_name]["det_ainv"][section_source],
            processed[case_name]["det_g"][section_source],
        )
        for case_name in (CASE_PRIMARY, CASE_COMPARATOR)
    }
    summary = {
        "source_summary": str(source),
        "source_summary_sha256": sha256(source),
        "case_count": len(rows),
        "state_semantics": {
            "input_vtu": "canonical materialized BEST mesh for each case",
            "metrics": "all fit metrics in cases.csv are best/* values paired with that best mesh",
            "inverse_converged": "source inverse/converged flag; not a stationarity interpretation",
        },
        "determinants": {
            "DetF": "det(deformed edge matrix) / det(reference edge matrix)",
            "DetAinv": "det(I + symmetric ActivationInv[xx, yy, zz, xy, yz, xz])",
            "DetG": "DetF * DetAinv",
        },
        "selection": {
            "primary_case": CASE_PRIMARY,
            "comparator_case": CASE_COMPARATOR,
            "muscle_id": SECTION_MUSCLE_ID,
            "criterion": "ActivationMask && MuscleId == 64 && MuscleFraction >= 0.5",
            "focus_source_cell_id": focus_source_id,
            "focus_primary_det_f": float(primary["det_f"][focus_source_id]),
            "pca_origin": origin.tolist(),
            "pca_axes_columns": axes.tolist(),
            "slab": {
                "longitudinal_axis": "PCA axis 0 (unclipped)",
                "transverse_half_width_fraction_of_q05_q95_span": config.slab_half_width_fraction,
                "transverse_half_width": half_width.tolist(),
                "source_cell_count": int(section_source.size),
                "source_cell_ids": section_source.tolist(),
            },
        },
        "zygomaticus_63_64_criterion": "ActivationMask && MuscleId in {63, 64}; no MuscleFraction threshold",
        "section_determinant_sign_metrics": section_metrics,
        "input_vtus": {
            case_name: {"path": str(item["path"]), "sha256": item["path_sha256"]}
            for case_name, item in processed.items()
        },
        "exports": {
            path.name: sha256(path) for path in sorted(config.output_dir.glob("*.vtu"))
        },
        "cases": rows,
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    table = [
        "# Human face muscle section folding",
        "",
        "All determinants are recomputed from the materialized endpoints. `DoubleInverted` means `DetF < 0` and `DetAinv < 0`; it is descriptive, not a constraint or repair.",
        "",
        "| case | best step | best RMS (mm) | active F-negative volume | zygomaticus F-negative volume | zygomaticus F&A double-inverted cells | roughness (laplacian RMS) | forward failures | inverse converged |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        table.extend(
            [
                f"| {row['case']} | {row['best_step']} | {row['best_error_rms_mm']:.4g} | {row['active_muscle_f_negative_rest_volume_fraction']:.4g} | {row['zygomaticus_63_64_f_negative_rest_volume_fraction']:.4g} | {row['zygomaticus_63_64_double_inverted_cells']} | {row['roughness_displacement_laplacian_rms']:.4g} | {row['forward_failures']} | {row['inverse_converged']} |"
            ]
        )
    (config.output_dir / "results.md").write_text("\n".join(table) + "\n")


if __name__ == "__main__":
    cherries.main(main)
