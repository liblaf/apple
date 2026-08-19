from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from _reference import (
    PREPARED_MESH,
    SOURCE_MANIFEST,
    SOURCE_MANIFEST_SHA256,
    SOURCE_MANIFEST_SIZE_BYTES,
    SOURCE_MATERIAL_SHA256,
    SOURCE_SKIN,
    SOURCE_SKIN_SHA256,
    SOURCE_SKIN_SIZE_BYTES,
    SOURCE_SOLVER_SHA256,
    SOURCE_TOPOLOGY_SHA256,
    enable_reference_modules,
)

from liblaf import cherries, melon
from liblaf.apple.common import ACTIVATION_INV, FRACTION, LAMBDA, MU

enable_reference_modules()

from _human_face_config import SKIN_E, SKIN_NU  # noqa: E402
from _material_heuristics import (  # noqa: E402
    file_sha256,
    skin_material_content_hash,
    skin_solver_content_hash,
    skin_topology_content_hash,
    weighted_quantile,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SOURCE_LABEL = "e100-p000"
EXPECTED_CANDIDATE_LABELS = ("e100-p200", "e005-p000", "e005-p200")
FORMULA_RTOL = 1.0e-13
FORMULA_ATOL = 1.0e-14
EDGE_WARNING_THRESHOLDS = {
    "E_jump_max_MPa": 0.06,
    "E_jump_weighted_rms_MPa": 0.012,
    "activation_jump_max": 0.08,
    "activation_jump_weighted_rms": 0.02,
}
LAME_CONVERSION = (
    "existing 3D isotropic convention: "
    "lambda = E * nu / ((1 + nu) * (1 - 2 * nu)); "
    "mu = E / (2 * (1 + nu))"
)


@dataclass(frozen=True)
class Candidate:
    label: str
    young_min_scale: float
    prestrain_gain: float


@dataclass(frozen=True)
class EdgeGraph:
    interior_i: np.ndarray
    interior_j: np.ndarray
    interior_length: np.ndarray
    boundary_cell: np.ndarray
    boundary_length: np.ndarray
    nonmanifold_eligible_edges: int


CANDIDATES = (
    Candidate("e100-p200", 1.0, 2.0),
    Candidate("e005-p000", 0.05, 0.0),
    Candidate("e005-p200", 0.05, 2.0),
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(PREPARED_MESH)
    input_manifest: Path = cherries.input(SOURCE_MANIFEST)
    input_skin: Path = cherries.input(SOURCE_SKIN)
    output_manifest: Path = cherries.output(
        "10-exaggerated-materials-manifest.json", mkdir=True
    )
    output_table: Path = cherries.output(
        "10-exaggerated-materials-table.md", mkdir=True
    )
    output_dir_name: str = "10-exaggerated-materials"


def file_identity(path: Path) -> dict[str, int | str]:
    return {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}


def array_sha256(name: str, values: np.ndarray, dtype: str = "<f8") -> str:
    canonical = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def require_exact_source_path(actual: Path, expected: Path, *, name: str) -> None:
    if actual.resolve() != expected.resolve():
        msg = f"{name} must be the fixed reference {expected.resolve()}, got {actual.resolve()}"
        raise ValueError(msg)


def require_file_identity(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    name: str,
) -> dict[str, int | str]:
    actual = file_identity(path)
    expected = {"size_bytes": expected_size, "sha256": expected_sha256}
    if actual != expected:
        msg = f"{name} identity mismatch: expected {expected}, got {actual}"
        raise ValueError(msg)
    return actual


def require_fields(surface: pv.PolyData) -> None:
    required_cell = {
        LAMBDA.vtk,
        MU.vtk,
        FRACTION.vtk,
        ACTIVATION_INV.vtk,
        "RestArea",
        "EligibleMaterialTriangle",
        "ExpansionWeight",
        "ContractionSeverityLogCapped",
        "SkinYoungModulusMPa",
        "SkinActivationInvDiag",
        "StressFreeAreaRatio",
    }
    missing = sorted(required_cell - set(surface.cell_data))
    if missing:
        msg = f"source skin is missing required cell arrays: {missing}"
        raise KeyError(msg)


def content_hashes(surface: pv.PolyData) -> dict[str, str]:
    return {
        "topology_sha256": skin_topology_content_hash(surface),
        "material_sha256": skin_material_content_hash(surface),
        "solver_sha256": skin_solver_content_hash(surface),
    }


def validate_source_manifest(  # noqa: C901, PLR0912
    manifest: dict[str, Any], skin_path: Path, skin: pv.PolyData
) -> dict[str, Any]:
    if manifest.get("schema_version") != 2:
        msg = f"source manifest schema must be 2, got {manifest.get('schema_version')}"
        raise ValueError(msg)
    if manifest.get("target") != "Smile" or manifest.get("complete") is not True:
        msg = "source manifest must be the complete fixed Smile material preparation"
        raise ValueError(msg)
    if manifest.get("validation_errors") or manifest.get("candidate_validation_errors"):
        msg = "source manifest contains validation errors"
        raise ValueError(msg)
    if Path(str(manifest.get("input_mesh"))).resolve() != PREPARED_MESH.resolve():
        msg = "source manifest does not reference the fixed prepared Smile mesh"
        raise ValueError(msg)
    if manifest.get("input_mesh_identity_verified_stable") is not True:
        msg = "source manifest did not verify a stable prepared-mesh identity"
        raise ValueError(msg)
    rows = [
        row
        for row in manifest.get("candidates", [])
        if row.get("label") == SOURCE_LABEL
    ]
    if len(rows) != 1:
        msg = f"source manifest must contain exactly one {SOURCE_LABEL} row"
        raise ValueError(msg)
    row = rows[0]
    declared_path = (Path(SOURCE_MANIFEST).parent / str(row["skin/path"])).resolve()
    if declared_path != skin_path.resolve():
        msg = f"source manifest resolves {SOURCE_LABEL} to {declared_path}, not {skin_path}"
        raise ValueError(msg)
    expected_file_identity = {
        "size_bytes": SOURCE_SKIN_SIZE_BYTES,
        "sha256": SOURCE_SKIN_SHA256,
    }
    for key in ("skin/file_identity", "readback/file_identity"):
        if row.get(key) != expected_file_identity:
            msg = f"source manifest {key} does not match the pinned VTP identity"
            raise ValueError(msg)
    hashes = content_hashes(skin)
    expected_hashes = {
        "topology_sha256": SOURCE_TOPOLOGY_SHA256,
        "material_sha256": SOURCE_MATERIAL_SHA256,
        "solver_sha256": SOURCE_SOLVER_SHA256,
    }
    if hashes != expected_hashes:
        msg = f"source VTP solver content mismatch: expected {expected_hashes}, got {hashes}"
        raise ValueError(msg)
    for prefix in ("content", "readback/content"):
        for name, expected in expected_hashes.items():
            key = f"{prefix}/{name}"
            if row.get(key) != expected:
                msg = f"source manifest {key} does not match actual VTP content"
                raise ValueError(msg)
    if row.get("validation/ok") is not True or row.get("readback/ok") is not True:
        msg = f"source manifest {SOURCE_LABEL} row is not validated"
        raise ValueError(msg)
    return row


def validate_source_formula(skin: pv.PolyData) -> dict[str, str]:
    require_fields(skin)
    weight = np.asarray(skin.cell_data["ExpansionWeight"], dtype=np.float64)
    contraction = np.asarray(
        skin.cell_data["ContractionSeverityLogCapped"], dtype=np.float64
    )
    young = np.asarray(skin.cell_data["SkinYoungModulusMPa"], dtype=np.float64)
    activation_diag = np.asarray(
        skin.cell_data["SkinActivationInvDiag"], dtype=np.float64
    )
    arrays = {
        "points": np.asarray(skin.points),
        "ExpansionWeight": weight,
        "ContractionSeverityLogCapped": contraction,
        "SkinYoungModulusMPa": young,
        "SkinActivationInvDiag": activation_diag,
        LAMBDA.vtk: np.asarray(skin.cell_data[LAMBDA.vtk]),
        MU.vtk: np.asarray(skin.cell_data[MU.vtk]),
        FRACTION.vtk: np.asarray(skin.cell_data[FRACTION.vtk]),
        ACTIVATION_INV.vtk: np.asarray(skin.cell_data[ACTIVATION_INV.vtk]),
    }
    nonfinite = sorted(
        name for name, values in arrays.items() if not np.isfinite(values).all()
    )
    if nonfinite:
        msg = f"source VTP contains non-finite solver or driver arrays: {nonfinite}"
        raise ValueError(msg)
    if weight.shape != (skin.n_cells,) or np.any((weight < 0.0) | (weight > 1.0)):
        msg = "source ExpansionWeight must be a per-cell field in [0, 1]"
        raise ValueError(msg)
    if contraction.shape != (skin.n_cells,) or np.any(contraction < 0.0):
        msg = "source ContractionSeverityLogCapped must be a nonnegative per-cell field"
        raise ValueError(msg)
    expected_young = np.full(skin.n_cells, SKIN_E, dtype=np.float64)
    expected_lambda = (
        expected_young * SKIN_NU / ((1.0 + SKIN_NU) * (1.0 - 2.0 * SKIN_NU))
    )
    expected_mu = expected_young / (2.0 * (1.0 + SKIN_NU))
    activation = np.asarray(skin.cell_data[ACTIVATION_INV.vtk], dtype=np.float64)
    formula_checks = (
        np.allclose(young, expected_young, rtol=FORMULA_RTOL, atol=FORMULA_ATOL),
        np.allclose(activation_diag, 0.0, rtol=0.0, atol=0.0),
        activation.shape == (skin.n_cells, 3),
        np.allclose(activation, 0.0, rtol=0.0, atol=0.0),
        np.allclose(
            skin.cell_data[LAMBDA.vtk],
            expected_lambda,
            rtol=FORMULA_RTOL,
            atol=FORMULA_ATOL,
        ),
        np.allclose(
            skin.cell_data[MU.vtk],
            expected_mu,
            rtol=FORMULA_RTOL,
            atol=FORMULA_ATOL,
        ),
        np.allclose(skin.cell_data[FRACTION.vtk], 1.0, rtol=0.0, atol=0.0),
    )
    if not all(formula_checks):
        msg = (
            f"source {SOURCE_LABEL} solver arrays do not implement the baseline formula"
        )
        raise ValueError(msg)
    return {
        "ExpansionWeight_sha256": array_sha256("ExpansionWeight", weight),
        "ContractionSeverityLogCapped_sha256": array_sha256(
            "ContractionSeverityLogCapped", contraction
        ),
    }


def triangle_cells(surface: pv.PolyData) -> np.ndarray:
    encoded = np.asarray(surface.faces, dtype=np.int64)
    if encoded.size == 0 or encoded.size % 4 != 0:
        msg = "source skin must be a non-empty triangular PolyData"
        raise ValueError(msg)
    encoded = encoded.reshape(-1, 4)
    if not np.all(encoded[:, 0] == 3):
        msg = "source skin contains a non-triangle cell"
        raise ValueError(msg)
    return encoded[:, 1:]


def make_edge_graph(surface: pv.PolyData, eligible: np.ndarray) -> EdgeGraph:
    triangles = triangle_cells(surface)
    cell_ids = np.arange(surface.n_cells, dtype=np.int64)
    edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    owners = np.tile(cell_ids, 3)
    edges.sort(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    owners = owners[order]
    starts = np.flatnonzero(np.r_[True, np.any(edges[1:] != edges[:-1], axis=1)])
    counts = np.diff(np.r_[starts, edges.shape[0]])
    group_edges = edges[starts]
    points = np.asarray(surface.points, dtype=np.float64)
    group_lengths = np.linalg.norm(
        points[group_edges[:, 1]] - points[group_edges[:, 0]], axis=1
    )
    if not np.isfinite(group_lengths).all() or np.any(group_lengths <= 0.0):
        msg = "source skin contains a non-finite or zero-length edge"
        raise ValueError(msg)

    paired = counts == 2
    paired_starts = starts[paired]
    paired_i = owners[paired_starts]
    paired_j = owners[paired_starts + 1]
    paired_lengths = group_lengths[paired]
    interior = eligible[paired_i] & eligible[paired_j]
    boundary_pair = eligible[paired_i] ^ eligible[paired_j]
    boundary_pair_cell = np.where(
        eligible[paired_i[boundary_pair]],
        paired_i[boundary_pair],
        paired_j[boundary_pair],
    )

    single = counts == 1
    single_cell = owners[starts[single]]
    single_active = eligible[single_cell]
    boundary_cell = np.concatenate(
        (boundary_pair_cell, single_cell[single_active])
    ).astype(np.int64, copy=False)
    boundary_length = np.concatenate(
        (paired_lengths[boundary_pair], group_lengths[single][single_active])
    )

    nonmanifold_eligible = 0
    for start, count in zip(starts[counts > 2], counts[counts > 2], strict=True):
        if np.any(eligible[owners[start : start + count]]):
            nonmanifold_eligible += 1
    return EdgeGraph(
        interior_i=paired_i[interior],
        interior_j=paired_j[interior],
        interior_length=paired_lengths[interior],
        boundary_cell=boundary_cell,
        boundary_length=boundary_length,
        nonmanifold_eligible_edges=nonmanifold_eligible,
    )


def weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return math.sqrt(float(np.dot(weights, np.square(values)) / weights.sum()))


def edge_jump_metrics(
    values: np.ndarray,
    graph: EdgeGraph,
    *,
    boundary_reference: float,
    eligible: np.ndarray,
) -> dict[str, float]:
    interior = np.abs(values[graph.interior_i] - values[graph.interior_j])
    boundary = np.abs(values[graph.boundary_cell] - boundary_reference)
    jumps = np.concatenate((interior, boundary))
    weights = np.concatenate((graph.interior_length, graph.boundary_length))
    value_range = float(values[eligible].max() - values[eligible].min())
    return {
        "max": float(jumps.max()),
        "edge_length_weighted_q99": weighted_quantile(jumps, weights, 0.99),
        "edge_length_weighted_rms": weighted_rms(jumps, weights),
        "face_boundary_max": float(boundary.max()) if boundary.size else 0.0,
        "max_fraction_of_eligible_range": (
            float(jumps.max() / value_range) if value_range > 0.0 else 0.0
        ),
    }


def make_candidate(
    source: pv.PolyData, candidate: Candidate
) -> tuple[pv.PolyData, dict[str, np.ndarray]]:
    weight = np.asarray(source.cell_data["ExpansionWeight"], dtype=np.float64)
    contraction = np.asarray(
        source.cell_data["ContractionSeverityLogCapped"], dtype=np.float64
    )
    young = SKIN_E * np.power(candidate.young_min_scale, weight)
    activation_diag = np.exp(0.5 * candidate.prestrain_gain * contraction) - 1.0
    activation_inv = np.zeros((source.n_cells, 3), dtype=np.float64)
    activation_inv[:, 0] = activation_diag
    activation_inv[:, 1] = activation_diag
    stress_free_area_ratio = np.reciprocal(np.square(1.0 + activation_diag))
    lambda_ = young * SKIN_NU / ((1.0 + SKIN_NU) * (1.0 - 2.0 * SKIN_NU))
    mu = young / (2.0 * (1.0 + SKIN_NU))

    skin = source.copy(deep=True)
    skin.cell_data[LAMBDA.vtk] = lambda_
    skin.cell_data[MU.vtk] = mu
    skin.cell_data[FRACTION.vtk] = np.ones(source.n_cells, dtype=np.float64)
    skin.cell_data[ACTIVATION_INV.vtk] = activation_inv
    skin.cell_data["SkinYoungModulusMPa"] = young
    skin.cell_data["SkinActivationInvDiag"] = activation_diag
    skin.cell_data["StressFreeAreaRatio"] = stress_free_area_ratio
    return skin, {
        "young": young,
        "lambda": lambda_,
        "mu": mu,
        "activation_diag": activation_diag,
        "activation_inv": activation_inv,
        "stress_free_area_ratio": stress_free_area_ratio,
    }


def validate_candidate_surface(  # noqa: C901
    skin: pv.PolyData,
    source: pv.PolyData,
    candidate: Candidate,
    *,
    expected_content: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    errors: list[str] = []
    try:
        require_fields(skin)
    except (KeyError, ValueError) as error:
        return [str(error)], {}
    if skin.n_points != source.n_points or skin.n_cells != source.n_cells:
        errors.append("point or triangle count differs from the pinned source VTP")
    weight = np.asarray(skin.cell_data["ExpansionWeight"], dtype=np.float64)
    contraction = np.asarray(
        skin.cell_data["ContractionSeverityLogCapped"], dtype=np.float64
    )
    source_weight = np.asarray(source.cell_data["ExpansionWeight"], dtype=np.float64)
    source_contraction = np.asarray(
        source.cell_data["ContractionSeverityLogCapped"], dtype=np.float64
    )
    if not np.array_equal(weight, source_weight):
        errors.append("ExpansionWeight differs from the pinned source VTP")
    if not np.array_equal(contraction, source_contraction):
        errors.append("ContractionSeverityLogCapped differs from the pinned source VTP")

    expected_young = SKIN_E * np.power(candidate.young_min_scale, weight)
    expected_activation_diag = (
        np.exp(0.5 * candidate.prestrain_gain * contraction) - 1.0
    )
    expected_activation = np.zeros((skin.n_cells, 3), dtype=np.float64)
    expected_activation[:, 0] = expected_activation_diag
    expected_activation[:, 1] = expected_activation_diag
    expected_lambda = (
        expected_young * SKIN_NU / ((1.0 + SKIN_NU) * (1.0 - 2.0 * SKIN_NU))
    )
    expected_mu = expected_young / (2.0 * (1.0 + SKIN_NU))
    expected_stress_free = np.reciprocal(np.square(1.0 + expected_activation_diag))
    actual = {
        "SkinYoungModulusMPa": np.asarray(
            skin.cell_data["SkinYoungModulusMPa"], dtype=np.float64
        ),
        "SkinActivationInvDiag": np.asarray(
            skin.cell_data["SkinActivationInvDiag"], dtype=np.float64
        ),
        "StressFreeAreaRatio": np.asarray(
            skin.cell_data["StressFreeAreaRatio"], dtype=np.float64
        ),
        LAMBDA.vtk: np.asarray(skin.cell_data[LAMBDA.vtk], dtype=np.float64),
        MU.vtk: np.asarray(skin.cell_data[MU.vtk], dtype=np.float64),
        FRACTION.vtk: np.asarray(skin.cell_data[FRACTION.vtk], dtype=np.float64),
        ACTIVATION_INV.vtk: np.asarray(
            skin.cell_data[ACTIVATION_INV.vtk], dtype=np.float64
        ),
    }
    nonfinite = sorted(
        name for name, values in actual.items() if not np.isfinite(values).all()
    )
    if nonfinite:
        errors.append(f"non-finite candidate arrays: {nonfinite}")
    formula_checks = {
        "SkinYoungModulusMPa": (actual["SkinYoungModulusMPa"], expected_young),
        "SkinActivationInvDiag": (
            actual["SkinActivationInvDiag"],
            expected_activation_diag,
        ),
        "StressFreeAreaRatio": (
            actual["StressFreeAreaRatio"],
            expected_stress_free,
        ),
        LAMBDA.vtk: (actual[LAMBDA.vtk], expected_lambda),
        MU.vtk: (actual[MU.vtk], expected_mu),
        ACTIVATION_INV.vtk: (actual[ACTIVATION_INV.vtk], expected_activation),
    }
    for name, (values, expected) in formula_checks.items():
        if values.shape != expected.shape or not np.allclose(
            values, expected, rtol=FORMULA_RTOL, atol=FORMULA_ATOL
        ):
            errors.append(f"{name} does not match the fixed candidate formula")
    if actual[FRACTION.vtk].shape != (skin.n_cells,) or not np.allclose(
        actual[FRACTION.vtk], 1.0, rtol=0.0, atol=0.0
    ):
        errors.append("Fraction must be exactly one on every skin triangle")
    if np.any(actual["SkinYoungModulusMPa"] <= 0.0):
        errors.append("Young's modulus must remain positive")
    if np.any(actual["SkinActivationInvDiag"] < 0.0):
        errors.append("prestrain activation must remain nonnegative")

    hashes = content_hashes(skin)
    if hashes.get("topology_sha256") != SOURCE_TOPOLOGY_SHA256:
        errors.append("candidate topology content differs from the pinned source VTP")
    if expected_content is not None and hashes != expected_content:
        errors.append("candidate content hashes changed during VTP readback")
    return errors, hashes


def area_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(values, weights) / weights.sum())


def area_weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return math.sqrt(area_weighted_mean(np.square(values), weights))


def candidate_metrics(
    skin: pv.PolyData,
    candidate: Candidate,
    graph: EdgeGraph,
) -> tuple[dict[str, Any], list[str]]:
    eligible = np.asarray(skin.cell_data["EligibleMaterialTriangle"], dtype=bool)
    rest_area = np.asarray(skin.cell_data["RestArea"], dtype=np.float64)
    weight = np.asarray(skin.cell_data["ExpansionWeight"], dtype=np.float64)
    contraction = np.asarray(
        skin.cell_data["ContractionSeverityLogCapped"], dtype=np.float64
    )
    young = np.asarray(skin.cell_data["SkinYoungModulusMPa"], dtype=np.float64)
    activation = np.asarray(skin.cell_data["SkinActivationInvDiag"], dtype=np.float64)
    stress_free = np.asarray(skin.cell_data["StressFreeAreaRatio"], dtype=np.float64)
    eligible_area = rest_area[eligible]
    e_jump = edge_jump_metrics(
        young,
        graph,
        boundary_reference=float(SKIN_E),
        eligible=eligible,
    )
    activation_jump = edge_jump_metrics(
        activation,
        graph,
        boundary_reference=0.0,
        eligible=eligible,
    )
    warnings: list[str] = []
    comparisons = (
        ("E max edge jump", e_jump["max"], EDGE_WARNING_THRESHOLDS["E_jump_max_MPa"]),
        (
            "E weighted-RMS edge jump",
            e_jump["edge_length_weighted_rms"],
            EDGE_WARNING_THRESHOLDS["E_jump_weighted_rms_MPa"],
        ),
        (
            "ActivationInv max edge jump",
            activation_jump["max"],
            EDGE_WARNING_THRESHOLDS["activation_jump_max"],
        ),
        (
            "ActivationInv weighted-RMS edge jump",
            activation_jump["edge_length_weighted_rms"],
            EDGE_WARNING_THRESHOLDS["activation_jump_weighted_rms"],
        ),
    )
    for name, actual, threshold in comparisons:
        if actual > threshold:
            warnings.append(
                f"{name} {actual:.6g} exceeds legacy threshold {threshold:.6g}; "
                "recorded as warning only"
            )
    return {
        "skin/base_E_MPa": float(SKIN_E),
        "skin/nu": float(SKIN_NU),
        "skin/eligible_triangles": int(eligible.sum()),
        "skin/expansion_triangles": int(np.count_nonzero(weight > 0.0)),
        "skin/contraction_triangles": int(np.count_nonzero(contraction > 0.0)),
        "skin/E_MPa_min": float(young.min()),
        "skin/E_MPa_max": float(young.max()),
        "skin/E_MPa_mean": float(young.mean()),
        "skin/E_MPa_area_weighted_mean": area_weighted_mean(
            young[eligible], eligible_area
        ),
        "skin/activation_inv_diag_max": float(activation.max()),
        "skin/activation_inv_diag_rms": float(
            np.linalg.norm(activation) / math.sqrt(activation.size)
        ),
        "skin/activation_inv_diag_area_weighted_mean": area_weighted_mean(
            activation[eligible], eligible_area
        ),
        "skin/activation_inv_diag_area_weighted_rms": area_weighted_rms(
            activation[eligible], eligible_area
        ),
        "skin/stress_free_area_ratio_min": float(stress_free.min()),
        "field/ExpansionWeight_max": float(weight.max()),
        "field/ContractionSeverityLogCapped_max": float(contraction.max()),
        "field/E_edge_jump_MPa": e_jump,
        "field/activation_edge_jump": activation_jump,
        "formula/young": "E = 0.2 * young_min_scale ** ExpansionWeight",
        "formula/prestrain": (
            "Ainv_diag = exp(0.5 * prestrain_gain * ContractionSeverityLogCapped) - 1"
        ),
        "skin/lame_conversion": LAME_CONVERSION,
        "candidate/label": candidate.label,
    }, warnings


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| candidate | E scale | gain | E min MPa | area-mean E MPa | Ainv max | min stress-free area ratio | E jump q99/max MPa | Ainv jump q99/max | hard gates | warnings | skin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        e_jump = row["field/E_edge_jump_MPa"]
        activation_jump = row["field/activation_edge_jump"]
        lines.append(
            f"| {row['label']} | {row['young_min_scale']:.3g} | "
            f"{row['prestrain_gain']:.3g} | {row['skin/E_MPa_min']:.6g} | "
            f"{row['skin/E_MPa_area_weighted_mean']:.6g} | "
            f"{row['skin/activation_inv_diag_max']:.6g} | "
            f"{row['skin/stress_free_area_ratio_min']:.6g} | "
            f"{e_jump['edge_length_weighted_q99']:.4g}/{e_jump['max']:.4g} | "
            f"{activation_jump['edge_length_weighted_q99']:.4g}/"
            f"{activation_jump['max']:.4g} | "
            f"{'ok' if row['validation/ok'] else 'failed'} | "
            f"{len(row['warnings'])} | `{row['skin/path']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    labels = tuple(str(row["label"]) for row in rows)
    if labels != EXPECTED_CANDIDATE_LABELS:
        errors.append(
            f"candidate labels/order {labels} != fixed design {EXPECTED_CANDIDATE_LABELS}"
        )
    if len({str(row["skin/path"]) for row in rows}) != len(rows):
        errors.append("candidate output paths are not unique")
    if len({str(row["content/solver_sha256"]) for row in rows}) != len(rows):
        errors.append("candidate solver-content hashes are not unique")
    if {str(row["content/topology_sha256"]) for row in rows} != {
        SOURCE_TOPOLOGY_SHA256
    }:
        errors.append("candidate topology hashes differ from the pinned source")
    errors.extend(
        f"{row['label']}: candidate hard gates failed"
        for row in rows
        if not bool(row["validation/ok"])
    )
    return errors


def run(cfg: Config) -> None:  # noqa: C901, PLR0915
    require_exact_source_path(cfg.input_mesh, PREPARED_MESH, name="input_mesh")
    require_exact_source_path(
        cfg.input_manifest, SOURCE_MANIFEST, name="input_manifest"
    )
    require_exact_source_path(cfg.input_skin, SOURCE_SKIN, name="input_skin")
    manifest_identity_before = require_file_identity(
        cfg.input_manifest,
        expected_size=SOURCE_MANIFEST_SIZE_BYTES,
        expected_sha256=SOURCE_MANIFEST_SHA256,
        name="source manifest",
    )
    skin_identity_before = require_file_identity(
        cfg.input_skin,
        expected_size=SOURCE_SKIN_SIZE_BYTES,
        expected_sha256=SOURCE_SKIN_SHA256,
        name="source baseline VTP",
    )
    source_manifest = json.loads(cfg.input_manifest.read_text(encoding="utf-8"))
    mesh_identity_before = file_identity(cfg.input_mesh)
    if mesh_identity_before != source_manifest.get("input_mesh_identity"):
        msg = (
            "prepared input mesh identity differs from the pinned source manifest: "
            f"expected {source_manifest.get('input_mesh_identity')}, "
            f"got {mesh_identity_before}"
        )
        raise ValueError(msg)
    source_skin = pv.read(cfg.input_skin)
    if not isinstance(source_skin, pv.PolyData):
        msg = f"source skin read as {type(source_skin).__name__}, expected PolyData"
        raise TypeError(msg)
    source_row = validate_source_manifest(source_manifest, cfg.input_skin, source_skin)
    source_driver_hashes = validate_source_formula(source_skin)
    eligible = np.asarray(source_skin.cell_data["EligibleMaterialTriangle"], dtype=bool)
    graph = make_edge_graph(source_skin, eligible)
    topology_errors: list[str] = []
    if graph.nonmanifold_eligible_edges:
        topology_errors.append(
            f"eligible material domain has {graph.nonmanifold_eligible_edges} nonmanifold edges"
        )
    lhs = int(3 * eligible.sum())
    rhs = int(2 * graph.interior_i.size + graph.boundary_cell.size)
    if lhs != rhs:
        topology_errors.append(
            f"eligible edge identity failed: 3F={lhs}, 2Eint+Eboundary={rhs}"
        )
    if topology_errors:
        msg = f"source edge topology validation failed: {topology_errors}"
        raise ValueError(msg)

    output_dir = cfg.output_manifest.parent / cfg.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for step, candidate in enumerate(CANDIDATES):
        cherries.set_step(step)
        skin, _ = make_candidate(source_skin, candidate)
        prewrite_errors, prewrite_hashes = validate_candidate_surface(
            skin, source_skin, candidate
        )
        if prewrite_errors:
            msg = f"{candidate.label} prewrite hard gates failed: {prewrite_errors}"
            raise RuntimeError(msg)
        metrics, warnings = candidate_metrics(skin, candidate, graph)
        path = output_dir / f"skin-{candidate.label}.vtp"
        melon.save(skin, path)
        cherries.log_output(path)
        readback = pv.read(path)
        if not isinstance(readback, pv.PolyData):
            msg = f"{path} read back as {type(readback).__name__}, expected PolyData"
            raise TypeError(msg)
        readback_errors, readback_hashes = validate_candidate_surface(
            readback,
            source_skin,
            candidate,
            expected_content=prewrite_hashes,
        )
        row_errors = [f"readback: {error}" for error in readback_errors]
        row = {
            "schema_version": SCHEMA_VERSION,
            "label": candidate.label,
            "young_min_scale": candidate.young_min_scale,
            "prestrain_gain": candidate.prestrain_gain,
            **metrics,
            "content/n_points": int(skin.n_points),
            "content/n_triangles": int(skin.n_cells),
            "content/topology_sha256": prewrite_hashes["topology_sha256"],
            "content/material_sha256": prewrite_hashes["material_sha256"],
            "content/solver_sha256": prewrite_hashes["solver_sha256"],
            "skin/path": str(path.relative_to(cfg.output_manifest.parent)),
            "skin/file_identity": file_identity(path),
            "readback/n_points": int(readback.n_points),
            "readback/n_triangles": int(readback.n_cells),
            "readback/content/topology_sha256": readback_hashes.get("topology_sha256"),
            "readback/content/material_sha256": readback_hashes.get("material_sha256"),
            "readback/content/solver_sha256": readback_hashes.get("solver_sha256"),
            "readback/formula_consistent": not readback_errors,
            "readback/ok": not readback_errors,
            "readback/errors": readback_errors,
            "warnings": warnings,
            "validation/errors": row_errors,
            "validation/ok": not row_errors,
        }
        rows.append(row)
        cherries.log_metrics(
            {
                "material": {
                    "E_MPa_min": row["skin/E_MPa_min"],
                    "activation_inv_diag_max": row["skin/activation_inv_diag_max"],
                    "stress_free_area_ratio_min": row[
                        "skin/stress_free_area_ratio_min"
                    ],
                    "warning_count": len(warnings),
                }
            }
        )
        logger.info(
            "%s E_min=%.6g MPa Ainv_max=%.6g hard_gates=%s warnings=%d",
            candidate.label,
            row["skin/E_MPa_min"],
            row["skin/activation_inv_diag_max"],
            row["validation/ok"],
            len(warnings),
        )

    manifest_identity_after = file_identity(cfg.input_manifest)
    skin_identity_after = file_identity(cfg.input_skin)
    mesh_identity_after = file_identity(cfg.input_mesh)
    manifest_errors = validate_rows(rows)
    if manifest_identity_after != manifest_identity_before:
        manifest_errors.append("source manifest changed during preparation")
    if skin_identity_after != skin_identity_before:
        manifest_errors.append("source baseline VTP changed during preparation")
    if mesh_identity_after != mesh_identity_before:
        manifest_errors.append("prepared input mesh changed during preparation")
    candidate_errors = {
        str(row["label"]): row["validation/errors"]
        for row in rows
        if row["validation/errors"]
    }
    warnings = {str(row["label"]): row["warnings"] for row in rows if row["warnings"]}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": not manifest_errors and not candidate_errors,
        "design": "exaggerated-heterogeneous-mechanism-screen",
        "experiment": "human-face Smile exaggerated heterogeneous skin materials",
        "purpose": (
            "mechanism-contrast screen; deliberately exaggerated and not a "
            "physiological calibration"
        ),
        "source": {
            "manifest/path": str(cfg.input_manifest),
            "manifest/file_identity": manifest_identity_before,
            "manifest/identity_verified_stable": (
                manifest_identity_before == manifest_identity_after
            ),
            "candidate": SOURCE_LABEL,
            "candidate/manifest_row_solver_sha256": source_row["content/solver_sha256"],
            "skin/path": str(cfg.input_skin),
            "skin/file_identity": skin_identity_before,
            "skin/identity_verified_stable": skin_identity_before
            == skin_identity_after,
            "content/topology_sha256": SOURCE_TOPOLOGY_SHA256,
            "content/material_sha256": SOURCE_MATERIAL_SHA256,
            "content/solver_sha256": SOURCE_SOLVER_SHA256,
            "driver_fields": source_driver_hashes,
        },
        "input_mesh": str(cfg.input_mesh),
        "input_mesh_identity": mesh_identity_before,
        "input_mesh_identity_verified_stable": (
            mesh_identity_before == mesh_identity_after
        ),
        "fixed_design": {
            "candidate_labels": list(EXPECTED_CANDIDATE_LABELS),
            "candidates": [
                {
                    "label": candidate.label,
                    "young_min_scale": candidate.young_min_scale,
                    "prestrain_gain": candidate.prestrain_gain,
                }
                for candidate in CANDIDATES
            ],
            "heterogeneous_fields": [
                "ExpansionWeight",
                "ContractionSeverityLogCapped",
            ],
            "young_rule": "E = 0.2 * young_min_scale ** ExpansionWeight",
            "prestrain_rule": (
                "Ainv_diag = exp(0.5 * prestrain_gain * "
                "ContractionSeverityLogCapped) - 1"
            ),
            "lame_conversion": LAME_CONVERSION,
        },
        "edge_jump_policy": {
            "status": "warning_only",
            "legacy_thresholds": EDGE_WARNING_THRESHOLDS,
            "warnings": warnings,
        },
        "surface_topology": {
            "eligible_triangles": int(eligible.sum()),
            "eligible_interior_edges": int(graph.interior_i.size),
            "eligible_boundary_edges": int(graph.boundary_cell.size),
            "nonmanifold_eligible_edges": graph.nonmanifold_eligible_edges,
            "identity_lhs_3F": lhs,
            "identity_rhs_2Eint_plus_Eboundary": rhs,
        },
        "n_candidates": len(rows),
        "validation_errors": manifest_errors,
        "candidate_validation_errors": candidate_errors,
        "candidates": rows,
    }
    cfg.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    write_table(cfg.output_table, rows)
    logger.info("Wrote %s", cfg.output_manifest)
    logger.info("Wrote %s", cfg.output_table)
    if manifest_errors or candidate_errors:
        msg = (
            "exaggerated material preparation hard gates failed: "
            f"manifest={manifest_errors}, candidates={candidate_errors}"
        )
        raise RuntimeError(msg)


if __name__ == "__main__":
    cherries.main(run)
