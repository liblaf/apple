# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
"""Build the pinned CPU fixture for the face inverse-physics comparison.

The fixture keeps the June tetrahedral volume and smile target, installs the
later conservative cut constraint, maps the corrected zero-prestrain skin,
and replaces the historical broad activation mask with a named facial-
expression domain.  Fiber directions are geometry estimates for experiments;
they are not anatomical measurements.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

HERE = Path(__file__).resolve().parent
GROUP_DIR = HERE.parent
REPO_ROOT = HERE.parents[5]

DEFAULT_VOLUME = (
    REPO_ROOT
    / "exp/2026/06/17/human-face-smile-prestrain-v2/data/10-human-face-prepared.vtu"
)
DEFAULT_CUT_REFERENCE = (
    REPO_ROOT
    / "exp/2026/08/17/human-face-smile-material-heuristic-sweep/data/10-material-candidates/skin-e100-p000.vtp"
)
DEFAULT_SKIN = (
    REPO_ROOT
    / "exp/2026/08/18/human-face-smile-plane-stress-skin/data/10-corrected-baseline/skin-isface-e0200-p000.vtp"
)

EXPECTED_VOLUME_SHA256 = (
    "8131d6944b322d7c1e21918688f297e2887b42bf4dbc19ce36259b007e8dc563"
)
EXPECTED_CUT_REFERENCE_SHA256 = (
    "ffd586e8e1625facc89e87be803fbac16b374ae3d64b34916fd280cd05104c5f"
)
EXPECTED_SKIN_SHA256 = (
    "4c7ddce893eed4a8d0590042488ae1b35f0cae23383db6bc9814427eb6f7cc6f"
)
EXPECTED_CUT_TRIANGLE_TOPOLOGY_SHA256 = (
    "8207cda8f9e11dbb4406f683e5ad818a6950e3515ac373719514094fb5b7fe5d"
)
EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256 = (
    "ca39cdc839855be34e75222964a1e5c129dd210e8800c684d7e6d1ce6424f138"
)

# Exact IDs and labels in the pinned June mesh.  They intentionally exclude
# chewing, ocular, neck, fascia, and tendon regions from the primary inverse.
FACIAL_EXPRESSION_MUSCLES: dict[int, str] = {
    28: "Occipitofrontalis epicranius001_Head_muscles_0",
    57: "Levator labii superioris001_Head_muscles_0_0",
    58: "Levator labii superioris001_Head_muscles_0_1",
    61: "Buccinator001_Head_muscles_0_0",
    62: "Buccinator001_Head_muscles_0_1",
    63: "Zygomaticus major001_Head_muscles_0_0",
    64: "Zygomaticus major001_Head_muscles_0_1",
    73: "Risorius001_Head_muscles_0_0",
    74: "Risorius001_Head_muscles_0_1",
    93: "Mentalis001_Head_muscles_0_0",
    94: "Mentalis001_Head_muscles_0_1",
    97: "Orbicularis oculi001_Head_muscles_0_0",
    98: "Orbicularis oculi001_Head_muscles_0_1",
    99: "Depressor anguli001_Head_muscles_0_0",
    100: "Depressor anguli001_Head_muscles_0_1",
    101: "Nasalis alarportion001_Head_muscles_0_0",
    102: "Nasalis alarportion001_Head_muscles_0_1",
    108: "Procerus001_Head_muscles_0",
    110: "Corrugator supercilii001_Head_muscles_0_0",
    111: "Corrugator supercilii001_Head_muscles_0_1",
    116: "Nasalis transverse portion001_Head_muscles_0",
    139: "Depressor septi001_Head_muscles_0",
    142: "Levator anguli oris001_Head_muscles_0_0",
    143: "Levator anguli oris001_Head_muscles_0_1",
    162: "Depressor labii inferioris001_Head_muscles_0_0",
    163: "Depressor labii inferioris001_Head_muscles_0_1",
    218: "Zygomaticus minor001_Head_muscles_0_0",
    219: "Zygomaticus minor001_Head_muscles_0_1",
    254: "Orbicularis oris001_Head_muscles_0",
    255: "Platysma001_Head_muscles_0_0",
    256: "Platysma001_Head_muscles_0_1",
    257: "Depressor supercilli001_Head_muscles_0_0",
    258: "Depressor supercilli001_Head_muscles_0_1",
    283: "Levator labii superioris alaeque nasi001_Head_muscles_0_0",
    284: "Levator labii superioris alaeque nasi001_Head_muscles_0_1",
}

RING_MUSCLE_IDS = frozenset({97, 98, 254})


@dataclass(frozen=True)
class FaceFixture:
    """CPU-prepared model inputs and their machine-readable audit summary."""

    mesh: pv.UnstructuredGrid
    skin: pv.PolyData
    active_cell_ids: np.ndarray
    control_ids: np.ndarray
    region_names: tuple[str, ...]
    region_fibers: np.ndarray
    confidence: np.ndarray
    summary: dict[str, Any]


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_identity(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned {label}: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _triangles(surface: pv.PolyData) -> np.ndarray:
    encoded = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
    if encoded.shape[0] != surface.n_cells or np.any(encoded[:, 0] != 3):
        raise ValueError("expected an all-triangle PolyData")
    return encoded[:, 1:]


def _tets(mesh: pv.UnstructuredGrid) -> np.ndarray:
    if pv.CellType.TETRA not in mesh.cells_dict:
        raise ValueError("expected an all-tetrahedron UnstructuredGrid")
    tets = np.asarray(mesh.cells_dict[pv.CellType.TETRA], dtype=np.int64)
    if tets.shape != (mesh.n_cells, 4):
        raise ValueError("mixed cell types are not supported by the face fixture")
    return tets


def _map_global_ids(mesh: pv.UnstructuredGrid, surface: pv.PolyData) -> np.ndarray:
    mesh_ids = np.asarray(mesh.point_data["GlobalPointId"], dtype=np.int64)
    requested = np.asarray(surface.point_data["GlobalPointId"], dtype=np.int64)
    if mesh_ids.shape != (mesh.n_points,) or np.unique(mesh_ids).size != mesh.n_points:
        raise ValueError("volume GlobalPointId must be a unique point vector")
    if (
        requested.shape != (surface.n_points,)
        or np.unique(requested).size != surface.n_points
    ):
        raise ValueError("surface GlobalPointId must be a unique point vector")
    order = np.argsort(mesh_ids)
    positions = np.searchsorted(mesh_ids[order], requested)
    if np.any(positions >= mesh.n_points) or not np.array_equal(
        mesh_ids[order[positions]], requested
    ):
        raise ValueError("surface GlobalPointId values do not map to the volume")
    mapped = order[positions]
    if not np.array_equal(
        np.asarray(surface.points, dtype=np.float64),
        np.asarray(mesh.points, dtype=np.float64)[mapped],
    ):
        raise ValueError("surface points differ from mapped volume points")
    return mapped


def configure_hard_fixed_cut(
    mesh: pv.UnstructuredGrid, cut_reference: pv.PolyData
) -> tuple[np.ndarray, dict[str, Any]]:
    """Install the pinned conservative constraint on the artificial cut."""
    if mesh.n_points != 228_660 or mesh.n_cells != 1_146_517:
        raise ValueError("pinned June volume topology changed")
    mesh.point_data["GlobalPointId"] = np.arange(mesh.n_points, dtype=np.int64)
    if cut_reference.n_cells != 128_172:
        raise ValueError("pinned full-boundary triangle count changed")

    triangles = _triangles(cut_reference)
    group_ids = np.asarray(cut_reference.point_data["GroupId"], dtype=np.int64)
    if group_ids.shape != (cut_reference.n_points,):
        raise ValueError("cut-reference GroupId must be a point vector")
    unassigned = group_ids == -1
    if int(unassigned.sum()) != 6_000:
        raise ValueError("pinned cut-marker point count changed")
    cut_triangles = np.any(unassigned[triangles], axis=1)
    if int(cut_triangles.sum()) != 13_165:
        raise ValueError("pinned artificial-cut triangle count changed")

    cut_keys = np.sort(
        np.asarray(cut_reference.point_data["GlobalPointId"], dtype=np.int64)[
            triangles[cut_triangles]
        ],
        axis=1,
    ).astype("<i8", copy=False)
    order = np.lexsort((cut_keys[:, 2], cut_keys[:, 1], cut_keys[:, 0]))
    topology_digest = hashlib.sha256(
        np.ascontiguousarray(cut_keys[order]).tobytes()
    ).hexdigest()
    if topology_digest != EXPECTED_CUT_TRIANGLE_TOPOLOGY_SHA256:
        raise ValueError("pinned artificial-cut topology digest changed")

    mapped = _map_global_ids(mesh, cut_reference)
    cut_source_ids = np.unique(triangles[cut_triangles])
    cut_ids = np.sort(mapped[cut_source_ids]).astype(np.int64, copy=False)
    if cut_ids.size != 6_980:
        raise ValueError("pinned artificial-cut incident-vertex count changed")
    cut_digest = hashlib.sha256(
        np.ascontiguousarray(cut_ids.astype("<i8", copy=False)).tobytes()
    ).hexdigest()
    if cut_digest != EXPECTED_CUT_INCIDENT_GLOBAL_IDS_SHA256:
        raise ValueError("pinned artificial-cut incident-point digest changed")

    historical = np.asarray(mesh.point_data["IsFixed"], dtype=bool)
    fixed_mask = np.asarray(mesh.point_data["FixedMask"], dtype=bool)
    fixed_value = np.asarray(mesh.point_data["FixedValue"], dtype=np.float64)
    is_face = np.asarray(mesh.point_data["IsFace"], dtype=bool)
    if historical.shape != (mesh.n_points,):
        raise ValueError("historical IsFixed must be a point vector")
    if not np.array_equal(fixed_mask, np.repeat(historical[:, None], 3, axis=1)):
        raise ValueError("historical FixedMask differs from IsFixed")
    if not np.array_equal(fixed_value, np.zeros_like(fixed_value)):
        raise ValueError("fixture requires exact-zero fixed values")
    if np.any(is_face[cut_ids]):
        raise ValueError("artificial-cut vertices unexpectedly overlap IsFace")
    preexisting = historical[cut_ids]
    if int(preexisting.sum()) != 380:
        raise ValueError("pinned preexisting cut fixation count changed")

    installed = historical.copy()
    installed[cut_ids] = True
    installed_mask = np.repeat(installed[:, None], 3, axis=1)
    incident = np.zeros(mesh.n_points, dtype=np.int8)
    incident[cut_ids] = 1
    preexisting_field = np.zeros(mesh.n_points, dtype=np.int8)
    preexisting_field[cut_ids[preexisting]] = 1
    added_field = np.zeros(mesh.n_points, dtype=np.int8)
    added_field[cut_ids[~preexisting]] = 1
    mesh.point_data["HistoricalIsFixed"] = historical.astype(np.int8)
    mesh.point_data["ArtificialCutIncident"] = incident
    mesh.point_data["CutBoundaryPreexistingFixed"] = preexisting_field
    mesh.point_data["CutBoundaryAddedFixed"] = added_field
    mesh.point_data["IsFixed"] = installed
    mesh.point_data["FixedMask"] = installed_mask
    mesh.point_data["FixedValue"] = np.zeros_like(fixed_value)
    if int(installed.sum()) != 33_636 or int(installed_mask.sum()) != 100_908:
        raise ValueError("pinned hard-fixed model count changed")

    return cut_ids, {
        "policy": "all-artificial-cut-incident-vertices-hard-fixed",
        "marker": "source skin triangle touches mapped GroupId=-1 vertex",
        "hard_fixed_is_ground_truth": False,
        "triangles": int(cut_triangles.sum()),
        "incident_vertices": int(cut_ids.size),
        "preexisting_fixed_vertices": int(preexisting.sum()),
        "newly_fixed_vertices": int((~preexisting).sum()),
        "model_total_fixed_vertices": int(installed.sum()),
        "model_total_fixed_dofs": int(installed_mask.sum()),
        "triangle_topology_sha256": topology_digest,
        "incident_global_ids_sha256": cut_digest,
    }


def map_skin_to_volume(skin: pv.PolyData, mesh: pv.UnstructuredGrid) -> np.ndarray:
    """Validate and return the corrected skin-to-volume point map."""
    if skin.n_points != 15_299 or skin.n_cells != 29_899:
        raise ValueError("pinned corrected skin topology changed")
    mapped = _map_global_ids(mesh, skin)
    E = np.asarray(skin.cell_data["SkinYoungModulusMPa"], dtype=np.float64)
    nu = np.asarray(skin.cell_data["SkinPoissonRatio"], dtype=np.float64)
    lambda_ = np.asarray(skin.cell_data["lambda"], dtype=np.float64)
    mu = np.asarray(skin.cell_data["mu"], dtype=np.float64)
    fraction = np.asarray(skin.cell_data["Fraction"], dtype=np.float64)
    activation = np.asarray(skin.cell_data["ActivationInv"], dtype=np.float64)
    if not np.array_equal(E, np.full(skin.n_cells, 0.2)):
        raise ValueError("corrected skin E differs from 0.2 MPa")
    if not np.array_equal(nu, np.full(skin.n_cells, 0.49)):
        raise ValueError("corrected skin nu differs from 0.49")
    if not np.allclose(lambda_, E * nu / (1.0 - nu**2), rtol=1e-13, atol=1e-14):
        raise ValueError("corrected skin Lambda is not the plane-stress value")
    if not np.allclose(mu, E / (2.0 * (1.0 + nu)), rtol=1e-13, atol=1e-14):
        raise ValueError("corrected skin Mu is not the plane-stress value")
    if not np.array_equal(fraction, np.ones(skin.n_cells)):
        raise ValueError("corrected skin Fraction differs from one")
    if activation.shape != (skin.n_cells, 3) or np.any(activation != 0.0):
        raise ValueError("corrected skin is not exact zero-prestrain")
    return mapped


def _muscle_names(mesh: pv.UnstructuredGrid) -> tuple[str, ...]:
    raw = np.asarray(mesh.field_data["MuscleName"]).reshape(-1)
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw
    )


def select_facial_expression_cells(
    mesh: pv.UnstructuredGrid,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Replace the historical broad mask with exact named expression muscles."""
    names = _muscle_names(mesh)
    for muscle_id, expected in FACIAL_EXPRESSION_MUSCLES.items():
        if muscle_id >= len(names) or names[muscle_id] != expected:
            actual = None if muscle_id >= len(names) else names[muscle_id]
            raise ValueError(
                f"MuscleId {muscle_id} label changed: expected {expected!r}, got {actual!r}"
            )
    muscle_id = np.asarray(mesh.cell_data["MuscleId"], dtype=np.int64)
    fraction = np.asarray(mesh.cell_data["MuscleFraction"], dtype=np.float64)
    historical = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    if muscle_id.shape != (mesh.n_cells,) or fraction.shape != (mesh.n_cells,):
        raise ValueError("MuscleId and MuscleFraction must be cell vectors")
    if not np.array_equal(historical, fraction > 1e-6):
        raise ValueError(
            "historical ActivationMask no longer matches MuscleFraction > 1e-6"
        )

    selected_ids = np.asarray(sorted(FACIAL_EXPRESSION_MUSCLES), dtype=np.int64)
    selected = np.isin(muscle_id, selected_ids) & (fraction > 1e-6)
    active_ids = np.flatnonzero(selected)
    if active_ids.size == 0:
        raise ValueError("named facial-expression activation domain is empty")
    control_ids = np.full(mesh.n_cells, -1, dtype=np.int32)
    for control_id, selected_muscle_id in enumerate(selected_ids):
        region = selected & (muscle_id == selected_muscle_id)
        if not np.any(region):
            raise ValueError(
                f"selected MuscleId {selected_muscle_id} has no active cells"
            )
        control_ids[region] = control_id

    excluded: list[dict[str, Any]] = []
    for excluded_id in np.unique(muscle_id[historical & ~selected]):
        region = historical & (muscle_id == excluded_id)
        excluded.append(
            {
                "muscle_id": int(excluded_id),
                "name": names[int(excluded_id)],
                "cells": int(region.sum()),
            }
        )
    mesh.cell_data["HistoricalActivationMask"] = historical.astype(np.int8)
    mesh.cell_data["ActivationMask"] = selected.astype(np.int8)
    mesh.cell_data["ActivationControlId"] = control_ids
    region_names = tuple(FACIAL_EXPRESSION_MUSCLES[int(i)] for i in selected_ids)
    return (
        active_ids,
        control_ids,
        region_names,
        {
            "predicate": "MuscleId in exact facial-expression whitelist and MuscleFraction > 1e-6",
            "historical_predicate": "MuscleFraction > 1e-6",
            "historical_active_cells": int(historical.sum()),
            "selected_active_cells": int(selected.sum()),
            "selected_fraction_of_historical": float(selected.sum() / historical.sum()),
            "selected_muscle_ids": selected_ids.tolist(),
            "selected_names": list(region_names),
            "excluded_historical_labels": excluded,
        },
    )


def _canonical_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("fiber estimate produced a zero or non-finite vector")
    result = vector / norm
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result = -result
    return result


def _weighted_pca(
    points: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("PCA weights must have a positive finite sum")
    center = np.sum(points * weights[:, None], axis=0) / total
    centered = points - center
    covariance = (centered * weights[:, None]).T @ centered / total
    values, vectors = np.linalg.eigh(covariance)
    if np.any(~np.isfinite(values)) or values[2] <= 0.0 or values[1] <= 0.0:
        raise ValueError("muscle centroid covariance is degenerate")
    return center, values, vectors


def estimate_region_fibers(
    mesh: pv.UnstructuredGrid, control_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Estimate geometry-only per-cell directions and explicit confidence."""
    tets = _tets(mesh)
    centers = np.asarray(mesh.points, dtype=np.float64)[tets].mean(axis=1)
    volume = np.asarray(mesh.cell_data["Volume"], dtype=np.float64)
    fraction = np.asarray(mesh.cell_data["MuscleFraction"], dtype=np.float64)
    selected_muscle_ids = np.asarray(sorted(FACIAL_EXPRESSION_MUSCLES), dtype=np.int64)
    if control_ids.shape != (mesh.n_cells,):
        raise ValueError("ActivationControlId must be a cell vector")

    fibers = np.zeros((mesh.n_cells, 3), dtype=np.float64)
    confidence = np.zeros(mesh.n_cells, dtype=np.float64)
    region_rows: list[dict[str, Any]] = []
    region_fibers = np.zeros((selected_muscle_ids.size, 3), dtype=np.float64)
    for control_id, selected_muscle_id in enumerate(selected_muscle_ids):
        cell_ids = np.flatnonzero(control_ids == control_id)
        weights = volume[cell_ids] * fraction[cell_ids]
        center, values, vectors = _weighted_pca(centers[cell_ids], weights)
        if selected_muscle_id in RING_MUSCLE_IDS:
            normal = vectors[:, 0]
            major = vectors[:, 2]
            minor = vectors[:, 1]
            centered = centers[cell_ids] - center
            major_q = centered @ major
            minor_q = centered @ minor
            gradient = (
                major_q[:, None] / values[2] * major[None, :]
                + minor_q[:, None] / values[1] * minor[None, :]
            )
            tangents = np.cross(normal[None, :], gradient)
            norms = np.linalg.norm(tangents, axis=1)
            if np.any(norms <= np.finfo(np.float64).eps):
                raise ValueError(
                    f"ellipse tangent is undefined for MuscleId {selected_muscle_id}"
                )
            tangents /= norms[:, None]
            for row in range(tangents.shape[0]):
                tangents[row] = _canonical_vector(tangents[row])
            planarity = float(np.clip(1.0 - values[0] / values[1], 0.0, 1.0))
            ring_balance = float(np.clip(values[1] / values[2], 0.0, 1.0))
            score = float(np.sqrt(planarity * ring_balance))
            fibers[cell_ids] = tangents
            region_fibers[control_id] = _canonical_vector(
                np.average(tangents, axis=0, weights=weights)
            )
            method = "weighted-centroid PCA plane plus per-cell ellipse tangent"
            confidence_basis = "sqrt((1-lambda_min/lambda_mid)*(lambda_mid/lambda_max))"
        else:
            axis = _canonical_vector(vectors[:, 2])
            score = float(np.clip(1.0 - values[1] / values[2], 0.0, 1.0))
            fibers[cell_ids] = axis
            region_fibers[control_id] = axis
            method = "weighted-centroid first principal axis"
            confidence_basis = "1-lambda_mid/lambda_max"
        confidence[cell_ids] = score
        region_rows.append(
            {
                "control_id": control_id,
                "muscle_id": int(selected_muscle_id),
                "name": FACIAL_EXPRESSION_MUSCLES[int(selected_muscle_id)],
                "cells": int(cell_ids.size),
                "method": method,
                "confidence": score,
                "confidence_basis": confidence_basis,
                "covariance_eigenvalues_m2": values.tolist(),
                "reference_fiber": region_fibers[control_id].tolist(),
            }
        )

    active = control_ids >= 0
    if not np.allclose(np.linalg.norm(fibers[active], axis=1), 1.0):
        raise ValueError("active-cell fiber estimates are not unit vectors")
    if np.any(fibers[~active] != 0.0) or np.any(confidence[~active] != 0.0):
        raise ValueError("inactive cells must have zero fiber metadata")
    mesh.cell_data["ActivationFiber"] = fibers
    mesh.cell_data["ActivationFiberConfidence"] = confidence
    mesh.field_data["ActivationRegionMuscleId"] = selected_muscle_ids
    mesh.field_data["ActivationRegionName"] = np.asarray(
        [FACIAL_EXPRESSION_MUSCLES[int(i)] for i in selected_muscle_ids]
    )
    mesh.field_data["ActivationRegionFiber"] = region_fibers
    mesh.field_data["ActivationRegionFiberConfidence"] = np.asarray(
        [row["confidence"] for row in region_rows], dtype=np.float64
    )
    return region_fibers, confidence, region_rows


def _validate_target(mesh: pv.UnstructuredGrid) -> dict[str, Any]:
    target = np.asarray(mesh.point_data["Smile"], dtype=np.float64)
    is_face = np.asarray(mesh.point_data["IsFace"], dtype=bool)
    loss_mask = np.asarray(mesh.point_data["SmileLossMask"], dtype=bool)
    finite = np.isfinite(target).all(axis=1)
    expected = is_face & finite
    if target.shape != (mesh.n_points, 3) or not np.array_equal(loss_mask, expected):
        raise ValueError("Smile target or IsFace-and-finite loss mask changed")
    return {
        "name": "Smile",
        "definition": "actual Smile vector; loss mask is IsFace & finite(Smile)",
        "loss_points": int(loss_mask.sum()),
        "fixed_overlap_points": int(
            (loss_mask & np.asarray(mesh.point_data["IsFixed"], dtype=bool)).sum()
        ),
        "displacement_rms_m": float(
            np.linalg.norm(target[loss_mask]) / np.sqrt(loss_mask.sum())
        ),
    }


def load_face_fixture(
    mesh_path: Path = DEFAULT_VOLUME,
    skin_path: Path = DEFAULT_SKIN,
    cut_reference_path: Path = DEFAULT_CUT_REFERENCE,
) -> FaceFixture:
    """Load, validate, and prepare the pinned face fixture on CPU."""
    mesh_path = Path(mesh_path).resolve()
    skin_path = Path(skin_path).resolve()
    cut_reference_path = Path(cut_reference_path).resolve()
    _require_identity(mesh_path, EXPECTED_VOLUME_SHA256, "June volume")
    _require_identity(skin_path, EXPECTED_SKIN_SHA256, "corrected skin")
    _require_identity(
        cut_reference_path, EXPECTED_CUT_REFERENCE_SHA256, "cut reference"
    )
    raw_mesh = pv.read(mesh_path)
    if not isinstance(raw_mesh, pv.UnstructuredGrid):
        raise TypeError("June volume did not read as UnstructuredGrid")
    mesh = raw_mesh.copy(deep=True)
    raw_skin = pv.read(skin_path)
    if not isinstance(raw_skin, pv.PolyData):
        raise TypeError("corrected skin did not read as PolyData")
    skin = raw_skin.copy(deep=True)
    raw_cut = pv.read(cut_reference_path)
    if not isinstance(raw_cut, pv.PolyData):
        raise TypeError("cut reference did not read as PolyData")

    _, cut_summary = configure_hard_fixed_cut(mesh, raw_cut)
    skin_map = map_skin_to_volume(skin, mesh)
    active_ids, control_ids, region_names, domain_summary = (
        select_facial_expression_cells(mesh)
    )
    region_fibers, confidence, region_rows = estimate_region_fibers(mesh, control_ids)
    target_summary = _validate_target(mesh)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "design": "june-volume-hard-fixed-cut-corrected-p000-skin-named-expression",
        "units": {"length": "m", "Young_modulus": "MPa"},
        "inputs": {
            "volume": {"path": str(mesh_path), "sha256": EXPECTED_VOLUME_SHA256},
            "cut_reference": {
                "path": str(cut_reference_path),
                "sha256": EXPECTED_CUT_REFERENCE_SHA256,
            },
            "skin": {"path": str(skin_path), "sha256": EXPECTED_SKIN_SHA256},
        },
        "volume": {
            "points": int(mesh.n_points),
            "tetrahedra": int(mesh.n_cells),
            "fraction_fields": [
                "AponeurosisFraction",
                "FatFraction",
                "MuscleFraction",
            ],
        },
        "cut_boundary": cut_summary,
        "skin": {
            "domain": "all three triangle vertices have IsFace=true",
            "points": int(skin.n_points),
            "triangles": int(skin.n_cells),
            "volume_mapping_points": int(skin_map.size),
            "E_MPa": 0.2,
            "nu": 0.49,
            "thickness_m": 0.001,
            "lame_conversion": "plane stress",
            "prestrain": "none; ActivationInv is exactly zero",
            "target_derived": False,
        },
        "activation_domain": domain_summary,
        "fiber_estimate": {
            "source": "rest geometry only; no expression target was read",
            "claim": "heuristic experiment coordinate, not anatomical calibration",
            "regions": region_rows,
        },
        "target": target_summary,
    }
    return FaceFixture(
        mesh=mesh,
        skin=skin,
        active_cell_ids=active_ids,
        control_ids=control_ids,
        region_names=region_names,
        region_fibers=region_fibers,
        confidence=confidence,
        summary=summary,
    )
