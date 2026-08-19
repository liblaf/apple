from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from _reference import GROUP_DIR, PREPARED_MESH, enable_reference_modules

from liblaf import cherries, melon
from liblaf.apple.common import (
    ACTIVATION_INV,
    FRACTION,
    GLOBAL_POINT_ID,
    LAMBDA,
    MU,
    lame_converter_plane_stress,
)

enable_reference_modules()

from _human_face_config import (  # noqa: E402
    IS_FACE,
    IS_FIXED,
    SKIN_E,
    SKIN_NU,
    SKIN_THICKNESS,
    SMILE_TARGET,
)
from _material_heuristics import (  # noqa: E402
    file_sha256,
    skin_material_content_hash,
    skin_solver_content_hash,
    skin_topology_content_hash,
)

logger = logging.getLogger(__name__)

DESIGN = "isface-plane-stress-corrected-baseline"
SCHEMA_VERSION = 3
CANDIDATE_LABEL = "isface-e0200-p000"
SOURCE_MESH = Path(
    "/home/liblaf/Projects/liblaf/melon/exp/2026/05/27/head/data/62-tetmesh-3191k.vtu"
)
TOPOLOGY_REFERENCE = (
    Path(__file__).resolve().parents[4] / "06/17/human-face-smile-prestrain-v2/data/"
    "10-smile-isface-skin-estimated-prestrain.vtp"
)
MELON_MASK_IMPLEMENTATION = (
    Path("/home/liblaf/Projects/liblaf/melon")
    / "exp/2026/05/27/head/src/42-gen-masks.py"
)

PREPARED_MESH_SIZE_BYTES = 76_792_914
PREPARED_MESH_SHA256 = (
    "8131d6944b322d7c1e21918688f297e2887b42bf4dbc19ce36259b007e8dc563"
)
SOURCE_MESH_SIZE_BYTES = 176_496_263
SOURCE_MESH_SHA256 = "824464f109a4e97c3176091bb21e8fdb533def6fc5616845bc36bb377c2a7752"
TOPOLOGY_REFERENCE_SIZE_BYTES = 20_278_382
TOPOLOGY_REFERENCE_SHA256 = (
    "156873bf661f9a32d8d4161d548e20c208008a6bee5e0308e15abc9403befeac"
)
MELON_MASK_IMPLEMENTATION_SIZE_BYTES = 5_766
MELON_MASK_IMPLEMENTATION_SHA256 = (
    "80b607635f150693aaffc9f960fe04f8e16d5be52efa80a87d1106dc2787d4ff"
)

FACE_GROUPS = (
    "Chin",
    "EyelidBottom",
    "EyelidOuterBottom",
    "EyelidOuterTop",
    "EyelidTop",
    "Face",
    "LipBottom",
    "LipOuterBottom",
    "LipOuterTop",
    "LipTop",
)

EXPECTED_FULL_BOUNDARY_TRIANGLES = 128_172
EXPECTED_FULL_BOUNDARY_UNASSIGNED_GROUP_POINTS = 6_000
EXPECTED_SOURCE_OUTER_TRIANGLES = 115_007
EXPECTED_ARTIFICIAL_CUT_TRIANGLES = 13_165
EXPECTED_SKIN_POINTS = 15_299
EXPECTED_SKIN_TRIANGLES = 29_899
EXPECTED_SKIN_AREA_M2 = 0.04287998059707303
EXPECTED_SKIN_COMPONENTS = 1
AREA_ATOL_M2 = 5.0e-13
FORMULA_RTOL = 1.0e-13
FORMULA_ATOL = 1.0e-14

LAME_CONVERSION = (
    "thin-membrane plane-stress reduction: "
    "lambda = E * nu / (1 - nu**2); "
    "mu = E / (2 * (1 + nu))"
)
VOLUME_LAME_CONVERSION = (
    "unchanged 3D isotropic volume convention: "
    "lambda = E * nu / ((1 + nu) * (1 - 2 * nu)); "
    "mu = E / (2 * (1 + nu))"
)
EXPECTED_OUTPUT_MANIFEST = GROUP_DIR / "data/10-corrected-baseline-manifest.json"
EXPECTED_OUTPUT_TABLE = GROUP_DIR / "data/10-corrected-baseline-table.md"
EXPECTED_OUTPUT_SKIN = (
    GROUP_DIR / "data/10-corrected-baseline/skin-isface-e0200-p000.vtp"
)

DOMAIN_ONE_CELL_ARRAYS = (
    "IsFaceTriangle",
    "SourceOuterTriangle",
)
DOMAIN_ZERO_CELL_ARRAYS = (
    "ArtificialCutTriangle",
    "FixedTriangle",
    "DisallowedGroupTriangle",
)
DOMAIN_DIAGNOSTIC_CELL_ARRAYS = (
    "TeethProximityTriangle",
    "GingivaProximityTriangle",
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(PREPARED_MESH)
    input_source_mesh: Path = cherries.input(SOURCE_MESH)
    input_topology_reference: Path = cherries.input(TOPOLOGY_REFERENCE)
    output_manifest: Path = cherries.output(
        "10-corrected-baseline-manifest.json", mkdir=True
    )
    output_table: Path = cherries.output("10-corrected-baseline-table.md", mkdir=True)
    output_skin: Path = cherries.output(
        "10-corrected-baseline/skin-isface-e0200-p000.vtp", mkdir=True
    )


def _file_identity(path: Path) -> dict[str, int | str]:
    return {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _require_exact_path(actual: Path, expected: Path, *, name: str) -> None:
    if actual.resolve() != expected.resolve():
        msg = f"{name} must be the pinned path {expected}, got {actual}"
        raise ValueError(msg)


def _validate_output_contract(cfg: Config) -> None:
    for actual, expected, name in (
        (cfg.output_manifest, EXPECTED_OUTPUT_MANIFEST, "output_manifest"),
        (cfg.output_table, EXPECTED_OUTPUT_TABLE, "output_table"),
        (cfg.output_skin, EXPECTED_OUTPUT_SKIN, "output_skin"),
    ):
        _require_exact_path(actual, expected, name=name)
    stale_outputs = [
        path
        for path in (cfg.output_manifest, cfg.output_table, cfg.output_skin)
        if path.exists()
    ]
    if stale_outputs:
        msg = (
            "refusing to overwrite an earlier or partial corrected-baseline "
            "preparation; review and remove or archive explicitly: "
            f"{[str(path) for path in stale_outputs]}"
        )
        raise FileExistsError(msg)


def _require_file_identity(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    name: str,
) -> dict[str, int | str]:
    if not path.is_file():
        msg = f"missing pinned {name}: {path}"
        raise FileNotFoundError(msg)
    actual = _file_identity(path)
    expected = {"size_bytes": expected_size, "sha256": expected_sha256}
    if actual != expected:
        msg = f"{name} identity mismatch: expected {expected}, got {actual}"
        raise ValueError(msg)
    return actual


def _require_authoritative_face_groups() -> dict[str, int | str]:
    identity = _require_file_identity(
        MELON_MASK_IMPLEMENTATION,
        expected_size=MELON_MASK_IMPLEMENTATION_SIZE_BYTES,
        expected_sha256=MELON_MASK_IMPLEMENTATION_SHA256,
        name="Melon face-mask implementation",
    )
    tree = ast.parse(
        MELON_MASK_IMPLEMENTATION.read_text(encoding="utf-8"),
        filename=str(MELON_MASK_IMPLEMENTATION),
    )
    values: tuple[str, ...] | None = None
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if isinstance(node.target, ast.Name) and node.target.id == "FACE_GROUPS":
            raw = ast.literal_eval(node.value)
            values = tuple(str(value) for value in raw)
            break
    if values != FACE_GROUPS:
        msg = (
            "local facial membrane allowlist differs from pinned Melon FACE_GROUPS: "
            f"local={FACE_GROUPS}, Melon={values}"
        )
        raise ValueError(msg)
    return identity


def _group_names(mesh: pv.UnstructuredGrid) -> tuple[str, ...]:
    if "GroupName" not in mesh.field_data:
        msg = "prepared mesh is missing authoritative GroupName field data"
        raise KeyError(msg)
    names: list[str] = []
    for raw in np.asarray(mesh.field_data["GroupName"]).reshape(-1):
        if isinstance(raw, bytes):
            names.append(raw.decode("utf-8"))
        else:
            names.append(str(raw))
    if len(names) != len(set(names)):
        msg = "prepared mesh GroupName values are not unique"
        raise ValueError(msg)
    missing = sorted(set(FACE_GROUPS) - set(names))
    if missing:
        msg = f"prepared mesh is missing Melon FACE_GROUPS: {missing}"
        raise ValueError(msg)
    return tuple(names)


def _point_group_ids(mesh: pv.UnstructuredGrid) -> np.ndarray:
    if "GroupId" not in mesh.point_data:
        msg = "prepared mesh is missing authoritative point GroupId"
        raise KeyError(msg)
    raw = np.asarray(mesh.point_data["GroupId"])
    ids = np.asarray(raw, dtype=np.int64)
    names = _group_names(mesh)
    if raw.shape != (mesh.n_points,) or not np.array_equal(raw, ids):
        msg = "prepared point GroupId must contain exact integer identifiers"
        raise ValueError(msg)
    # GroupId=-1 is the pinned marker for points introduced on the artificial
    # InFaceConvex cut. It is valid on the complete volume/boundary but must
    # never enter the filtered all-vertex IsFace membrane.
    if np.any((ids < -1) | (ids >= len(names))):
        msg = "prepared point GroupId escapes GroupName field data"
        raise ValueError(msg)
    return ids


def _read_unstructured(path: Path, *, name: str) -> pv.UnstructuredGrid:
    mesh = pv.read(path)
    if not isinstance(mesh, pv.UnstructuredGrid):
        msg = f"{name} read as {type(mesh).__name__}, expected UnstructuredGrid"
        raise TypeError(msg)
    return mesh


def _read_polydata(path: Path, *, name: str) -> pv.PolyData:
    surface = pv.read(path)
    if not isinstance(surface, pv.PolyData):
        msg = f"{name} read as {type(surface).__name__}, expected PolyData"
        raise TypeError(msg)
    return surface


def _triangles(surface: pv.PolyData) -> np.ndarray:
    faces = np.asarray(surface.faces, dtype=np.int64)
    if surface.n_cells == 0 or faces.size != 4 * surface.n_cells:
        msg = "skin must be a non-empty triangle-only PolyData"
        raise ValueError(msg)
    encoded = faces.reshape(-1, 4)
    if not np.all(encoded[:, 0] == 3):
        msg = "skin contains a non-triangle face"
        raise ValueError(msg)
    return encoded[:, 1:]


def _canonical_face_keys(point_ids: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(point_ids, dtype=np.int64)[triangles], axis=1)


def _structured_face_keys(keys: np.ndarray) -> np.ndarray:
    canonical = np.ascontiguousarray(keys, dtype="<i8")
    if canonical.ndim != 2 or canonical.shape[1] != 3:
        msg = f"expected triangular face keys, got {canonical.shape}"
        raise ValueError(msg)
    dtype = np.dtype([("v0", "<i8"), ("v1", "<i8"), ("v2", "<i8")])
    return canonical.view(dtype).reshape(-1)


def _face_key_membership(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.isin(
        _structured_face_keys(query),
        _structured_face_keys(reference),
        assume_unique=False,
    )


def _face_key_hash(keys: np.ndarray) -> str:
    canonical = np.ascontiguousarray(keys, dtype="<i8")
    order = np.lexsort((canonical[:, 2], canonical[:, 1], canonical[:, 0]))
    canonical = canonical[order]
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _triangle_area(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def _topology_metrics(triangles: np.ndarray) -> dict[str, int]:
    n_faces = triangles.shape[0]
    edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    owners = np.tile(np.arange(n_faces, dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    owners = owners[order]
    starts = np.r_[0, np.flatnonzero(np.any(edges[1:] != edges[:-1], axis=1)) + 1]
    stops = np.r_[starts[1:], edges.shape[0]]
    counts = stops - starts

    parent = np.arange(n_faces, dtype=np.int64)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for start, stop in zip(starts, stops, strict=True):
        first = int(owners[start])
        for owner in owners[start + 1 : stop]:
            union(first, int(owner))

    components = len({find(face) for face in range(n_faces)})
    return {
        "components": components,
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "interior_edges": int(np.count_nonzero(counts == 2)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "identity_lhs_3F": int(3 * n_faces),
        "identity_rhs_edge_incidence": int(counts.sum()),
    }


def _content_hashes(surface: pv.PolyData) -> dict[str, str]:
    return {
        "topology_sha256": skin_topology_content_hash(surface),
        "material_sha256": skin_material_content_hash(surface),
        "solver_sha256": skin_solver_content_hash(surface),
    }


def _volume_global_ids(mesh: pv.UnstructuredGrid) -> np.ndarray:
    if GLOBAL_POINT_ID.vtk in mesh.point_data:
        ids = np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    else:
        ids = np.arange(mesh.n_points, dtype=np.int64)
    if ids.shape != (mesh.n_points,) or np.unique(ids).size != mesh.n_points:
        msg = "prepared mesh has invalid or duplicate GlobalPointId values"
        raise ValueError(msg)
    return ids


def _domain_audit(  # noqa: C901, PLR0912, PLR0915
    mesh: pv.UnstructuredGrid,
    source: pv.UnstructuredGrid,
    topology_reference: pv.PolyData,
) -> dict[str, Any]:
    required_prepared = {
        IS_FACE,
        IS_FIXED,
        "GroupId",
        "IsTeeth",
        "IsGingiva",
        SMILE_TARGET,
    }
    missing_prepared = sorted(required_prepared - set(mesh.point_data))
    if missing_prepared:
        msg = f"prepared mesh is missing domain fields: {missing_prepared}"
        raise KeyError(msg)
    if "vtkOriginalPointIds" not in mesh.point_data:
        msg = "prepared mesh is missing its source-point identity"
        raise KeyError(msg)

    volume_global_ids = _volume_global_ids(mesh)
    source_point_ids = np.asarray(
        mesh.point_data["vtkOriginalPointIds"], dtype=np.int64
    )
    if (
        source_point_ids.shape != (mesh.n_points,)
        or np.unique(source_point_ids).size != mesh.n_points
        or source_point_ids.min() < 0
        or source_point_ids.max() >= source.n_points
    ):
        msg = "prepared vtkOriginalPointIds do not map one-to-one into source mesh"
        raise ValueError(msg)
    if not np.array_equal(
        np.asarray(mesh.points, dtype=np.float64),
        np.asarray(source.points, dtype=np.float64)[source_point_ids],
    ):
        msg = "prepared points do not exactly match their pinned source identities"
        raise ValueError(msg)

    boundary = mesh.extract_surface(algorithm=None).triangulate()
    boundary_triangles = _triangles(boundary)
    if "vtkOriginalPointIds" not in boundary.point_data:
        msg = "prepared boundary is missing local point identity"
        raise KeyError(msg)
    boundary_local_ids = np.asarray(
        boundary.point_data["vtkOriginalPointIds"], dtype=np.int64
    )
    if boundary.n_cells != EXPECTED_FULL_BOUNDARY_TRIANGLES:
        msg = (
            f"prepared boundary has {boundary.n_cells} triangles, expected "
            f"{EXPECTED_FULL_BOUNDARY_TRIANGLES}"
        )
        raise ValueError(msg)

    source_outer = source.extract_surface(algorithm=None).triangulate()
    source_outer_triangles = _triangles(source_outer)
    if "vtkOriginalPointIds" not in source_outer.point_data:
        msg = "source outer surface is missing source point identity"
        raise KeyError(msg)
    source_outer_point_ids = np.asarray(
        source_outer.point_data["vtkOriginalPointIds"], dtype=np.int64
    )
    source_outer_keys = _canonical_face_keys(
        source_outer_point_ids, source_outer_triangles
    )
    boundary_source_ids = source_point_ids[boundary_local_ids]
    boundary_source_keys = _canonical_face_keys(boundary_source_ids, boundary_triangles)
    on_source_outer = _face_key_membership(boundary_source_keys, source_outer_keys)
    artificial_cut = ~on_source_outer

    group_names = _group_names(mesh)
    point_group_ids = _point_group_ids(mesh)
    allowed_group_ids = np.asarray(
        [group_names.index(name) for name in FACE_GROUPS], dtype=np.int64
    )
    boundary_group_ids = point_group_ids[boundary_local_ids]
    allowed_group_point = np.isin(boundary_group_ids, allowed_group_ids)
    face_point = np.asarray(mesh.point_data[IS_FACE], dtype=bool)[boundary_local_ids]
    fixed_point = np.asarray(mesh.point_data[IS_FIXED], dtype=bool)[boundary_local_ids]
    teeth_point = np.asarray(mesh.point_data["IsTeeth"], dtype=bool)[boundary_local_ids]
    gingiva_point = np.asarray(mesh.point_data["IsGingiva"], dtype=bool)[
        boundary_local_ids
    ]
    finite_target_point = np.isfinite(
        np.asarray(mesh.point_data[SMILE_TARGET], dtype=np.float64)[boundary_local_ids]
    ).all(axis=1)

    is_face = np.all(face_point[boundary_triangles], axis=1)
    allowed_group = np.all(allowed_group_point[boundary_triangles], axis=1)
    disallowed_group = ~allowed_group
    fixed = np.any(fixed_point[boundary_triangles], axis=1)
    teeth_proximity = np.any(teeth_point[boundary_triangles], axis=1)
    gingiva_proximity = np.any(gingiva_point[boundary_triangles], axis=1)
    finite_target = np.all(finite_target_point[boundary_triangles], axis=1)

    candidate_boundary_point_ids = np.unique(boundary_triangles[is_face])
    observed_group_ids = np.unique(boundary_group_ids[candidate_boundary_point_ids])
    candidate_group_ids_valid = bool(
        np.all((observed_group_ids >= 0) & (observed_group_ids < len(group_names)))
    )
    observed_group_names = (
        tuple(sorted(group_names[index] for index in observed_group_ids))
        if candidate_group_ids_valid
        else ()
    )

    boundary_global_ids = volume_global_ids[boundary_local_ids]
    expected_face_keys = _canonical_face_keys(
        boundary_global_ids, boundary_triangles[is_face]
    )
    reference_triangles = _triangles(topology_reference)
    reference_global_ids = np.asarray(
        topology_reference.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64
    )
    if (
        reference_global_ids.shape != (topology_reference.n_points,)
        or np.unique(reference_global_ids).size != topology_reference.n_points
    ):
        msg = "topology reference has invalid or duplicate GlobalPointId values"
        raise ValueError(msg)
    reference_face_keys = _canonical_face_keys(
        reference_global_ids, reference_triangles
    )
    expected_hash = _face_key_hash(expected_face_keys)
    reference_hash = _face_key_hash(reference_face_keys)

    reference_positions = np.searchsorted(
        np.sort(volume_global_ids), reference_global_ids
    )
    volume_order = np.argsort(volume_global_ids)
    if np.any(reference_positions >= volume_global_ids.size) or not np.array_equal(
        volume_global_ids[volume_order[reference_positions]], reference_global_ids
    ):
        msg = "topology-reference GlobalPointId values do not map into prepared mesh"
        raise ValueError(msg)
    reference_local_ids = volume_order[reference_positions]

    errors: list[str] = []
    observed_source_outer = int(np.count_nonzero(on_source_outer))
    observed_cut = int(np.count_nonzero(artificial_cut))
    observed_unassigned_boundary_points = int(
        np.count_nonzero(boundary_group_ids == -1)
    )
    if observed_source_outer != EXPECTED_SOURCE_OUTER_TRIANGLES:
        errors.append(
            f"source-outer count {observed_source_outer} != "
            f"{EXPECTED_SOURCE_OUTER_TRIANGLES}"
        )
    if observed_cut != EXPECTED_ARTIFICIAL_CUT_TRIANGLES:
        errors.append(
            f"artificial-cut count {observed_cut} != "
            f"{EXPECTED_ARTIFICIAL_CUT_TRIANGLES}"
        )
    if (
        observed_unassigned_boundary_points
        != EXPECTED_FULL_BOUNDARY_UNASSIGNED_GROUP_POINTS
    ):
        errors.append(
            "full-boundary unassigned GroupId point count "
            f"{observed_unassigned_boundary_points} != "
            f"{EXPECTED_FULL_BOUNDARY_UNASSIGNED_GROUP_POINTS}"
        )
    if int(np.count_nonzero(is_face)) != EXPECTED_SKIN_TRIANGLES:
        errors.append("all-vertex IsFace triangle count changed")
    if not candidate_group_ids_valid:
        errors.append("all-vertex IsFace domain contains an invalid GroupId")
    if not np.array_equal(is_face, allowed_group):
        errors.append("IsFace does not exactly equal the pinned Melon FACE_GROUPS mask")
    if set(observed_group_names) != set(FACE_GROUPS):
        errors.append(
            "membrane GroupName coverage differs from the exact Melon FACE_GROUPS"
        )
    if np.count_nonzero(is_face & artificial_cut):
        errors.append("all-vertex IsFace domain overlaps the artificial cut")
    if np.count_nonzero(is_face & fixed):
        errors.append("all-vertex IsFace domain overlaps fixed/bone points")
    if np.count_nonzero(is_face & disallowed_group):
        errors.append("all-vertex IsFace domain contains a disallowed anatomy group")
    if not np.all(finite_target[is_face]):
        errors.append("all-vertex IsFace domain contains a non-finite Smile target")
    if expected_hash != reference_hash:
        errors.append("topology reference is not the exact all-vertex IsFace face set")
    if topology_reference.n_points != EXPECTED_SKIN_POINTS:
        errors.append("topology-reference point count changed")
    if topology_reference.n_cells != EXPECTED_SKIN_TRIANGLES:
        errors.append("topology-reference triangle count changed")
    if not np.array_equal(
        np.asarray(topology_reference.points, dtype=np.float64),
        np.asarray(mesh.points, dtype=np.float64)[reference_local_ids],
    ):
        errors.append("topology-reference points do not match prepared GlobalPointId")
    if errors:
        msg = "skin-domain identity gates failed: " + "; ".join(errors)
        raise RuntimeError(msg)

    return {
        "selection": "all three triangle vertices have IsFace=true",
        "claim_scope": "facial-ROI membrane, not a complete anatomical epidermis",
        "full_boundary_triangles": int(boundary.n_cells),
        "source_outer_triangles": observed_source_outer,
        "artificial_cut_triangles": observed_cut,
        "full_boundary_unassigned_group_points": (observed_unassigned_boundary_points),
        "skin_triangles": int(np.count_nonzero(is_face)),
        "skin_artificial_cut_overlap_triangles": int(
            np.count_nonzero(is_face & artificial_cut)
        ),
        "skin_fixed_overlap_triangles": int(np.count_nonzero(is_face & fixed)),
        "skin_disallowed_group_overlap_triangles": int(
            np.count_nonzero(is_face & disallowed_group)
        ),
        "skin_teeth_proximity_overlap_triangles": int(
            np.count_nonzero(is_face & teeth_proximity)
        ),
        "skin_gingiva_proximity_overlap_triangles": int(
            np.count_nonzero(is_face & gingiva_proximity)
        ),
        "skin_nonfinite_target_triangles": int(
            np.count_nonzero(is_face & ~finite_target)
        ),
        "source_outer_face_key_sha256": _face_key_hash(source_outer_keys),
        "prepared_boundary_source_face_key_sha256": _face_key_hash(
            boundary_source_keys
        ),
        "isface_global_face_key_sha256": expected_hash,
        "topology_reference_global_face_key_sha256": reference_hash,
        "anatomy_group_gate": (
            "every membrane vertex GroupName belongs to the exact pinned Melon "
            "FACE_GROUPS allowlist"
        ),
        "face_group_allowlist": list(FACE_GROUPS),
        "face_group_ids": allowed_group_ids.tolist(),
        "observed_skin_group_names": list(observed_group_names),
        "teeth_and_gingiva_proximity_policy": (
            "diagnostic only: these are 2 mm proximity masks, not anatomy labels"
        ),
        "validation_errors": [],
        "validation_ok": True,
    }


def _make_skin(
    topology_reference: pv.PolyData,
    mesh: pv.UnstructuredGrid,
) -> pv.PolyData:
    skin = pv.PolyData(
        np.asarray(topology_reference.points, dtype=np.float64).copy(),
        np.asarray(topology_reference.faces, dtype=np.int64).copy(),
    )
    global_ids = np.asarray(
        topology_reference.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64
    ).copy()
    volume_global_ids = _volume_global_ids(mesh)
    order = np.argsort(volume_global_ids)
    positions = np.searchsorted(volume_global_ids[order], global_ids)
    local_ids = order[positions]
    source_point_ids = np.asarray(
        mesh.point_data["vtkOriginalPointIds"], dtype=np.int64
    )[local_ids]

    skin.point_data[GLOBAL_POINT_ID.vtk] = global_ids
    skin.point_data["SourcePointId"] = source_point_ids
    for name in ("GroupId", IS_FACE, IS_FIXED, "IsTeeth", "IsGingiva"):
        skin.point_data[name] = np.asarray(mesh.point_data[name])[local_ids]
    skin.field_data["GroupName"] = np.asarray(mesh.field_data["GroupName"])

    E = np.asarray(SKIN_E, dtype=np.float64)
    nu = np.asarray(SKIN_NU, dtype=np.float64)
    lambda_tensor, mu_tensor = lame_converter_plane_stress(E, nu)
    lambda_ = float(lambda_tensor.item())
    mu = float(mu_tensor.item())
    triangles = _triangles(skin)
    rest_area = _triangle_area(np.asarray(skin.points, dtype=np.float64), triangles)

    skin.cell_data[LAMBDA.vtk] = np.full(skin.n_cells, lambda_, dtype=np.float64)
    skin.cell_data[MU.vtk] = np.full(skin.n_cells, mu, dtype=np.float64)
    skin.cell_data[FRACTION.vtk] = np.ones(skin.n_cells, dtype=np.float64)
    skin.cell_data[ACTIVATION_INV.vtk] = np.zeros((skin.n_cells, 3), dtype=np.float64)
    skin.cell_data["SkinYoungModulusMPa"] = np.full(
        skin.n_cells, float(E), dtype=np.float64
    )
    skin.cell_data["SkinPoissonRatio"] = np.full(
        skin.n_cells, float(nu), dtype=np.float64
    )
    skin.cell_data["SkinActivationInvDiag"] = np.zeros(skin.n_cells, dtype=np.float64)
    skin.cell_data["StressFreeAreaRatio"] = np.ones(skin.n_cells, dtype=np.float64)
    skin.cell_data["RestArea"] = rest_area
    for name in DOMAIN_ONE_CELL_ARRAYS:
        skin.cell_data[name] = np.ones(skin.n_cells, dtype=np.int8)
    for name in DOMAIN_ZERO_CELL_ARRAYS:
        skin.cell_data[name] = np.zeros(skin.n_cells, dtype=np.int8)
    teeth_point = np.asarray(skin.point_data["IsTeeth"], dtype=bool)
    gingiva_point = np.asarray(skin.point_data["IsGingiva"], dtype=bool)
    skin.cell_data["TeethProximityTriangle"] = np.any(
        teeth_point[triangles], axis=1
    ).astype(np.int8)
    skin.cell_data["GingivaProximityTriangle"] = np.any(
        gingiva_point[triangles], axis=1
    ).astype(np.int8)
    return skin


def _validate_skin(  # noqa: C901, PLR0912, PLR0915
    skin: pv.PolyData,
    mesh: pv.UnstructuredGrid,
    topology_reference: pv.PolyData,
    *,
    expected_content: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    try:
        triangles = _triangles(skin)
    except (TypeError, ValueError) as error:
        return [str(error)], {}

    required_point = {
        GLOBAL_POINT_ID.vtk,
        "SourcePointId",
        "GroupId",
        IS_FACE,
        IS_FIXED,
        "IsTeeth",
        "IsGingiva",
    }
    required_cell = {
        LAMBDA.vtk,
        MU.vtk,
        FRACTION.vtk,
        ACTIVATION_INV.vtk,
        "SkinYoungModulusMPa",
        "SkinPoissonRatio",
        "SkinActivationInvDiag",
        "StressFreeAreaRatio",
        "RestArea",
        *DOMAIN_ONE_CELL_ARRAYS,
        *DOMAIN_ZERO_CELL_ARRAYS,
        *DOMAIN_DIAGNOSTIC_CELL_ARRAYS,
    }
    missing_point = sorted(required_point - set(skin.point_data))
    missing_cell = sorted(required_cell - set(skin.cell_data))
    errors.extend(f"missing point array: {name}" for name in missing_point)
    errors.extend(f"missing cell array: {name}" for name in missing_cell)
    if errors:
        return errors, {}

    if skin.n_points != EXPECTED_SKIN_POINTS:
        errors.append(f"actual Koiter point count {skin.n_points} != 15299")
    if skin.n_cells != EXPECTED_SKIN_TRIANGLES:
        errors.append(f"actual Koiter triangle count {skin.n_cells} != 29899")
    if skin.n_cells >= EXPECTED_FULL_BOUNDARY_TRIANGLES:
        errors.append("Koiter input is not a physically filtered PolyData")

    global_ids = np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    volume_global_ids = _volume_global_ids(mesh)
    local_ids = np.zeros(skin.n_points, dtype=np.int64)
    if (
        global_ids.shape != (skin.n_points,)
        or np.unique(global_ids).size != skin.n_points
        or global_ids.min() < volume_global_ids.min()
        or global_ids.max() > volume_global_ids.max()
    ):
        errors.append("skin GlobalPointId is missing, duplicated, or out of range")
    else:
        order = np.argsort(volume_global_ids)
        positions = np.searchsorted(volume_global_ids[order], global_ids)
        if np.any(positions >= volume_global_ids.size) or not np.array_equal(
            volume_global_ids[order[positions]], global_ids
        ):
            errors.append("skin GlobalPointId does not map to prepared mesh")
        else:
            local_ids = order[positions]
            if not np.array_equal(
                np.asarray(skin.points, dtype=np.float64),
                np.asarray(mesh.points, dtype=np.float64)[local_ids],
            ):
                errors.append("skin points do not match prepared GlobalPointId")

    source_ids = np.asarray(skin.point_data["SourcePointId"], dtype=np.int64)
    expected_source_ids = np.asarray(
        mesh.point_data["vtkOriginalPointIds"], dtype=np.int64
    )[local_ids]
    if not np.array_equal(source_ids, expected_source_ids):
        errors.append("skin SourcePointId does not match prepared source identity")
    for name in ("GroupId", IS_FACE, IS_FIXED, "IsTeeth", "IsGingiva"):
        actual = np.asarray(skin.point_data[name])
        expected = np.asarray(mesh.point_data[name])[local_ids]
        if not np.array_equal(actual, expected):
            errors.append(f"skin {name} differs from prepared GlobalPointId")

    group_names = _group_names(mesh)
    group_ids = np.asarray(skin.point_data["GroupId"], dtype=np.int64)
    unique_group_ids = np.unique(group_ids)
    if np.any((unique_group_ids < 0) | (unique_group_ids >= len(group_names))):
        errors.append("skin contains an invalid or unassigned GroupId")
        observed_group_names: set[str] = set()
    else:
        observed_group_names = {group_names[index] for index in unique_group_ids}
    if not observed_group_names <= set(FACE_GROUPS):
        errors.append("skin contains a GroupName outside pinned Melon FACE_GROUPS")

    reference_topology_hash = skin_topology_content_hash(topology_reference)
    content = _content_hashes(skin)
    if content["topology_sha256"] != reference_topology_hash:
        errors.append("skin topology differs from the pinned IsFace reference")
    if expected_content is not None and content != expected_content:
        errors.append("skin content hashes changed during VTP readback")

    E = np.full(skin.n_cells, float(SKIN_E), dtype=np.float64)
    nu = np.full(skin.n_cells, float(SKIN_NU), dtype=np.float64)
    expected_lambda = E * nu / (1.0 - np.square(nu))
    expected_mu = E / (2.0 * (1.0 + nu))
    expected_scalar = {
        "SkinYoungModulusMPa": E,
        "SkinPoissonRatio": nu,
        LAMBDA.vtk: expected_lambda,
        MU.vtk: expected_mu,
        FRACTION.vtk: np.ones(skin.n_cells, dtype=np.float64),
        "SkinActivationInvDiag": np.zeros(skin.n_cells, dtype=np.float64),
        "StressFreeAreaRatio": np.ones(skin.n_cells, dtype=np.float64),
    }
    for name, expected in expected_scalar.items():
        actual = np.asarray(skin.cell_data[name], dtype=np.float64)
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=FORMULA_RTOL, atol=FORMULA_ATOL
        ):
            errors.append(f"{name} differs from homogeneous E=.2 p000 plane stress")
    activation = np.asarray(skin.cell_data[ACTIVATION_INV.vtk], dtype=np.float64)
    if activation.shape != (skin.n_cells, 3) or not np.array_equal(
        activation, np.zeros_like(activation)
    ):
        errors.append("ActivationInv must be exactly zero for p000")

    for name in DOMAIN_ONE_CELL_ARRAYS:
        values = np.asarray(skin.cell_data[name], dtype=np.int8)
        if values.shape != (skin.n_cells,) or not np.all(values == 1):
            errors.append(f"{name} must be one on every Koiter triangle")
    for name in DOMAIN_ZERO_CELL_ARRAYS:
        values = np.asarray(skin.cell_data[name], dtype=np.int8)
        if values.shape != (skin.n_cells,) or np.any(values != 0):
            errors.append(f"{name} must be zero on every Koiter triangle")
    for name, point_name in (
        ("TeethProximityTriangle", "IsTeeth"),
        ("GingivaProximityTriangle", "IsGingiva"),
    ):
        expected = np.any(
            np.asarray(skin.point_data[point_name], dtype=bool)[triangles], axis=1
        ).astype(np.int8)
        if not np.array_equal(
            np.asarray(skin.cell_data[name], dtype=np.int8), expected
        ):
            errors.append(f"{name} differs from the point-proximity diagnostic")
    if not np.all(np.asarray(skin.point_data[IS_FACE], dtype=bool)):
        errors.append(f"{IS_FACE} must be true on every Koiter point")
    if np.any(np.asarray(skin.point_data[IS_FIXED], dtype=bool)):
        errors.append(f"{IS_FIXED} overlaps the Koiter point set")

    area = _triangle_area(np.asarray(skin.points, dtype=np.float64), triangles)
    rest_area = np.asarray(skin.cell_data["RestArea"], dtype=np.float64)
    if not np.array_equal(area, rest_area):
        errors.append("stored RestArea differs from live triangle geometry")
    total_area = float(math.fsum(float(value) for value in area))
    if not math.isclose(
        total_area, EXPECTED_SKIN_AREA_M2, rel_tol=0.0, abs_tol=AREA_ATOL_M2
    ):
        errors.append(f"Koiter area {total_area:.17g} != {EXPECTED_SKIN_AREA_M2:.17g}")
    topology = _topology_metrics(triangles)
    if topology["components"] != EXPECTED_SKIN_COMPONENTS:
        errors.append("Koiter domain is not one edge-connected component")
    if topology["nonmanifold_edges"] != 0:
        errors.append("Koiter domain has nonmanifold edges")
    if topology["identity_lhs_3F"] != topology["identity_rhs_edge_incidence"]:
        errors.append("Koiter edge-incidence identity failed")

    face_keys = _canonical_face_keys(global_ids, triangles)
    metrics: dict[str, Any] = {
        "content/n_points": int(skin.n_points),
        "content/n_triangles": int(skin.n_cells),
        "content/area_m2": total_area,
        "content/global_face_key_sha256": _face_key_hash(face_keys),
        **{f"content/{key}": value for key, value in content.items()},
        **{f"topology/{key}": value for key, value in topology.items()},
        "material/E_MPa": float(SKIN_E),
        "material/nu": float(SKIN_NU),
        "material/lambda_MPa": float(expected_lambda[0]),
        "material/mu_MPa": float(expected_mu[0]),
        "material/lame_conversion": LAME_CONVERSION,
        "material/prestrain": "p000: ActivationInv is exactly zero",
        "domain/face_group_allowlist": list(FACE_GROUPS),
        "domain/observed_group_names": sorted(observed_group_names),
        "domain/teeth_proximity_triangles": int(
            np.count_nonzero(skin.cell_data["TeethProximityTriangle"])
        ),
        "domain/gingiva_proximity_triangles": int(
            np.count_nonzero(skin.cell_data["GingivaProximityTriangle"])
        ),
        "validation/errors": sorted(set(errors)),
        "validation/ok": not errors,
    }
    return sorted(set(errors)), metrics


def _write_table(path: Path, row: dict[str, Any]) -> None:
    lines = [
        "| candidate | triangles | area m2 | components | E MPa | nu | lambda MPa | mu MPa | prestrain | hard gates | skin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        (
            f"| {row['label']} | {row['content/n_triangles']} | "
            f"{row['content/area_m2']:.13g} | {row['topology/components']} | "
            f"{row['material/E_MPa']:.6g} | {row['material/nu']:.6g} | "
            f"{row['material/lambda_MPa']:.9g} | "
            f"{row['material/mu_MPa']:.9g} | p000 | "
            f"{'ok' if row['validation/ok'] else 'failed'} | "
            f"`{row['skin/path']}` |"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: Config) -> None:
    # Explicitly approved to prepare the corrected p000 material input. This
    # entrypoint does not run a forward or inverse solve.
    _validate_output_contract(cfg)
    _require_exact_path(cfg.input_mesh, PREPARED_MESH, name="input_mesh")
    _require_exact_path(cfg.input_source_mesh, SOURCE_MESH, name="input_source_mesh")
    _require_exact_path(
        cfg.input_topology_reference,
        TOPOLOGY_REFERENCE,
        name="input_topology_reference",
    )
    input_identities_before = {
        "prepared_mesh": _require_file_identity(
            cfg.input_mesh,
            expected_size=PREPARED_MESH_SIZE_BYTES,
            expected_sha256=PREPARED_MESH_SHA256,
            name="prepared mesh",
        ),
        "source_mesh": _require_file_identity(
            cfg.input_source_mesh,
            expected_size=SOURCE_MESH_SIZE_BYTES,
            expected_sha256=SOURCE_MESH_SHA256,
            name="source head mesh",
        ),
        "topology_reference": _require_file_identity(
            cfg.input_topology_reference,
            expected_size=TOPOLOGY_REFERENCE_SIZE_BYTES,
            expected_sha256=TOPOLOGY_REFERENCE_SHA256,
            name="IsFace topology reference",
        ),
        "melon_face_mask_implementation": _require_authoritative_face_groups(),
    }

    mesh = _read_unstructured(cfg.input_mesh, name="prepared mesh")
    source = _read_unstructured(cfg.input_source_mesh, name="source head mesh")
    topology_reference = _read_polydata(
        cfg.input_topology_reference, name="IsFace topology reference"
    )
    domain = _domain_audit(mesh, source, topology_reference)
    skin = _make_skin(topology_reference, mesh)
    prewrite_errors, prewrite = _validate_skin(
        skin, mesh, topology_reference, expected_content=None
    )
    if prewrite_errors:
        msg = f"corrected baseline prewrite gates failed: {prewrite_errors}"
        raise RuntimeError(msg)
    if prewrite["content/n_triangles"] != EXPECTED_SKIN_TRIANGLES:
        msg = "refusing to pass anything except 29,899 triangles to Koiter"
        raise RuntimeError(msg)
    if (
        prewrite["content/global_face_key_sha256"]
        != domain["isface_global_face_key_sha256"]
    ):
        msg = "candidate face identity differs from the audited IsFace domain"
        raise RuntimeError(msg)

    melon.save(skin, cfg.output_skin)
    cherries.log_output(cfg.output_skin)
    readback = _read_polydata(cfg.output_skin, name="corrected baseline skin")
    expected_content = {
        key.removeprefix("content/"): value
        for key, value in prewrite.items()
        if key
        in {
            "content/topology_sha256",
            "content/material_sha256",
            "content/solver_sha256",
        }
    }
    readback_errors, readback_metrics = _validate_skin(
        readback,
        mesh,
        topology_reference,
        expected_content=expected_content,
    )
    if readback_errors:
        msg = f"corrected baseline readback gates failed: {readback_errors}"
        raise RuntimeError(msg)

    input_identities_after = {
        "prepared_mesh": _file_identity(cfg.input_mesh),
        "source_mesh": _file_identity(cfg.input_source_mesh),
        "topology_reference": _file_identity(cfg.input_topology_reference),
        "melon_face_mask_implementation": _file_identity(MELON_MASK_IMPLEMENTATION),
    }
    if input_identities_after != input_identities_before:
        msg = "a pinned source changed during corrected-baseline preparation"
        raise RuntimeError(msg)

    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "label": CANDIDATE_LABEL,
        "young_min_scale": 1.0,
        "prestrain_gain": 0.0,
        "skin/path": str(cfg.output_skin.relative_to(cfg.output_manifest.parent)),
        "skin/file_identity": _file_identity(cfg.output_skin),
        "skin/nu": float(SKIN_NU),
        "skin/thickness_m": float(SKIN_THICKNESS),
        "skin/lame_conversion": LAME_CONVERSION,
        "skin/domain": "all-vertex IsFace filtered PolyData",
        **prewrite,
        "readback/n_points": readback_metrics["content/n_points"],
        "readback/n_triangles": readback_metrics["content/n_triangles"],
        "readback/area_m2": readback_metrics["content/area_m2"],
        "readback/content/topology_sha256": readback_metrics["content/topology_sha256"],
        "readback/content/material_sha256": readback_metrics["content/material_sha256"],
        "readback/content/solver_sha256": readback_metrics["content/solver_sha256"],
        "readback/errors": [],
        "readback/ok": True,
        "validation/errors": [],
        "validation/ok": True,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "design": DESIGN,
        "experiment": "human-face Smile IsFace plane-stress corrected baseline",
        "purpose": (
            "establish one corrected homogeneous p000 baseline before any "
            "prestrain or heterogeneous-E inverse experiment"
        ),
        "input_mesh": str(cfg.input_mesh),
        "input_mesh_identity": input_identities_before["prepared_mesh"],
        "inputs": {
            "source_mesh/path": str(cfg.input_source_mesh),
            "source_mesh/file_identity": input_identities_before["source_mesh"],
            "topology_reference/path": str(cfg.input_topology_reference),
            "topology_reference/file_identity": input_identities_before[
                "topology_reference"
            ],
            "melon_face_mask_implementation/path": str(MELON_MASK_IMPLEMENTATION),
            "melon_face_mask_implementation/file_identity": (
                input_identities_before["melon_face_mask_implementation"]
            ),
            "identities_verified_stable": True,
        },
        "fixed_design": {
            "candidate_labels": [CANDIDATE_LABEL],
            "skin_domain": "all-vertex IsFace physically filtered PolyData",
            "skin_triangles": EXPECTED_SKIN_TRIANGLES,
            "skin_area_m2": EXPECTED_SKIN_AREA_M2,
            "skin_components": EXPECTED_SKIN_COMPONENTS,
            "skin_E_MPa": float(SKIN_E),
            "skin_nu": float(SKIN_NU),
            "skin_prestrain": "p000",
            "skin_lame_conversion": LAME_CONVERSION,
            "volume_lame_conversion": VOLUME_LAME_CONVERSION,
            "inverse_activation_initialization": "fresh exact zero",
            "inverse_optimizer": "Adam",
            "inverse_lr": 0.3,
            "inverse_optimizer_steps": 40,
            "inverse_evaluations": 41,
        },
        "domain_contract": domain,
        "constitutive_contract": {
            "skin": LAME_CONVERSION,
            "volume": VOLUME_LAME_CONVERSION,
            "skin/E_MPa": float(SKIN_E),
            "skin/nu": float(SKIN_NU),
            "skin/thickness_m": float(SKIN_THICKNESS),
            "skin/prestrain": "none; ActivationInv is exactly zero",
            "heterogeneous_material_fields": False,
        },
        "n_candidates": 1,
        "candidate_validation_errors": {},
        "validation_errors": [],
        "candidates": [row],
    }
    cfg.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    _write_table(cfg.output_table, row)
    cherries.log_output(cfg.output_manifest)
    cherries.log_output(cfg.output_table)
    logger.info("Wrote corrected baseline skin %s", cfg.output_skin)
    logger.info("Wrote %s", cfg.output_manifest)


if __name__ == "__main__":
    cherries.main(run)
