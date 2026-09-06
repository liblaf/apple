"""Render a fixed initial-frame crinkle clip of the HFP1 head.

This post-hoc pipeline reads the exact 41 saved inverse states, selects the
negative-Y crinkle-clip tetrahedra once at step 0, and advects that immutable
cell cohort through the remaining states.  The selection plane is never
reapplied after initialization.
"""

from __future__ import annotations

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
import csv
import hashlib
import itertools
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pydantic_settings as ps
import pyvista as pv
from PIL import Image

from liblaf import cherries

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
DESIGN = "hfp1-initial-frame-negative-y-crinkle-clip-evolution"
STEPS = tuple(range(41))
FPS = 30
RESOLUTION = (1_200, 1_200)
EXPECTED_PARAVIEW_VERSION = "6.1.1"
EXPECTED_POINTS = 228_660
EXPECTED_TETS = 1_146_517
EXPECTED_SKIN_POINTS = 15_299
EXPECTED_SKIN_TRIANGLES = 29_899
EXPECTED_ANCHOR_GLOBAL_POINT_ID = 52_222
ANCHOR_POINT_M = (
    1.4077719415114796,
    2.1730086794286745,
    0.09695972390415916,
)
PLANE_Y = ANCHOR_POINT_M[1]
FRACTION_FIELDS = ("FatFraction", "MuscleFraction", "AponeurosisFraction")
EXPECTED_WHOLE_COUNTS = {
    "Fat": 945_754,
    "Muscle": 170_548,
    "Aponeurosis": 30_215,
}
EXPECTED_CRINKLE_TETS = 423_522
EXPECTED_CRINKLE_POINTS = 85_619
EXPECTED_CRINKLE_COUNTS = {
    "Fat": 360_904,
    "Muscle": 58_419,
    "Aponeurosis": 4_199,
}
EXPECTED_INITIAL_STRICT_BOUNDARY_TETS = 16_732
EXPECTED_INITIAL_INCLUSIVE_BOUNDARY_TETS = 16_743
EXPECTED_SELECTION_IDS_SHA256 = (
    "2cd6b6618b04b1b9ef5e365c26c1a4b7cf3cbf3b39c9b78000e88bbd05f8d204"
)
EXPECTED_SELECTION_TOPOLOGY_SHA256 = (
    "e54791ee6386c8237475206fe07b32eebb9d253090b90dbfca3c1312ed58d18d"
)
EXPECTED_SKIN_TRACE_POINTS = 286
EXPECTED_SKIN_TRACE_LINES = 285
MATERIALS = {
    "0": {
        "name": "Fat",
        "fraction_field": "FatFraction",
        "constitutive_model": "Stable Neo-Hookean",
        "young_modulus_MPa": 0.003,
        "poisson_ratio": 0.49,
        "rgb": [0.929, 0.694, 0.125],
    },
    "1": {
        "name": "Muscle",
        "fraction_field": "MuscleFraction",
        "constitutive_model": "active Stable Neo-Hookean",
        "young_modulus_MPa": 0.03,
        "poisson_ratio": 0.49,
        "rgb": [0.796, 0.153, 0.153],
    },
    "2": {
        "name": "Aponeurosis",
        "fraction_field": "AponeurosisFraction",
        "constitutive_model": "Stable Neo-Hookean",
        "young_modulus_MPa": 0.1,
        "poisson_ratio": 0.35,
        "rgb": [0.122, 0.467, 0.706],
    },
}
SKIN_RGB = [0.0, 0.38, 0.38]
SKIN_LINE_WIDTH_PX = 1.5
CELL_EDGE_RGB = [0.10, 0.11, 0.13]
CELL_EDGE_WIDTH_PX = 0.45
EXPECTED_ORBICULARIS_UNION_BOUNDS_M = [
    1.3687930653465954,
    1.4446770011922974,
    2.1402443920667777,
    2.1824845957078955,
    0.06611088768428088,
    0.09695972390415916,
]
EXPECTED_CAMERA_FOCUS_M = [
    1.4067350332694464,
    PLANE_Y,
    0.08153530579422003,
]
CAMERA_PARALLEL_SCALE_M = 0.055

GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
SOURCE_DIR = (
    REPO_ROOT
    / "exp/2026/08/19/human-face-smile-fat-floor-skin-energy-prestrain-inverse/data"
)
SOURCE_ENDPOINT = SOURCE_DIR / "20-hfp1.vtu"
SOURCE_HISTORY = SOURCE_DIR / "20-hfp1-steps.vtkhdf"
SOURCE_SUMMARY = SOURCE_DIR / "20-hfp1-summary-final.json"
SOURCE_SKIN = (
    SOURCE_DIR / "10-prepared-material-cases-v2/skin-hfp1-selective-efat-c020.vtp"
)
PARAVIEW_SCRIPT = Path(__file__).with_name(
    "60-hfp1-full-head-material-section-evolution-paraview.py"
)
PVBATCH = Path("/usr/bin/pvbatch")
OUTPUT_DIR = GROUP_DIR / "data/60-hfp1-full-head-material-section-evolution"

EXPECTED_IDENTITIES = {
    "endpoint": {
        "size_bytes": 147_652_097,
        "sha256": "f93bf583819048b5d81a674c4f409450e3cd1200e0d3811b3dc98811480d53dd",
    },
    "history": {
        "size_bytes": 2_072_672_205,
        "sha256": "27f016f4a4b5cc4f54552ea7410c0a2feb758c646b24265e01680f31e29b86ce",
    },
    "summary": {
        "size_bytes": 111_700,
        "sha256": "73238e1a1cdb4f8f398b4d1430874abd22ef768d564225d4e3b07307bdb41540",
    },
    "skin": {
        "size_bytes": 2_072_304,
        "sha256": "89e0b349b1ba8002bc654325ba2f025c492b6e096c242c4c194ddede72cd117d",
    },
}


def output(name: str) -> str:
    return f"60-hfp1-full-head-material-section-evolution/{name}"


class Config(cherries.BaseConfig):
    """Pinned sources and outputs for the fixed crinkle-clip render."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    source_endpoint: Path = cherries.input(SOURCE_ENDPOINT)
    source_history: Path = cherries.input(SOURCE_HISTORY)
    source_summary: Path = cherries.input(SOURCE_SUMMARY)
    source_skin: Path = cherries.input(SOURCE_SKIN)
    paraview_script: Path = cherries.input(PARAVIEW_SCRIPT)
    output_material_series: Path = cherries.output(
        output("material-crinkle-history.vtu.series"), mkdir=True
    )
    output_skin_series: Path = cherries.output(
        output("skin-initial-trace-history.vtp.series"), mkdir=True
    )
    output_selection: Path = cherries.output(
        output("initial-crinkle-selection.npz"), mkdir=True
    )
    output_trajectory: Path = cherries.output(output("trajectory.csv"), mkdir=True)
    output_contract: Path = cherries.output(output("contract.json"), mkdir=True)
    output_renderer_receipt: Path = cherries.output(
        output("renderer-receipt.json"), mkdir=True
    )
    output_video: Path = cherries.output(
        output("60-hfp1-full-head-material-section-evolution.mp4"), mkdir=True
    )
    output_poster: Path = cherries.output(
        output("60-hfp1-full-head-material-section-evolution-poster.png"),
        mkdir=True,
    )
    output_pvsm: Path = cherries.output(
        output("60-hfp1-full-head-material-section-evolution.pvsm"), mkdir=True
    )
    output_receipt: Path = cherries.output(output("receipt.json"), mkdir=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, int | str]:
    return {"size_bytes": path.stat().st_size, "sha256": sha256(path)}


def ordered_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(sha256(path).encode())
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_dataset(dataset: pv.DataSet, path: Path) -> dict[str, int | str]:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    dataset.save(temporary, binary=True)
    temporary.replace(path)
    return identity(path)


def write_series(path: Path, frames: list[Path]) -> dict[str, int | str]:
    write_json(
        path,
        {
            "file-series-version": "1.0",
            "files": [
                {"name": str(frame.relative_to(path.parent)), "time": step}
                for step, frame in zip(STEPS, frames, strict=True)
            ],
        },
    )
    return identity(path)


def expected_paths() -> dict[str, Path]:
    return {
        "source_endpoint": SOURCE_ENDPOINT,
        "source_history": SOURCE_HISTORY,
        "source_summary": SOURCE_SUMMARY,
        "source_skin": SOURCE_SKIN,
        "paraview_script": PARAVIEW_SCRIPT,
        "output_material_series": OUTPUT_DIR / "material-crinkle-history.vtu.series",
        "output_skin_series": OUTPUT_DIR / "skin-initial-trace-history.vtp.series",
        "output_selection": OUTPUT_DIR / "initial-crinkle-selection.npz",
        "output_trajectory": OUTPUT_DIR / "trajectory.csv",
        "output_contract": OUTPUT_DIR / "contract.json",
        "output_renderer_receipt": OUTPUT_DIR / "renderer-receipt.json",
        "output_video": OUTPUT_DIR / "60-hfp1-full-head-material-section-evolution.mp4",
        "output_poster": OUTPUT_DIR
        / "60-hfp1-full-head-material-section-evolution-poster.png",
        "output_pvsm": OUTPUT_DIR / "60-hfp1-full-head-material-section-evolution.pvsm",
        "output_receipt": OUTPUT_DIR / "receipt.json",
    }


def validate_config(cfg: Config) -> None:
    for name, expected in expected_paths().items():
        actual = getattr(cfg, name)
        if actual.resolve() != expected.resolve():
            raise ValueError(f"{name} must be {expected}, got {actual}")
    if not PVBATCH.is_file():
        raise FileNotFoundError(PVBATCH)
    stale = list(OUTPUT_DIR.rglob("*")) if OUTPUT_DIR.exists() else []
    stale = [path for path in stale if path.is_file()]
    if stale:
        raise FileExistsError(f"refusing stale output files: {stale}")


def validate_source(path: Path, key: str) -> dict[str, int | str]:
    actual = identity(path)
    if actual != EXPECTED_IDENTITIES[key]:
        raise ValueError(f"{key} identity changed: {actual}")
    return actual


def array_bounds(points: np.ndarray) -> list[float]:
    low, high = points.min(axis=0), points.max(axis=0)
    return [
        float(low[0]),
        float(high[0]),
        float(low[1]),
        float(high[1]),
        float(low[2]),
        float(high[2]),
    ]


def union_bounds(bounds: list[list[float]]) -> list[float]:
    return [
        min(item[0] for item in bounds),
        max(item[1] for item in bounds),
        min(item[2] for item in bounds),
        max(item[3] for item in bounds),
        min(item[4] for item in bounds),
        max(item[5] for item in bounds),
    ]


def line_component_count(section: pv.PolyData) -> int:
    parent = np.arange(section.n_points, dtype=np.int64)

    def find(point: int) -> int:
        while parent[point] != point:
            parent[point] = parent[parent[point]]
            point = int(parent[point])
        return point

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    packed = np.asarray(section.lines, dtype=np.int64)
    cursor = 0
    used: set[int] = set()
    while cursor < packed.size:
        count = int(packed[cursor])
        points = packed[cursor + 1 : cursor + count + 1]
        if count < 2 or points.size != count:
            raise ValueError("skin section contains a malformed line")
        used.update(int(point) for point in points)
        for left, right in itertools.pairwise(points):
            union(int(left), int(right))
        cursor += count + 1
    if cursor != packed.size or used != set(range(section.n_points)):
        raise ValueError("skin section connectivity changed")
    return len({find(point) for point in used})


def arrays_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for source in arrays:
        array = np.ascontiguousarray(source)
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def source_id_sha256(source_ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(source_ids, dtype="<i8").tobytes()).hexdigest()


def tetra_topology_sha256(grid: pv.UnstructuredGrid) -> str:
    packed = np.asarray(grid.cells, dtype=np.int64)
    if packed.size != 5 * grid.n_cells or not np.all(packed[::5] == 4):
        raise ValueError("crinkle selection is not packed tetrahedra")
    local_connectivity = packed.reshape(-1, 5)[:, 1:]
    global_point_ids = np.asarray(grid.point_data["GlobalPointId"], dtype=np.int64)
    source_cell_ids = np.asarray(grid.cell_data["SourceCellId"], dtype=np.int64)
    canonical = np.column_stack(
        [source_cell_ids, global_point_ids[local_connectivity]]
    ).astype("<i8", copy=False)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def build_initial_skin_trace(
    initial_points: np.ndarray,
    skin_global_ids: np.ndarray,
    skin_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    skin_points = initial_points[skin_global_ids]
    signed = skin_points[:, 1] - PLANE_Y
    if np.any(signed == 0.0):
        raise ValueError("skin vertex lies exactly on the selection plane")
    edge_to_point: dict[tuple[int, int], int] = {}
    edge_endpoints: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    segments: list[tuple[int, int]] = []
    source_triangles: list[int] = []
    for triangle_id, face in enumerate(skin_faces):
        crossings: list[int] = []
        for left_local, right_local in ((0, 1), (1, 2), (2, 0)):
            left_skin = int(face[left_local])
            right_skin = int(face[right_local])
            left_signed = float(signed[left_skin])
            right_signed = float(signed[right_skin])
            if left_signed * right_signed >= 0.0:
                continue
            left_global = int(skin_global_ids[left_skin])
            right_global = int(skin_global_ids[right_skin])
            if left_global > right_global:
                left_global, right_global = right_global, left_global
            key = (left_global, right_global)
            point_id = edge_to_point.get(key)
            if point_id is None:
                point_id = len(edge_endpoints)
                edge_to_point[key] = point_id
                edge_endpoints.append(key)
                denominator = (
                    initial_points[right_global, 1] - initial_points[left_global, 1]
                )
                weight = (PLANE_Y - initial_points[left_global, 1]) / denominator
                if not 0.0 < weight < 1.0:
                    raise ValueError("invalid initial skin-edge interpolation weight")
                edge_weights.append(float(weight))
            crossings.append(point_id)
        if crossings:
            if len(crossings) != 2 or crossings[0] == crossings[1]:
                raise ValueError("initial plane has a degenerate skin intersection")
            segments.append((crossings[0], crossings[1]))
            source_triangles.append(triangle_id)
    endpoints = np.asarray(edge_endpoints, dtype=np.int64)
    weights = np.asarray(edge_weights, dtype=np.float64)
    lines = np.asarray(segments, dtype=np.int64)
    triangles = np.asarray(source_triangles, dtype=np.int64)
    if (
        endpoints.shape != (EXPECTED_SKIN_TRACE_POINTS, 2)
        or weights.shape != (EXPECTED_SKIN_TRACE_POINTS,)
        or lines.shape != (EXPECTED_SKIN_TRACE_LINES, 2)
        or triangles.shape != (EXPECTED_SKIN_TRACE_LINES,)
    ):
        raise ValueError("initial skin-trace topology changed")
    return endpoints, weights, lines, triangles


def write_selection(
    path: Path,
    source_tetra_ids: np.ndarray,
    skin_edge_global_point_ids: np.ndarray,
    skin_edge_weights: np.ndarray,
    skin_lines: np.ndarray,
    skin_source_triangle_ids: np.ndarray,
) -> dict[str, int | str]:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    np.savez_compressed(
        temporary,
        source_tetra_ids=np.asarray(source_tetra_ids, dtype=np.int64),
        skin_edge_global_point_ids=np.asarray(
            skin_edge_global_point_ids, dtype=np.int64
        ),
        skin_edge_weights=np.asarray(skin_edge_weights, dtype=np.float64),
        skin_lines=np.asarray(skin_lines, dtype=np.int64),
        skin_source_triangle_ids=np.asarray(skin_source_triangle_ids, dtype=np.int64),
    )
    temporary.replace(path)
    return identity(path)


def crinkle_record(
    grid: pv.UnstructuredGrid,
    selected_source_ids: np.ndarray,
) -> dict[str, Any]:
    if (
        grid.n_points != EXPECTED_CRINKLE_POINTS
        or grid.n_cells != EXPECTED_CRINKLE_TETS
        or not np.all(grid.celltypes == pv.CellType.TETRA)
    ):
        raise ValueError("fixed crinkle topology changed")
    source_ids = np.asarray(grid.cell_data["SourceCellId"], dtype=np.int64)
    if not np.array_equal(source_ids, selected_source_ids):
        raise ValueError("fixed crinkle SourceCellId cohort changed")
    global_point_ids = np.asarray(grid.point_data["GlobalPointId"], dtype=np.int64)
    if (
        global_point_ids.shape != (EXPECTED_CRINKLE_POINTS,)
        or np.unique(global_point_ids).size != EXPECTED_CRINKLE_POINTS
    ):
        raise ValueError("fixed crinkle GlobalPointId map changed")
    fractions = np.column_stack(
        [np.asarray(grid.cell_data[name], dtype=np.float64) for name in FRACTION_FIELDS]
    )
    dominant = np.asarray(grid.cell_data["DominantMaterial"], dtype=np.int32)
    fraction_sum_error = float(np.max(np.abs(fractions.sum(axis=1) - 1.0)))
    if (
        fractions.shape != (EXPECTED_CRINKLE_TETS, 3)
        or np.any(~np.isfinite(fractions))
        or np.any((fractions < 0.0) | (fractions > 1.0))
        or fraction_sum_error > 1e-12
        or not np.array_equal(dominant, np.argmax(fractions, axis=1))
    ):
        raise ValueError("crinkle material categories changed")
    counts = {
        str(MATERIALS[str(index)]["name"]): int(np.count_nonzero(dominant == index))
        for index in range(3)
    }
    if counts != EXPECTED_CRINKLE_COUNTS:
        raise ValueError(f"crinkle dominant counts changed: {counts}")
    topology_sha256 = tetra_topology_sha256(grid)
    if topology_sha256 != EXPECTED_SELECTION_TOPOLOGY_SHA256:
        raise ValueError(f"fixed crinkle connectivity changed: {topology_sha256}")
    return {
        "points": int(grid.n_points),
        "tetrahedra": int(grid.n_cells),
        "dominant_counts": counts,
        "fraction_sum_max_abs_error": fraction_sum_error,
        "source_cell_ids_sha256": source_id_sha256(source_ids),
        "topology_sha256": topology_sha256,
        "bounds_m": array_bounds(np.asarray(grid.points, dtype=np.float64)),
    }


def skin_trace_record(
    trace: pv.PolyData,
    expected_lines: np.ndarray,
    expected_source_triangles: np.ndarray,
) -> dict[str, Any]:
    packed = np.asarray(trace.lines, dtype=np.int64)
    if (
        trace.n_points != EXPECTED_SKIN_TRACE_POINTS
        or trace.n_cells != EXPECTED_SKIN_TRACE_LINES
        or trace.n_lines != trace.n_cells
        or packed.size != 3 * trace.n_cells
        or not np.array_equal(packed.reshape(-1, 3)[:, 0], np.full(trace.n_cells, 2))
        or not np.array_equal(packed.reshape(-1, 3)[:, 1:], expected_lines)
    ):
        raise ValueError("fixed initial skin trace topology changed")
    edge_ids = np.asarray(trace.point_data["InitialSkinEdgeId"], dtype=np.int64)
    source_triangles = np.asarray(
        trace.cell_data["SourceSkinTriangleId"], dtype=np.int64
    )
    if not np.array_equal(edge_ids, np.arange(trace.n_points, dtype=np.int64)):
        raise ValueError("fixed initial skin edge IDs changed")
    if not np.array_equal(source_triangles, expected_source_triangles):
        raise ValueError("fixed initial skin source triangles changed")
    components = line_component_count(trace)
    if components != 1:
        raise ValueError("fixed initial skin trace is disconnected")
    points = np.asarray(trace.points, dtype=np.float64)
    return {
        "points": int(trace.n_points),
        "lines": int(trace.n_cells),
        "components": components,
        "topology_sha256": arrays_sha256(
            edge_ids, expected_lines, expected_source_triangles
        ),
        "bounds_m": array_bounds(points),
        "max_abs_distance_from_initial_plane_m": float(
            np.max(np.abs(points[:, 1] - PLANE_Y))
        ),
    }


def write_trajectory(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    fieldnames = [
        "step",
        "crinkle_points",
        "crinkle_tetrahedra",
        "fat_tetrahedra",
        "muscle_tetrahedra",
        "aponeurosis_tetrahedra",
        "currently_strictly_straddling_plane_tetrahedra",
        "currently_touching_or_straddling_plane_tetrahedra",
        "skin_points",
        "skin_lines",
        "skin_components",
        "x_min_m",
        "x_max_m",
        "y_min_m",
        "y_max_m",
        "z_min_m",
        "z_max_m",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_pngs(paths: list[Path]) -> None:
    if len(paths) != len(STEPS):
        raise ValueError("PNG frame count changed")
    for step, path in zip(STEPS, paths, strict=True):
        if path.name != f"frame-{step:03d}.png" or path.stat().st_size <= 50_000:
            raise ValueError(f"invalid rendered frame: {path}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.size != RESOLUTION or image.mode != "RGB":
                raise ValueError(f"rendered frame properties changed: {path}")


def validate_renderer_receipt(
    receipt: dict[str, Any],
    contract: dict[str, Any],
    frame_paths: list[Path],
    pvsm: Path,
) -> None:
    required = {
        "schema_version",
        "design",
        "complete",
        "paraview_version",
        "frame_count",
        "TimestepValues",
        "frames",
        "ordered_png_sha256",
        "camera",
        "plane",
        "selection",
        "material_representation",
        "opaque_selected_volume_surface",
        "cell_edges_rendered",
        "external_tetra_faces_rendered",
        "cell_edge_rgb",
        "cell_edge_width_px",
        "determinant_metrics_rendered",
        "skin",
        "pvsm",
    }
    render = contract["render"]
    if (
        set(receipt) != required
        or receipt["schema_version"] != SCHEMA_VERSION
        or receipt["design"] != DESIGN
        or receipt["complete"] is not True
        or receipt["paraview_version"] != EXPECTED_PARAVIEW_VERSION
        or receipt["frame_count"] != len(STEPS)
        or receipt["TimestepValues"] != [float(step) for step in STEPS]
        or receipt["camera"] != contract["camera"]
        or receipt["plane"] != contract["plane"]
        or receipt["selection"] != contract["selection"]
        or receipt["material_representation"] != "Surface With Edges"
        or receipt["opaque_selected_volume_surface"] is not True
        or receipt["cell_edges_rendered"] is not True
        or receipt["external_tetra_faces_rendered"] is not True
        or receipt["cell_edge_rgb"] != render["cell_edge_rgb"]
        or receipt["cell_edge_width_px"] != render["cell_edge_width_px"]
        or receipt["determinant_metrics_rendered"] is not False
        or receipt["skin"] != contract["skin"]
        or receipt["ordered_png_sha256"] != ordered_digest(frame_paths)
        or receipt["pvsm"] != {"path": str(pvsm.resolve()), "identity": identity(pvsm)}
    ):
        raise ValueError("renderer receipt contract changed")
    records = receipt["frames"]
    if not isinstance(records, list) or len(records) != len(STEPS):
        raise ValueError("renderer frame receipt count changed")
    for step, (record, frame_path, source) in enumerate(
        zip(records, frame_paths, contract["frames"], strict=True)
    ):
        material = record.get("material_crinkle", {})
        skin = record.get("skin_trace", {})
        if (
            record.get("step") != step
            or Path(str(record.get("path"))).resolve() != frame_path.resolve()
            or record.get("identity") != identity(frame_path)
            or material.get("points") != source["material_crinkle"]["points"]
            or material.get("tetrahedra") != source["material_crinkle"]["tetrahedra"]
            or material.get("dominant_counts")
            != source["material_crinkle"]["dominant_counts"]
            or material.get("source_cell_ids_sha256")
            != contract["selection"]["source_cell_ids_sha256"]
            or material.get("topology_sha256")
            != contract["selection"]["tetra_topology_sha256"]
            or material.get("argmax_of_continuous_fractions") is not True
            or float(material.get("fraction_sum_max_abs_error", float("inf"))) > 1e-12
            or skin.get("points") != source["skin_trace"]["points"]
            or skin.get("lines") != source["skin_trace"]["lines"]
            or skin.get("components") != source["skin_trace"]["components"]
            or skin.get("topology_sha256") != source["skin_trace"]["topology_sha256"]
        ):
            raise ValueError(f"renderer frame receipt changed at step {step}")


def encode_video(frames_dir: Path, video: Path) -> dict[str, Any]:
    temporary = video.with_name(f".{video.stem}.tmp{video.suffix}")
    if video.exists() or temporary.exists():
        raise FileExistsError(video)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "warning",
            "-framerate",
            str(FPS),
            "-start_number",
            "0",
            "-i",
            str(frames_dir / "frame-%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=True,
    )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(temporary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    facts = json.loads(probe.stdout)
    stream = facts["streams"][0]
    count = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    duration = float(facts["format"]["duration"])
    if (
        stream["codec_name"] != "h264"
        or stream["pix_fmt"] != "yuv420p"
        or int(stream["width"]) != RESOLUTION[0]
        or int(stream["height"]) != RESOLUTION[1]
        or stream["r_frame_rate"] != "30/1"
        or stream["avg_frame_rate"] != "30/1"
        or count != len(STEPS)
        or abs(duration - len(STEPS) / FPS) > 1 / FPS
    ):
        raise ValueError(f"video contract failed: {facts}")
    framehash = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(temporary),
            "-map",
            "0:v:0",
            "-f",
            "framehash",
            "-hash",
            "sha256",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    decoded_hashes = [
        line.rsplit(",", maxsplit=1)[-1].strip()
        for line in framehash.stdout.splitlines()
        if line and not line.startswith("#")
    ]
    if len(decoded_hashes) != len(STEPS) or len(set(decoded_hashes)) != len(STEPS):
        raise ValueError("encoded video does not contain 41 unique decoded frames")
    facts["decoded_frame_sha256"] = decoded_hashes
    facts["decoded_frames_unique"] = True
    temporary.replace(video)
    return facts


def copy_poster(source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    if source == target or target.exists() or temporary.exists():
        raise FileExistsError(target)
    os.link(source, temporary)
    temporary.replace(target)
    if identity(source) != identity(target):
        raise ValueError("poster does not exactly match the final frame")


def main(cfg: Config) -> None:
    validate_config(cfg)
    source_identities = {
        key: validate_source(path, key)
        for key, path in (
            ("endpoint", cfg.source_endpoint),
            ("history", cfg.source_history),
            ("summary", cfg.source_summary),
            ("skin", cfg.source_skin),
        )
    }
    summary = read_json(cfg.source_summary)
    expected_summary = {
        "case": "20-hfp1",
        "best_step": 40,
        "inverse/evaluations": 41,
        "history_frames": 41,
        "inverse/converged": False,
        "inverse/forward_fail_count": 0,
        "inverse/adjoint_fail_count": 0,
        "inverse/stop_reason": "step_limit_smooth_decrease",
    }
    if {key: summary.get(key) for key in expected_summary} != expected_summary:
        raise ValueError("HFP1 source summary changed")

    endpoint = pv.read(cfg.source_endpoint)
    if not isinstance(endpoint, pv.UnstructuredGrid):
        raise TypeError(f"endpoint is {type(endpoint).__name__}")
    if endpoint.n_points != EXPECTED_POINTS or endpoint.n_cells != EXPECTED_TETS:
        raise ValueError("HFP1 endpoint topology changed")
    if not np.all(endpoint.celltypes == pv.CellType.TETRA):
        raise ValueError("HFP1 endpoint is not tetra-only")
    reference_points = np.asarray(endpoint.points, dtype=np.float64).copy()
    endpoint_deformed = np.asarray(
        endpoint.point_data["DeformedPoint"], dtype=np.float64
    ).copy()
    global_ids = np.asarray(endpoint.point_data["GlobalPointId"], dtype=np.int64)
    if not np.array_equal(global_ids, np.arange(EXPECTED_POINTS, dtype=np.int64)):
        raise ValueError("endpoint GlobalPointId ordering changed")
    fractions = np.column_stack(
        [
            np.asarray(endpoint.cell_data[name], dtype=np.float64)
            for name in FRACTION_FIELDS
        ]
    ).copy()
    if (
        np.any(~np.isfinite(fractions))
        or np.any((fractions < 0.0) | (fractions > 1.0))
        or float(np.max(np.abs(fractions.sum(axis=1) - 1.0))) != 0.0
    ):
        raise ValueError("volume material fractions changed")
    dominant = np.argmax(fractions, axis=1).astype(np.int32)
    whole_counts = {
        str(MATERIALS[str(index)]["name"]): int(np.count_nonzero(dominant == index))
        for index in range(3)
    }
    if whole_counts != EXPECTED_WHOLE_COUNTS:
        raise ValueError(f"whole-head dominant counts changed: {whole_counts}")
    mixed_tets = int(np.count_nonzero((fractions > 0.001).sum(axis=1) > 1))
    if mixed_tets != 232_741:
        raise ValueError(f"mixed-tet count changed: {mixed_tets}")
    active = np.asarray(endpoint.cell_data["ActivationMask"], dtype=bool)
    muscle_id = np.asarray(endpoint.cell_data["MuscleId"], dtype=np.int64)
    oo_selected = active & (muscle_id == 254) & (fractions[:, 1] >= 0.5)
    packed_cells = np.asarray(endpoint.cells, dtype=np.int64).copy()
    celltypes = np.asarray(endpoint.celltypes, dtype=np.uint8).copy()
    full_grid = pv.UnstructuredGrid(packed_cells, celltypes, reference_points)
    full_grid.point_data["GlobalPointId"] = np.arange(EXPECTED_POINTS, dtype=np.int64)
    full_grid.cell_data["SourceCellId"] = np.arange(EXPECTED_TETS, dtype=np.int64)
    full_grid.cell_data["DominantMaterial"] = dominant
    for index, name in enumerate(FRACTION_FIELDS):
        full_grid.cell_data[name] = fractions[:, index]
    tetrahedra = np.asarray(full_grid.cells).reshape(-1, 5)[:, 1:]
    oo_points = np.unique(tetrahedra[oo_selected].ravel())
    del endpoint, packed_cells, celltypes, global_ids

    skin_source = pv.read(cfg.source_skin)
    if not isinstance(skin_source, pv.PolyData):
        raise TypeError(f"skin source is {type(skin_source).__name__}")
    if (
        skin_source.n_points != EXPECTED_SKIN_POINTS
        or skin_source.n_cells != EXPECTED_SKIN_TRIANGLES
    ):
        raise ValueError("skin membrane topology changed")
    skin_global_ids = np.asarray(
        skin_source.point_data["GlobalPointId"], dtype=np.int64
    ).copy()
    if not np.array_equal(
        np.asarray(skin_source.points), reference_points[skin_global_ids]
    ):
        raise ValueError("skin membrane is not mapped to the full head")
    packed_skin_faces = np.asarray(skin_source.faces, dtype=np.int64).copy()
    if packed_skin_faces.size != 4 * EXPECTED_SKIN_TRIANGLES or not np.all(
        packed_skin_faces[::4] == 3
    ):
        raise ValueError("skin membrane is not triangle-only")
    skin_faces = packed_skin_faces.reshape(-1, 4)[:, 1:]
    del skin_source

    inputs_dir = OUTPUT_DIR / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    material_paths: list[Path] = []
    skin_paths: list[Path] = []
    frame_records: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    crinkle_bounds: list[list[float]] = []
    oo_bounds: list[list[float]] = []
    crinkle_grid: pv.UnstructuredGrid | None = None
    selected_source_ids: np.ndarray | None = None
    selected_global_point_ids: np.ndarray | None = None
    skin_edge_endpoints: np.ndarray | None = None
    skin_edge_weights: np.ndarray | None = None
    skin_lines: np.ndarray | None = None
    skin_source_triangles: np.ndarray | None = None
    selection_identity: dict[str, int | str] | None = None
    skin_topology_sha256: str | None = None
    final_points: np.ndarray | None = None

    with h5py.File(cfg.source_history, "r") as hdf:
        root = hdf["VTKHDF"]
        time_values = np.asarray(root["Steps/Values"], dtype=np.float64)
        inverse_steps = np.asarray(root["FieldData/inverse_step"], dtype=np.int64)
        if not np.array_equal(time_values, np.asarray(STEPS, dtype=float)):
            raise ValueError("VTKHDF time values changed")
        if not np.array_equal(inverse_steps, np.asarray(STEPS, dtype=np.int64)):
            raise ValueError("VTKHDF inverse steps changed")
        point_offsets = np.asarray(
            root["Steps/PointDataOffsets/DeformedPoint"], dtype=np.int64
        )
        if point_offsets.shape != (len(STEPS),):
            raise ValueError("VTKHDF point offsets changed")
        payload = root["PointData/DeformedPoint"]
        for step in STEPS:
            offset = int(point_offsets[step])
            deformed_points = np.asarray(
                payload[offset : offset + EXPECTED_POINTS], dtype=np.float64
            )
            if deformed_points.shape != (EXPECTED_POINTS, 3):
                raise ValueError(f"step {step} point block changed")
            if step == 0:
                anchor_local = int(np.argmax(deformed_points[oo_points, 2]))
                anchor_global = int(oo_points[anchor_local])
                anchor_point = deformed_points[anchor_global]
                if anchor_global != EXPECTED_ANCHOR_GLOBAL_POINT_ID or not np.allclose(
                    anchor_point, ANCHOR_POINT_M, atol=1e-12, rtol=0.0
                ):
                    raise ValueError("initial maximum-Z Orbicularis anchor changed")
                initial_tetra_y = deformed_points[tetrahedra, 1]
                selection_mask = initial_tetra_y.min(axis=1) <= PLANE_Y
                selected_source_ids = np.flatnonzero(selection_mask).astype(np.int64)
                if (
                    selected_source_ids.shape != (EXPECTED_CRINKLE_TETS,)
                    or source_id_sha256(selected_source_ids)
                    != EXPECTED_SELECTION_IDS_SHA256
                    or np.any(initial_tetra_y[~selection_mask].min(axis=1) <= PLANE_Y)
                ):
                    raise ValueError("initial negative-Y crinkle selection changed")
                strict_boundary = int(
                    np.count_nonzero(
                        (initial_tetra_y[:, 0:].min(axis=1) < PLANE_Y)
                        & (initial_tetra_y[:, 0:].max(axis=1) > PLANE_Y)
                    )
                )
                inclusive_boundary = int(
                    np.count_nonzero(
                        (initial_tetra_y.min(axis=1) <= PLANE_Y)
                        & (initial_tetra_y.max(axis=1) >= PLANE_Y)
                    )
                )
                if (
                    strict_boundary != EXPECTED_INITIAL_STRICT_BOUNDARY_TETS
                    or inclusive_boundary != EXPECTED_INITIAL_INCLUSIVE_BOUNDARY_TETS
                ):
                    raise ValueError("initial crinkle-boundary tetrahedra changed")
                full_grid.points = deformed_points
                crinkle_grid = full_grid.extract_cells(selected_source_ids)
                for association, name in (
                    (crinkle_grid.point_data, "vtkOriginalPointIds"),
                    (crinkle_grid.cell_data, "vtkOriginalCellIds"),
                ):
                    if name in association:
                        del association[name]
                selected_global_point_ids = np.asarray(
                    crinkle_grid.point_data["GlobalPointId"], dtype=np.int64
                ).copy()
                if not np.array_equal(
                    np.asarray(crinkle_grid.cell_data["SourceCellId"], dtype=np.int64),
                    selected_source_ids,
                ):
                    raise ValueError(
                        "initial extraction reordered the source tetrahedra"
                    )
                if (
                    crinkle_grid.n_points != EXPECTED_CRINKLE_POINTS
                    or tetra_topology_sha256(crinkle_grid)
                    != EXPECTED_SELECTION_TOPOLOGY_SHA256
                ):
                    raise ValueError("initial crinkle extraction topology changed")
                (
                    skin_edge_endpoints,
                    skin_edge_weights,
                    skin_lines,
                    skin_source_triangles,
                ) = build_initial_skin_trace(
                    deformed_points, skin_global_ids, skin_faces
                )
                skin_topology_sha256 = arrays_sha256(
                    np.arange(EXPECTED_SKIN_TRACE_POINTS, dtype=np.int64),
                    skin_lines,
                    skin_source_triangles,
                )
                selection_identity = write_selection(
                    cfg.output_selection,
                    selected_source_ids,
                    skin_edge_endpoints,
                    skin_edge_weights,
                    skin_lines,
                    skin_source_triangles,
                )
            oo_bounds.append(array_bounds(deformed_points[oo_points]))
            if (
                crinkle_grid is None
                or selected_source_ids is None
                or selected_global_point_ids is None
                or skin_edge_endpoints is None
                or skin_edge_weights is None
                or skin_lines is None
                or skin_source_triangles is None
            ):
                raise RuntimeError("step-0 crinkle selection was not initialized")
            crinkle_grid.points = deformed_points[selected_global_point_ids]
            if not np.array_equal(
                np.asarray(crinkle_grid.points),
                deformed_points[selected_global_point_ids],
            ):
                raise ValueError("crinkle point mapping changed")
            material_metrics = crinkle_record(crinkle_grid, selected_source_ids)
            selected_tetra_y = deformed_points[tetrahedra[selected_source_ids], 1]
            strict_current = int(
                np.count_nonzero(
                    (selected_tetra_y.min(axis=1) < PLANE_Y)
                    & (selected_tetra_y.max(axis=1) > PLANE_Y)
                )
            )
            inclusive_current = int(
                np.count_nonzero(
                    (selected_tetra_y.min(axis=1) <= PLANE_Y)
                    & (selected_tetra_y.max(axis=1) >= PLANE_Y)
                )
            )
            material_metrics["currently_strictly_straddling_plane_tetrahedra"] = (
                strict_current
            )
            material_metrics["currently_touching_or_straddling_plane_tetrahedra"] = (
                inclusive_current
            )
            material_path = inputs_dir / f"material-crinkle-step-{step:03d}.vtu"
            material_identity = save_dataset(crinkle_grid, material_path)
            material_paths.append(material_path)
            crinkle_bounds.append(material_metrics["bounds_m"])

            trace_points = (1.0 - skin_edge_weights[:, None]) * deformed_points[
                skin_edge_endpoints[:, 0]
            ] + skin_edge_weights[:, None] * deformed_points[skin_edge_endpoints[:, 1]]
            packed_trace_lines = np.column_stack(
                [np.full(len(skin_lines), 2, dtype=np.int64), skin_lines]
            ).ravel()
            skin_trace = pv.PolyData(trace_points, lines=packed_trace_lines)
            skin_trace.point_data["InitialSkinEdgeId"] = np.arange(
                skin_trace.n_points, dtype=np.int64
            )
            skin_trace.cell_data["SourceSkinTriangleId"] = skin_source_triangles
            skin_trace.cell_data["SkinMembrane"] = np.ones(
                skin_trace.n_cells, dtype=np.int8
            )
            skin_metrics = skin_trace_record(
                skin_trace, skin_lines, skin_source_triangles
            )
            if (
                step == 0
                and skin_metrics["max_abs_distance_from_initial_plane_m"] > 1e-12
            ):
                raise ValueError("initial skin trace left the selection plane")
            skin_path = inputs_dir / f"skin-initial-trace-step-{step:03d}.vtp"
            skin_identity = save_dataset(skin_trace, skin_path)
            skin_paths.append(skin_path)

            frame_records.append(
                {
                    "step": step,
                    "material_crinkle": {
                        "path": str(material_path.resolve()),
                        "identity": material_identity,
                        **material_metrics,
                    },
                    "skin_trace": {
                        "path": str(skin_path.resolve()),
                        "identity": skin_identity,
                        **skin_metrics,
                    },
                }
            )
            counts = material_metrics["dominant_counts"]
            bounds = material_metrics["bounds_m"]
            trajectory_rows.append(
                {
                    "step": step,
                    "crinkle_points": material_metrics["points"],
                    "crinkle_tetrahedra": material_metrics["tetrahedra"],
                    "fat_tetrahedra": counts["Fat"],
                    "muscle_tetrahedra": counts["Muscle"],
                    "aponeurosis_tetrahedra": counts["Aponeurosis"],
                    "currently_strictly_straddling_plane_tetrahedra": strict_current,
                    "currently_touching_or_straddling_plane_tetrahedra": inclusive_current,
                    "skin_points": skin_metrics["points"],
                    "skin_lines": skin_metrics["lines"],
                    "skin_components": skin_metrics["components"],
                    "x_min_m": bounds[0],
                    "x_max_m": bounds[1],
                    "y_min_m": bounds[2],
                    "y_max_m": bounds[3],
                    "z_min_m": bounds[4],
                    "z_max_m": bounds[5],
                }
            )
            cherries.set_step(step)
            cherries.log_metrics(
                {
                    "crinkle/tetrahedra": material_metrics["tetrahedra"],
                    "crinkle/fat_tetrahedra": counts["Fat"],
                    "crinkle/muscle_tetrahedra": counts["Muscle"],
                    "crinkle/aponeurosis_tetrahedra": counts["Aponeurosis"],
                    "crinkle/currently_straddling": strict_current,
                    "skin/lines": skin_metrics["lines"],
                }
            )
            logger.info(
                "Prepared step %02d: %d fixed tetrahedra (%d fat, %d muscle, %d aponeurosis)",
                step,
                material_metrics["tetrahedra"],
                counts["Fat"],
                counts["Muscle"],
                counts["Aponeurosis"],
            )
            if step == STEPS[-1]:
                final_points = deformed_points.copy()

    if (
        final_points is None
        or not np.array_equal(final_points, endpoint_deformed)
        or crinkle_grid is None
        or selected_source_ids is None
        or selected_global_point_ids is None
        or selection_identity is None
        or skin_topology_sha256 is None
        or not np.array_equal(
            np.asarray(crinkle_grid.points),
            endpoint_deformed[selected_global_point_ids],
        )
    ):
        raise ValueError("history step 40 does not reproduce the endpoint")
    all_bounds = union_bounds(crinkle_bounds)
    expected_union = [
        1.3451348154345881,
        1.4689473780830882,
        2.110828902932927,
        2.182122865309264,
        -0.02446428876198042,
        0.10129456819534544,
    ]
    if not np.allclose(all_bounds, expected_union, atol=1e-12, rtol=0.0):
        raise ValueError(f"crinkle union bounds changed: {all_bounds}")
    all_oo_bounds = union_bounds(oo_bounds)
    if not np.allclose(
        all_oo_bounds,
        EXPECTED_ORBICULARIS_UNION_BOUNDS_M,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(f"all-state Orbicularis bounds changed: {all_oo_bounds}")
    focus_x = 0.5 * (all_oo_bounds[0] + all_oo_bounds[1])
    focus_z = 0.5 * (all_oo_bounds[4] + all_oo_bounds[5])
    if not np.allclose(
        [focus_x, PLANE_Y, focus_z],
        EXPECTED_CAMERA_FOCUS_M,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("all-state Orbicularis camera focus changed")
    camera = {
        "focus": [focus_x, PLANE_Y, focus_z],
        "position": [focus_x, PLANE_Y + 0.30, focus_z],
        "view_up": [0.0, 0.0, 1.0],
        "look_direction": [0.0, -1.0, 0.0],
        "projection": "parallel",
        "parallel_scale": CAMERA_PARALLEL_SCALE_M,
        "scope": "expanded Orbicularis-oris region; display crop only after the full-head initial crinkle clip",
        "focus_source_bounds_m": all_oo_bounds,
    }
    x_min = camera["focus"][0] - camera["parallel_scale"]
    x_max = camera["focus"][0] + camera["parallel_scale"]
    z_min = camera["focus"][2] - camera["parallel_scale"]
    z_max = camera["focus"][2] + camera["parallel_scale"]
    if not (
        x_min <= all_oo_bounds[0] <= all_oo_bounds[1] <= x_max
        and z_min <= all_oo_bounds[4] <= all_oo_bounds[5] <= z_max
        and (all_bounds[0] < x_min or all_bounds[1] > x_max)
        and all_bounds[4] < z_min
    ):
        raise ValueError("camera no longer expresses the intended expanded mouth crop")

    material_series_identity = write_series(cfg.output_material_series, material_paths)
    skin_series_identity = write_series(cfg.output_skin_series, skin_paths)
    write_trajectory(cfg.output_trajectory, trajectory_rows)
    version = (
        subprocess.run(
            [str(PVBATCH), "--version"], check=True, capture_output=True, text=True
        )
        .stdout.strip()
        .split()[-1]
    )
    if version != EXPECTED_PARAVIEW_VERSION:
        raise ValueError(f"ParaView version changed: {version}")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "case": {
            "id": "HFP1",
            "source_case": "20-hfp1",
            "steps": list(STEPS),
            "inverse_converged": False,
            "stop_reason": "step_limit_smooth_decrease",
            "physics_rerun": False,
        },
        "plane": {
            "origin": [0.0, PLANE_Y, 0.0],
            "normal": [0.0, 1.0, 0.0],
            "selection_step": 0,
            "used_for_initial_selection_only": True,
            "reapplied_after_initial_frame": False,
            "anchor": {
                "definition": "Y coordinate of the initial full-Orbicularis maximum-Z vertex",
                "step": 0,
                "global_point_id": EXPECTED_ANCHOR_GLOBAL_POINT_ID,
                "point_m": list(ANCHOR_POINT_M),
            },
        },
        "topology": {
            "full_head_points": EXPECTED_POINTS,
            "full_head_tetrahedra": EXPECTED_TETS,
            "spatial_crop_before_initial_clip": False,
            "whole_volume_dominant_counts": whole_counts,
            "mixed_tetrahedra_over_0p001_in_multiple_fractions": mixed_tets,
            "fraction_fields": list(FRACTION_FIELDS),
            "fraction_sum_max_abs_error": 0.0,
        },
        "selection": {
            "method": "initial-frame-crinkle-clip",
            "selection_step": 0,
            "plane_normal": [0.0, 1.0, 0.0],
            "plane_y_m": PLANE_Y,
            "retained_half_space": "y <= plane_y",
            "predicate": "min(initial tetra vertex y) <= plane_y",
            "paraview_equivalent": {"Crinkleclip": 1, "Invert": 1},
            "selected_points": EXPECTED_CRINKLE_POINTS,
            "selected_tetrahedra": EXPECTED_CRINKLE_TETS,
            "selected_dominant_counts": EXPECTED_CRINKLE_COUNTS,
            "initial_strict_plane_straddling_tetrahedra": EXPECTED_INITIAL_STRICT_BOUNDARY_TETS,
            "initial_touching_or_straddling_tetrahedra": EXPECTED_INITIAL_INCLUSIVE_BOUNDARY_TETS,
            "source_cell_ids_sha256": EXPECTED_SELECTION_IDS_SHA256,
            "tetra_topology_sha256": EXPECTED_SELECTION_TOPOLOGY_SHA256,
            "cell_ids_fixed_across_frames": True,
            "coordinates_only_change_after_selection": True,
            "per_frame_reclip": False,
            "artifact": {
                "path": str(cfg.output_selection.resolve()),
                "identity": selection_identity,
            },
        },
        "materials": MATERIALS,
        "material_view": {
            "field": "DominantMaterial",
            "definition": "argmax(FatFraction, MuscleFraction, AponeurosisFraction)",
            "tie_break": "Fat, then Muscle, then Aponeurosis",
            "visualization_only": True,
            "physics_interpretation": "solver uses continuous fraction-weighted volume energies",
        },
        "skin": {
            "semantics": "step-0 plane intersection of the separate Koiter skin membrane, advected with frozen edge interpolation weights",
            "not_a_volume_material": True,
            "selection_step": 0,
            "per_frame_reclip": False,
            "fixed_points": EXPECTED_SKIN_TRACE_POINTS,
            "fixed_lines": EXPECTED_SKIN_TRACE_LINES,
            "topology_sha256": skin_topology_sha256,
            "rgb": SKIN_RGB,
            "line_width_px": SKIN_LINE_WIDTH_PX,
        },
        "crinkle_union_bounds_m": all_bounds,
        "camera": camera,
        "render": {
            "resolution": list(RESOLUTION),
            "fps": FPS,
            "frame_count": len(STEPS),
            "one_panel": True,
            "determinant_metrics_rendered": False,
            "material_representation": "Surface With Edges",
            "cell_edges_rendered": True,
            "external_tetra_faces_rendered": True,
            "cell_edge_rgb": CELL_EDGE_RGB,
            "cell_edge_width_px": CELL_EDGE_WIDTH_PX,
            "ambient": 0.55,
            "diffuse": 0.45,
            "opaque_selected_volume_surface": True,
            "full_head_before_initial_clip": True,
            "camera_crop_only": True,
            "complete_crinkle_clip_in_view": False,
            "no_interpolation_or_duplication": True,
            "no_deformation_exaggeration": True,
        },
        "inputs": {
            "material_series": {
                "path": str(cfg.output_material_series.resolve()),
                "identity": material_series_identity,
            },
            "skin_series": {
                "path": str(cfg.output_skin_series.resolve()),
                "identity": skin_series_identity,
            },
            "selection": {
                "path": str(cfg.output_selection.resolve()),
                "identity": selection_identity,
            },
        },
        "frames": frame_records,
        "outputs": {
            "frames_dir": str((OUTPUT_DIR / "frames").resolve()),
            "pvsm": str(cfg.output_pvsm.resolve()),
            "renderer_receipt": str(cfg.output_renderer_receipt.resolve()),
        },
    }
    write_json(cfg.output_contract, contract)
    completed = subprocess.run(
        [
            str(PVBATCH),
            str(cfg.paraview_script),
            "--contract",
            str(cfg.output_contract),
            "--receipt",
            str(cfg.output_renderer_receipt),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        logger.info("pvbatch stdout:\n%s", completed.stdout.rstrip())
    if completed.stderr:
        logger.info("pvbatch stderr:\n%s", completed.stderr.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(f"pvbatch failed with exit code {completed.returncode}")

    renderer_receipt = read_json(cfg.output_renderer_receipt)
    frame_paths = sorted((OUTPUT_DIR / "frames").glob("frame-*.png"))
    validate_pngs(frame_paths)
    if len({sha256(path) for path in frame_paths}) != len(STEPS):
        raise ValueError("rendered PNG frames are not all unique")
    validate_renderer_receipt(
        renderer_receipt,
        contract,
        frame_paths,
        cfg.output_pvsm,
    )
    video_probe = encode_video(OUTPUT_DIR / "frames", cfg.output_video)
    copy_poster(frame_paths[-1], cfg.output_poster)
    if not cfg.output_pvsm.is_file() or cfg.output_pvsm.stat().st_size <= 10_000:
        raise ValueError("ParaView state is missing or too small")
    write_json(
        cfg.output_receipt,
        {
            "schema_version": SCHEMA_VERSION,
            "design": DESIGN,
            "complete": True,
            "sources": {
                key: {"path": str(path.resolve()), "identity": source_identities[key]}
                for key, path in (
                    ("endpoint", cfg.source_endpoint),
                    ("history", cfg.source_history),
                    ("summary", cfg.source_summary),
                    ("skin", cfg.source_skin),
                )
            },
            "contract": {
                "path": str(cfg.output_contract.resolve()),
                "identity": identity(cfg.output_contract),
            },
            "renderer_receipt": {
                "path": str(cfg.output_renderer_receipt.resolve()),
                "identity": identity(cfg.output_renderer_receipt),
            },
            "trajectory": {
                "path": str(cfg.output_trajectory.resolve()),
                "identity": identity(cfg.output_trajectory),
            },
            "outputs": {
                "video": {
                    "path": str(cfg.output_video.resolve()),
                    "identity": identity(cfg.output_video),
                },
                "poster": {
                    "path": str(cfg.output_poster.resolve()),
                    "identity": identity(cfg.output_poster),
                },
                "pvsm": {
                    "path": str(cfg.output_pvsm.resolve()),
                    "identity": identity(cfg.output_pvsm),
                },
                "video_probe": video_probe,
            },
            "history": {
                "source_frame_count": len(STEPS),
                "png_frame_count": len(frame_paths),
                "video_frame_count": len(STEPS),
                "one_frame_per_recorded_step": True,
                "ordered_png_sha256": ordered_digest(frame_paths),
                "final_frame_matches_endpoint_exactly": True,
            },
        },
    )
    logger.info("Wrote exact HFP1 fixed crinkle-clip video to %s", OUTPUT_DIR)


if __name__ == "__main__":
    cherries.main(main)
