"""Build an exact 41-frame HFP1 max-Z-anchored mouth-section video.

This post-hoc pipeline reads only saved inverse states.  PyVista prepares
compact temporal geometry; ParaView 6.1.1 renders every pixel; FFmpeg encodes
one recorded state per frame at 30 FPS.
"""

from __future__ import annotations

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
import csv
import hashlib
import itertools
import json
import logging
import math
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

SCHEMA_VERSION = 1
DESIGN = "hfp1-orbicularis-oris-initial-max-z-coplanar-section-evolution"
STEPS = tuple(range(41))
FPS = 30
RESOLUTION = (1_200, 1_800)
MUSCLE_ID = 254
MUSCLE_NAME = "Orbicularis oris001_Head_muscles_0"
EXPECTED_POINTS = 228_660
EXPECTED_TETS = 1_146_517
EXPECTED_SELECTED_POINTS = 3_248
EXPECTED_SELECTED_TETS = 10_484
EXPECTED_CONTEXT_POINTS = 15_299
EXPECTED_CONTEXT_CELLS = 29_899
EXPECTED_LIP_CONTEXT_POINTS = 2_275
EXPECTED_LIP_CONTEXT_CELLS = 4_296
EXPECTED_ANCHOR_GLOBAL_POINT_ID = 52_222
PARAVIEW_VERSION = "6.1.1"
SCALARS = ("DetF", "DetAinv", "DetG")

GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
SOURCE_DIR = (
    REPO_ROOT
    / "exp/2026/08/19/human-face-smile-fat-floor-skin-energy-prestrain-inverse/data"
)
SOURCE_ENDPOINT = SOURCE_DIR / "20-hfp1.vtu"
SOURCE_HISTORY = SOURCE_DIR / "20-hfp1-steps.vtkhdf"
SOURCE_SUMMARY = SOURCE_DIR / "20-hfp1-summary-final.json"
SOURCE_CONTEXT = SOURCE_DIR / "26-paraview-fat-floor-terminal/inputs/hfp1.vtp"
PARAVIEW_SCRIPT = Path(__file__).with_name(
    "50-hfp1-orbicularis-oris-topdown-evolution-paraview.py"
)
PVBATCH = Path("/usr/bin/pvbatch")
OUTPUT_DIR = GROUP_DIR / "data/50-hfp1-orbicularis-oris-topdown-evolution"

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
    "context": {
        "size_bytes": 2_634_336,
        "sha256": "a595a874ca4ce42c2884a8f6c7705e857cfbe51531a44537ea0c49020972249f",
    },
}


def output(name: str) -> str:
    return f"50-hfp1-orbicularis-oris-topdown-evolution/{name}"


class Config(cherries.BaseConfig):
    """Pinned inputs and outputs for the exact saved-state render."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    source_endpoint: Path = cherries.input(SOURCE_ENDPOINT)
    source_history: Path = cherries.input(SOURCE_HISTORY)
    source_summary: Path = cherries.input(SOURCE_SUMMARY)
    source_context: Path = cherries.input(SOURCE_CONTEXT)
    paraview_script: Path = cherries.input(PARAVIEW_SCRIPT)
    output_reference: Path = cherries.output(output("inputs/reference.vtu"), mkdir=True)
    output_muscle_series: Path = cherries.output(
        output("muscle-history.vtu.series"), mkdir=True
    )
    output_skin_series: Path = cherries.output(
        output("skin-section-history.vtp.series"), mkdir=True
    )
    output_trajectory: Path = cherries.output(output("trajectory.csv"), mkdir=True)
    output_contract: Path = cherries.output(output("contract.json"), mkdir=True)
    output_renderer_receipt: Path = cherries.output(
        output("renderer-receipt.json"), mkdir=True
    )
    output_video: Path = cherries.output(
        output("50-hfp1-orbicularis-oris-topdown-evolution.mp4"), mkdir=True
    )
    output_poster: Path = cherries.output(
        output("50-hfp1-orbicularis-oris-topdown-evolution-poster.png"), mkdir=True
    )
    output_pvsm: Path = cherries.output(
        output("50-hfp1-orbicularis-oris-topdown-evolution.pvsm"), mkdir=True
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
    if temporary.exists() or path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_dataset(dataset: pv.DataSet, path: Path) -> dict[str, int | str]:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if temporary.exists() or path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    dataset.save(temporary, binary=True)
    temporary.replace(path)
    return identity(path)


def expected_paths() -> dict[str, Path]:
    return {
        "source_endpoint": SOURCE_ENDPOINT,
        "source_history": SOURCE_HISTORY,
        "source_summary": SOURCE_SUMMARY,
        "source_context": SOURCE_CONTEXT,
        "paraview_script": PARAVIEW_SCRIPT,
        "output_reference": OUTPUT_DIR / "inputs/reference.vtu",
        "output_muscle_series": OUTPUT_DIR / "muscle-history.vtu.series",
        "output_skin_series": OUTPUT_DIR / "skin-section-history.vtp.series",
        "output_trajectory": OUTPUT_DIR / "trajectory.csv",
        "output_contract": OUTPUT_DIR / "contract.json",
        "output_renderer_receipt": OUTPUT_DIR / "renderer-receipt.json",
        "output_video": OUTPUT_DIR / "50-hfp1-orbicularis-oris-topdown-evolution.mp4",
        "output_poster": OUTPUT_DIR
        / "50-hfp1-orbicularis-oris-topdown-evolution-poster.png",
        "output_pvsm": OUTPUT_DIR / "50-hfp1-orbicularis-oris-topdown-evolution.pvsm",
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


def tetrahedra(mesh: pv.UnstructuredGrid) -> np.ndarray:
    if mesh.n_cells != EXPECTED_TETS or not np.all(mesh.celltypes == pv.CellType.TETRA):
        raise ValueError("HFP1 endpoint is not the expected tetra-only mesh")
    packed = np.asarray(mesh.cells)
    if packed.size != 5 * mesh.n_cells:
        raise ValueError("HFP1 connectivity size changed")
    packed = packed.reshape(-1, 5)
    if not np.all(packed[:, 0] == 4):
        raise ValueError("HFP1 tetra connectivity is malformed")
    return packed[:, 1:]


def activation_determinant(values: np.ndarray) -> np.ndarray:
    if values.shape != (EXPECTED_SELECTED_TETS, 6):
        raise ValueError(f"ActivationInv shape changed: {values.shape}")
    matrix = np.zeros((values.shape[0], 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = 1.0 + values[:, 0]
    matrix[:, 1, 1] = 1.0 + values[:, 1]
    matrix[:, 2, 2] = 1.0 + values[:, 2]
    matrix[:, 0, 1] = matrix[:, 1, 0] = values[:, 3]
    matrix[:, 1, 2] = matrix[:, 2, 1] = values[:, 4]
    matrix[:, 0, 2] = matrix[:, 2, 0] = values[:, 5]
    return np.linalg.det(matrix)


def compact_grid(
    points: np.ndarray,
    local_cells: np.ndarray,
    selected: np.ndarray,
    fields: dict[str, np.ndarray],
) -> pv.UnstructuredGrid:
    packed = np.column_stack((np.full(selected.size, 4), local_cells)).astype(np.int64)
    result = pv.UnstructuredGrid(
        packed.ravel(),
        np.full(selected.size, pv.CellType.TETRA, dtype=np.uint8),
        points,
    )
    result.cell_data["SourceCellId"] = selected.astype(np.int64)
    for name, values in fields.items():
        if values.shape[0] != selected.size:
            raise ValueError(f"{name} length changed")
        result.cell_data[name] = values
    if (
        result.n_points != EXPECTED_SELECTED_POINTS
        or result.n_cells != EXPECTED_SELECTED_TETS
    ):
        raise ValueError("compact full-Orbicularis topology changed")
    return result


def determinant_metrics(
    rest_volume: np.ndarray, det_f: np.ndarray, det_ainv: np.ndarray
) -> dict[str, Any]:
    det_g = det_f * det_ainv
    double = (det_f < 0.0) & (det_ainv < 0.0)
    total = float(rest_volume.sum())
    result: dict[str, Any] = {
        "cells": int(det_f.size),
        "double_inverted_cells": int(double.sum()),
        "double_inverted_rest_volume_fraction": float(
            rest_volume[double].sum() / total
        ),
    }
    for name, values in zip(SCALARS, (det_f, det_ainv, det_g), strict=True):
        negative = values < 0.0
        result[name] = {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "negative_cells": int(negative.sum()),
            "negative_rest_volume_fraction": float(rest_volume[negative].sum() / total),
        }
    return result


def array_bounds(points: np.ndarray) -> list[float]:
    low = points.min(axis=0)
    high = points.max(axis=0)
    return [
        float(low[0]),
        float(high[0]),
        float(low[1]),
        float(high[1]),
        float(low[2]),
        float(high[2]),
    ]


def union_bounds(items: list[list[float]]) -> list[float]:
    return [
        min(item[0] for item in items),
        max(item[1] for item in items),
        min(item[2] for item in items),
        max(item[3] for item in items),
        min(item[4] for item in items),
        max(item[5] for item in items),
    ]


def superior_camera(bounds: list[float]) -> dict[str, Any]:
    focus = [
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    ]
    x_span, z_span = bounds[1] - bounds[0], bounds[5] - bounds[4]
    panel_aspect = RESOLUTION[0] / (RESOLUTION[1] / len(SCALARS))
    scale = 0.62 * max(z_span, x_span / panel_aspect)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("invalid superior camera scale")
    return {
        "focus": focus,
        "position": [focus[0], focus[1] + 0.30, focus[2]],
        "view_up": [0.0, 0.0, 1.0],
        "parallel_scale": float(scale),
        "projection": "parallel",
        "look_direction": [0.0, -1.0, 0.0],
        "orientation": "+Y camera looking toward -Y; +Z is anterior/up in image",
    }


def camera_view_xz_bounds(camera: dict[str, Any]) -> tuple[float, float, float, float]:
    focus = np.asarray(camera["focus"], dtype=float)
    scale = float(camera["parallel_scale"])
    panel_aspect = RESOLUTION[0] / (RESOLUTION[1] / len(SCALARS))
    return (
        float(focus[0] - scale * panel_aspect),
        float(focus[0] + scale * panel_aspect),
        float(focus[2] - scale),
        float(focus[2] + scale),
    )


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
    used: set[int] = set()
    cursor = 0
    while cursor < packed.size:
        count = int(packed[cursor])
        points = packed[cursor + 1 : cursor + 1 + count]
        if count < 2 or points.size != count:
            raise ValueError("skin section contains a malformed line")
        used.update(int(point) for point in points)
        for left, right in itertools.pairwise(points):
            union(int(left), int(right))
        cursor += count + 1
    if cursor != packed.size or used != set(range(section.n_points)):
        raise ValueError("skin section line connectivity changed")
    return len({find(point) for point in used})


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


def write_trajectory(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or path.exists():
        raise FileExistsError(path)
    fieldnames = [
        "step",
        "det_f_min",
        "det_f_negative_cells",
        "det_ainv_min",
        "det_ainv_negative_cells",
        "det_g_min",
        "det_g_negative_cells",
        "double_inverted_cells",
        "skin_section_points",
        "skin_section_cells",
        "skin_section_components",
        "skin_section_camera_view_points",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_pngs(paths: list[Path]) -> None:
    if len(paths) != len(STEPS):
        raise ValueError("PNG frame count changed")
    for step, path in zip(STEPS, paths, strict=True):
        expected = f"frame-{step:03d}.png"
        if path.name != expected or not path.is_file() or path.stat().st_size <= 50_000:
            raise ValueError(f"invalid rendered frame: {path}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.size != RESOLUTION or image.mode != "RGB":
                raise ValueError(f"rendered frame properties changed: {path}")


def encode_video(frames_dir: Path, video: Path) -> dict[str, Any]:
    temporary = video.with_name(f".{video.stem}.tmp{video.suffix}")
    if temporary.exists() or video.exists():
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
    temporary.replace(video)
    return facts


def copy_poster(source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    if temporary.exists() or target.exists():
        raise FileExistsError(target)
    os.link(source, temporary)
    temporary.replace(target)
    if identity(source) != identity(target):
        raise ValueError("poster does not exactly match final frame")


def main(cfg: Config) -> None:
    validate_config(cfg)
    source_identities = {
        key: validate_source(path, key)
        for key, path in (
            ("endpoint", cfg.source_endpoint),
            ("history", cfg.source_history),
            ("summary", cfg.source_summary),
            ("context", cfg.source_context),
        )
    }
    summary = read_json(cfg.source_summary)
    expected_summary = {
        "case": "20-hfp1",
        "best_step": 40,
        "best/step": 40,
        "inverse/evaluations": 41,
        "inverse/max_steps": 40,
        "history_frames": 41,
        "inverse/converged": False,
        "inverse/forward_fail_count": 0,
        "inverse/adjoint_fail_count": 0,
        "inverse/stop_reason": "step_limit_smooth_decrease",
    }
    actual_summary = {key: summary.get(key) for key in expected_summary}
    if actual_summary != expected_summary:
        raise ValueError(f"HFP1 source summary changed: {actual_summary}")

    endpoint = pv.read(cfg.source_endpoint)
    if not isinstance(endpoint, pv.UnstructuredGrid):
        raise TypeError(f"endpoint is {type(endpoint).__name__}")
    if endpoint.n_points != EXPECTED_POINTS or endpoint.n_cells != EXPECTED_TETS:
        raise ValueError("HFP1 endpoint topology changed")
    names = np.asarray(endpoint.field_data["MuscleName"])
    if names[MUSCLE_ID] != MUSCLE_NAME:
        raise ValueError(f"MuscleId 254 changed: {names[MUSCLE_ID]!r}")
    cells = tetrahedra(endpoint)
    reference = np.asarray(endpoint.points, dtype=np.float64)
    active = np.asarray(endpoint.cell_data["ActivationMask"], dtype=bool)
    muscle_id = np.asarray(endpoint.cell_data["MuscleId"], dtype=np.int64)
    fraction = np.asarray(endpoint.cell_data["MuscleFraction"], dtype=np.float64)
    selected = np.flatnonzero(active & (muscle_id == MUSCLE_ID) & (fraction >= 0.5))
    if selected.size != EXPECTED_SELECTED_TETS:
        raise ValueError(f"full Orbicularis selection changed: {selected.size}")
    used = np.unique(cells[selected].ravel())
    if used.size != EXPECTED_SELECTED_POINTS:
        raise ValueError(f"selected point count changed: {used.size}")
    selected_cells = cells[selected]
    local_cells = np.searchsorted(used, selected_cells)
    reference_points = reference[used]
    reference_edges = np.stack(
        [
            reference[selected_cells[:, index]] - reference[selected_cells[:, 0]]
            for index in (1, 2, 3)
        ],
        axis=2,
    )
    reference_det = np.linalg.det(reference_edges)
    if np.any(~np.isfinite(reference_det)) or np.any(reference_det == 0.0):
        raise ValueError("selected reference mesh contains a degenerate tetrahedron")
    rest_volume = np.abs(reference_det) / 6.0
    reference_grid = compact_grid(
        reference_points,
        local_cells,
        selected,
        {
            "RestVolume": rest_volume,
            "MuscleFraction": fraction[selected],
        },
    )
    reference_identity = save_dataset(reference_grid, cfg.output_reference)

    context = pv.read(cfg.source_context)
    if not isinstance(context, pv.PolyData):
        raise TypeError(f"context is {type(context).__name__}")
    if (
        context.n_points != EXPECTED_CONTEXT_POINTS
        or context.n_cells != EXPECTED_CONTEXT_CELLS
    ):
        raise ValueError("HFP1 context topology changed")
    global_point_ids = np.asarray(context.point_data["GlobalPointId"], dtype=np.int64)
    if (
        global_point_ids.shape != (EXPECTED_CONTEXT_POINTS,)
        or np.unique(global_point_ids).size != EXPECTED_CONTEXT_POINTS
        or global_point_ids.min() < 0
        or global_point_ids.max() >= EXPECTED_POINTS
    ):
        raise ValueError("context GlobalPointId mapping changed")
    endpoint_deformed = np.asarray(
        endpoint.point_data["DeformedPoint"], dtype=np.float64
    )
    if not np.array_equal(
        np.asarray(context.points), endpoint_deformed[global_point_ids]
    ):
        raise ValueError("context topology is not pinned to the HFP1 endpoint")
    is_lip = np.asarray(endpoint.point_data["IsLip"], dtype=bool)
    if is_lip.shape != (EXPECTED_POINTS,) or int(is_lip.sum()) != 3_408:
        raise ValueError("endpoint IsLip mask changed")
    context_faces = np.asarray(context.faces, dtype=np.int64)
    if context_faces.size != 4 * EXPECTED_CONTEXT_CELLS:
        raise ValueError("context face connectivity changed")
    context_faces = context_faces.reshape(-1, 4)
    if not np.all(context_faces[:, 0] == 3):
        raise ValueError("context is not triangle-only")
    context_faces = context_faces[:, 1:]
    context_is_lip = is_lip[global_point_ids]
    lip_faces = context_faces[np.all(context_is_lip[context_faces], axis=1)]
    lip_context_points = np.unique(lip_faces.ravel())
    lip_local_faces = np.searchsorted(lip_context_points, lip_faces)
    if lip_context_points.size != EXPECTED_LIP_CONTEXT_POINTS or lip_faces.shape != (
        EXPECTED_LIP_CONTEXT_CELLS,
        3,
    ):
        raise ValueError("all-IsLip external-surface topology changed")
    lip_context = pv.PolyData(
        np.asarray(context.points)[lip_context_points],
        np.column_stack(
            (np.full(EXPECTED_LIP_CONTEXT_CELLS, 3), lip_local_faces)
        ).ravel(),
    )
    lip_context.clear_data()
    lip_global_point_ids = global_point_ids[lip_context_points]
    lip_sort_order = np.argsort(lip_global_point_ids)
    sorted_lip_global_ids = lip_global_point_ids[lip_sort_order]
    inputs_dir = cfg.output_reference.parent
    inputs_dir.mkdir(parents=True, exist_ok=True)
    muscle_paths: list[Path] = []
    skin_paths: list[Path] = []
    muscle_records: list[dict[str, Any]] = []
    skin_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    extrema = {name: [0.0, 0.0] for name in SCALARS}
    final_points: np.ndarray | None = None
    final_activation: np.ndarray | None = None

    with h5py.File(cfg.source_history, "r") as hdf:
        root = hdf["VTKHDF"]
        time_values = np.asarray(root["Steps/Values"], dtype=np.float64)
        inverse_steps = np.asarray(root["FieldData/inverse_step"], dtype=np.int64)
        if not np.array_equal(time_values, np.asarray(STEPS, dtype=float)):
            raise ValueError(f"VTKHDF time values changed: {time_values}")
        if not np.array_equal(inverse_steps, np.asarray(STEPS, dtype=np.int64)):
            raise ValueError(f"VTKHDF inverse_step changed: {inverse_steps}")
        point_offsets = np.asarray(
            root["Steps/PointDataOffsets/DeformedPoint"], dtype=np.int64
        )
        cell_offsets = np.asarray(
            root["Steps/CellDataOffsets/ActivationInv"], dtype=np.int64
        )
        if point_offsets.shape != (len(STEPS),) or cell_offsets.shape != (len(STEPS),):
            raise ValueError("VTKHDF dynamic offsets changed")
        point_payload = root["PointData/DeformedPoint"]
        activation_payload = root["CellData/ActivationInv"]
        state_bounds = [
            array_bounds(
                np.asarray(point_payload[int(offset) + used], dtype=np.float64)
            )
            for offset in point_offsets
        ]
        bounds = union_bounds([array_bounds(reference_points), *state_bounds])
        camera = superior_camera(bounds)
        initial_points = np.asarray(
            point_payload[int(point_offsets[0]) + used], dtype=np.float64
        )
        anchor_local_index = int(np.argmax(initial_points[:, 2]))
        anchor_point = initial_points[anchor_local_index]
        section_y = float(anchor_point[1])
        if (
            not math.isclose(
                float(anchor_point[2]), state_bounds[0][5], rel_tol=0.0, abs_tol=1e-12
            )
            or int(used[anchor_local_index]) != EXPECTED_ANCHOR_GLOBAL_POINT_ID
        ):
            raise ValueError("initial maximum-Z Orbicularis anchor changed")
        muscle_plane_offsets = [item[3] - section_y for item in state_bounds]
        camera_x_min, camera_x_max, camera_z_min, camera_z_max = camera_view_xz_bounds(
            camera
        )
        for step in STEPS:
            point_offset = int(point_offsets[step])
            cell_offset = int(cell_offsets[step])
            deformed_points = np.asarray(
                point_payload[point_offset + used], dtype=np.float64
            )
            activation = np.asarray(
                activation_payload[cell_offset + selected], dtype=np.float64
            )
            deformed_edges = np.stack(
                [
                    deformed_points[local_cells[:, index]]
                    - deformed_points[local_cells[:, 0]]
                    for index in (1, 2, 3)
                ],
                axis=2,
            )
            det_f = np.linalg.det(deformed_edges) / reference_det
            det_ainv = activation_determinant(activation)
            det_g = det_f * det_ainv
            if any(np.any(~np.isfinite(values)) for values in (det_f, det_ainv, det_g)):
                raise ValueError(f"non-finite determinant at step {step}")
            double = ((det_f < 0.0) & (det_ainv < 0.0)).astype(np.int8)
            frame = compact_grid(
                deformed_points,
                local_cells,
                selected,
                {
                    "RestVolume": rest_volume,
                    "MuscleFraction": fraction[selected],
                    "DetF": det_f,
                    "DetAinv": det_ainv,
                    "DetG": det_g,
                    "DoubleInverted": double,
                },
            )
            frame_path = inputs_dir / f"muscle-step-{step:03d}.vtu"
            frame_identity = save_dataset(frame, frame_path)
            metrics = determinant_metrics(rest_volume, det_f, det_ainv)
            muscle_paths.append(frame_path)
            muscle_records.append(
                {
                    "step": step,
                    "path": str(frame_path.resolve()),
                    "identity": frame_identity,
                    "metrics": metrics,
                }
            )
            if array_bounds(deformed_points) != state_bounds[step]:
                raise ValueError(f"step {step} pre-scanned muscle bounds changed")
            for name, values in zip(SCALARS, (det_f, det_ainv, det_g), strict=True):
                extrema[name][0] = min(extrema[name][0], float(values.min()))
                extrema[name][1] = max(extrema[name][1], float(values.max()))

            sorted_skin_points = np.asarray(
                point_payload[point_offset + sorted_lip_global_ids], dtype=np.float64
            )
            skin_points = np.empty_like(sorted_skin_points)
            skin_points[lip_sort_order] = sorted_skin_points
            dynamic_skin = lip_context.copy(deep=True)
            dynamic_skin.points = skin_points
            dynamic_skin.clear_data()
            section = dynamic_skin.slice(
                normal=(0.0, 1.0, 0.0), origin=(0.0, section_y, 0.0)
            )
            section_points = np.asarray(section.points)
            if (
                not isinstance(section, pv.PolyData)
                or section.n_cells < 1
                or section.n_lines != section.n_cells
                or np.max(np.abs(section_points[:, 1] - section_y)) > 1e-10
            ):
                raise ValueError(f"invalid skin section at step {step}")
            components = line_component_count(section)
            camera_view_points = int(
                (
                    (section_points[:, 0] >= camera_x_min)
                    & (section_points[:, 0] <= camera_x_max)
                    & (section_points[:, 2] >= camera_z_min)
                    & (section_points[:, 2] <= camera_z_max)
                ).sum()
            )
            if components != 1 or camera_view_points != section.n_points:
                raise ValueError(
                    f"unstable or off-camera skin section at step {step}: "
                    f"components={components}, camera_view_points={camera_view_points}"
                )
            section.clear_data()
            section.cell_data["SkinSection"] = np.ones(section.n_cells, dtype=np.int8)
            skin_path = inputs_dir / f"skin-section-step-{step:03d}.vtp"
            skin_identity = save_dataset(section, skin_path)
            skin_paths.append(skin_path)
            skin_records.append(
                {
                    "step": step,
                    "path": str(skin_path.resolve()),
                    "identity": skin_identity,
                    "points": section.n_points,
                    "cells": section.n_cells,
                    "components": components,
                    "camera_view_points": camera_view_points,
                }
            )
            rows.append(
                {
                    "step": step,
                    "det_f_min": metrics["DetF"]["minimum"],
                    "det_f_negative_cells": metrics["DetF"]["negative_cells"],
                    "det_ainv_min": metrics["DetAinv"]["minimum"],
                    "det_ainv_negative_cells": metrics["DetAinv"]["negative_cells"],
                    "det_g_min": metrics["DetG"]["minimum"],
                    "det_g_negative_cells": metrics["DetG"]["negative_cells"],
                    "double_inverted_cells": metrics["double_inverted_cells"],
                    "skin_section_points": section.n_points,
                    "skin_section_cells": section.n_cells,
                    "skin_section_components": components,
                    "skin_section_camera_view_points": camera_view_points,
                }
            )
            cherries.set_step(step)
            cherries.log_metrics(
                {
                    "orbicularis/det_f_negative_cells": metrics["DetF"][
                        "negative_cells"
                    ],
                    "orbicularis/det_ainv_negative_cells": metrics["DetAinv"][
                        "negative_cells"
                    ],
                    "orbicularis/det_g_negative_cells": metrics["DetG"][
                        "negative_cells"
                    ],
                    "orbicularis/double_inverted_cells": metrics[
                        "double_inverted_cells"
                    ],
                    "skin_section/cells": section.n_cells,
                    "skin_section/camera_view_points": camera_view_points,
                }
            )
            if step == STEPS[-1]:
                final_points = deformed_points.copy()
                final_activation = activation.copy()

    if final_points is None or final_activation is None:
        raise RuntimeError("history loop did not produce a final state")
    if not np.array_equal(final_points, endpoint_deformed[used]):
        raise ValueError("history step 40 does not reproduce endpoint geometry")
    endpoint_activation = np.asarray(
        endpoint.cell_data["ActivationInv"], dtype=np.float64
    )[selected]
    if not np.array_equal(final_activation, endpoint_activation):
        raise ValueError("history step 40 does not reproduce endpoint activation")

    muscle_series_identity = write_series(cfg.output_muscle_series, muscle_paths)
    skin_series_identity = write_series(cfg.output_skin_series, skin_paths)
    write_trajectory(cfg.output_trajectory, rows)
    scalar_ranges = dict(extrema)
    if any(not (limits[0] < 0.0 < limits[1]) for limits in scalar_ranges.values()):
        raise ValueError(f"global scalar ranges do not bracket zero: {scalar_ranges}")
    version_result = subprocess.run(
        [str(PVBATCH), "--version"], check=True, capture_output=True, text=True
    )
    if version_result.stdout.strip().split()[-1] != PARAVIEW_VERSION:
        raise ValueError(f"ParaView version changed: {version_result.stdout}")

    contract = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "case": {
            "id": "HFP1",
            "source_case": "20-hfp1",
            "steps": list(STEPS),
            "evaluations": len(STEPS),
            "inverse_converged": False,
            "stop_reason": "step_limit_smooth_decrease",
            "physics_rerun": False,
            "sources": {
                key: {"path": str(path.resolve()), "identity": source_identities[key]}
                for key, path in (
                    ("endpoint", cfg.source_endpoint),
                    ("history", cfg.source_history),
                    ("summary", cfg.source_summary),
                    ("context", cfg.source_context),
                )
            },
        },
        "selection": {
            "muscle_id": MUSCLE_ID,
            "muscle_name": MUSCLE_NAME,
            "predicate": "ActivationMask && MuscleId == 254 && MuscleFraction >= 0.5",
            "spatial_crop": False,
            "cells": EXPECTED_SELECTED_TETS,
            "points": EXPECTED_SELECTED_POINTS,
        },
        "inputs": {
            "reference": {
                "path": str(cfg.output_reference.resolve()),
                "identity": reference_identity,
                "points": EXPECTED_SELECTED_POINTS,
                "cells": EXPECTED_SELECTED_TETS,
            },
            "series": {
                "path": str(cfg.output_muscle_series.resolve()),
                "identity": muscle_series_identity,
                "frames": len(STEPS),
                "steps": list(STEPS),
            },
            "skin_section_series": {
                "path": str(cfg.output_skin_series.resolve()),
                "identity": skin_series_identity,
                "frames": len(STEPS),
                "steps": list(STEPS),
            },
        },
        "frames": muscle_records,
        "skin_section": {
            "plane_y": section_y,
            "anchor": {
                "step": 0,
                "definition": "maximum Z point of the full selected Orbicularis at the initial saved frame; its Y coordinate fixes the section plane",
                "global_point_id": int(used[anchor_local_index]),
                "point_m": [float(value) for value in anchor_point],
            },
            "surface_selection": {
                "predicate": "all three external-surface triangle vertices have IsLip == true",
                "points": EXPECTED_LIP_CONTEXT_POINTS,
                "triangles": EXPECTED_LIP_CONTEXT_CELLS,
                "nasal_geometry_included": False,
            },
            "frames": skin_records,
        },
        "geometry": {
            "reference_bounds_m": array_bounds(reference_points),
            "all_muscle_state_union_bounds_m": bounds,
            "deformation_exaggeration": 1.0,
        },
        "scalar_ranges": scalar_ranges,
        "camera": camera,
        "render": {
            "resolution": list(RESOLUTION),
            "fps": FPS,
            "frame_count": len(STEPS),
            "no_interpolation_or_duplication": True,
            "no_deformation_exaggeration": True,
            "geometry_mode": "muscle-and-lip-skin-coplanar-initial-max-z-y-section",
            "camera_scope": "mouth-from-full-orbicularis-bounds",
            "skin_line_width_px": 2.0,
        },
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
            str(cfg.paraview_script.resolve()),
            "--contract",
            str(cfg.output_contract.resolve()),
            "--receipt",
            str(cfg.output_renderer_receipt.resolve()),
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
    if (
        renderer_receipt.get("complete") is not True
        or renderer_receipt.get("paraview_version") != PARAVIEW_VERSION
        or renderer_receipt.get("frame_count") != len(STEPS)
    ):
        raise ValueError("renderer receipt is incomplete")
    png_paths = [OUTPUT_DIR / "frames" / f"frame-{step:03d}.png" for step in STEPS]
    validate_pngs(png_paths)
    video_probe = encode_video(OUTPUT_DIR / "frames", cfg.output_video)
    copy_poster(png_paths[-1], cfg.output_poster)
    if not cfg.output_pvsm.is_file() or cfg.output_pvsm.stat().st_size <= 10_000:
        raise ValueError("temporal ParaView state is missing or too small")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "purpose": "post-hoc saved-state visualization; no physics rerun",
        "case": contract["case"],
        "selection": contract["selection"],
        "history": {
            "first_step": STEPS[0],
            "last_step": STEPS[-1],
            "source_frame_count": len(STEPS),
            "png_frame_count": len(png_paths),
            "video_frame_count": len(STEPS),
            "exact_consecutive": True,
            "one_frame_per_recorded_step": True,
            "no_interpolation_or_duplication": True,
            "final_frame_matches_endpoint_exactly": True,
            "ordered_muscle_vtu_sha256": ordered_digest(muscle_paths),
            "ordered_skin_vtp_sha256": ordered_digest(skin_paths),
            "ordered_png_sha256": ordered_digest(png_paths),
        },
        "skin_section": {
            "definition": "semantic IsLip external surface intersected each step by the fixed Y-plane through the initial full-Orbicularis maximum-Z point",
            "plane_y_m": section_y,
            "normal": [0.0, 1.0, 0.0],
            "anchor": contract["skin_section"]["anchor"],
            "surface_selection": contract["skin_section"]["surface_selection"],
            "dynamic_deformed_surface_each_step": True,
            "muscle_display_cut_at_same_plane": True,
            "full_muscle_selection_metrics_retained": True,
            "all_state_muscle_plane_offset_range_m": [
                min(muscle_plane_offsets),
                max(muscle_plane_offsets),
            ],
            "camera_side_of_all_muscle_states": False,
            "all_lip_arcs_retained": True,
            "nasal_geometry_excluded_by_semantic_mask": True,
            "plane_cut_is_not_surface_silhouette": True,
            "render_color": "teal",
            "frame_point_count_range": [
                min(record["points"] for record in skin_records),
                max(record["points"] for record in skin_records),
            ],
            "frame_cell_count_range": [
                min(record["cells"] for record in skin_records),
                max(record["cells"] for record in skin_records),
            ],
            "frame_component_count_range": [
                min(record["components"] for record in skin_records),
                max(record["components"] for record in skin_records),
            ],
            "frame_camera_view_point_count_range": [
                min(record["camera_view_points"] for record in skin_records),
                max(record["camera_view_points"] for record in skin_records),
            ],
        },
        "visualization": {
            "geometry_mode": "muscle-and-lip-skin-coplanar-initial-max-z-y-section",
            "camera_scope": "mouth-from-full-orbicularis-bounds",
            "full_orbicularis_metrics_in_labels": True,
            "skin_line_uses_true_fixed_plane_without_display_offset": True,
            "skin_line_width_px": 2.0,
        },
        "camera": camera,
        "scalar_ranges": scalar_ranges,
        "trajectory": {
            "path": str(cfg.output_trajectory.resolve()),
            "identity": identity(cfg.output_trajectory),
            "first": rows[0],
            "last": rows[-1],
        },
        "contract": {
            "path": str(cfg.output_contract.resolve()),
            "identity": identity(cfg.output_contract),
        },
        "renderer_receipt": {
            "path": str(cfg.output_renderer_receipt.resolve()),
            "identity": identity(cfg.output_renderer_receipt),
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
    }
    write_json(cfg.output_receipt, receipt)
    logger.info("Wrote exact HFP1 full-Orbicularis evolution to %s", OUTPUT_DIR)


if __name__ == "__main__":
    cherries.main(main)
