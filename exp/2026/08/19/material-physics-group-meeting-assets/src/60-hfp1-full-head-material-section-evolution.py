"""Render the complete HFP1 head section through the fixed mouth plane.

This post-hoc pipeline reads the exact 41 saved inverse states, intersects the
entire tetrahedral head and the separate skin membrane with one fixed plane,
and delegates the one-panel categorical material render to ParaView.
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

SCHEMA_VERSION = 1
DESIGN = "hfp1-full-head-dominant-material-fixed-y-section-evolution"
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
EXPECTED_STEP_COUNTS = {
    0: {"Fat": 13_778, "Muscle": 2_703, "Aponeurosis": 251},
    40: {"Fat": 14_617, "Muscle": 3_673, "Aponeurosis": 150},
}
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
CELL_EDGE_RGB = [0.12, 0.13, 0.15]
CELL_EDGE_WIDTH_PX = 0.35
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
    """Pinned sources and outputs for the full-head section render."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    source_endpoint: Path = cherries.input(SOURCE_ENDPOINT)
    source_history: Path = cherries.input(SOURCE_HISTORY)
    source_summary: Path = cherries.input(SOURCE_SUMMARY)
    source_skin: Path = cherries.input(SOURCE_SKIN)
    paraview_script: Path = cherries.input(PARAVIEW_SCRIPT)
    output_material_series: Path = cherries.output(
        output("material-section-history.vtp.series"), mkdir=True
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
        "output_material_series": OUTPUT_DIR / "material-section-history.vtp.series",
        "output_skin_series": OUTPUT_DIR / "skin-section-history.vtp.series",
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


def section_record(section: pv.PolyData) -> dict[str, Any]:
    points = np.asarray(section.points, dtype=np.float64)
    if (
        section.n_points < 1
        or section.n_cells < 1
        or np.max(np.abs(points[:, 1] - PLANE_Y)) > 1e-10
    ):
        raise ValueError("material section is empty or left the fixed plane")
    dominant = np.asarray(section.cell_data["DominantMaterial"], dtype=np.int32)
    if dominant.shape != (section.n_cells,) or np.any((dominant < 0) | (dominant > 2)):
        raise ValueError("material section category changed")
    source_ids = np.asarray(section.cell_data["SourceCellId"], dtype=np.int64)
    if (
        source_ids.shape != (section.n_cells,)
        or np.unique(source_ids).size != section.n_cells
    ):
        raise ValueError("material section no longer has one polygon per source tet")
    sized = section.compute_cell_sizes(length=False, area=True, volume=False)
    areas = np.asarray(sized.cell_data["Area"], dtype=np.float64)
    if (
        areas.shape != (section.n_cells,)
        or np.any(~np.isfinite(areas))
        or np.any(areas <= 0)
    ):
        raise ValueError("material section has invalid polygon areas")
    counts: dict[str, int] = {}
    areas_by_material: dict[str, float] = {}
    for index in range(3):
        name = str(MATERIALS[str(index)]["name"])
        mask = dominant == index
        counts[name] = int(mask.sum())
        areas_by_material[name] = float(areas[mask].sum())
    if any(value == 0 for value in counts.values()):
        raise ValueError(f"full section misses a volume material: {counts}")
    return {
        "points": int(section.n_points),
        "cells": int(section.n_cells),
        "dominant_counts": counts,
        "dominant_areas_m2": areas_by_material,
        "bounds_m": array_bounds(points),
        "source_tets": int(np.unique(source_ids).size),
    }


def write_trajectory(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    fieldnames = [
        "step",
        "section_points",
        "section_cells",
        "fat_cells",
        "muscle_cells",
        "aponeurosis_cells",
        "fat_area_m2",
        "muscle_area_m2",
        "aponeurosis_area_m2",
        "skin_points",
        "skin_cells",
        "skin_components",
        "x_min_m",
        "x_max_m",
        "z_min_m",
        "z_max_m",
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
        "material_representation",
        "filled_material_surface",
        "cell_edges_rendered",
        "internal_tet_section_edges_rendered",
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
        or receipt["material_representation"] != "Surface With Edges"
        or receipt["filled_material_surface"] is not True
        or receipt["cell_edges_rendered"] is not True
        or receipt["internal_tet_section_edges_rendered"] is not True
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
        material = record.get("material_section", {})
        skin = record.get("skin_section", {})
        if (
            record.get("step") != step
            or Path(str(record.get("path"))).resolve() != frame_path.resolve()
            or record.get("identity") != identity(frame_path)
            or material.get("points") != source["material_section"]["points"]
            or material.get("cells") != source["material_section"]["cells"]
            or material.get("dominant_counts")
            != source["material_section"]["dominant_counts"]
            or material.get("argmax_of_continuous_fractions") is not True
            or float(material.get("fraction_sum_max_abs_error", float("inf"))) > 1e-12
            or skin.get("points") != source["skin_section"]["points"]
            or skin.get("cells") != source["skin_section"]["cells"]
            or skin.get("components") != source["skin_section"]["components"]
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
    skin_template = pv.PolyData(
        np.asarray(skin_source.points).copy(), np.asarray(skin_source.faces).copy()
    )
    del skin_source
    skin_sort_order = np.argsort(skin_global_ids)
    sorted_skin_global_ids = skin_global_ids[skin_sort_order]

    inputs_dir = OUTPUT_DIR / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    material_paths: list[Path] = []
    skin_paths: list[Path] = []
    frame_records: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    section_bounds: list[list[float]] = []
    oo_bounds: list[list[float]] = []
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
            oo_bounds.append(array_bounds(deformed_points[oo_points]))
            full_grid.points = deformed_points
            section = full_grid.slice(
                normal=(0.0, 1.0, 0.0),
                origin=(0.0, PLANE_Y, 0.0),
                generate_triangles=False,
            )
            material_metrics = section_record(section)
            expected_counts = EXPECTED_STEP_COUNTS.get(step)
            if (
                expected_counts is not None
                and material_metrics["dominant_counts"] != expected_counts
            ):
                raise ValueError(
                    f"step {step} material counts changed: "
                    f"{material_metrics['dominant_counts']}"
                )
            material_path = inputs_dir / f"material-section-step-{step:03d}.vtp"
            material_identity = save_dataset(section, material_path)
            material_paths.append(material_path)
            section_bounds.append(material_metrics["bounds_m"])

            sorted_skin_points = np.asarray(
                payload[offset + sorted_skin_global_ids], dtype=np.float64
            )
            skin_points = np.empty_like(sorted_skin_points)
            skin_points[skin_sort_order] = sorted_skin_points
            dynamic_skin = skin_template.copy(deep=True)
            dynamic_skin.points = skin_points
            skin_section = dynamic_skin.slice(
                normal=(0.0, 1.0, 0.0), origin=(0.0, PLANE_Y, 0.0)
            )
            skin_section_points = np.asarray(skin_section.points)
            if (
                skin_section.n_cells < 1
                or skin_section.n_lines != skin_section.n_cells
                or np.max(np.abs(skin_section_points[:, 1] - PLANE_Y)) > 1e-10
            ):
                raise ValueError(f"invalid skin section at step {step}")
            skin_components = line_component_count(skin_section)
            if skin_components != 1:
                raise ValueError(f"skin section is disconnected at step {step}")
            skin_section.clear_data()
            skin_section.cell_data["SkinMembrane"] = np.ones(
                skin_section.n_cells, dtype=np.int8
            )
            skin_path = inputs_dir / f"skin-section-step-{step:03d}.vtp"
            skin_identity = save_dataset(skin_section, skin_path)
            skin_paths.append(skin_path)

            frame_records.append(
                {
                    "step": step,
                    "material_section": {
                        "path": str(material_path.resolve()),
                        "identity": material_identity,
                        **material_metrics,
                    },
                    "skin_section": {
                        "path": str(skin_path.resolve()),
                        "identity": skin_identity,
                        "points": int(skin_section.n_points),
                        "cells": int(skin_section.n_cells),
                        "components": skin_components,
                        "bounds_m": array_bounds(skin_section_points),
                    },
                }
            )
            counts = material_metrics["dominant_counts"]
            areas = material_metrics["dominant_areas_m2"]
            bounds = material_metrics["bounds_m"]
            trajectory_rows.append(
                {
                    "step": step,
                    "section_points": material_metrics["points"],
                    "section_cells": material_metrics["cells"],
                    "fat_cells": counts["Fat"],
                    "muscle_cells": counts["Muscle"],
                    "aponeurosis_cells": counts["Aponeurosis"],
                    "fat_area_m2": areas["Fat"],
                    "muscle_area_m2": areas["Muscle"],
                    "aponeurosis_area_m2": areas["Aponeurosis"],
                    "skin_points": skin_section.n_points,
                    "skin_cells": skin_section.n_cells,
                    "skin_components": skin_components,
                    "x_min_m": bounds[0],
                    "x_max_m": bounds[1],
                    "z_min_m": bounds[4],
                    "z_max_m": bounds[5],
                }
            )
            cherries.set_step(step)
            cherries.log_metrics(
                {
                    "section/cells": material_metrics["cells"],
                    "section/fat_cells": counts["Fat"],
                    "section/muscle_cells": counts["Muscle"],
                    "section/aponeurosis_cells": counts["Aponeurosis"],
                    "skin/cells": skin_section.n_cells,
                }
            )
            logger.info(
                "Prepared step %02d: %d full-head polygons (%d fat, %d muscle, %d aponeurosis)",
                step,
                material_metrics["cells"],
                counts["Fat"],
                counts["Muscle"],
                counts["Aponeurosis"],
            )
            if step == STEPS[-1]:
                final_points = deformed_points.copy()

    if final_points is None or not np.array_equal(final_points, endpoint_deformed):
        raise ValueError("history step 40 does not reproduce the endpoint")
    all_bounds = union_bounds(section_bounds)
    expected_union = [
        1.345802605307577,
        1.4681043565709346,
        PLANE_Y,
        PLANE_Y,
        -0.022516579048682724,
        0.10056717297035135,
    ]
    if not np.allclose(all_bounds, expected_union, atol=1e-9, rtol=0.0):
        raise ValueError(f"full-section union bounds changed: {all_bounds}")
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
        "scope": "expanded Orbicularis-oris region; display crop only after full-head sectioning",
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
            "fixed_for_all_frames": True,
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
            "spatial_crop_before_section": False,
            "whole_volume_dominant_counts": whole_counts,
            "mixed_tetrahedra_over_0p001_in_multiple_fractions": mixed_tets,
            "fraction_fields": list(FRACTION_FIELDS),
            "fraction_sum_max_abs_error": 0.0,
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
            "semantics": "separate Koiter skin membrane intersected by the same plane",
            "not_a_volume_material": True,
            "rgb": SKIN_RGB,
            "line_width_px": SKIN_LINE_WIDTH_PX,
        },
        "section_union_bounds_m": all_bounds,
        "camera": camera,
        "render": {
            "resolution": list(RESOLUTION),
            "fps": FPS,
            "frame_count": len(STEPS),
            "one_panel": True,
            "determinant_metrics_rendered": False,
            "material_representation": "Surface With Edges",
            "cell_edges_rendered": True,
            "internal_tet_section_edges_rendered": True,
            "cell_edge_rgb": CELL_EDGE_RGB,
            "cell_edge_width_px": CELL_EDGE_WIDTH_PX,
            "filled_material_polygons": True,
            "full_head_section_before_camera": True,
            "camera_crop_only": True,
            "complete_section_in_view": False,
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
    logger.info("Wrote exact HFP1 full-head material section video to %s", OUTPUT_DIR)


if __name__ == "__main__":
    cherries.main(main)
