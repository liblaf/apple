"""Prepare and render the full HFP1 Orbicularis oris from head superior view.

This is a post-hoc visualization of the saved step-40 endpoint.  It never
instantiates or reruns the forward or inverse physics.
"""

from __future__ import annotations

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from PIL import Image

from liblaf import cherries

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DESIGN = "hfp1-step40-full-orbicularis-oris-superior-view"
MUSCLE_ID = 254
MUSCLE_NAME = "Orbicularis oris001_Head_muscles_0"
EXPECTED_POINTS = 228_660
EXPECTED_TETS = 1_146_517
EXPECTED_SELECTED_POINTS = 3_248
EXPECTED_SELECTED_TETS = 10_484
EXPECTED_PARAVIEW_VERSION = "6.1.1"
CONTEXT_RESOLUTION = (1_800, 1_400)
DETERMINANT_RESOLUTION = (2_700, 1_000)

GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]
SOURCE_DIR = (
    REPO_ROOT
    / "exp/2026/08/19/human-face-smile-fat-floor-skin-energy-prestrain-inverse/data"
)
SOURCE_ENDPOINT = SOURCE_DIR / "20-hfp1.vtu"
SOURCE_SUMMARY = SOURCE_DIR / "20-hfp1-summary-final.json"
SOURCE_CONTEXT = SOURCE_DIR / "26-paraview-fat-floor-terminal/inputs/hfp1.vtp"
PARAVIEW_SCRIPT = Path(__file__).with_name(
    "40-hfp1-orbicularis-oris-topdown-paraview.py"
)
PVBATCH = Path("/usr/bin/pvbatch")
OUTPUT_DIR = GROUP_DIR / "data/40-hfp1-orbicularis-oris-topdown"

EXPECTED_IDENTITIES = {
    "endpoint": {
        "size_bytes": 147_652_097,
        "sha256": "f93bf583819048b5d81a674c4f409450e3cd1200e0d3811b3dc98811480d53dd",
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


def _output(name: str) -> str:
    return f"40-hfp1-orbicularis-oris-topdown/{name}"


class Config(cherries.BaseConfig):
    """Pinned source and outputs for the post-hoc render."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    source_endpoint: Path = cherries.input(SOURCE_ENDPOINT)
    source_summary: Path = cherries.input(SOURCE_SUMMARY)
    source_context: Path = cherries.input(SOURCE_CONTEXT)
    paraview_script: Path = cherries.input(PARAVIEW_SCRIPT)
    output_reference: Path = cherries.output(
        _output("40-hfp1-orbicularis-oris-reference.vtu"), mkdir=True
    )
    output_deformed: Path = cherries.output(
        _output("40-hfp1-orbicularis-oris-deformed.vtu"), mkdir=True
    )
    output_context_png: Path = cherries.output(
        _output("40-hfp1-orbicularis-oris-topdown-context.png"), mkdir=True
    )
    output_context_pvsm: Path = cherries.output(
        _output("40-hfp1-orbicularis-oris-topdown-context.pvsm"), mkdir=True
    )
    output_determinants_png: Path = cherries.output(
        _output("40-hfp1-orbicularis-oris-topdown-determinants.png"), mkdir=True
    )
    output_determinants_pvsm: Path = cherries.output(
        _output("40-hfp1-orbicularis-oris-topdown-determinants.pvsm"), mkdir=True
    )
    output_contract: Path = cherries.output(_output("contract.json"), mkdir=True)
    output_renderer_receipt: Path = cherries.output(
        _output("renderer-receipt.json"), mkdir=True
    )
    output_receipt: Path = cherries.output(_output("receipt.json"), mkdir=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, int | str]:
    return {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    result = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(result, dict):
        raise TypeError(f"{path} is not a JSON object")
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _expected_paths() -> dict[str, Path]:
    return {
        "source_endpoint": SOURCE_ENDPOINT,
        "source_summary": SOURCE_SUMMARY,
        "source_context": SOURCE_CONTEXT,
        "paraview_script": PARAVIEW_SCRIPT,
        "output_reference": OUTPUT_DIR / "40-hfp1-orbicularis-oris-reference.vtu",
        "output_deformed": OUTPUT_DIR / "40-hfp1-orbicularis-oris-deformed.vtu",
        "output_context_png": OUTPUT_DIR
        / "40-hfp1-orbicularis-oris-topdown-context.png",
        "output_context_pvsm": OUTPUT_DIR
        / "40-hfp1-orbicularis-oris-topdown-context.pvsm",
        "output_determinants_png": OUTPUT_DIR
        / "40-hfp1-orbicularis-oris-topdown-determinants.png",
        "output_determinants_pvsm": OUTPUT_DIR
        / "40-hfp1-orbicularis-oris-topdown-determinants.pvsm",
        "output_contract": OUTPUT_DIR / "contract.json",
        "output_renderer_receipt": OUTPUT_DIR / "renderer-receipt.json",
        "output_receipt": OUTPUT_DIR / "receipt.json",
    }


def _validate_config(cfg: Config) -> None:
    for name, expected in _expected_paths().items():
        actual = getattr(cfg, name)
        if actual.resolve() != expected.resolve():
            raise ValueError(f"{name} must be {expected}, got {actual}")
    if not PVBATCH.is_file():
        raise FileNotFoundError(PVBATCH)
    generated = [
        path for name, path in _expected_paths().items() if name.startswith("output_")
    ]
    stale = [path for path in generated if path.exists()]
    if stale:
        raise FileExistsError(f"refusing stale outputs: {stale}")


def _validate_input(path: Path, key: str) -> dict[str, int | str]:
    actual = _identity(path)
    if actual != EXPECTED_IDENTITIES[key]:
        raise ValueError(f"{key} identity changed: {actual}")
    return actual


def _tetrahedra(mesh: pv.UnstructuredGrid) -> np.ndarray:
    if mesh.n_cells != EXPECTED_TETS or not np.all(mesh.celltypes == pv.CellType.TETRA):
        raise ValueError("HFP1 endpoint is not the expected tetra-only mesh")
    packed = np.asarray(mesh.cells)
    if packed.size != 5 * mesh.n_cells:
        raise ValueError("HFP1 tetra connectivity has an unexpected packed size")
    cells = packed.reshape(-1, 5)
    if not np.all(cells[:, 0] == 4):
        raise ValueError("HFP1 tetra connectivity is malformed")
    return cells[:, 1:]


def _det_f(
    reference: np.ndarray, deformed: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    reference_edges = np.stack(
        [reference[cells[:, index]] - reference[cells[:, 0]] for index in (1, 2, 3)],
        axis=2,
    )
    deformed_edges = np.stack(
        [deformed[cells[:, index]] - deformed[cells[:, 0]] for index in (1, 2, 3)],
        axis=2,
    )
    reference_det = np.linalg.det(reference_edges)
    if np.any(~np.isfinite(reference_det)) or np.any(reference_det == 0.0):
        raise ValueError("HFP1 endpoint contains a degenerate reference tetrahedron")
    return np.linalg.det(deformed_edges) / reference_det, np.abs(reference_det) / 6.0


def _det_ainv(values: np.ndarray) -> np.ndarray:
    if values.shape != (EXPECTED_TETS, 6):
        raise ValueError(f"ActivationInv shape changed: {values.shape}")
    matrix = np.zeros((values.shape[0], 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = 1.0 + values[:, 0]
    matrix[:, 1, 1] = 1.0 + values[:, 1]
    matrix[:, 2, 2] = 1.0 + values[:, 2]
    matrix[:, 0, 1] = matrix[:, 1, 0] = values[:, 3]
    matrix[:, 1, 2] = matrix[:, 2, 1] = values[:, 4]
    matrix[:, 0, 2] = matrix[:, 2, 0] = values[:, 5]
    return np.linalg.det(matrix)


def _compact(
    points: np.ndarray,
    cells: np.ndarray,
    selected: np.ndarray,
    fields: dict[str, np.ndarray],
) -> pv.UnstructuredGrid:
    used = np.unique(cells[selected].ravel())
    local = np.searchsorted(used, cells[selected])
    packed = np.column_stack((np.full(selected.size, 4), local)).astype(np.int64)
    result = pv.UnstructuredGrid(
        packed.ravel(),
        np.full(selected.size, pv.CellType.TETRA, dtype=np.uint8),
        points[used],
    )
    result.cell_data["SourceCellId"] = selected.astype(np.int64)
    for name, values in fields.items():
        if values.shape[0] != selected.size:
            raise ValueError(f"prepared field {name} has the wrong length")
        result.cell_data[name] = values
    if (
        result.n_points != EXPECTED_SELECTED_POINTS
        or result.n_cells != EXPECTED_SELECTED_TETS
    ):
        raise ValueError(
            f"full Orbicularis topology changed: {result.n_points}, {result.n_cells}"
        )
    return result


def _save_grid(grid: pv.UnstructuredGrid, path: Path) -> dict[str, int | str]:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    grid.save(temporary, binary=True)
    temporary.replace(path)
    return _identity(path)


def _bounds(points: np.ndarray) -> list[float]:
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


def _union_bounds(*items: list[float]) -> list[float]:
    return [
        min(item[0] for item in items),
        max(item[1] for item in items),
        min(item[2] for item in items),
        max(item[3] for item in items),
        min(item[4] for item in items),
        max(item[5] for item in items),
    ]


def _superior_camera(
    bounds: list[float], resolution: tuple[int, int]
) -> dict[str, Any]:
    focus = [
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    ]
    x_span = bounds[1] - bounds[0]
    z_span = bounds[5] - bounds[4]
    aspect = resolution[0] / resolution[1]
    parallel_scale = 0.58 * max(z_span, x_span / aspect)
    if not np.isfinite(parallel_scale) or parallel_scale <= 0.0:
        raise ValueError("invalid superior-view camera scale")
    return {
        "focus": focus,
        "position": [focus[0], focus[1] + 0.30, focus[2]],
        "view_up": [0.0, 0.0, 1.0],
        "parallel_scale": float(parallel_scale),
        "projection": "parallel",
        "look_direction": [0.0, -1.0, 0.0],
        "orientation": "+Y camera looking toward -Y; +Z is anterior/up in image",
    }


def _metrics(
    volume: np.ndarray, det_f: np.ndarray, det_ainv: np.ndarray
) -> dict[str, Any]:
    det_g = det_f * det_ainv
    double = (det_f < 0.0) & (det_ainv < 0.0)
    total_volume = float(volume.sum())
    output: dict[str, Any] = {
        "cells": int(det_f.size),
        "rest_volume_m3": total_volume,
        "double_inverted_cells": int(double.sum()),
        "double_inverted_rest_volume_fraction": float(
            volume[double].sum() / total_volume
        ),
    }
    for name, values in (
        ("DetF", det_f),
        ("DetAinv", det_ainv),
        ("DetG", det_g),
    ):
        negative = values < 0.0
        output[name] = {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "negative_cells": int(negative.sum()),
            "negative_rest_volume_fraction": float(
                volume[negative].sum() / total_volume
            ),
        }
    return output


def _paraview_version() -> str:
    completed = subprocess.run(
        [str(PVBATCH), "--version"], check=True, capture_output=True, text=True
    )
    words = completed.stdout.strip().split()
    if not words:
        raise RuntimeError("pvbatch --version produced no output")
    return words[-1]


def _validate_png(path: Path, resolution: tuple[int, int]) -> dict[str, Any]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        size = tuple(int(value) for value in image.size)
        mode = image.mode
    if size != resolution:
        raise ValueError(f"PNG resolution changed: {path}: {size}")
    if path.stat().st_size <= 50_000:
        raise ValueError(f"PNG is unexpectedly small: {path}")
    return {"resolution": list(size), "mode": mode, **_identity(path)}


def main(cfg: Config) -> None:
    _validate_config(cfg)
    endpoint_identity = _validate_input(cfg.source_endpoint, "endpoint")
    summary_identity = _validate_input(cfg.source_summary, "summary")
    context_identity = _validate_input(cfg.source_context, "context")
    summary = _read_json(cfg.source_summary)
    expected_summary = {
        "case": "20-hfp1",
        "best_step": 40,
        "best/step": 40,
        "inverse/evaluations": 41,
        "inverse/max_steps": 40,
        "inverse/converged": False,
        "inverse/forward_fail_count": 0,
        "inverse/adjoint_fail_count": 0,
        "inverse/stop_reason": "step_limit_smooth_decrease",
    }
    actual_summary = {key: summary.get(key) for key in expected_summary}
    if actual_summary != expected_summary:
        raise ValueError(f"HFP1 step-40 summary changed: {actual_summary}")

    mesh = pv.read(cfg.source_endpoint)
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise TypeError(f"HFP1 endpoint is {type(mesh).__name__}")
    if mesh.n_points != EXPECTED_POINTS or mesh.n_cells != EXPECTED_TETS:
        raise ValueError(f"HFP1 topology changed: {mesh.n_points}, {mesh.n_cells}")
    required_point = {"DeformedPoint"}
    required_cell = {
        "ActivationInv",
        "ActivationMask",
        "MuscleFraction",
        "MuscleId",
    }
    if not required_point <= set(mesh.point_data):
        raise KeyError("HFP1 endpoint lacks DeformedPoint")
    if not required_cell <= set(mesh.cell_data):
        raise KeyError(
            f"HFP1 endpoint lacks fields: {required_cell - set(mesh.cell_data)}"
        )
    names = np.asarray(mesh.field_data["MuscleName"])
    if names[MUSCLE_ID] != MUSCLE_NAME:
        raise ValueError(f"MuscleId 254 changed: {names[MUSCLE_ID]!r}")

    cells = _tetrahedra(mesh)
    reference_points = np.asarray(mesh.points, dtype=np.float64)
    deformed_points = np.asarray(mesh.point_data["DeformedPoint"], dtype=np.float64)
    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    muscle_id = np.asarray(mesh.cell_data["MuscleId"], dtype=np.int64)
    muscle_fraction = np.asarray(mesh.cell_data["MuscleFraction"], dtype=np.float64)
    selected = np.flatnonzero(
        active & (muscle_id == MUSCLE_ID) & (muscle_fraction >= 0.5)
    )
    if selected.size != EXPECTED_SELECTED_TETS:
        raise ValueError(f"full Orbicularis selection changed: {selected.size}")

    det_f_all, rest_volume_all = _det_f(reference_points, deformed_points, cells)
    det_ainv_all = _det_ainv(
        np.asarray(mesh.cell_data["ActivationInv"], dtype=np.float64)
    )
    det_f = det_f_all[selected]
    det_ainv = det_ainv_all[selected]
    det_g = det_f * det_ainv
    rest_volume = rest_volume_all[selected]
    double = ((det_f < 0.0) & (det_ainv < 0.0)).astype(np.int8)
    fields = {
        "RestVolume": rest_volume,
        "MuscleFraction": muscle_fraction[selected],
        "DetF": det_f,
        "DetAinv": det_ainv,
        "DetG": det_g,
        "DoubleInverted": double,
    }
    reference = _compact(reference_points, cells, selected, fields)
    deformed = _compact(deformed_points, cells, selected, fields)
    reference_identity = _save_grid(reference, cfg.output_reference)
    deformed_identity = _save_grid(deformed, cfg.output_deformed)

    context = pv.read(cfg.source_context)
    if not isinstance(context, pv.PolyData):
        raise TypeError(f"HFP1 context is {type(context).__name__}")
    if context.n_points != 15_299 or context.n_cells != 29_899:
        raise ValueError(
            f"HFP1 context topology changed: {context.n_points}, {context.n_cells}"
        )

    reference_bounds = _bounds(np.asarray(reference.points))
    deformed_bounds = _bounds(np.asarray(deformed.points))
    muscle_bounds = _union_bounds(reference_bounds, deformed_bounds)
    context_bounds = [float(value) for value in context.bounds]
    result_metrics = _metrics(rest_volume, det_f, det_ainv)
    scalar_ranges = {
        name: [result_metrics[name]["minimum"], result_metrics[name]["maximum"]]
        for name in ("DetF", "DetAinv", "DetG")
    }
    version = _paraview_version()
    if version != EXPECTED_PARAVIEW_VERSION:
        raise ValueError(f"ParaView version changed: {version}")

    contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "case": {
            "id": "HFP1",
            "source_case": "20-hfp1",
            "step": 40,
            "evaluations": 41,
            "inverse_converged": False,
            "stop_reason": "step_limit_smooth_decrease",
            "physics_rerun": False,
        },
        "inputs": {
            "source_endpoint": {
                "path": str(cfg.source_endpoint.resolve()),
                "identity": endpoint_identity,
                "points": EXPECTED_POINTS,
                "cells": EXPECTED_TETS,
            },
            "source_summary": {
                "path": str(cfg.source_summary.resolve()),
                "identity": summary_identity,
            },
            "reference": {
                "path": str(cfg.output_reference.resolve()),
                "identity": reference_identity,
                "points": EXPECTED_SELECTED_POINTS,
                "cells": EXPECTED_SELECTED_TETS,
            },
            "deformed": {
                "path": str(cfg.output_deformed.resolve()),
                "identity": deformed_identity,
                "points": EXPECTED_SELECTED_POINTS,
                "cells": EXPECTED_SELECTED_TETS,
            },
            "context_surface": {
                "path": str(cfg.source_context.resolve()),
                "identity": context_identity,
                "points": 15_299,
                "cells": 29_899,
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
        "geometry": {
            "reference_bounds_m": reference_bounds,
            "deformed_bounds_m": deformed_bounds,
            "union_bounds_m": muscle_bounds,
            "deformation_exaggeration": 1.0,
        },
        "metrics": result_metrics,
        "scalar_ranges": scalar_ranges,
        "camera": {
            "context": _superior_camera(context_bounds, CONTEXT_RESOLUTION),
            "muscle": _superior_camera(muscle_bounds, DETERMINANT_RESOLUTION),
        },
        "image_resolution": {
            "context": list(CONTEXT_RESOLUTION),
            "determinants": list(DETERMINANT_RESOLUTION),
        },
        "renderer": {
            "authority": "native ParaView only; no PyVista pixel rendering",
            "path": str(PVBATCH.resolve()),
            "version": version,
            "script": str(cfg.paraview_script.resolve()),
            "script_identity": _identity(cfg.paraview_script),
        },
        "outputs": {
            "context_png": str(cfg.output_context_png.resolve()),
            "context_pvsm": str(cfg.output_context_pvsm.resolve()),
            "determinants_png": str(cfg.output_determinants_png.resolve()),
            "determinants_pvsm": str(cfg.output_determinants_pvsm.resolve()),
            "renderer_receipt": str(cfg.output_renderer_receipt.resolve()),
        },
    }
    _write_json(cfg.output_contract, contract)
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

    renderer_receipt = _read_json(cfg.output_renderer_receipt)
    if renderer_receipt.get("complete") is not True:
        raise ValueError("renderer receipt is incomplete")
    outputs = {
        "context_png": {
            "path": str(cfg.output_context_png.resolve()),
            **_validate_png(cfg.output_context_png, CONTEXT_RESOLUTION),
        },
        "context_pvsm": {
            "path": str(cfg.output_context_pvsm.resolve()),
            **_identity(cfg.output_context_pvsm),
        },
        "determinants_png": {
            "path": str(cfg.output_determinants_png.resolve()),
            **_validate_png(cfg.output_determinants_png, DETERMINANT_RESOLUTION),
        },
        "determinants_pvsm": {
            "path": str(cfg.output_determinants_pvsm.resolve()),
            **_identity(cfg.output_determinants_pvsm),
        },
    }
    for key in ("context_pvsm", "determinants_pvsm"):
        if int(outputs[key]["size_bytes"]) <= 10_000:
            raise ValueError(f"PVSM is unexpectedly small: {outputs[key]['path']}")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": True,
        "contract": {
            "path": str(cfg.output_contract.resolve()),
            "identity": _identity(cfg.output_contract),
        },
        "renderer_receipt": {
            "path": str(cfg.output_renderer_receipt.resolve()),
            "identity": _identity(cfg.output_renderer_receipt),
            "summary": renderer_receipt,
        },
        "outputs": outputs,
    }
    _write_json(cfg.output_receipt, receipt)
    cherries.set_step(40)
    cherries.log_metrics(
        {
            "orbicularis/cells": EXPECTED_SELECTED_TETS,
            "orbicularis/det_f_negative_cells": result_metrics["DetF"][
                "negative_cells"
            ],
            "orbicularis/det_ainv_negative_cells": result_metrics["DetAinv"][
                "negative_cells"
            ],
            "orbicularis/det_g_negative_cells": result_metrics["DetG"][
                "negative_cells"
            ],
            "orbicularis/double_inverted_cells": result_metrics[
                "double_inverted_cells"
            ],
        }
    )
    logger.info("Wrote full HFP1 Orbicularis oris superior views to %s", OUTPUT_DIR)


if __name__ == "__main__":
    cherries.main(main)
