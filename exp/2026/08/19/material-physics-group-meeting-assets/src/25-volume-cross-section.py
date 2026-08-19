from __future__ import annotations

# ruff: noqa: EM101, EM102, TRY003
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
DESIGN = "whole-anatomy-dominant-material-coronal-cross-section"
GROUP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[6]

PREPARED_MESH = (
    REPO_ROOT
    / "exp/2026/06/17/human-face-smile-prestrain-v2/data/10-human-face-prepared.vtu"
)
PARAVIEW_SCRIPT = GROUP_DIR / "src/25-volume-cross-section-paraview.py"
PVBATCH = Path("/usr/bin/pvbatch")

OUTPUT_DIR = GROUP_DIR / "data/25-volume-cross-section"
CONTRACT = GROUP_DIR / "data/25-volume-cross-section-contract.json"
RENDERER_RECEIPT = (
    GROUP_DIR / "data/25-volume-cross-section-renderer-receipt.json"
)
FINAL_RECEIPT = GROUP_DIR / "data/25-volume-cross-section-receipt.json"
PNG = OUTPUT_DIR / "25-volume-cross-section-dominant-material.png"
PVSM = OUTPUT_DIR / "25-volume-cross-section-dominant-material.pvsm"

EXPECTED_MESH_IDENTITY = {
    "size_bytes": 76_792_914,
    "sha256": "8131d6944b322d7c1e21918688f297e2887b42bf4dbc19ce36259b007e8dc563",
}
EXPECTED_POINTS = 228_660
EXPECTED_TETS = 1_146_517
EXPECTED_PARAVIEW_VERSION = "6.1.1"
FIELDS = ("FatFraction", "MuscleFraction", "AponeurosisFraction")
IMAGE_RESOLUTION = (2000, 1600)

MATERIALS = {
    "0": {
        "name": "Fat",
        "constitutive_model": "Stable Neo-Hookean",
        "young_modulus_MPa": 0.003,
        "poisson_ratio": 0.49,
        "fraction_field": "FatFraction",
        "rgb": [0.929, 0.694, 0.125],
    },
    "1": {
        "name": "Muscle",
        "constitutive_model": "active Stable Neo-Hookean",
        "young_modulus_MPa": 0.03,
        "poisson_ratio": 0.49,
        "fraction_field": "MuscleFraction",
        "rgb": [0.796, 0.153, 0.153],
    },
    "2": {
        "name": "Aponeurosis",
        "constitutive_model": "Stable Neo-Hookean",
        "young_modulus_MPa": 0.1,
        "poisson_ratio": 0.35,
        "fraction_field": "AponeurosisFraction",
        "rgb": [0.122, 0.467, 0.706],
    },
}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(PREPARED_MESH)
    paraview_script: Path = cherries.input(PARAVIEW_SCRIPT)
    output_contract: Path = cherries.output(
        "25-volume-cross-section-contract.json", mkdir=True
    )
    output_renderer_receipt: Path = cherries.output(
        "25-volume-cross-section-renderer-receipt.json", mkdir=True
    )
    output_final_receipt: Path = cherries.output(
        "25-volume-cross-section-receipt.json", mkdir=True
    )
    output_png: Path = cherries.output(
        "25-volume-cross-section/25-volume-cross-section-dominant-material.png",
        mkdir=True,
    )
    output_pvsm: Path = cherries.output(
        "25-volume-cross-section/25-volume-cross-section-dominant-material.pvsm",
        mkdir=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, int | str]:
    return {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    result = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(result, dict):
        raise TypeError(f"{path} is not a JSON object")
    return result


def _require_exact_path(actual: Path, expected: Path, name: str) -> None:
    if actual.resolve() != expected.resolve():
        raise ValueError(f"{name} must be {expected}, got {actual}")


def _validate_config(cfg: Config) -> None:
    for actual, expected, name in (
        (cfg.input_mesh, PREPARED_MESH, "input_mesh"),
        (cfg.paraview_script, PARAVIEW_SCRIPT, "paraview_script"),
        (cfg.output_contract, CONTRACT, "output_contract"),
        (cfg.output_renderer_receipt, RENDERER_RECEIPT, "output_renderer_receipt"),
        (cfg.output_final_receipt, FINAL_RECEIPT, "output_final_receipt"),
        (cfg.output_png, PNG, "output_png"),
        (cfg.output_pvsm, PVSM, "output_pvsm"),
    ):
        _require_exact_path(actual, expected, name)


def _validate_mesh(path: Path) -> tuple[pv.UnstructuredGrid, dict[str, Any]]:
    if _identity(path) != EXPECTED_MESH_IDENTITY:
        raise ValueError(f"prepared mesh identity changed: {_identity(path)}")
    mesh = pv.read(path)
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise TypeError(f"prepared mesh is {type(mesh).__name__}, expected UnstructuredGrid")
    if mesh.n_points != EXPECTED_POINTS or mesh.n_cells != EXPECTED_TETS:
        raise ValueError(
            f"prepared topology changed: {mesh.n_points} points, {mesh.n_cells} cells"
        )
    missing = [field for field in FIELDS if field not in mesh.cell_data]
    if missing:
        raise KeyError(f"prepared mesh lacks active fraction fields: {missing}")
    fractions = np.column_stack(
        [np.asarray(mesh.cell_data[field], dtype=np.float64) for field in FIELDS]
    )
    if not np.all(np.isfinite(fractions)):
        raise ValueError("active material fractions contain non-finite values")
    if np.any((fractions < 0.0) | (fractions > 1.0)):
        raise ValueError("active material fractions escape [0, 1]")
    sum_error = float(np.max(np.abs(fractions.sum(axis=1) - 1.0)))
    if sum_error != 0.0:
        raise ValueError(f"active fractions no longer sum bit-exactly: {sum_error}")
    dominant = np.argmax(fractions, axis=1)
    dominant_counts = {
        MATERIALS[str(index)]["name"]: int(np.count_nonzero(dominant == index))
        for index in range(3)
    }
    bounds = [float(value) for value in mesh.bounds]
    return mesh, {
        "points": int(mesh.n_points),
        "tetrahedra": int(mesh.n_cells),
        "bounds_m": bounds,
        "active_fraction_sum_max_abs_error": sum_error,
        "whole_volume_dominant_tet_counts": dominant_counts,
        "dominance_tie_break": "numpy argmax field order: Fat, Muscle, Aponeurosis",
    }


def _paraview_version() -> str:
    completed = subprocess.run(
        [str(PVBATCH), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    words = completed.stdout.strip().split()
    if not words:
        raise RuntimeError("pvbatch --version produced no output")
    return words[-1]


def _validate_png(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        size = tuple(int(value) for value in image.size)
        mode = image.mode
    if size != IMAGE_RESOLUTION:
        raise ValueError(f"PNG resolution changed: {size}")
    if path.stat().st_size <= 50_000:
        raise ValueError("PNG is unexpectedly small")
    return {"resolution": list(size), "mode": mode, **_identity(path)}


def main(cfg: Config) -> None:
    _validate_config(cfg)
    if not cfg.paraview_script.is_file():
        raise FileNotFoundError(cfg.paraview_script)
    if not PVBATCH.is_file():
        raise FileNotFoundError(PVBATCH)

    mesh, mesh_validation = _validate_mesh(cfg.input_mesh)
    bounds = np.asarray(mesh.bounds, dtype=np.float64).reshape(3, 2)
    center = bounds.mean(axis=1)
    normal = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    up = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    aspect = IMAGE_RESOLUTION[0] / IMAGE_RESOLUTION[1]
    horizontal_range = float(np.ptp(bounds[0]))
    vertical_range = float(np.ptp(bounds[1]))
    parallel_scale = 0.58 * max(vertical_range, horizontal_range / aspect)

    version = _paraview_version()
    if version != EXPECTED_PARAVIEW_VERSION:
        raise ValueError(f"ParaView version changed: {version}")

    contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": False,
        "input": {
            "path": str(cfg.input_mesh.resolve()),
            "identity": EXPECTED_MESH_IDENTITY,
            "validation": mesh_validation,
        },
        "cross_section": {
            "name": "coronal mid-plane",
            "origin_m": center.tolist(),
            "normal": normal.tolist(),
            "view_up": up.tolist(),
            "camera_focus_m": center.tolist(),
            "camera_parallel_scale_m": float(parallel_scale),
        },
        "categorical_view": {
            "field": "DominantMaterial",
            "definition": "argmax(FatFraction, MuscleFraction, AponeurosisFraction)",
            "visualization_only": True,
            "physics_interpretation": (
                "The solver uses continuous fraction-weighted constitutive energies; "
                "the categorical dominant-material field is not a solver input."
            ),
            "active_fraction_sum_statement": (
                "FatFraction + MuscleFraction + AponeurosisFraction = 1 exactly"
            ),
            "materials": MATERIALS,
        },
        "renderer": {
            "path": str(PVBATCH.resolve()),
            "version": version,
            "script": str(cfg.paraview_script.resolve()),
            "script_identity": _identity(cfg.paraview_script),
            "image_resolution": list(IMAGE_RESOLUTION),
            "native_paraview_rendering": True,
        },
        "outputs": {
            "png": str(cfg.output_png.resolve()),
            "pvsm": str(cfg.output_pvsm.resolve()),
            "renderer_receipt": str(cfg.output_renderer_receipt.resolve()),
        },
    }
    _write_json(cfg.output_contract, contract)

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

    renderer_receipt = _read_json(cfg.output_renderer_receipt)
    if renderer_receipt.get("complete") is not True:
        raise ValueError("renderer receipt is incomplete")
    png_validation = _validate_png(cfg.output_png)
    if not cfg.output_pvsm.is_file() or cfg.output_pvsm.stat().st_size <= 10_000:
        raise ValueError("PVSM is missing or unexpectedly small")

    contract["complete"] = True
    _write_json(cfg.output_contract, contract)
    final_receipt = {
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
        "outputs": {
            "png": {"path": str(cfg.output_png.resolve()), **png_validation},
            "pvsm": {
                "path": str(cfg.output_pvsm.resolve()),
                **_identity(cfg.output_pvsm),
            },
        },
    }
    _write_json(cfg.output_final_receipt, final_receipt)
    logger.info("Wrote native ParaView cross-section: %s", cfg.output_png)


if __name__ == "__main__":
    cherries.main(main)
