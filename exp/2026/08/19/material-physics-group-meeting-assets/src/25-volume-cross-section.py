from __future__ import annotations

# ruff: noqa: EM101, EM102, TRY003
import hashlib
import itertools
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

SCHEMA_VERSION = 2
DESIGN = "whole-anatomy-dominant-material-three-midplane-cross-sections"
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
RENDERER_RECEIPT = GROUP_DIR / "data/25-volume-cross-section-renderer-receipt.json"
FINAL_RECEIPT = GROUP_DIR / "data/25-volume-cross-section-receipt.json"

EXPECTED_MESH_IDENTITY = {
    "size_bytes": 76_792_914,
    "sha256": "8131d6944b322d7c1e21918688f297e2887b42bf4dbc19ce36259b007e8dc563",
}
EXPECTED_POINTS = 228_660
EXPECTED_TETS = 1_146_517
EXPECTED_PARAVIEW_VERSION = "6.1.1"
FIELDS = ("FatFraction", "MuscleFraction", "AponeurosisFraction")
IMAGE_RESOLUTION = (2000, 1600)
PLANE_SPECS = {
    "midsagittal": {
        "label": "midsagittal mid-plane",
        "normal": (1.0, 0.0, 0.0),
        "view_up": (0.0, 1.0, 0.0),
    },
    "coronal": {
        "label": "coronal mid-plane",
        "normal": (0.0, 0.0, 1.0),
        "view_up": (0.0, 1.0, 0.0),
    },
    "axial": {
        "label": "axial mid-plane",
        "normal": (0.0, 1.0, 0.0),
        "view_up": (0.0, 0.0, 1.0),
    },
}
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


def _relative_output(plane: str, suffix: str) -> str:
    stem = f"25-volume-cross-section-{plane}-dominant-material"
    if suffix == "render-input.vtp":
        return f"25-volume-cross-section/25-volume-cross-section-{plane}-{suffix}"
    return f"25-volume-cross-section/{stem}.{suffix}"


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
    output_midsagittal_render_input: Path = cherries.output(
        _relative_output("midsagittal", "render-input.vtp"), mkdir=True
    )
    output_midsagittal_png: Path = cherries.output(
        _relative_output("midsagittal", "png"), mkdir=True
    )
    output_midsagittal_pvsm: Path = cherries.output(
        _relative_output("midsagittal", "pvsm"), mkdir=True
    )
    output_coronal_render_input: Path = cherries.output(
        _relative_output("coronal", "render-input.vtp"), mkdir=True
    )
    output_coronal_png: Path = cherries.output(
        _relative_output("coronal", "png"), mkdir=True
    )
    output_coronal_pvsm: Path = cherries.output(
        _relative_output("coronal", "pvsm"), mkdir=True
    )
    output_axial_render_input: Path = cherries.output(
        _relative_output("axial", "render-input.vtp"), mkdir=True
    )
    output_axial_png: Path = cherries.output(
        _relative_output("axial", "png"), mkdir=True
    )
    output_axial_pvsm: Path = cherries.output(
        _relative_output("axial", "pvsm"), mkdir=True
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


def _plane_paths(cfg: Config, plane: str) -> dict[str, Path]:
    return {
        "render_input": getattr(cfg, f"output_{plane}_render_input"),
        "png": getattr(cfg, f"output_{plane}_png"),
        "pvsm": getattr(cfg, f"output_{plane}_pvsm"),
    }


def _validate_config(cfg: Config) -> None:
    fixed = {
        "input_mesh": PREPARED_MESH,
        "paraview_script": PARAVIEW_SCRIPT,
        "output_contract": CONTRACT,
        "output_renderer_receipt": RENDERER_RECEIPT,
        "output_final_receipt": FINAL_RECEIPT,
    }
    for name, expected in fixed.items():
        actual = getattr(cfg, name)
        if actual.resolve() != expected.resolve():
            raise ValueError(f"{name} must be {expected}, got {actual}")
    for plane in PLANE_SPECS:
        expected = {
            "render_input": OUTPUT_DIR
            / f"25-volume-cross-section-{plane}-render-input.vtp",
            "png": OUTPUT_DIR
            / f"25-volume-cross-section-{plane}-dominant-material.png",
            "pvsm": OUTPUT_DIR
            / f"25-volume-cross-section-{plane}-dominant-material.pvsm",
        }
        for key, actual in _plane_paths(cfg, plane).items():
            if actual.resolve() != expected[key].resolve():
                raise ValueError(f"{plane} {key} must be {expected[key]}, got {actual}")


def _validate_mesh(path: Path) -> tuple[pv.UnstructuredGrid, dict[str, Any]]:
    identity = _identity(path)
    if identity != EXPECTED_MESH_IDENTITY:
        raise ValueError(f"prepared mesh identity changed: {identity}")
    mesh = pv.read(path)
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise TypeError(
            f"prepared mesh is {type(mesh).__name__}, expected UnstructuredGrid"
        )
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
    dominant = np.argmax(fractions, axis=1).astype(np.int32)
    mesh.cell_data["DominantMaterial"] = dominant
    return mesh, {
        "points": int(mesh.n_points),
        "tetrahedra": int(mesh.n_cells),
        "bounds_m": [float(value) for value in mesh.bounds],
        "active_fraction_sum_max_abs_error": sum_error,
        "whole_volume_dominant_tet_counts": {
            MATERIALS[str(index)]["name"]: int(np.count_nonzero(dominant == index))
            for index in range(3)
        },
        "dominance_tie_break": "numpy argmax field order: Fat, Muscle, Aponeurosis",
    }


def _camera_scale(bounds: np.ndarray, normal: np.ndarray, up: np.ndarray) -> float:
    right = np.cross(up, normal)
    right /= np.linalg.norm(right)
    corners = np.asarray(
        list(itertools.product(*[(low, high) for low, high in bounds])),
        dtype=np.float64,
    )
    horizontal_range = float(np.ptp(corners @ right))
    vertical_range = float(np.ptp(corners @ up))
    aspect = IMAGE_RESOLUTION[0] / IMAGE_RESOLUTION[1]
    return 0.58 * max(vertical_range, horizontal_range / aspect)


def _write_section(
    mesh: pv.UnstructuredGrid,
    *,
    plane: str,
    spec: dict[str, Any],
    center: np.ndarray,
    bounds: np.ndarray,
    path: Path,
) -> tuple[dict[str, Any], dict[str, int | str]]:
    normal = np.asarray(spec["normal"], dtype=np.float64)
    up = np.asarray(spec["view_up"], dtype=np.float64)
    section = mesh.slice(normal=normal, origin=center)
    if section.n_points == 0 or section.n_cells == 0:
        raise ValueError(f"prepared {plane} cross-section is empty")
    dominant = np.asarray(section.cell_data["DominantMaterial"], dtype=np.int32)
    counts = {
        MATERIALS[str(index)]["name"]: int(np.count_nonzero(dominant == index))
        for index in range(3)
    }
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"{plane} cross-section misses a material: {counts}")
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    section.save(temporary, binary=True)
    temporary.replace(path)
    return (
        {
            "name": str(spec["label"]),
            "origin_m": center.tolist(),
            "normal": normal.tolist(),
            "view_up": up.tolist(),
            "camera_focus_m": center.tolist(),
            "camera_parallel_scale_m": _camera_scale(bounds, normal, up),
            "points": int(section.n_points),
            "cells": int(section.n_cells),
            "dominant_category_cell_counts": counts,
        },
        _identity(path),
    )


def _paraview_version() -> str:
    completed = subprocess.run(
        [str(PVBATCH), "--version"], check=True, capture_output=True, text=True
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
        raise ValueError(f"PNG is unexpectedly small: {path}")
    return {"resolution": list(size), "mode": mode, **_identity(path)}


def main(cfg: Config) -> None:
    _validate_config(cfg)
    if not cfg.paraview_script.is_file() or not PVBATCH.is_file():
        raise FileNotFoundError("ParaView renderer or pvbatch is missing")
    mesh, mesh_validation = _validate_mesh(cfg.input_mesh)
    bounds = np.asarray(mesh.bounds, dtype=np.float64).reshape(3, 2)
    center = bounds.mean(axis=1)
    cross_sections: dict[str, Any] = {}
    view_outputs: dict[str, Any] = {}
    render_input_identities: dict[str, dict[str, int | str]] = {}
    for plane, spec in PLANE_SPECS.items():
        paths = _plane_paths(cfg, plane)
        section, identity = _write_section(
            mesh,
            plane=plane,
            spec=spec,
            center=center,
            bounds=bounds,
            path=paths["render_input"],
        )
        cross_sections[plane] = section
        render_input_identities[plane] = identity
        view_outputs[plane] = {
            "render_input": {
                "path": str(paths["render_input"].resolve()),
                "identity": identity,
            },
            "png": str(paths["png"].resolve()),
            "pvsm": str(paths["pvsm"].resolve()),
        }

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
        "cross_sections": cross_sections,
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
            "views": view_outputs,
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
    final_outputs: dict[str, Any] = {}
    for plane in PLANE_SPECS:
        paths = _plane_paths(cfg, plane)
        png_validation = _validate_png(paths["png"])
        if not paths["pvsm"].is_file() or paths["pvsm"].stat().st_size <= 10_000:
            raise ValueError(f"PVSM is missing or unexpectedly small: {paths['pvsm']}")
        final_outputs[plane] = {
            "render_input": {
                "path": str(paths["render_input"].resolve()),
                **render_input_identities[plane],
            },
            "png": {"path": str(paths["png"].resolve()), **png_validation},
            "pvsm": {
                "path": str(paths["pvsm"].resolve()),
                **_identity(paths["pvsm"]),
            },
        }

    contract["complete"] = True
    _write_json(cfg.output_contract, contract)
    _write_json(
        cfg.output_final_receipt,
        {
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
            "outputs": final_outputs,
        },
    )
    logger.info("Wrote three native ParaView cross-section views to %s", OUTPUT_DIR)


if __name__ == "__main__":
    cherries.main(main)
