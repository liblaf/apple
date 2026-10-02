# ruff: noqa: EM101, EM102, TRY003
"""Extend the immutable surface audit with completed inverse endpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyvista as pv
import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
DEFAULT_MANIFEST = EXPERIMENT / "docs/42-extended-surface-roughness-manifest.json"
DEFAULT_OUTPUT = EXPERIMENT / "data/42-extended-surface-roughness/summary.json"
DEFAULT_MARKDOWN = EXPERIMENT / "docs/42-extended-surface-roughness.md"


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 receipt."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_script(name: str, path: Path) -> ModuleType:
    """Load a neighboring numbered script as a module."""
    source_dir = str(HERE)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def common_variation(
    comparison: ModuleType,
    fixture_path: Path,
    skin_path: Path,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate every endpoint on the current 120,020-cell comparison graph."""
    reference = pv.read(fixture_path)
    skin = pv.read(skin_path)
    if not isinstance(reference, pv.UnstructuredGrid) or not isinstance(
        skin, pv.PolyData
    ):
        raise TypeError("fixture and skin types changed")
    target = np.asarray(reference.point_data["Smile"], dtype=float)
    top = np.flatnonzero(
        np.asarray(reference.point_data["IsFace"], dtype=bool)
        & np.isfinite(target).all(1)
    )
    weights = comparison.surface_weights(reference, skin, top)
    tet = comparison.cells(reference)
    points = np.asarray(reference.points, dtype=float)
    dm = np.transpose(points[tet[:, 1:]] - points[tet[:, :1]], (0, 2, 1))
    dm_inv = np.linalg.inv(dm)
    ids = np.flatnonzero(np.asarray(reference.cell_data["ActivationMask"], dtype=bool))
    region = np.asarray(reference.cell_data["ActivationControlId"], dtype=int)[ids]
    fraction = np.asarray(reference.cell_data["MuscleFraction"], dtype=float)
    volume = float((np.linalg.det(dm) / 6)[ids].dot(fraction[ids]))
    ei_np, ej_np, ew_np = comparison.active_graph(points, tet, ids, region, fraction)
    ei, ej, ew = (torch.as_tensor(value) for value in (ei_np, ej_np, ew_np))
    return [
        comparison.common_metrics(
            reference,
            top,
            weights,
            target,
            dm_inv,
            tet,
            ids,
            ei,
            ej,
            ew,
            volume,
            record,
        )
        for record in records
    ]


def table_record(
    roughness_case: dict[str, Any], variation: dict[str, Any]
) -> dict[str, Any]:
    """Select the common numerical fields used in the human-readable table."""
    expression = roughness_case["surface"]["expression"]
    common = {
        "fit_rms_mm": expression["residual_rms_mm"],
        "motion_rms_mm": expression["displacement_rms_mm"],
        "target_projection_amplitude": expression["target_projection_amplitude"],
    }
    for key, value in common.items():
        if not math.isclose(value, variation[key], rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(
                f"common metric mismatch for {roughness_case['id']}: {key}"
            )
    highpass = {}
    for scale in ("2mm", "5mm", "10mm"):
        values = roughness_case["surface"]["scales"][scale][
            "normal_displacement_highpass"
        ]
        highpass[scale] = {
            roi: values[roi]["rms_mm"] for roi in ("full_face", "mouth_10mm")
        }
    return {
        "id": roughness_case["id"],
        "label": roughness_case["label"],
        **common,
        "normal_displacement_highpass_rms_mm": highpass,
        "same_muscle_activation_variation_common_current_mask": variation[
            "same_muscle_activation_variation_common_current_mask"
        ],
    }


def format_number(value: float | None) -> str:
    """Format a compact table number."""
    return "n/a" if value is None else f"{value:.4g}"


def markdown(
    table: list[dict[str, Any]], run_summaries: dict[str, dict[str, Any]]
) -> str:
    """Render the selected common metrics as a reviewable Markdown table."""
    header = (
        "| case | fit RMS (mm) | motion RMS (mm) | projection | full-face "
        "displacement HP RMS at 2/5/10 mm (mm) | mouth displacement HP RMS at "
        "2/5/10 mm (mm) | current-mask field variation |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    rows = []
    for item in table:
        highpass = item["normal_displacement_highpass_rms_mm"]
        full = "/".join(
            format_number(highpass[s]["full_face"]) for s in ("2mm", "5mm", "10mm")
        )
        mouth = "/".join(
            format_number(highpass[s]["mouth_10mm"]) for s in ("2mm", "5mm", "10mm")
        )
        rows.append(
            "| {label} | {fit} | {motion} | {projection} | {full} | {mouth} | {variation} |".format(
                label=item["label"],
                fit=format_number(item["fit_rms_mm"]),
                motion=format_number(item["motion_rms_mm"]),
                projection=format_number(item["target_projection_amplitude"]),
                full=full,
                mouth=mouth,
                variation=format_number(
                    item["same_muscle_activation_variation_common_current_mask"]
                ),
            )
        )
    raw6 = run_summaries["current-raw6-smooth-no-skin"]
    region5 = run_summaries["current-region5-no-floor"]
    return "\n".join(
        [
            "# Extended saved-endpoint surface roughness",
            "",
            "These numbers support inspection of the saved 3D surfaces; they do not replace it. High-pass RMS uses the rest-normal displacement field and the same rest-skin operator for every endpoint.",
            "",
            header,
            *rows,
            "",
            "Field variation is evaluated only on the current fixture's 120,020-cell activation mask and same-MuscleId shared-face graph. The historical endpoint is restricted to that same support; no full historical-mask graph is reported as the current comparison graph.",
            "",
            f"Raw6-S stopped when its line search stalled at step {raw6['final_step']}; its saved endpoint has {raw6['inverted_tets']} inverted tetrahedra, and its independent reset reached {raw6['reset_steps']:,} steps without success. Region5Modes reached its step-{region5['final_step']} budget with {region5['inverted_tets']} inverted tetrahedra; its independent reset succeeded with displacement difference {region5['rest_reset_difference_over_D']:.6g} D. These are saved endpoints, not claims of inverse convergence.",
            "",
        ]
    )


def run_summary_record(path: Path) -> dict[str, Any]:
    """Select stopping and reset evidence from one immutable run summary."""
    payload = json.loads(path.read_text())
    return {
        "path": str(path),
        "sha256": sha256(path),
        "status": payload["status"],
        "final_step": payload["final"]["step"],
        "final_kkt": payload["final"]["kkt"],
        "inverted_tets": payload["final"]["inverted_tets"],
        "reset_success": payload["reset_forward"]["success"],
        "reset_steps": payload["reset_forward"]["steps"],
        "rest_reset_difference_over_D": payload["rest_reset_difference_over_D"],
    }


def main(manifest_path: Path, output_path: Path, markdown_path: Path) -> None:
    """Run the extended CPU-only audit and write JSON plus Markdown."""
    manifest_path = manifest_path.resolve()
    output_path = output_path.resolve()
    markdown_path = markdown_path.resolve()
    roughness = load_script("surface_roughness_41", HERE / "41-surface-roughness.py")
    comparison = load_script(
        "diagnosis_comparison_40", HERE / "40-compare-diagnosis.py"
    )
    roughness.main(manifest_path, output_path)
    summary = json.loads(output_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    fixture_path = (manifest_path.parent / manifest["fixture_vtu"]).resolve()
    skin_path = (manifest_path.parent / manifest["skin_vtp"]).resolve()
    records = [
        {
            "id": case["id"],
            "status": "immutable saved endpoint",
            "path": item["endpoint"]["path"],
        }
        for case, item in zip(manifest["cases"], summary["cases"], strict=True)
    ]
    variations = common_variation(comparison, fixture_path, skin_path, records)
    table = [
        table_record(case, variation)
        for case, variation in zip(summary["cases"], variations, strict=True)
    ]
    summary["inputs"]["extension_script"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256(Path(__file__)),
    }
    summary["comparison_table"] = table
    summary["field_variation_support"] = {
        "metric": "same_muscle_activation_variation_common_current_mask",
        "activation_cells": 120020,
        "support": "current fixture activation mask and same-MuscleId shared-face graph",
        "historical_scope": "historical endpoint restricted to current-mask support; full historical-mask graph not evaluated",
    }
    run_summaries = {
        case["id"]: run_summary_record(
            (manifest_path.parent / case["summary_json"]).resolve()
        )
        for case in manifest["cases"]
        if "summary_json" in case
    }
    summary["run_summaries"] = run_summaries
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(markdown(table, run_summaries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    main(args.manifest, args.output, args.markdown)
