# ruff: noqa: EM101, EM102, TRY003
"""Collect the final ten-case CPU-only surface and field comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib as mpl
import numpy as np
import pyvista as pv
import torch

mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
DEFAULT_MANIFEST = EXPERIMENT / "docs/44-final-surface-comparison-manifest.json"
DEFAULT_OUTPUT = EXPERIMENT / "data/44-final-surface-comparison"
EXPECTED_IDS = (
    "historical-saved-no-skin",
    "manual-c50-no-skin-baseline",
    "manual-c50-selected-muscle-lame-x10",
    "manual-c50-fat-lame-x0.1",
    "manual-c50-aponeurosis-lame-x0.1",
    "current-raw6-smooth-no-skin",
    "current-region5-no-floor",
    "current-raw6-no-skin",
    "historical-adam-raw6",
    "historical-adam-raw6-s",
)
HISTORICAL_GRAPH_IDS = {
    "historical-saved-no-skin",
    "historical-adam-raw6",
    "historical-adam-raw6-s",
}


class IncompleteInputsError(RuntimeError):
    """Raised when a declared immutable endpoint or receipt is missing."""


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    """Describe one immutable input or output file."""
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_script(name: str, path: Path) -> ModuleType:
    """Load one neighboring numbered script without invoking its CLI."""
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


def resolve(manifest_path: Path, relative: str) -> Path:
    """Resolve a manifest path and reject moving latest-state names."""
    if "latest" in Path(relative).name.lower():
        raise ValueError(f"moving latest state is forbidden: {relative}")
    return (manifest_path.parent / relative).resolve()


def read_manifest(path: Path) -> dict[str, Any]:  # noqa: C901, PLR0912
    """Validate the declared final case inventory and its static schemas."""
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("final comparison manifest must use schema version 1")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise TypeError("manifest cases must be a list")
    ids = tuple(case.get("id") for case in cases)
    if ids != EXPECTED_IDS:
        raise ValueError(f"final comparison case order changed: {ids}")
    if tuple(float(value) for value in payload["scales_mm"]) != (2.0, 5.0, 10.0):
        raise ValueError("final comparison requires 2, 5, and 10 mm scales")
    for key in ("fixture_vtu", "skin_vtp", "historical_fixture_vtu"):
        if not isinstance(payload.get(key), str):
            raise TypeError(f"manifest requires {key}")
        resolve(path, payload[key])
    for case in cases:
        for key in ("label", "category", "endpoint_vtu"):
            if not isinstance(case.get(key), str) or not case[key]:
                raise TypeError(f"case {case['id']} requires {key}")
        resolve(path, case["endpoint_vtu"])
        schema = case.get("summary_schema")
        if schema is None:
            if "declared_status" not in case:
                raise ValueError(f"case {case['id']} has no status source")
            continue
        if schema not in {"historical_saved", "current_inverse", "historical_adam"}:
            raise ValueError(f"unknown summary schema for {case['id']}: {schema}")
        resolve(path, case["summary_json"])
        if schema == "historical_adam":
            resolve(path, case["solver_receipts_jsonl"])
    return payload


def required_paths(manifest_path: Path, case: dict[str, Any]) -> dict[str, Path]:
    """Return every file required before a case is final-audit ready."""
    paths = {"endpoint_vtu": resolve(manifest_path, case["endpoint_vtu"])}
    if "summary_json" in case:
        paths["summary_json"] = resolve(manifest_path, case["summary_json"])
    if "solver_receipts_jsonl" in case:
        paths["solver_receipts_jsonl"] = resolve(
            manifest_path, case["solver_receipts_jsonl"]
        )
    return paths


def validate_endpoint(
    endpoint_path: Path,
    fixture: pv.UnstructuredGrid,
    skin: pv.PolyData,
    roughness: ModuleType,
) -> None:
    """Require exact rest geometry, topology, and cell ordering."""
    endpoint = pv.read(endpoint_path)
    if not isinstance(endpoint, pv.UnstructuredGrid):
        raise TypeError(f"endpoint is not an UnstructuredGrid: {endpoint_path}")
    roughness.assert_rest_geometry(fixture, skin, endpoint)
    if endpoint.n_cells != fixture.n_cells:
        raise ValueError(f"endpoint cell count changed: {endpoint_path}")
    if not np.array_equal(endpoint.celltypes, fixture.celltypes) or not np.array_equal(
        endpoint.cells, fixture.cells
    ):
        raise ValueError(f"endpoint topology or cell order changed: {endpoint_path}")
    matrices = np.asarray(endpoint.cell_data["ActivationInverseMatrix"], dtype=float)
    if matrices.shape != (fixture.n_cells, 9) or not np.isfinite(matrices).all():
        raise ValueError(f"endpoint active-map field is invalid: {endpoint_path}")


def historical_saved_status(path: Path) -> dict[str, Any]:
    """Read the exported June endpoint status without treating it as a rerun."""
    payload = json.loads(path.read_text())
    metrics = payload["recorded_june_metrics"]
    return {
        "summary_schema": "historical_saved",
        "source_summary": file_record(path),
        "status": payload["status"],
        "endpoint_selection": {
            "label": "recorded June best saved step",
            "step": metrics["best_step"],
        },
        "inverse_converged": metrics["inverse_converged"],
        "stop_reason": metrics["stop_reason"],
        "forward_failures": metrics["forward_failures"],
        "adjoint_failures": metrics["adjoint_failures"],
        "rest_reset": {"performed": False, "result": None},
    }


def current_inverse_status(path: Path) -> dict[str, Any]:
    """Read a current L-BFGS runner summary and preserve its reset result."""
    payload = json.loads(path.read_text())
    final = payload["final"]
    return {
        "summary_schema": "current_inverse",
        "source_summary": file_record(path),
        "status": payload["status"],
        "endpoint_selection": {
            "label": "saved final endpoint",
            "step": final["step"],
        },
        "final": final,
        "accepted_trajectory_min_detF": payload["accepted_trajectory_min_detF"],
        "geometry_rejection_enabled": payload["geometry_rejection_enabled"],
        "rest_reset_difference_over_D": payload["rest_reset_difference_over_D"],
        "reset_forward": payload["reset_forward"],
    }


def receipt_summary(
    path: Path,
    *,
    last_evaluated_step: int,
    best_valid_step: int,
    last_solver_valid_step: int,
) -> dict[str, Any]:
    """Hash a historical solver-receipt stream and retain failure step IDs."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not records:
        raise ValueError(f"empty solver receipt stream: {path}")
    steps = [record["step"] for record in records]
    if steps != list(range(last_evaluated_step + 1)):
        raise ValueError(
            f"solver receipts must contain every unique evaluated step "
            f"0..{last_evaluated_step} in order: {path}"
        )
    valid_steps = [
        record["step"]
        for record in records
        if record["forward"]["success"] is True and record["adjoint"]["success"] is True
    ]
    if best_valid_step not in valid_steps:
        raise ValueError(f"best step has no successful solver receipt: {path}")
    if not valid_steps or last_solver_valid_step != valid_steps[-1]:
        raise ValueError(f"last solver-valid step disagrees with receipts: {path}")
    forward_failures = [
        record["step"] for record in records if not record["forward"]["success"]
    ]
    adjoint_failures = [
        record["step"] for record in records if not record["adjoint"]["success"]
    ]
    return {
        **file_record(path),
        "records": len(records),
        "first_step": records[0]["step"],
        "last_step": records[-1]["step"],
        "solver_valid_records": len(valid_steps),
        "best_valid_step_verified": best_valid_step,
        "last_solver_valid_step_verified": last_solver_valid_step,
        "forward_failure_steps": forward_failures,
        "adjoint_failure_steps": adjoint_failures,
    }


def historical_adam_status(summary_path: Path, receipts_path: Path) -> dict[str, Any]:
    """Read the terminal historical runner schema without inventing reset data."""
    payload = json.loads(summary_path.read_text())
    convergence = payload["convergence"]
    best = payload["best"]
    if convergence["claimed"] is not False:
        raise ValueError("historical Adam runner must not claim convergence")
    if best["step"] != convergence["best_valid_step"]:
        raise ValueError("historical Adam best-step labels disagree")
    if (
        payload["last_evaluated"]["step"] != convergence["last_evaluated_step"]
        or payload["last_solver_valid"]["step"] != convergence["last_solver_valid_step"]
        or best["solver_valid"] is not True
        or payload["last_solver_valid"]["solver_valid"] is not True
    ):
        raise ValueError("historical Adam evaluated/valid row labels disagree")
    receipts = receipt_summary(
        receipts_path,
        last_evaluated_step=convergence["last_evaluated_step"],
        best_valid_step=convergence["best_valid_step"],
        last_solver_valid_step=convergence["last_solver_valid_step"],
    )
    continuation = payload["continuation"]
    source_step = continuation["source_global_step"]
    last_step = convergence["last_evaluated_step"]
    if (
        source_step != 50
        or continuation["local_step_range"] != [0, last_step]
        or continuation["nominal_global_step_range"] != [50, 50 + last_step]
        or continuation["optimizer_moments_at_source"]
        != "explicitly reset for both methods"
        or continuation["uninterrupted_original_trajectory_claimed"] is not False
        or continuation["best_selection_scope"] != "continuation evaluations only"
        or convergence["declared_steps"] != 150
    ):
        raise ValueError(
            "historical continuation step or optimizer-reset contract differs"
        )
    bootstrap = continuation["bootstrap_re_equilibration"]
    if not (
        bootstrap["source_seed_reused"]
        and bootstrap["forward"]["success"]
        and bootstrap["adjoint"]["success"]
    ):
        raise ValueError("historical continuation bootstrap was not solver-valid")
    return {
        "summary_schema": "historical_adam",
        "source_summary": file_record(summary_path),
        "status": payload["status"],
        "convergence": convergence,
        "continuation": continuation,
        "endpoint_selection": {
            "label": "runner best solver-valid endpoint",
            "step": convergence["best_valid_step"],
            "step_coordinate": "local continuation evaluation after shared Adam reset",
            "source_global_step": source_step,
            "nominal_global_step": source_step + convergence["best_valid_step"],
            "runner_field": "convergence.best_valid_step",
        },
        "best": best,
        "last_solver_valid": payload["last_solver_valid"],
        "last_evaluated": payload["last_evaluated"],
        "solver_receipts": receipts,
        "numerical_failures": payload["numerical_failures"],
        "optimizer_events": payload["optimizer"]["events"],
        "geometry_rejection_enabled": payload["geometry_rejection_enabled"],
        "rest_reset": {
            "performed": False,
            "result": None,
            "difference_over_D": None,
        },
    }


def status_record(manifest_path: Path, case: dict[str, Any]) -> dict[str, Any]:
    """Dispatch to the declared runner-summary schema."""
    schema = case.get("summary_schema")
    if schema is None:
        return {
            "summary_schema": "declared_static",
            "status": case["declared_status"],
        }
    summary_path = resolve(manifest_path, case["summary_json"])
    if schema == "historical_saved":
        return historical_saved_status(summary_path)
    if schema == "current_inverse":
        return current_inverse_status(summary_path)
    if schema == "historical_adam":
        receipts_path = resolve(manifest_path, case["solver_receipts_jsonl"])
        return historical_adam_status(summary_path, receipts_path)
    raise AssertionError(schema)


def preflight(
    manifest_path: Path,
    manifest: dict[str, Any],
    roughness: ModuleType,
) -> dict[str, Any]:
    """Validate every currently complete immutable case and list missing files."""
    fixture_path = resolve(manifest_path, manifest["fixture_vtu"])
    skin_path = resolve(manifest_path, manifest["skin_vtp"])
    historical_fixture_path = resolve(manifest_path, manifest["historical_fixture_vtu"])
    for path in (fixture_path, skin_path, historical_fixture_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    fixture = pv.read(fixture_path)
    skin = pv.read(skin_path)
    if not isinstance(fixture, pv.UnstructuredGrid) or not isinstance(
        skin, pv.PolyData
    ):
        raise TypeError("fixture and skin types changed")
    cases = []
    for case in manifest["cases"]:
        paths = required_paths(manifest_path, case)
        missing = [name for name, path in paths.items() if not path.is_file()]
        endpoint_exists = paths["endpoint_vtu"].is_file()
        if endpoint_exists:
            validate_endpoint(paths["endpoint_vtu"], fixture, skin, roughness)
        ready = not missing
        record: dict[str, Any] = {
            "id": case["id"],
            "ready": ready,
            "endpoint_geometry_validated": endpoint_exists,
            "missing": missing,
        }
        if ready:
            record["endpoint"] = file_record(paths["endpoint_vtu"])
            record["status"] = status_record(manifest_path, case)["status"]
        cases.append(record)
    ready_count = sum(case["ready"] for case in cases)
    return {
        "status": "ready" if ready_count == len(cases) else "waiting_for_final_inputs",
        "gpu_used": False,
        "manifest": file_record(manifest_path),
        "ready_cases": ready_count,
        "declared_cases": len(cases),
        "cases": cases,
    }


def current_common_metrics(
    comparison: ModuleType,
    fixture_path: Path,
    skin_path: Path,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate all endpoints on the same current-fixture graph and target."""
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
    if len(ids) != 120_020 or len(ei_np) != 210_187:
        raise ValueError("current comparison graph cardinality changed")
    ei, ej, ew = (torch.as_tensor(value) for value in (ei_np, ej_np, ew_np))
    metrics = [
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
    support = {
        "name": "current_fixture_same_muscle_graph",
        "fixture": file_record(fixture_path),
        "active_tetrahedra": len(ids),
        "muscle_labels": len(np.unique(region)),
        "same_label_shared_face_edges": len(ei_np),
        "smooth_length_m": 0.005,
        "scope": "every endpoint restricted to the current 120,020-cell activation mask",
    }
    return metrics, support


def historical_graph_variation(
    comparison: ModuleType,
    fixture_path: Path,
    endpoints: dict[str, Path],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Evaluate only historical cases on the full 103-label historical graph."""
    fixture = pv.read(fixture_path)
    if not isinstance(fixture, pv.UnstructuredGrid):
        raise TypeError("historical fixture type changed")
    tet = comparison.cells(fixture)
    points = np.asarray(fixture.points, dtype=float)
    ids = np.flatnonzero(np.asarray(fixture.cell_data["ActivationMask"], dtype=bool))
    region = np.asarray(fixture.cell_data["ActivationControlId"], dtype=int)[ids]
    fraction = np.asarray(fixture.cell_data["MuscleFraction"], dtype=float)
    ei_np, ej_np, ew_np = comparison.active_graph(points, tet, ids, region, fraction)
    volume = float(
        np.asarray(fixture.cell_data["Volume"], dtype=float)[ids].dot(fraction[ids])
    )
    if (
        len(ids) != 288_235
        or len(np.unique(region)) != 103
        or len(ei_np) != 501_409
        or not np.all(region[ei_np] == region[ej_np])
        or not np.all(ew_np > 0)
    ):
        raise ValueError("historical comparison graph contract changed")
    ei, ej, ew = (torch.as_tensor(value) for value in (ei_np, ej_np, ew_np))
    values = {}
    for identity, endpoint_path in endpoints.items():
        endpoint = pv.read(endpoint_path)
        if not isinstance(endpoint, pv.UnstructuredGrid):
            raise TypeError(f"historical endpoint type changed: {endpoint_path}")
        if endpoint.n_cells != fixture.n_cells:
            raise ValueError(f"historical endpoint cell count changed: {endpoint_path}")
        matrices = np.asarray(
            endpoint.cell_data["ActivationInverseMatrix"], dtype=float
        ).reshape(-1, 3, 3)
        if matrices.shape != (fixture.n_cells, 3, 3) or not np.isfinite(matrices).all():
            raise ValueError(f"historical active-map field invalid: {endpoint_path}")
        field = torch.as_tensor(
            (matrices[ids] - np.eye(3)).reshape(len(ids), 9) / math.sqrt(1.5)
        )
        values[identity] = float(comparison.smooth_energy(field, ei, ej, ew, volume))
    support = {
        "name": "historical_fixture_same_muscle_graph",
        "fixture": file_record(fixture_path),
        "active_tetrahedra": len(ids),
        "muscle_labels": len(np.unique(region)),
        "same_label_shared_face_edges": len(ei_np),
        "fraction_weighted_active_volume_m3": volume,
        "smooth_length_m": 0.005,
        "scope": "only historical saved and matched historical Adam endpoints",
    }
    return values, support


def highpass_record(surface: dict[str, Any]) -> dict[str, Any]:
    """Select displacement and residual RMS at both ROIs and all scales."""
    return {
        scale: {
            field: {
                roi: surface["scales"][scale][field][roi]["rms_mm"]
                for roi in ("full_face", "mouth_10mm")
            }
            for field in (
                "normal_displacement_highpass",
                "normal_residual_highpass",
            )
        }
        for scale in ("2mm", "5mm", "10mm")
    }


def comparison_record(
    case: dict[str, Any],
    surface_case: dict[str, Any],
    common: dict[str, Any],
    status: dict[str, Any],
    historical_variation: float | None,
) -> dict[str, Any]:
    """Assemble one common table row and cross-check duplicate metrics."""
    expression = surface_case["surface"]["expression"]
    expected = {
        "fit_rms_mm": expression["residual_rms_mm"],
        "motion_rms_mm": expression["displacement_rms_mm"],
        "target_projection_amplitude": expression["target_projection_amplitude"],
    }
    for key, value in expected.items():
        if not math.isclose(value, common[key], rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f"surface/common mismatch for {case['id']}: {key}")
    current_variation = common["same_muscle_activation_variation_common_current_mask"]
    if current_variation is None or not math.isfinite(current_variation):
        raise ValueError(f"missing current-mask field variation: {case['id']}")
    return {
        "id": case["id"],
        "label": case["label"],
        "category": case["category"],
        "endpoint": surface_case["endpoint"],
        "status": status,
        **expected,
        "detF_min": common["detF_min"],
        "inverted_tetrahedra": common["inverted_tets"],
        "highpass_rms_mm": highpass_record(surface_case["surface"]),
        "field_variation": {
            "current_fixture_same_muscle_graph": current_variation,
            "historical_fixture_same_muscle_graph": historical_variation,
        },
    }


def write_csv(path: Path, table: list[dict[str, Any]]) -> None:
    """Write a flat review table with both high-pass fields."""
    fields = [
        "id",
        "label",
        "category",
        "status",
        "endpoint_step",
        "fit_rms_mm",
        "motion_rms_mm",
        "target_projection_amplitude",
        "detF_min",
        "inverted_tetrahedra",
        "current_graph_field_variation",
        "historical_graph_field_variation",
    ]
    fields.extend(
        f"{prefix}_hp_{roi}_{scale}_rms_mm"
        for prefix in ("displacement", "residual")
        for roi in ("full", "mouth")
        for scale in ("2mm", "5mm", "10mm")
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in table:
            row = {
                "id": item["id"],
                "label": item["label"],
                "category": item["category"],
                "status": item["status"]["status"],
                "endpoint_step": item["status"]
                .get("endpoint_selection", {})
                .get("step"),
                "fit_rms_mm": item["fit_rms_mm"],
                "motion_rms_mm": item["motion_rms_mm"],
                "target_projection_amplitude": item["target_projection_amplitude"],
                "detF_min": item["detF_min"],
                "inverted_tetrahedra": item["inverted_tetrahedra"],
                "current_graph_field_variation": item["field_variation"][
                    "current_fixture_same_muscle_graph"
                ],
                "historical_graph_field_variation": item["field_variation"][
                    "historical_fixture_same_muscle_graph"
                ],
            }
            for scale in ("2mm", "5mm", "10mm"):
                highpass = item["highpass_rms_mm"][scale]
                for prefix, field in (
                    ("displacement", "normal_displacement_highpass"),
                    ("residual", "normal_residual_highpass"),
                ):
                    row[f"{prefix}_hp_full_{scale}_rms_mm"] = highpass[field][
                        "full_face"
                    ]
                    row[f"{prefix}_hp_mouth_{scale}_rms_mm"] = highpass[field][
                        "mouth_10mm"
                    ]
            writer.writerow(row)


def write_chart(path_png: Path, path_pdf: Path, table: list[dict[str, Any]]) -> None:
    """Write a secondary numerical tradeoff chart for the saved 3D endpoints."""
    colors = {
        "historical_saved": "#6c757d",
        "manual": "#2a9d8f",
        "manual_material": "#52b788",
        "current_inverse": "#e76f51",
        "historical_adam": "#457b9d",
    }
    labels = {
        "historical-saved-no-skin": "June saved",
        "manual-c50-no-skin-baseline": "manual",
        "manual-c50-selected-muscle-lame-x10": "muscle x10",
        "manual-c50-fat-lame-x0.1": "fat x0.1",
        "manual-c50-aponeurosis-lame-x0.1": "apo x0.1",
        "current-raw6-smooth-no-skin": "current Raw6-S",
        "current-region5-no-floor": "Region5",
        "current-raw6-no-skin": "current Raw6",
        "historical-adam-raw6": "matched Raw6",
        "historical-adam-raw6-s": "matched Raw6-S",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for item in table:
        color = colors[item["category"]]
        label = labels[item["id"]]
        mouth_hp = item["highpass_rms_mm"]["5mm"]["normal_displacement_highpass"][
            "mouth_10mm"
        ]
        axes[0].scatter(item["fit_rms_mm"], mouth_hp, color=color, s=38)
        axes[0].annotate(label, (item["fit_rms_mm"], mouth_hp), fontsize=7)
        axes[1].scatter(
            item["target_projection_amplitude"],
            item["motion_rms_mm"],
            color=color,
            s=38,
        )
        axes[1].annotate(
            label,
            (item["target_projection_amplitude"], item["motion_rms_mm"]),
            fontsize=7,
        )
    axes[0].set(
        xlabel="Target fit RMS (mm)",
        ylabel="Mouth displacement high-pass RMS, 5 mm scale (mm)",
        title="Fit and surface high-frequency content",
    )
    axes[1].set(
        xlabel="Target projection amplitude",
        ylabel="Surface motion RMS (mm)",
        title="Expression amplitude",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Saved-endpoint numerical comparison; inspect 3D geometry separately"
    )
    figure.savefig(path_png, dpi=240)
    figure.savefig(path_pdf)
    plt.close(figure)


def build(manifest_path: Path, output_dir: Path) -> None:
    """Build final artifacts only after every immutable input is complete."""
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    manifest = read_manifest(manifest_path)
    roughness = load_script(
        "final_surface_roughness_41", HERE / "41-surface-roughness.py"
    )
    check = preflight(manifest_path, manifest, roughness)
    if check["status"] != "ready":
        missing = {
            case["id"]: case["missing"] for case in check["cases"] if case["missing"]
        }
        raise IncompleteInputsError(f"final immutable inputs are incomplete: {missing}")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    comparison = load_script(
        "final_diagnosis_comparison_40", HERE / "40-compare-diagnosis.py"
    )
    with tempfile.TemporaryDirectory(
        prefix=".44-final-surface-", dir=output_dir.parent
    ) as temporary:
        stage = Path(temporary) / "artifact"
        stage.mkdir()
        roughness_path = stage / "roughness.json"
        roughness.main(manifest_path, roughness_path)
        roughness_summary = json.loads(roughness_path.read_text())
        records = [
            {
                "id": case["id"],
                "status": "immutable saved endpoint",
                "path": surface["endpoint"]["path"],
            }
            for case, surface in zip(
                manifest["cases"], roughness_summary["cases"], strict=True
            )
        ]
        current_metrics, current_support = current_common_metrics(
            comparison,
            resolve(manifest_path, manifest["fixture_vtu"]),
            resolve(manifest_path, manifest["skin_vtp"]),
            records,
        )
        endpoint_paths = {
            case["id"]: resolve(manifest_path, case["endpoint_vtu"])
            for case in manifest["cases"]
            if case["id"] in HISTORICAL_GRAPH_IDS
        }
        historical_values, historical_support = historical_graph_variation(
            comparison,
            resolve(manifest_path, manifest["historical_fixture_vtu"]),
            endpoint_paths,
        )
        statuses = {
            case["id"]: status_record(manifest_path, case) for case in manifest["cases"]
        }
        table = [
            comparison_record(
                case,
                surface,
                common,
                statuses[case["id"]],
                historical_values.get(case["id"]),
            )
            for case, surface, common in zip(
                manifest["cases"],
                roughness_summary["cases"],
                current_metrics,
                strict=True,
            )
        ]
        csv_path = stage / "comparison-table.csv"
        png_path = stage / "tradeoff.png"
        pdf_path = stage / "tradeoff.pdf"
        write_csv(csv_path, table)
        write_chart(png_path, pdf_path, table)
        summary = {
            "schema_version": 1,
            "scope": (
                "CPU-only comparison of ten immutable saved endpoints; numerical "
                "tables and charts support but do not replace 3D inspection"
            ),
            "manifest": file_record(manifest_path),
            "collector": file_record(Path(__file__)),
            "roughness_audit": roughness_summary,
            "variation_supports": {
                current_support["name"]: current_support,
                historical_support["name"]: historical_support,
            },
            "comparison_table": table,
            "outputs": {
                "csv": {**file_record(csv_path), "path": "comparison-table.csv"},
                "tradeoff_png": {**file_record(png_path), "path": "tradeoff.png"},
                "tradeoff_pdf": {**file_record(pdf_path), "path": "tradeoff.pdf"},
            },
            "interpretation_limits": [
                "High-pass metrics are descriptive and do not identify artifacts by themselves.",
                "Field variation values are comparable only when their named graph support matches.",
                "Historical full-graph variation is not reported for current-fixture or manual cases.",
                "Saved endpoints and fixed budgets do not establish inverse convergence or anatomical validity.",
            ],
            "gpu_used": False,
        }
        summary_path = stage / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        roughness_path.unlink()
        stage.rename(output_dir)


def main() -> None:
    """Run preflight or build the final immutable comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = read_manifest(manifest_path)
    if args.preflight:
        roughness = load_script(
            "preflight_surface_roughness_41", HERE / "41-surface-roughness.py"
        )
        print(json.dumps(preflight(manifest_path, manifest, roughness), indent=2))
        return
    build(manifest_path, args.output)


if __name__ == "__main__":
    main()
