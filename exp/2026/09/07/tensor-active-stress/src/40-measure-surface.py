# ruff: noqa: EM101, EM102, TRY003
"""Measure frozen face endpoints with the established rest-skin roughness audit."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
EARLIER_EXPERIMENT = EXPERIMENT.parent / "face-actuation-diagnosis"
FROZEN_ROUGHNESS_HELPER = EARLIER_EXPERIMENT / "src/41-surface-roughness.py"
DEFAULT_MANIFEST = EXPERIMENT / "docs/40-measurement-manifest.json"
SCALES_MM = (2.0, 5.0, 10.0)
PRIMARY_SCALE = "5mm"
MOUTH_RADIUS_MM = 10.0
ZERO_GUARD_ABSOLUTE_MM = 1.0e-9
ZERO_GUARD_TARGET_FRACTION = 1.0e-8
CASE_ID = re.compile(r"[a-z0-9][a-z0-9-]*")
COMPLETED = False


class Config(cherries.BaseConfig):
    """Immutable endpoint manifest and Cherries-managed measurement output."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    manifest: Path = DEFAULT_MANIFEST
    output_dir: Path = EXPERIMENT / "data/40-surface-measurements"


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 receipt."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash an array's contiguous bytes after its dtype is fixed by the caller."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def record(path: Path) -> dict[str, Any]:
    """Describe an immutable file."""
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, value: Any) -> None:
    """Atomically write strict JSON."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_script(name: str, path: Path) -> ModuleType:
    """Load a frozen numbered script while suppressing import-time library noise."""
    if not path.is_file():
        raise FileNotFoundError(path)
    source_dir = str(path.parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        spec.loader.exec_module(module)
    return module


def validate_protocol(manifest: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    """Require the protocol frozen before endpoints are compared."""
    scales = tuple(float(value) for value in manifest.get("scales_mm", ()))
    if scales != SCALES_MM:
        raise ValueError("scales_mm must be exactly [2, 5, 10]")
    if float(manifest.get("primary_scale_mm", math.nan)) != 5.0:
        raise ValueError("primary_scale_mm must be 5")
    if float(manifest.get("mouth_radius_mm", math.nan)) != MOUTH_RADIUS_MM:
        raise ValueError("mouth_radius_mm must be 10")
    if manifest.get("include_target") is not True:
        raise ValueError("include_target must be true")
    for row in cases:
        if CASE_ID.fullmatch(row["id"]) is None:
            raise ValueError(f"unsafe case id: {row['id']!r}")


def sanitize_smile_target(
    fixture: pv.UnstructuredGrid, skin: pv.PolyData
) -> tuple[pv.UnstructuredGrid, dict[str, Any]]:
    """Use the finite target on every actual frozen-skin vertex, without filling it."""
    original = np.asarray(fixture.point_data["Smile"], dtype=np.float64)
    visible = np.asarray(fixture.point_data["IsFace"], dtype=bool)
    valid = np.asarray(fixture.point_data["TargetFinite"], dtype=bool)
    finite = np.isfinite(original).all(axis=1)
    if not np.array_equal(valid, finite):
        raise ValueError("TargetFinite does not exactly identify finite Smile rows")
    invalid_isface = np.flatnonzero(visible & ~valid)
    global_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    if len(invalid_isface) != 8:
        raise ValueError("frozen target's IsFace invalid-vertex count changed")
    if np.any(global_ids < 0) or np.any(global_ids >= fixture.n_points):
        raise ValueError("skin GlobalPointId is outside the frozen fixture")
    if not visible[global_ids].all():
        raise ValueError("frozen skin includes a non-IsFace fixture point")
    if not valid[global_ids].all():
        raise ValueError("frozen skin includes a non-finite Smile target value")
    if np.intersect1d(invalid_isface, global_ids).size:
        raise ValueError("invalid IsFace target unexpectedly intersects frozen skin")
    effective = np.zeros_like(original)
    effective[global_ids] = original[global_ids]
    sanitized = fixture.copy(deep=True)
    sanitized.point_data["Smile"] = effective
    return sanitized, {
        "source_array": "Smile",
        "valid_selector": "TargetFinite",
        "rule": (
            "use the original finite Smile displacement at every frozen skin "
            "GlobalPointId; no frozen skin target value is filled or dropped"
        ),
        "isface_vertices": int(np.count_nonzero(visible)),
        "frozen_skin_vertices": len(global_ids),
        "finite_frozen_skin_vertices": int(np.count_nonzero(valid[global_ids])),
        "invalid_isface_vertices": len(invalid_isface),
        "invalid_isface_global_point_ids": invalid_isface.tolist(),
        "invalid_isface_vertices_on_frozen_skin": 0,
        "original_smile_array_sha256": array_sha256(original),
        "effective_smile_array_sha256": array_sha256(effective),
    }


def make_target_endpoint(fixture: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """Construct the finite, documented Smile target endpoint in memory."""
    rest = np.asarray(fixture.points, dtype=np.float64)
    target = np.asarray(fixture.point_data["Smile"], dtype=np.float64)
    visible = np.asarray(fixture.point_data["IsFace"], dtype=bool)
    if not np.isfinite(target[visible]).all():
        raise ValueError("effective visible Smile target contains non-finite values")
    displacement = np.zeros_like(rest)
    displacement[visible] = target[visible]
    endpoint = fixture.copy(deep=True)
    endpoint.points = rest + displacement
    endpoint.point_data["RestPosition"] = rest
    endpoint.point_data["Displacement"] = displacement
    endpoint.point_data["TargetDisplacement"] = displacement
    return endpoint


def ratio_record(
    numerator_mm: float, denominator_mm: float, guard_mm: float
) -> dict[str, Any]:
    """Return a dimensionless ratio, or an explicit undefined receipt near zero."""
    values = (numerator_mm, denominator_mm, guard_mm)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("ratio inputs must be finite and nonnegative")
    defined = denominator_mm > guard_mm
    return {
        "ratio": numerator_mm / denominator_mm if defined else None,
        "defined": defined,
        "numerator_highpass_rms_mm": numerator_mm,
        "denominator_normal_field_rms_mm": denominator_mm,
        "denominator_guard_mm": guard_mm,
        "undefined_reason": None
        if defined
        else "normal-field RMS is at or below the frozen numerical zero guard",
    }


def add_normalized_ratios(surface: dict[str, Any]) -> dict[str, Any]:
    """Normalize high-pass RMS by the same field and ROI's total normal RMS."""
    target_rms = float(surface["expression"]["target_rms_mm"])
    guard = max(ZERO_GUARD_ABSOLUTE_MM, ZERO_GUARD_TARGET_FRACTION * target_rms)
    normalized: dict[str, Any] = {
        "definition": (
            "high-pass RMS divided by total RMS of the same rest-normal scalar "
            "field on the same ROI"
        ),
        "zero_guard": {
            "absolute_mm": ZERO_GUARD_ABSOLUTE_MM,
            "target_rms_fraction": ZERO_GUARD_TARGET_FRACTION,
            "target_rms_mm": target_rms,
            "applied_mm": guard,
            "comparison": "ratio is defined only when denominator > applied_mm",
        },
        "scales": {},
    }
    fields = (
        ("normal_displacement", "normal_displacement_highpass"),
        ("normal_residual", "normal_residual_highpass"),
    )
    for scale in ("2mm", "5mm", "10mm"):
        scale_output: dict[str, Any] = {}
        for field, highpass_field in fields:
            scale_output[field] = {}
            for roi in ("full_face", "mouth_10mm"):
                scale_output[field][roi] = ratio_record(
                    float(surface["scales"][scale][highpass_field][roi]["rms_mm"]),
                    float(surface[field][roi]["rms_mm"]),
                    guard,
                )
        normalized["scales"][scale] = scale_output
    surface["normalized_highpass"] = normalized
    surface["primary_5mm"] = {
        field: {
            roi: {
                "highpass_rms_mm": surface["scales"][PRIMARY_SCALE][
                    f"{field}_highpass"
                ][roi]["rms_mm"],
                "highpass_over_total_normal_rms": normalized["scales"][PRIMARY_SCALE][
                    field
                ][roi],
            }
            for roi in ("full_face", "mouth_10mm")
        }
        for field in ("normal_displacement", "normal_residual")
    }
    return surface


def save_frozen_map(
    path: Path, fixture: pv.UnstructuredGrid, skin: pv.PolyData, ops: Any
) -> dict[str, Any]:
    """Save the exact rest-surface indices and operators shared by every case."""
    rest = np.asarray(fixture.points, dtype=np.float64)
    target = np.asarray(fixture.point_data["Smile"], dtype=np.float64)
    arrays = {
        "global_point_id": np.asarray(ops.global_ids, dtype=np.int64),
        "triangles": np.asarray(ops.triangles, dtype=np.int64),
        "rest_skin_points_m": np.asarray(ops.points, dtype=np.float64),
        "rest_normals": np.asarray(ops.normals, dtype=np.float64),
        "lumped_area_m2": np.asarray(ops.mass, dtype=np.float64),
        "mouth_10mm_mask": np.asarray(ops.mouth, dtype=np.bool_),
        "lip_seed_mask": np.asarray(ops.lip_seed, dtype=np.bool_),
        "membrane_boundary_mask": np.asarray(ops.boundary, dtype=np.bool_),
        "intrinsic_distance_to_lip_m": np.asarray(ops.lip_distance, dtype=np.float64),
        "target_displacement_m": target[np.asarray(ops.global_ids, dtype=np.int64)],
        "fixture_rest_points_m": rest,
        "skin_points_m": np.asarray(skin.points, dtype=np.float64),
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    return {
        **record(path),
        "array_sha256": {name: array_sha256(value) for name, value in arrays.items()},
    }


def measure_case(
    helper: ModuleType,
    audit: ModuleType,
    fixture: pv.UnstructuredGrid,
    skin: pv.PolyData,
    ops: Any,
    endpoint: pv.UnstructuredGrid,
    case_id: str,
    label: str,
    endpoint_record: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Measure and save one endpoint using the frozen earlier implementation."""
    helper.assert_rest_geometry(fixture, skin, endpoint)
    surface, output_skin = audit.surface_diagnostics(
        fixture, endpoint, skin, ops, tuple(value / 1000.0 for value in SCALES_MM)
    )
    helper.assert_nonnegative_highpass(surface)
    if not helper.finite_numbers(surface):
        raise FloatingPointError(f"non-finite surface metric for {case_id}")
    surface = add_normalized_ratios(surface)
    surface_path = output_dir / f"{case_id}-surface.vtp"
    output_skin.save(surface_path)
    return {
        "id": case_id,
        "label": label,
        "endpoint": endpoint_record,
        "surface_map": record(surface_path),
        "surface": surface,
    }


def main(cfg: Config) -> None:
    """Measure immutable final VTUs and the exact target without a physics solve."""
    global COMPLETED  # noqa: PLW0603
    output_dir = cfg.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"choose an empty output directory: {output_dir}")
    manifest_path = cfg.manifest.resolve()
    cherries.log_input(manifest_path)
    helper = load_script("frozen_surface_roughness_41", FROZEN_ROUGHNESS_HELPER)
    manifest, cases = helper.read_manifest(manifest_path)
    validate_protocol(manifest, cases)
    fixture_path = helper.resolve_input(manifest_path, manifest["fixture_vtu"])
    skin_path = helper.resolve_input(manifest_path, manifest["skin_vtp"])
    source_fixture = pv.read(fixture_path)
    skin = pv.read(skin_path)
    if not isinstance(source_fixture, pv.UnstructuredGrid) or not isinstance(
        skin, pv.PolyData
    ):
        raise TypeError("fixture must be UnstructuredGrid and skin must be PolyData")
    fixture, target_contract = sanitize_smile_target(source_fixture, skin)
    audit = helper.load_sibling_audit()
    ops = audit.cotangent_operators(skin)
    invariance = helper.constant_invariance(
        audit, ops, tuple(value / 1000.0 for value in SCALES_MM)
    )
    frozen_map = save_frozen_map(
        output_dir / "frozen-surface-map.npz", fixture, skin, ops
    )

    output_cases = [
        measure_case(
            helper,
            audit,
            fixture,
            skin,
            ops,
            make_target_endpoint(fixture),
            "target",
            "Exact Smile target on the frozen rest skin",
            {
                "kind": "in-memory target from frozen fixture Smile",
                "fixture_sha256": sha256(fixture_path),
                "target_contract": target_contract,
                "target_visible_array_sha256": frozen_map["array_sha256"][
                    "target_displacement_m"
                ],
            },
            output_dir,
        )
    ]
    for row in cases:
        endpoint_path = helper.resolve_input(manifest_path, row["endpoint_vtu"])
        endpoint = pv.read(endpoint_path)
        if not isinstance(endpoint, pv.UnstructuredGrid):
            raise TypeError(f"endpoint is not an UnstructuredGrid: {endpoint_path}")
        output_cases.append(
            measure_case(
                helper,
                audit,
                fixture,
                skin,
                ops,
                endpoint,
                row["id"],
                row["label"],
                {"kind": "immutable saved final VTU", **record(endpoint_path)},
                output_dir,
            )
        )

    summary = {
        "schema_version": 1,
        "status": "completed",
        "scope": (
            "CPU-only saved-endpoint surface measurement; no forward, adjoint, "
            "optimization, collision, or anatomy claim"
        ),
        "protocol": {
            "scales_mm": list(SCALES_MM),
            "primary_scale_mm": 5.0,
            "rois": ["full_face", "mouth_10mm"],
            "mouth_radius_mm": MOUTH_RADIUS_MM,
            "fields": ["normal_displacement", "normal_target_residual"],
            "normal": "rest-state area-weighted vertex normal",
            "lowpass": "solve (M + t*K)y=Mx with t=scale^2/4",
            "highpass": "rest-normal scalar minus its cotangent-diffused low-pass",
            "boundary": (
                f"natural Neumann/no-flux on {ops.boundary_edges} open membrane edges"
            ),
            "normalization": (
                "high-pass RMS / total RMS of the same normal field and ROI"
            ),
            "zero_guard": {
                "absolute_mm": ZERO_GUARD_ABSOLUTE_MM,
                "target_rms_fraction": ZERO_GUARD_TARGET_FRACTION,
            },
            "constant_invariance": invariance,
        },
        "inputs": {
            "manifest": record(manifest_path),
            "fixture_vtu": record(fixture_path),
            "skin_vtp": record(skin_path),
            "frozen_roughness_helper_41": record(FROZEN_ROUGHNESS_HELPER),
            "verified_surface_implementation": record(helper.SIBLING_AUDIT),
            "measurement_script": record(Path(__file__)),
            "frozen_surface_map": frozen_map,
        },
        "target_contract": target_contract,
        "frozen_mapping": {
            "rule": (
                "skin GlobalPointId maps the one frozen rest skin to fixture "
                "points; every endpoint RestPosition must equal fixture points exactly"
            ),
            "skin_vertices": int(skin.n_points),
            "skin_triangles": int(skin.n_cells),
            "mouth_vertices": int(np.count_nonzero(ops.mouth)),
            "lip_seed_vertices": int(np.count_nonzero(ops.lip_seed)),
            "boundary_edges": int(ops.boundary_edges),
        },
        "primary_metrics": (
            "5 mm high-pass RMS and high-pass/total-normal-RMS ratio for "
            "displacement and target residual, on full face and mouth_10mm"
        ),
        "interpretation_limit": (
            "High-pass content is descriptive. The target can contain real fine "
            "detail, and a lower value alone does not identify an artifact."
        ),
        "cases": output_cases,
    }
    if not helper.finite_numbers(summary):
        raise FloatingPointError("measurement summary contains non-finite values")
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            cherries.log_output(path)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
