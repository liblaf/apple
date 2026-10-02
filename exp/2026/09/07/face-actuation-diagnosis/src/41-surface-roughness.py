# ruff: noqa: EM101, EM102, TRY003, TRY004
"""CPU-only rest-skin normal-field roughness audit for immutable endpoints."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyvista as pv

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
SIBLING_AUDIT = (
    EXPERIMENT.parent / "face-activation-materials/src/40-audit-face-results.py"
)
DEFAULT_MANIFEST = EXPERIMENT / "docs/41-surface-roughness-manifest.json"
DEFAULT_OUTPUT = EXPERIMENT / "data/41-surface-roughness/summary.json"


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 receipt."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sibling_audit() -> ModuleType:
    """Load the verified surface implementation without editing or copying it."""
    if not SIBLING_AUDIT.is_file():
        raise FileNotFoundError(SIBLING_AUDIT)
    source_dir = str(SIBLING_AUDIT.parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    spec = importlib.util.spec_from_file_location(
        "verified_surface_audit", SIBLING_AUDIT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load verified audit: {SIBLING_AUDIT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # Its Cherries Config declares sibling experiment defaults at import time.
    # This CPU utility only consumes its pure surface functions.
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        spec.loader.exec_module(module)
    return module


def read_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read explicit immutable endpoint paths, rejecting moving latest files."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("roughness manifest must use schema version 1")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("roughness manifest needs at least one case")
    ids = [row.get("id") for row in cases]
    if any(not isinstance(item, str) or not item for item in ids) or len(
        set(ids)
    ) != len(ids):
        raise ValueError("roughness manifest case IDs must be unique nonempty strings")
    for key in ("fixture_vtu", "skin_vtp"):
        if not isinstance(payload.get(key), str):
            raise ValueError(f"roughness manifest needs {key}")
    for row in cases:
        endpoint = row.get("endpoint_vtu")
        if not isinstance(row.get("label"), str) or not isinstance(endpoint, str):
            raise ValueError(
                "every roughness case needs label and endpoint_vtu strings"
            )
        if "latest" in Path(endpoint).name.lower():
            raise ValueError("roughness endpoints must be immutable, never latest")
    return payload, cases


def resolve_input(manifest: Path, relative: str) -> Path:
    """Resolve and require a manifest-referenced immutable file."""
    path = (manifest.parent / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def assert_rest_geometry(
    fixture: pv.UnstructuredGrid, skin: pv.PolyData, endpoint: pv.UnstructuredGrid
) -> None:
    """Require the same frozen volume rest geometry and IsFace point map."""
    if endpoint.n_points != fixture.n_points:
        raise ValueError("endpoint point count differs from frozen fixture")
    rest = np.asarray(fixture.points, dtype=np.float64)
    endpoint_rest = np.asarray(endpoint.point_data["RestPosition"], dtype=np.float64)
    if not np.array_equal(endpoint_rest, rest):
        raise ValueError("endpoint RestPosition differs from frozen fixture points")
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    if np.any(ids < 0) or np.any(ids >= fixture.n_points):
        raise ValueError("skin GlobalPointId is not a fixture point index")
    if not np.array_equal(np.asarray(skin.points, dtype=np.float64), rest[ids]):
        raise ValueError("rest IsFace skin no longer maps exactly to the fixture")


def finite_numbers(value: Any) -> bool:
    """Check JSON-compatible nested metrics for finite floating-point values."""
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def assert_nonnegative_highpass(surface: dict[str, Any]) -> None:
    """Require finite, nonnegative high-pass RMS measurements at every scale."""
    for scale in surface["scales"].values():
        for field in ("normal_displacement_highpass", "normal_residual_highpass"):
            for roi in ("full_face", "mouth_10mm"):
                rms = float(scale[field][roi]["rms_mm"])
                if not math.isfinite(rms) or rms < 0.0:
                    raise ValueError("high-pass RMS must be finite and nonnegative")


def constant_invariance(
    audit: ModuleType, ops: Any, scales_m: tuple[float, ...]
) -> dict[str, float]:
    """Verify natural-Neumann heat diffusion leaves a constant scalar unchanged."""
    constant = np.ones(len(ops.mass), dtype=np.float64)
    result: dict[str, float] = {}
    for scale in scales_m:
        _, high = audit.diffuse_scalar(constant, ops, scale)
        error = float(np.max(np.abs(high)))
        if not math.isfinite(error) or error > 1.0e-10:
            raise ValueError("surface diffusion does not preserve constants")
        result[f"{scale * 1000:g}mm_max_abs_highpass"] = error
    return result


def main(manifest_path: Path, output_path: Path) -> None:
    """Audit every explicitly listed endpoint against one frozen rest skin."""
    manifest_path = manifest_path.resolve()
    manifest, cases = read_manifest(manifest_path)
    fixture_path = resolve_input(manifest_path, manifest["fixture_vtu"])
    skin_path = resolve_input(manifest_path, manifest["skin_vtp"])
    scales_mm = tuple(float(value) for value in manifest.get("scales_mm", (2, 5, 10)))
    if scales_mm != (2.0, 5.0, 10.0):
        raise ValueError("the initial roughness audit requires scales 2, 5, and 10 mm")
    scales_m = tuple(value / 1000.0 for value in scales_mm)
    fixture = pv.read(fixture_path)
    skin = pv.read(skin_path)
    if not isinstance(fixture, pv.UnstructuredGrid) or not isinstance(
        skin, pv.PolyData
    ):
        raise TypeError("fixture must be UnstructuredGrid and skin must be PolyData")
    audit = load_sibling_audit()
    ops = audit.cotangent_operators(skin)
    invariance = constant_invariance(audit, ops, scales_m)
    output_cases = []
    for row in cases:
        endpoint_path = resolve_input(manifest_path, row["endpoint_vtu"])
        endpoint = pv.read(endpoint_path)
        if not isinstance(endpoint, pv.UnstructuredGrid):
            raise TypeError(f"endpoint is not an UnstructuredGrid: {endpoint_path}")
        assert_rest_geometry(fixture, skin, endpoint)
        surface, _ = audit.surface_diagnostics(fixture, endpoint, skin, ops, scales_m)
        if not finite_numbers(surface):
            raise FloatingPointError("surface metrics contain non-finite values")
        assert_nonnegative_highpass(surface)
        output_cases.append(
            {
                "id": row["id"],
                "label": row["label"],
                "endpoint": {
                    "path": str(endpoint_path),
                    "sha256": sha256(endpoint_path),
                    "bytes": endpoint_path.stat().st_size,
                },
                "surface": surface,
            }
        )
    summary = {
        "schema_version": 1,
        "scope": "CPU-only saved-endpoint surface roughness audit; no collision or forward solve",
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "fixture_vtu": {"path": str(fixture_path), "sha256": sha256(fixture_path)},
            "skin_vtp": {"path": str(skin_path), "sha256": sha256(skin_path)},
            "verified_surface_helper": {
                "path": str(SIBLING_AUDIT),
                "sha256": sha256(SIBLING_AUDIT),
            },
            "audit_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__)),
            },
        },
        "frozen_rest_geometry": {
            "skin": "same rest IsFace skin and GlobalPointId mapping for every endpoint",
            "rest_point_sha256": hashlib.sha256(
                np.ascontiguousarray(fixture.points).tobytes()
            ).hexdigest(),
            "skin_point_sha256": hashlib.sha256(
                np.ascontiguousarray(skin.points).tobytes()
            ).hexdigest(),
        },
        "operator": {
            "implementation": "verified sibling 40-audit-face-results.py SurfaceOperators, diffuse_scalar, surface_diagnostics",
            "scales_mm": list(scales_mm),
            "lowpass": "solve (M + t K)y = Mx, t=scale^2/4",
            "highpass": "rest-normal scalar minus its heat-diffused low-pass",
            "boundary": f"natural Neumann/no-flux on {ops.boundary_edges} open membrane edges",
            "mouth": "intrinsic edge distance <= 10 mm from rest GroupName Lip* vertices",
            "units": "normal displacement, normal target residual, and high-pass RMS in mm",
            "constant_invariance": invariance,
        },
        "interpretation_limit": (
            "High-frequency target-residual or displacement content may be real target detail; "
            "these descriptive high-pass metrics do not identify artifacts by themselves."
        ),
        "cases": output_cases,
    }
    if not finite_numbers(summary):
        raise FloatingPointError("roughness summary contains non-finite values")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.manifest, args.output)
