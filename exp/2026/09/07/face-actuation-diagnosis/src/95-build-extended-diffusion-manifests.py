#!/usr/bin/env python3
# ruff: noqa: EM101, EM102, TRY003
"""Build exact six-case manifests after the extended diffusion run validates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BASELINE_SUMMARY = ROOT / "data/80-forward-field-diffusion/summary.json"
EXTENDED_SUMMARY = ROOT / "data/94-extended-field-diffusion/summary.json"
RENDER_MANIFEST = ROOT / "docs/95-extended-diffusion-render-manifest.json"
SURFACE_MANIFEST = ROOT / "docs/95-extended-diffusion-surface-manifest.json"

CASE_ORDER = ("baseline", "mild", "strong", "r010", "r001", "component_constant")


def write_new(path: Path, value: dict[str, Any]) -> None:
    """Write one new JSON artifact, refusing to replace prior evidence."""
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def ratio(case: dict[str, Any], source_roughness: float) -> float:
    """Return the recorded achieved field roughness ratio."""
    field = case.get("field")
    if isinstance(field, dict) and "roughness_ratio" in field:
        return float(field["roughness_ratio"])
    return float(case["roughness"]) / source_roughness


def main() -> None:
    """Validate both producers, then emit the render and surface manifests."""
    baseline = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    extended = json.loads(EXTENDED_SUMMARY.read_text(encoding="utf-8"))
    if baseline.get("status") != "completed" or extended.get("status") != "completed":
        raise ValueError("both forward producers must have completed successfully")
    source_roughness = float(baseline["field_preparation"]["source_roughness"])
    baseline_cases = {case["id"]: case for case in baseline["cases"]}
    extended_cases = {case["id"]: case for case in extended["cases"]}
    cases = {**baseline_cases, **extended_cases}
    if tuple(cases) != CASE_ORDER:
        raise ValueError(f"unexpected producer case ordering: {tuple(cases)}")
    if any(case.get("status") != "equilibrium_valid" for case in cases.values()):
        raise ValueError(
            "every published case must be an independently valid equilibrium"
        )
    if len({case["seed_sha256"] for case in cases.values()}) != 1:
        raise ValueError("published cases do not use one common displacement seed")

    labels: dict[str, str] = {}
    for case_id, case in cases.items():
        achieved = ratio(case, source_roughness)
        if case_id == "baseline":
            prefix = "Unchanged Raw6 field"
        elif case_id == "component_constant":
            prefix = "Component-constant within each connected muscle"
        else:
            prefix = "Conservative field diffusion"
        labels[case_id] = f"{prefix} · achieved R/R0 = {achieved:.6f}"

    endpoints = {
        "baseline": "../data/80-forward-field-diffusion/baseline/final.vtu",
        "mild": "../data/80-forward-field-diffusion/mild/final.vtu",
        "strong": "../data/80-forward-field-diffusion/strong/final.vtu",
        "r010": "../data/94-extended-field-diffusion/r010/final.vtu",
        "r001": "../data/94-extended-field-diffusion/r001/final.vtu",
        "component_constant": "../data/94-extended-field-diffusion/component_constant/final.vtu",
    }
    target_skin = {
        "vtu": "../data/12-historical-fixture/volume.vtu",
        "selector": "IsFace",
        "displacement_array": "Smile",
        "valid_selector": "TargetFinite",
    }
    render = {
        "title": "Face actuation diagnosis · extended conservative field diffusion",
        "scope": (
            "Six independently solved equilibria from one frozen Raw6 activation field "
            "and one common displacement seed; no output geometry smoothing"
        ),
        "producer_summaries": [
            "../data/80-forward-field-diffusion/summary.json",
            "../data/94-extended-field-diffusion/summary.json",
        ],
        "cases": [
            {
                "id": case_id,
                "label": labels[case_id],
                "reference_vtu": "../data/12-historical-fixture/volume.vtu",
                "endpoint_vtu": endpoints[case_id],
                "point_convention": "positions_are_state",
                "material_array": "DominantMaterialPhase",
                "target_label": "Historical fixture Smile target skin · post-hoc context only",
                "target_skin": target_skin,
            }
            for case_id in CASE_ORDER
        ],
    }
    surface = {
        "schema_version": 1,
        "scope": (
            "CPU-only post-hoc surface audit across six independently solved equilibria; "
            "R/R0 changes the activation field, never the saved output geometry"
        ),
        "producer_summaries": render["producer_summaries"],
        "fixture_vtu": "../data/12-historical-fixture/volume.vtu",
        "skin_vtp": "../data/12-historical-fixture/skin.vtp",
        "scales_mm": [2.0, 5.0, 10.0],
        "cases": [
            {
                "id": case_id,
                "label": labels[case_id],
                "endpoint_vtu": endpoints[case_id],
            }
            for case_id in CASE_ORDER
        ],
    }
    write_new(RENDER_MANIFEST, render)
    write_new(SURFACE_MANIFEST, surface)


if __name__ == "__main__":
    main()
