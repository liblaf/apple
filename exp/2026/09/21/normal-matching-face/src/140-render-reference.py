# Copyright (c) 2026 liblaf
"""Render the selected fixed-reference loss beside comparable face fits."""
# ruff: noqa: SLF001

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
from experiment import Profile

from liblaf import cherries

SPEC = {
    "off": (
        ("40-continuation", "smooth-off-l2", "L2 only"),
        ("60-strong-normal", "smooth-off-normal", "beta .25"),
        ("132-reference-continuation", "smooth-off-normal", "2 mm / 5 deg"),
        ("90-beta1", "smooth-off-normal", "beta 1"),
    ),
    "on": (
        ("40-continuation", "smooth-on-l2", "L2 only"),
        ("60-strong-normal", "smooth-on-normal", "beta .25"),
        ("132-reference-continuation", "smooth-on-normal", "2 mm / 5 deg"),
        ("90-beta1", "smooth-on-normal", "beta 1"),
    ),
}
OLD_SPEC = importlib.util.spec_from_file_location(
    "old_render", Path(__file__).with_name("30-render.py")
)
assert OLD_SPEC
assert OLD_SPEC.loader
old = importlib.util.module_from_spec(OLD_SPEC)
OLD_SPEC.loader.exec_module(old)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    baseline_dir: Path = Path("40-continuation")
    strong_dir: Path = Path("60-strong-normal")
    reference_dir: Path = Path("132-reference-continuation")
    beta1_dir: Path = Path("90-beta1")
    verification: Path = Path("135-verification/checks.json")
    analysis: Path = Path("136-analysis")
    output: Path = Path("140-reference-figures")
    shared_step: int | None = None
    error_limit_mm: float | None = 10


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved_steps(folder: Path) -> set[int]:
    values = {
        int(path.stem.removeprefix("step-")) for path in folder.glob("step-*.npz")
    }
    assert values, folder
    return values


def mesh_context(root: Path) -> dict[str, np.ndarray]:
    names = (
        "rest_points",
        "skin_ids",
        "triangles",
        "target_displacement_skin",
        "skin_vertex_weights",
        "tets",
        "active_ids",
        "fixed_mask",
        "fixed_values",
        "edge_i",
        "edge_j",
        "edge_weight",
    )
    with np.load(root / "mesh.npz", allow_pickle=False) as saved:
        return {name: np.asarray(saved[name]) for name in names}


def check_geometry(roots: dict[str, Path]) -> dict[str, Any]:
    contexts = {name: mesh_context(root) for name, root in roots.items()}
    baseline = contexts["40-continuation"]
    for name, context in contexts.items():
        for field, expected in baseline.items():
            assert np.array_equal(context[field], expected), f"{name}: {field}"
    return {
        "reference_root": "40-continuation",
        "arrays_exactly_equal": True,
        "mesh_sha256": {
            name: digest(root / "mesh.npz") for name, root in roots.items()
        },
        "array_shapes": {name: list(value.shape) for name, value in baseline.items()},
    }


def slug(label: str) -> str:
    return "-".join(
        part
        for part in label.lower().replace("/", " ").replace(".", "").split()
        if part
    )


def main(cfg: Config) -> None:  # noqa: PLR0915
    audit = json.loads(cherries.input(cfg.verification).read_text())
    assert audit["passed"] is True
    assert audit["all_completed"] is True
    analysis = json.loads((cherries.input(cfg.analysis) / "analysis.json").read_text())
    assert analysis["gate"] == {
        "path": str(cfg.verification),
        "passed": True,
        "all_completed": True,
    }
    assert analysis["provenance"]["source_paths"] == {
        "baseline": str(cfg.baseline_dir),
        "strong": str(cfg.strong_dir),
        "reference": str(cfg.reference_dir),
        "beta1": str(cfg.beta1_dir),
    }
    required_variants = {
        "smooth-off-beta0",
        "smooth-off-beta25",
        "smooth-off-beta1",
        "smooth-on-beta0",
        "smooth-on-beta25",
        "smooth-on-beta1",
        "reference-off",
        "reference-on",
    }
    assert required_variants <= set(analysis["variants"])

    roots = {
        "40-continuation": cherries.input(cfg.baseline_dir),
        "60-strong-normal": cherries.input(cfg.strong_dir),
        "132-reference-continuation": cherries.input(cfg.reference_dir),
        "90-beta1": cherries.input(cfg.beta1_dir),
    }
    geometry = check_geometry(roots)
    folders = {
        (row, label): roots[root] / branch
        for row, items in SPEC.items()
        for root, branch, label in items
    }
    common = set.intersection(*(saved_steps(folder) for folder in folders.values()))
    step = max(common) if cfg.shared_step is None else cfg.shared_step
    assert step in common
    assert step <= int(analysis["latest_common_trace_step"])
    if cfg.shared_step is None:
        assert step == int(analysis["latest_common_saved_step"])

    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    context = mesh_context(roots["40-continuation"])
    rest = np.asarray(context["rest_points"], dtype=float)
    skin = np.asarray(context["skin_ids"], dtype=int)
    triangles = np.asarray(context["triangles"], dtype=int)
    target = rest[skin] + np.asarray(context["target_displacement_skin"], dtype=float)
    receipt = json.loads(old.RECEIPT.read_text())
    cameras = {item["id"]: item["camera"] for item in receipt["views"]}
    full = {
        **cameras["side-context"],
        "parallel_scale": 1.12 * cameras["side-context"]["parallel_scale"],
    }
    mouth = cameras["region1-mouth-corner"]
    states: list[tuple[str, np.ndarray, str]] = [("target", target, "Target")]
    for row, items in SPEC.items():
        for root, branch, label in items:
            folder = roots[root] / branch
            states.append(
                (
                    f"{row}-{slug(label)}",
                    rest[skin] + old._last(folder, len(rest), step)[skin],
                    f"{label} | smooth {row}\nstep {step} | {old._inversions(folder, step)} inverted tets",
                )
            )
    meshes = {}
    errors: list[float] = []
    for key, points, _title in states:
        mesh = old._poly(points, triangles)
        error = 1000 * np.linalg.norm(points - target, axis=1)
        mesh.point_data["PositionErrorMM"] = error
        meshes[key] = mesh
        errors.extend(error)
    percentile = float(np.percentile(errors, 99))
    limit = percentile if cfg.error_limit_mm is None else cfg.error_limit_mm
    assert limit > 0

    for view, camera in (("full", full), ("mouth", mouth)):
        paths = []
        selected = [states[0], *states[1:5], states[0], *states[5:]]
        for key, _points, title in selected:
            path = out / f"{view}-{key}.png"
            old._snapshot(meshes[key], camera, title, path)
            paths.append(path)
        old._plate(
            paths,
            f"{view}: target, L2, beta .25, selected 2 mm / 5 deg, and beta 1 at step {step}",
            out / f"{view}-comparison.png",
            (2, 5),
        )
    error_paths = []
    for key, _points, title in states[1:]:
        path = out / f"error-{key}.png"
        old._snapshot(meshes[key], full, title, path, error=True, limit=limit)
        error_paths.append(path)
    old._plate(
        error_paths,
        f"Position-error maps at step {step}: common {limit:.3g} mm scale",
        out / "position-error-maps.png",
        (2, 4),
    )
    (out / "summary.json").write_text(
        json.dumps(
            {
                "requested_shared_step": cfg.shared_step,
                "shared_step": step,
                "source_paths": {
                    "baseline": str(cfg.baseline_dir),
                    "strong": str(cfg.strong_dir),
                    "reference": str(cfg.reference_dir),
                    "beta1": str(cfg.beta1_dir),
                    "verification": str(cfg.verification),
                    "analysis": str(cfg.analysis),
                },
                "geometry_fixture_checks": geometry,
                "error_limit_mm": limit,
                "error_99th_percentile_mm_including_target": percentile,
                "error_limit_mm_requested": cfg.error_limit_mm,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
