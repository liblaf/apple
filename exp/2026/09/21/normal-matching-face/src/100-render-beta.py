# Copyright (c) 2026 liblaf
"""Render saved beta 0, .05, and .25 states at a matched checkpoint."""
# ruff: noqa: SLF001

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pydantic_settings as ps
from experiment import Profile

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
SPEC = {
    "off": (
        ("40-continuation", "smooth-off-l2", "beta 0"),
        ("40-continuation", "smooth-off-normal", "beta .05"),
        ("60-strong-normal", "smooth-off-normal", "beta .25"),
        ("90-beta1", "smooth-off-normal", "beta 1"),
    ),
    "on": (
        ("40-continuation", "smooth-on-l2", "beta 0"),
        ("40-continuation", "smooth-on-normal", "beta .05"),
        ("60-strong-normal", "smooth-on-normal", "beta .25"),
        ("90-beta1", "smooth-on-normal", "beta 1"),
    ),
}
spec = importlib.util.spec_from_file_location(
    "old_render", Path(__file__).with_name("30-render.py")
)
assert spec
assert spec.loader
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    strong_dir: Path = Path("60-strong-normal")
    beta1_dir: Path = Path("90-beta1")
    baseline_dir: Path = Path("40-continuation")
    analysis: Path = Path("96-analysis")
    output: Path = Path("100-beta1-figures")
    shared_step: int | None = None
    error_limit_mm: float | None = None


def steps(folder: Path) -> set[int]:
    return {int(x.stem.removeprefix("step-")) for x in folder.glob("step-*.npz")}


def main(c: Config) -> None:
    roots = {
        "40-continuation": cherries.input(c.baseline_dir),
        "60-strong-normal": cherries.input(c.strong_dir),
        "90-beta1": cherries.input(c.beta1_dir),
    }
    analysis = json.loads((cherries.input(c.analysis) / "analysis.json").read_text())
    assert analysis["gate"]["passed"] is True
    assert analysis["provenance"]["source_paths"] == {
        "baseline": str(c.baseline_dir),
        "strong": str(c.strong_dir),
        "beta1": str(c.beta1_dir),
    }
    folders = {
        (row, name): roots[d] / b for row, items in SPEC.items() for d, b, name in items
    }
    shared = set.intersection(*(steps(x) for x in folders.values()))
    step = max(shared) if c.shared_step is None else c.shared_step
    assert step in shared
    assert step <= int(analysis["latest_common_trace_step"])
    if c.shared_step is None:
        assert step == int(analysis["latest_common_saved_step"])
    out = cherries.output(c.output)
    out.mkdir(parents=True, exist_ok=False)
    with np.load(roots["60-strong-normal"] / "mesh.npz", allow_pickle=False) as z:
        rest = np.asarray(z["rest_points"], float)
        skin = np.asarray(z["skin_ids"], int)
        tri = np.asarray(z["triangles"], int)
        target_u = np.asarray(z["target_displacement_skin"], float)
    reference, target = rest[skin], rest[skin] + target_u
    receipt = json.loads(old.RECEIPT.read_text())
    cams = {x["id"]: x["camera"] for x in receipt["views"]}
    full = {
        **cams["side-context"],
        "parallel_scale": 1.12 * cams["side-context"]["parallel_scale"],
    }
    mouth = cams["region1-mouth-corner"]
    states = [("target", target_u, "Target")]
    for row, items in SPEC.items():
        for d, b, label in items:
            states.append(
                (
                    f"{row}-{label}",
                    old._last(roots[d] / b, len(rest), step),
                    (
                        f"{label} | smooth {row}\nstep {step} | "
                        f"{old._inversions(roots[d] / b, step)} inverted tets"
                    ),
                )
            )
    meshes = {}
    errs = []
    for key, u, _title in states:
        mesh = old._poly(reference + u[skin] if key != "target" else target, tri)
        err = 1000 * np.linalg.norm(mesh.points - target, axis=1)
        mesh.point_data["PositionErrorMM"] = err
        meshes[key] = mesh
        errs.extend(err)
    percentile = float(np.percentile(errs, 99))
    limit = percentile if c.error_limit_mm is None else c.error_limit_mm
    assert limit > 0
    for view, cam in (("full", full), ("mouth", mouth)):
        images = []
        selected = [states[0], *states[1:5], states[0], *states[5:]]
        for key, _u, title in selected:
            p = out / f"{view}-{key}.png"
            old._snapshot(meshes[key], cam, title, p)
            images.append(p)
        old._plate(
            images,
            f"{view}: target and beta 0/.05/.25/1 at matched checkpoint {step}",
            out / f"{view}-comparison.png",
            (2, 5),
        )
    errors = []
    for key, _u, title in states[1:]:
        p = out / f"error-{key}.png"
        old._snapshot(meshes[key], full, title, p, error=True, limit=limit)
        errors.append(p)
    old._plate(
        errors,
        f"Position-error maps at matched checkpoint {step}: common {limit:.3g} mm scale",
        out / "position-error-maps.png",
        (2, 4),
    )
    (out / "summary.json").write_text(
        json.dumps(
            {
                "shared_step": step,
                "source_paths": {
                    "baseline": str(c.baseline_dir),
                    "strong": str(c.strong_dir),
                    "beta1": str(c.beta1_dir),
                    "analysis": str(c.analysis),
                },
                "error_limit_mm": limit,
                "error_99th_percentile_mm_including_target": percentile,
                "error_limit_mm_requested": c.error_limit_mm,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
