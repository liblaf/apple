"""Aggregate completed, synchronized inverse-update timing receipts.

The input tree records nested inclusive and exclusive scopes.  This analyzer
uses only exclusive node time for additive category totals, so parent/child
scopes are never double counted.  It does not run physics or infer absent
profiles.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from liblaf import cherries

EXPERIMENT = Path(__file__).resolve().parent.parent
EPSILON_SECONDS = 1e-8


class Config(cherries.BaseConfig):
    regularized_dir: Path = EXPERIMENT / "data/inverse-duration-regularized-002"
    hard_dir: Path = EXPERIMENT / "data/inverse-duration-hard-001"
    historical_ledger: Path = EXPERIMENT / "data/historical-duration-audit-001"
    output_dir: Path = EXPERIMENT / "data/inverse-duration-analysis-001"


@dataclass(frozen=True)
class Profile:
    label: str
    case_name: str
    variant: str
    timing: dict[str, Any]
    timing_path: Path
    warm_start: dict[str, Any]
    run_protocol: dict[str, Any]


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"required artifact is missing: {path}"
    result = json.loads(path.read_text())
    assert isinstance(result, dict)
    return result


def validate_tree(tree: dict[str, Any]) -> None:
    """Check all exclusive values and every parent/child accounting identity."""
    for name, node in tree.items():
        assert isinstance(name, str)
        assert isinstance(node["count"], int)
        assert node["count"] > 0
        inclusive = float(node["inclusive_seconds"])
        exclusive = float(node["exclusive_seconds"])
        assert math.isfinite(inclusive)
        assert inclusive >= 0
        assert math.isfinite(exclusive)
        assert exclusive >= -EPSILON_SECONDS
        children = node["children"]
        assert isinstance(children, dict)
        children_inclusive = sum(
            float(value["inclusive_seconds"]) for value in children.values()
        )
        assert abs(exclusive - (inclusive - children_inclusive)) <= EPSILON_SECONDS
        assert children_inclusive <= inclusive + EPSILON_SECONDS
        validate_tree(children)


def walk_exclusive(
    tree: dict[str, Any], ancestors: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    for name, node in tree.items():
        path = (*ancestors, name)
        yield path, node
        yield from walk_exclusive(node["children"], path)


def find_case(run_dir: Path) -> Path:
    cases = [
        child
        for child in run_dir.iterdir()
        if child.is_dir()
        and (child / "historical/timing.json").is_file()
        and (child / "candidate/timing.json").is_file()
    ]
    assert len(cases) == 1, f"expected one completed case in {run_dir}, got {cases}"
    return cases[0]


def load_run(run_dir: Path, *, label: str) -> list[Profile]:
    protocol = read_json(run_dir / "summary.json")
    assert protocol["schema"] == "complete-inverse-update-duration-profile-v1"
    assert protocol["success"] is True, f"incomplete duration run: {run_dir}"
    case_dir = find_case(run_dir)
    warm_start = read_json(case_dir / "warm-start.json")
    assert warm_start["forward_count"] == 0
    assert "excluded from update" in warm_start["scope"]
    profiles = []
    for variant in ("historical", "candidate"):
        timing_path = case_dir / variant / "timing.json"
        timing = read_json(timing_path)
        assert timing["success"] is True, timing.get("failure")
        assert timing["variant"] == variant
        profiling = timing["profiling"]
        assert profiling["schema"] == "inverse-hierarchical-timing-v1"
        assert profiling["cuda_sync"] is True
        assert not profiling["missing_hooks"], profiling["missing_hooks"]
        tree = profiling["tree"]
        validate_tree(tree)
        assert "inverse_update" in tree
        root = tree["inverse_update"]
        assert root["count"] == 1
        assert (
            root["inclusive_seconds"]
            <= timing["outer_profile_wall_seconds"] + EPSILON_SECONDS
        )
        if variant == "historical":
            assert timing["gpu_contact"] is None
        else:
            gpu = timing["gpu_contact"]
            assert gpu is not None
            assert gpu["uploads"] > 0
            assert gpu["products"] > 0
        profiles.append(
            Profile(
                label, case_dir.name, variant, timing, timing_path, warm_start, protocol
            )
        )
    return profiles


def phase(path: tuple[str, ...]) -> str:
    if "forward" in path:
        return "forward"
    if "adjoint" in path:
        return "adjoint"
    return "other"


def forward_detail(path: tuple[str, ...]) -> str:
    inside_ccd = "collision/max_step_size" in path
    categories = (
        ("CCD broad phase", ("ipc/candidates_build",)) if inside_ccd else None,
        ("CCD narrow phase", ("ipc/ccd",)) if inside_ccd else None,
        ("CCD remainder", ("collision/max_step_size",)),
        ("directional curvature (FEM + contact)", ("model/hess_quad",)),
        (
            "contact rebuild",
            (
                "collision/state_at",
                "collision/update",
                "ipc/normal_collisions_build",
                "ipc/candidates_build",
            ),
        ),
        (
            "contact fun / grad / diag",
            ("collision/fun", "collision/grad", "collision/hess_diag"),
        ),
        ("contact Hessian assembly", ("ipc/barrier_hessian", "gpu_contact/assembly")),
        ("FEM operators", ("warp_model/",)),
        (
            "contact HVP / transfer",
            ("collision/hess_prod", "gpu_contact/hess_prod", "gpu_contact/upload"),
        ),
    )
    for item in categories:
        if item is None:
            continue
        category, markers = item
        if any(
            item.startswith(marker) if marker.endswith("/") else item == marker
            for marker in markers
            for item in path
        ):
            return category
    return "linear algebra / control remainder"


def adjoint_detail(path: tuple[str, ...]) -> str:
    categories = (
        ("directional curvature (FEM + contact)", ("model/hess_quad",)),
        ("contact Hessian assembly", ("ipc/barrier_hessian", "gpu_contact/assembly")),
        (
            "contact HVP / transfer",
            ("collision/hess_prod", "gpu_contact/hess_prod", "gpu_contact/upload"),
        ),
        ("FEM operators", ("warp_model/",)),
        (
            "contact fun / grad / diag",
            ("collision/fun", "collision/grad", "collision/hess_diag"),
        ),
        ("adjoint linear solve control", ("adjoint_linear",)),
    )
    for category, markers in categories:
        if any(
            item.startswith(marker) if marker.endswith("/") else item == marker
            for marker in markers
            for item in path
        ):
            return category
    return "adjoint control remainder"


def aggregate(profile: Profile) -> dict[str, Any]:
    tree = profile.timing["profiling"]["tree"]
    root = tree["inverse_update"]
    totals = {"forward": 0.0, "adjoint": 0.0, "other": 0.0}
    forward = dict.fromkeys(
        (
            "CCD broad phase",
            "CCD narrow phase",
            "CCD remainder",
            "contact rebuild",
            "contact fun / grad / diag",
            "contact Hessian assembly",
            "FEM operators",
            "directional curvature (FEM + contact)",
            "contact HVP / transfer",
            "linear algebra / control remainder",
        ),
        0.0,
    )
    adjoint = dict.fromkeys(
        (
            "contact Hessian assembly",
            "contact HVP / transfer",
            "FEM operators",
            "contact fun / grad / diag",
            "directional curvature (FEM + contact)",
            "adjoint linear solve control",
            "adjoint control remainder",
        ),
        0.0,
    )
    exact_ccd = {
        "inclusive_seconds": 0.0,
        "count": 0,
        "broad_seconds": 0.0,
        "broad_count": 0,
        "narrow_seconds": 0.0,
        "narrow_count": 0,
    }
    stage_inclusive = dict.fromkeys(
        ("coarse_pncg", "newton", "newton_cg", "adjoint_linear"), 0.0
    )
    for path, node in walk_exclusive({"inverse_update": root}):
        node_phase = phase(path)
        exclusive = max(0.0, float(node["exclusive_seconds"]))
        totals[node_phase] += exclusive
        if node_phase == "forward":
            forward[forward_detail(path)] += exclusive
        elif node_phase == "adjoint":
            adjoint[adjoint_detail(path)] += exclusive
        terminal = path[-1]
        if terminal == "collision/max_step_size":
            exact_ccd["inclusive_seconds"] += float(node["inclusive_seconds"])
            exact_ccd["count"] += int(node["count"])
        elif terminal == "ipc/candidates_build" and "collision/max_step_size" in path:
            exact_ccd["broad_seconds"] += float(node["inclusive_seconds"])
            exact_ccd["broad_count"] += int(node["count"])
        elif terminal == "ipc/ccd" and "collision/max_step_size" in path:
            exact_ccd["narrow_seconds"] += float(node["inclusive_seconds"])
            exact_ccd["narrow_count"] += int(node["count"])
        if terminal in stage_inclusive:
            stage_inclusive[terminal] += float(node["inclusive_seconds"])
    total_exclusive = sum(totals.values())
    assert abs(total_exclusive - float(root["inclusive_seconds"])) <= EPSILON_SECONDS
    assert abs(sum(forward.values()) - totals["forward"]) <= EPSILON_SECONDS
    assert abs(sum(adjoint.values()) - totals["adjoint"]) <= EPSILON_SECONDS
    receipt_forward = profile.timing["forward"]
    trace = receipt_forward.get("trace", [])
    retry_seconds = sum(
        retry["seconds"]
        for row in trace
        for retry in row.get("regularization_retries", [])
    )
    linear_seconds = sum(row.get("linear", {}).get("seconds", 0.0) for row in trace)
    return {
        "label": profile.label,
        "case": profile.case_name,
        "variant": profile.variant,
        "root_inclusive_seconds": float(root["inclusive_seconds"]),
        "outer_profile_wall_seconds": float(
            profile.timing["outer_profile_wall_seconds"]
        ),
        "top_level_exclusive_seconds": totals,
        "forward_exclusive_seconds": forward,
        "adjoint_exclusive_seconds": adjoint,
        "ccd_exact": {
            **exact_ccd,
            "seconds_per_call": (
                None
                if exact_ccd["count"] == 0
                else exact_ccd["inclusive_seconds"] / exact_ccd["count"]
            ),
            "broad_seconds_per_call": (
                None
                if exact_ccd["broad_count"] == 0
                else exact_ccd["broad_seconds"] / exact_ccd["broad_count"]
            ),
            "narrow_seconds_per_call": (
                None
                if exact_ccd["narrow_count"] == 0
                else exact_ccd["narrow_seconds"] / exact_ccd["narrow_count"]
            ),
        },
        "forward_receipt": {
            "method": receipt_forward.get("method"),
            "coarse_steps": receipt_forward.get("coarse_steps"),
            "newton_steps": receipt_forward.get("newton_steps"),
            "trace_linear_seconds": linear_seconds,
            "trace_regularization_retry_seconds": retry_seconds,
            "tree_stage_inclusive_seconds": stage_inclusive,
        },
        "warm_start": {
            "sha256": profile.warm_start["sha256"],
            "scope": profile.warm_start["scope"],
        },
    }


def stacked(
    ax: plt.Axes, rows: list[dict[str, Any]], key: str, title: str, ylabel: str
) -> None:
    names = [
        name
        for name in rows[0][key]
        if any(row[key][name] > EPSILON_SECONDS for row in rows)
    ]
    positions = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    colors = plt.get_cmap("tab20")(np.linspace(0, 0.95, len(names)))
    for name, color in zip(names, colors, strict=True):
        values = np.asarray([row[key][name] for row in rows])
        ax.bar(positions, values, bottom=bottom, label=name, color=color)
        bottom += values
    labels = [f"{row['label']}\n{row['variant']}" for row in rows]
    ax.set_xticks(positions, labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(visible=True, axis="y", alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")


def main(cfg: Config) -> None:
    assert not cfg.output_dir.exists(), cfg.output_dir
    profiles = [
        *load_run(cfg.regularized_dir, label="regularized"),
        *load_run(cfg.hard_dir, label="hard"),
    ]
    assert len(profiles) == 4
    for label in ("regularized", "hard"):
        warm_hashes = {
            profile.warm_start["sha256"]
            for profile in profiles
            if profile.label == label
        }
        assert len(warm_hashes) == 1, f"{label} variants did not share a warm start"
    aggregates = [aggregate(profile) for profile in profiles]
    historical = read_json(cfg.historical_ledger / "summary.json")
    assert historical["schema"] == "smile-historical-duration-audit-v2"
    cfg.output_dir.mkdir(parents=True)
    figure, axes = plt.subplots(3, 1, figsize=(15, 16), constrained_layout=True)
    stacked(
        axes[0],
        aggregates,
        "top_level_exclusive_seconds",
        "Full update: exclusive time",
        "seconds",
    )
    stacked(
        axes[1],
        aggregates,
        "forward_exclusive_seconds",
        "Forward: exclusive cost breakdown",
        "seconds",
    )
    stacked(
        axes[2],
        aggregates,
        "adjoint_exclusive_seconds",
        "Adjoint: exclusive cost breakdown",
        "seconds",
    )
    png = cfg.output_dir / "inverse-duration-breakdown.png"
    figure.savefig(png, dpi=220)
    plt.close(figure)
    source_hashes = {
        "script": sha256(Path(__file__)),
        "regularized/summary.json": sha256(cfg.regularized_dir / "summary.json"),
        "hard/summary.json": sha256(cfg.hard_dir / "summary.json"),
        "historical-duration-audit/summary.json": sha256(
            cfg.historical_ledger / "summary.json"
        ),
        **{
            f"profile/{profile.label}/{profile.variant}/timing.json": sha256(
                profile.timing_path
            )
            for profile in profiles
        },
    }
    summary = {
        "schema": "inverse-duration-analysis-v1",
        "success": True,
        "scope": "postprocessing of four completed synchronized full-update profiles; exclusive node times are additive, inclusive times are explanatory only",
        "source_hashes": source_hashes,
        "profiles": aggregates,
        "warm_start_scope": {
            label: {
                "shared_sha256": next(
                    profile.warm_start["sha256"]
                    for profile in profiles
                    if profile.label == label
                ),
                "scope": next(
                    profile.warm_start["scope"]
                    for profile in profiles
                    if profile.label == label
                ),
                "comparison": "historical and candidate use the same reconstructed prior-state warm adjoint within this case; reconstruction is excluded from update timing",
            }
            for label in ("regularized", "hard")
        },
        "historical_duration_ledger": {
            "primary_scope": historical["primary_scope"],
            "arms": historical["arms"],
        },
        "figures": {"inverse_duration_breakdown": png.name},
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    cherries.log_metrics({"profiles": len(profiles), "timed_updates": len(aggregates)})
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main)
