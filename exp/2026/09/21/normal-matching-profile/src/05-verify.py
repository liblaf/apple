"""Independent validation gates for the 2-D normal-matching continuation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import normal_study as ns
import numpy as np
import pydantic_settings as ps
from experiment import Profile

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("05-verification")
    derivatives_only: bool = False
    forward_tolerance: float = 1e-10
    forward_max_iterations: int = 250


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directions(
    function: Callable[[np.ndarray], float],
    point: np.ndarray,
    gradient: np.ndarray,
    rng: np.random.Generator,
    *,
    count: int = 3,
    step: float = 1e-6,
) -> list[dict[str, float]]:
    result = []
    for _ in range(count):
        direction = rng.normal(size=point.size)
        direction /= np.linalg.norm(direction)
        numeric = float(
            (function(point + step * direction) - function(point - step * direction))
            / (2 * step)
        )
        analytic = float(gradient @ direction)
        absolute = abs(analytic - numeric)
        result.append(
            {
                "analytic": analytic,
                "centered_difference": numeric,
                "absolute_error": absolute,
                "relative_error": absolute / max(abs(analytic), abs(numeric), 1e-12),
            }
        )
    return result


def curve_checks() -> dict[str, Any]:
    rng = np.random.default_rng(20260921)
    target = np.column_stack((np.linspace(0, 1, 17), rng.normal(scale=0.04, size=17)))
    points = target + rng.normal(scale=0.01, size=target.shape)
    weights = np.full(len(points) - 1, 1 / (len(points) - 1))
    value, gradient, _ = ns.curve_normal_loss(points, target, weights)
    derivative = directions(
        lambda flat: ns.curve_normal_loss(flat.reshape(points.shape), target, weights)[
            0
        ],
        points.ravel(),
        gradient.ravel(),
        rng,
    )
    maximum = max(item["relative_error"] for item in derivative)
    assert maximum < 2e-8, derivative

    target_value, target_gradient, _ = ns.curve_normal_loss(target, target, weights)
    assert target_value <= 1e-29, target_value
    assert np.linalg.norm(target_gradient, np.inf) <= 1e-13, target_gradient
    offset = np.array([2.3, -0.7])
    translated = ns.curve_normal_loss(points + offset, target + offset, weights)[0]
    scaled = ns.curve_normal_loss(3.7 * points, 3.7 * target, weights)[0]
    assert np.isclose(translated, value, rtol=0, atol=2e-15)
    assert np.isclose(scaled, value, rtol=0, atol=2e-15)
    straight = np.column_stack((np.linspace(0, 1, 17), np.zeros(17)))
    reversed_target = straight[::-1]
    reversal, _, _ = ns.curve_normal_loss(straight, reversed_target, weights)
    assert np.isclose(reversal, 2.0, rtol=0, atol=2e-15), reversal
    return {
        "objective": value,
        "derivatives": derivative,
        "maximum_relative_derivative_error": maximum,
        "target_loss": target_value,
        "target_gradient_inf": float(np.linalg.norm(target_gradient, np.inf)),
        "translation_invariance_abs_error": abs(translated - value),
        "positive_scaling_invariance_abs_error": abs(scaled - value),
        "orientation_reversal_loss": reversal,
    }


def mapped_normal_check() -> dict[str, Any]:
    rng = np.random.default_rng(20260922)
    mesh = ns.ph.build_mesh(20, 10)
    u = rng.normal(scale=0.01, size=mesh.nfree)
    value, gradient, _ = ns.normal_loss(mesh, u, 0.05)
    derivative = directions(
        lambda trial: ns.normal_loss(mesh, trial, 0.05)[0], u, gradient, rng
    )
    maximum = max(item["relative_error"] for item in derivative)
    assert maximum < 2e-8, derivative
    top = ns.gs.top_nodes(mesh)
    x = mesh.p[top, 0]
    target = np.zeros_like(mesh.p)
    target[top, 1] = 4 * 0.05 * x * (1 - x)
    target_u = ns.pack(mesh, target)
    zero, zero_gradient, _ = ns.normal_loss(mesh, target_u, 0.05)
    assert zero <= 1e-28
    assert np.linalg.norm(zero_gradient, np.inf) <= 1e-13
    return {
        "objective": value,
        "derivatives": derivative,
        "maximum_relative_derivative_error": maximum,
        "target_loss": zero,
        "target_gradient_inf": float(np.linalg.norm(zero_gradient, np.inf)),
    }


def smooth_controls(mesh: Any, mode: str) -> np.ndarray:
    centers = mesh.p[mesh.tri[mesh.muscle]].mean(axis=1)
    x, y = centers.T
    wave = 0.02 * np.sin(2 * np.pi * x) + 0.01 * np.cos(6 * np.pi * y)
    if mode == "unconstrained":
        return np.column_stack((0.08 + wave, 0.05 - wave, 0.02 + wave / 2)).ravel()
    if mode == "contraction_only":
        return np.column_stack((0.12 + wave, 0.09 - wave, 0.015 + wave / 4)).ravel()
    if mode == "learned_direction":
        return np.column_stack((0.12 + wave, 0.37 + 0.2 * wave)).ravel()
    assert mode == "x_contraction"
    return 0.12 + wave


def implicit_derivatives() -> dict[str, Any]:
    mesh = ns.ph.build_mesh(20, 10)
    rng = np.random.default_rng(20260923)
    result: dict[str, Any] = {}
    for kind in ("gradient", "normal"):
        modes: dict[str, Any] = {}
        for mode in sorted(ns.am.MODES):
            q = smooth_controls(mesh, mode)
            scales = ns.normalization(mesh, 0.05)
            _, _, analytic, values = ns.evaluate(
                mesh, q, mode, 0.05, kind, 0.25, scales, np.zeros(mesh.nfree)
            )
            check = directions(
                lambda trial, mode=mode, kind=kind, scales=scales: ns.evaluate(
                    mesh,
                    trial,
                    mode,
                    0.05,
                    kind,
                    0.25,
                    scales,
                    np.zeros(mesh.nfree),
                )[3]["objective"],
                q,
                analytic,
                rng,
            )
            maximum = max(item["relative_error"] for item in check)
            assert maximum < 3e-5, (kind, mode, check)
            modes[mode] = {
                "objective": values["objective"],
                "control_size": int(q.size),
                "derivatives": check,
                "maximum_relative_derivative_error": maximum,
            }
        result[kind] = modes
    return result


def current_source_hashes() -> dict[str, str]:
    paths = [
        *ns.numerical_sources(),
        Path(__file__),
        Path(__file__).with_name("experiment.py"),
    ]
    return {str(path): sha256(path) for path in paths}


def historical_source_hashes() -> dict[str, str]:
    protocol = json.loads((ns.BASELINE / "protocol.json").read_text())
    expected = protocol["source_sha256"]
    current = {
        str(ns.gs.study.__file__): sha256(Path(ns.gs.study.__file__)),
        str(ns.am.__file__): sha256(Path(ns.am.__file__)),
        str(ns.ph.__file__): sha256(Path(ns.ph.__file__)),
        str(ns.ph.MESH_SOURCE): sha256(ns.ph.MESH_SOURCE),
    }
    by_name = {Path(key).name: value for key, value in expected.items()}
    for path, digest in current.items():
        assert by_name[Path(path).name] == digest, (
            path,
            digest,
            by_name.get(Path(path).name),
        )
    return {Path(path).name: digest for path, digest in current.items()}


def trace_row(path: Path, step: int) -> dict[str, str]:
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["step"]) == step]
    assert len(rows) == 1, (path, step, len(rows))
    return rows[0]


def historical_replays() -> list[dict[str, Any]]:
    mesh = ns.ph.build_mesh(100, 10)
    result = []
    for folder in sorted(ns.BASELINE.glob("h*-w0")):
        history = np.load(folder / "history.npz", allow_pickle=False)
        index = np.flatnonzero(history["steps"] == 200)
        assert len(index) == 1, (folder, history["steps"])
        i = int(index[0])
        q, full_u = history["controls"][i], history["u"][i]
        mode, height = str(history["mode"]), float(history["height"])
        B = ns.gs.study.matrices(mesh, q, mode)
        packed = ns.pack(mesh, full_u)
        _, residual, _, physical_j = ns.ph.assemble(mesh, packed, B)
        raw_l2 = ns.ph.loss(mesh, packed, height, "l2")[0]
        row = trace_row(folder / "trace.csv", 200)
        assert np.isclose(raw_l2, float(row["raw_loss"]), rtol=1e-10, atol=1e-12)
        assert np.isclose(
            float(physical_j.min()), float(row["min_J"]), rtol=1e-10, atol=1e-12
        )
        assert int(float(row["top_edge_backtracking_count"])) == 0, (folder, row)
        assert physical_j.min() >= 0.1089, (folder, physical_j.min())
        assert np.linalg.norm(residual, np.inf) <= 1e-10
        result.append(
            {
                "case": folder.name,
                "history_step": 200,
                "raw_l2": raw_l2,
                "min_physical_J": float(physical_j.min()),
                "top_edge_backtracking_count": int(
                    float(row["top_edge_backtracking_count"])
                ),
                "equilibrium_residual_inf": float(np.linalg.norm(residual, np.inf)),
                "history_control_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
                "history_u_sha256": hashlib.sha256(full_u.tobytes()).hexdigest(),
            }
        )
    assert len(result) == 8
    return result


def main(cfg: Config) -> None:
    report = {
        "thresholds": {
            "curve_and_mapped_relative_derivative": 2e-8,
            "implicit_relative_derivative": 3e-5,
            "historical_minimum_J": 0.1089,
            "historical_equilibrium_residual_inf": 1e-10,
        },
        "curve_normal_loss": curve_checks(),
        "mapped_normal_loss": mapped_normal_check(),
        "full_implicit_total_objectives": implicit_derivatives(),
        "source_sha256": {
            "historical_reused": historical_source_hashes(),
            "current": current_source_hashes(),
        },
        "historical_step200_replays": historical_replays(),
    }
    if cfg.derivatives_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    report["passed"] = True
    write_json(output / "checks.json", report)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
