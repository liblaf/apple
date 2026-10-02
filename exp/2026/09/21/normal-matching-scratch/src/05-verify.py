"""Preflight checks for neutral-start normal matching with activation smoothing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import normal_smooth_study as ns
import numpy as np
import pydantic_settings as ps
from experiment import Profile

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("05-verification")
    derivatives_only: bool = False


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directional_checks(
    function: Callable[[np.ndarray], float],
    point: np.ndarray,
    gradient: np.ndarray,
    rng: np.random.Generator,
    *,
    count: int = 3,
    step: float = 1e-6,
) -> list[dict[str, float]]:
    records = []
    for _ in range(count):
        direction = rng.normal(size=point.size)
        direction /= np.linalg.norm(direction)
        numeric = (
            function(point + step * direction) - function(point - step * direction)
        ) / (2 * step)
        analytic = float(gradient @ direction)
        error = abs(analytic - numeric)
        records.append(
            {
                "analytic": analytic,
                "centered_difference": float(numeric),
                "relative_error": float(
                    error / max(abs(analytic), abs(numeric), 1e-12)
                ),
            }
        )
    return records


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


def normal_curve_checks() -> dict[str, Any]:
    rng = np.random.default_rng(20260921)
    target = np.column_stack((np.linspace(0, 1, 17), rng.normal(scale=0.04, size=17)))
    points = target + rng.normal(scale=0.01, size=target.shape)
    weights = np.full(len(points) - 1, 1 / (len(points) - 1))
    value, gradient, _ = ns.ns.curve_normal_loss(points, target, weights)
    derivative = directional_checks(
        lambda flat: ns.ns.curve_normal_loss(
            flat.reshape(points.shape), target, weights
        )[0],
        points.ravel(),
        gradient.ravel(),
        rng,
    )
    maximum = max(record["relative_error"] for record in derivative)
    assert maximum < 2e-8, derivative
    target_value, target_gradient, _ = ns.ns.curve_normal_loss(target, target, weights)
    assert target_value <= 1e-29
    assert np.linalg.norm(target_gradient, np.inf) <= 1e-13
    translation = np.array([2.3, -0.7])
    translated = ns.ns.curve_normal_loss(
        points + translation, target + translation, weights
    )[0]
    scaled = ns.ns.curve_normal_loss(3.7 * points, 3.7 * target, weights)[0]
    assert np.isclose(translated, value, rtol=0, atol=2e-15)
    assert np.isclose(scaled, value, rtol=0, atol=2e-15)
    straight = np.column_stack((np.linspace(0, 1, 17), np.zeros(17)))
    reversal = ns.ns.curve_normal_loss(straight, straight[::-1], weights)[0]
    assert np.isclose(reversal, 2.0, rtol=0, atol=2e-15)
    return {
        "maximum_relative_derivative_error": maximum,
        "target_loss": target_value,
        "target_gradient_inf": float(np.linalg.norm(target_gradient, np.inf)),
        "translation_invariance_abs_error": abs(translated - value),
        "positive_scale_invariance_abs_error": abs(scaled - value),
        "orientation_reversal_loss": reversal,
    }


def mapped_normal_check() -> dict[str, Any]:
    mesh = ns.ph.build_mesh(20, 10)
    rng = np.random.default_rng(20260922)
    u = rng.normal(scale=0.01, size=mesh.nfree)
    _, gradient, _ = ns.ns.normal_loss(mesh, u, 0.05)
    derivative = directional_checks(
        lambda trial: ns.ns.normal_loss(mesh, trial, 0.05)[0], u, gradient, rng
    )
    maximum = max(record["relative_error"] for record in derivative)
    assert maximum < 2e-8, derivative
    return {"maximum_relative_derivative_error": maximum}


def regularizer_pullback_checks() -> dict[str, Any]:
    mesh = ns.ph.build_mesh(20, 10)
    rng = np.random.default_rng(20260923)
    output = {}
    for mode in sorted(ns.am.MODES):
        q = smooth_controls(mesh, mode)
        B = ns.gs.study.matrices(mesh, q, mode)
        value, g_B = ns.am.smoothness(B[mesh.muscle], np.asarray(mesh.edges))
        analytic = ns.am.pullback(q, mode, g_B)
        derivative = directional_checks(
            lambda trial, mode=mode: ns.am.smoothness(
                ns.gs.study.matrices(mesh, trial, mode)[mesh.muscle],
                np.asarray(mesh.edges),
            )[0],
            q,
            analytic,
            rng,
        )
        maximum = max(record["relative_error"] for record in derivative)
        assert maximum < 2e-8, (mode, derivative)
        output[mode] = {
            "roughness": value,
            "maximum_relative_derivative_error": maximum,
        }
    return output


def implicit_objectives() -> dict[str, Any]:
    mesh = ns.ph.build_mesh(20, 10)
    scales = ns.ns.normalization(mesh, 0.05)
    rng = np.random.default_rng(20260924)
    output: dict[str, Any] = {}
    variants = (
        ("l2-off", "l2", 0.0, 0.0),
        ("normal-off", "normal", 0.05, 0.0),
        ("l2-on", "l2", 0.0, 1.0),
        ("normal-on", "normal", 0.05, 1.0),
    )
    for label, kind, beta, smooth_weight in variants:
        modes = {}
        for mode in sorted(ns.am.MODES):
            q = smooth_controls(mesh, mode)
            _, _, gradient, values = ns.evaluate(
                mesh,
                q,
                mode,
                0.05,
                kind,
                beta,
                smooth_weight,
                scales,
                np.zeros(mesh.nfree),
            )
            derivative = directional_checks(
                lambda trial, mode=mode, kind=kind, beta=beta, smooth_weight=smooth_weight: (
                    ns.evaluate(
                        mesh,
                        trial,
                        mode,
                        0.05,
                        kind,
                        beta,
                        smooth_weight,
                        scales,
                        np.zeros(mesh.nfree),
                    )[3]["objective"]
                ),
                q,
                gradient,
                rng,
            )
            maximum = max(record["relative_error"] for record in derivative)
            assert maximum < 3e-5, (label, mode, derivative)
            modes[mode] = {
                "objective": values["objective"],
                "maximum_relative_derivative_error": maximum,
            }
        output[label] = modes
    return output


def neutral_state_checks() -> dict[str, Any]:
    mesh = ns.ph.build_mesh(100, 10)
    output = {}
    scales = ns.ns.normalization(mesh, ns.HEIGHT)
    for mode in sorted(ns.am.MODES):
        q = ns.am.initialize(int(mesh.muscle.sum()), mode)
        B = ns.gs.study.matrices(mesh, q, mode)
        assert np.array_equal(B, np.broadcast_to(np.eye(2), B.shape))
        modes = {}
        for variant, smooth_weight, kind, beta in ns.VARIANTS:
            state, fitted_B, _, values = ns.evaluate(
                mesh,
                q,
                mode,
                ns.HEIGHT,
                kind,
                beta,
                smooth_weight,
                scales,
                np.zeros(mesh.nfree),
            )
            assert np.array_equal(state.u, np.zeros(mesh.nfree))
            assert np.array_equal(fitted_B, B)
            assert values["roughness"] == 0.0
            assert values["weighted_smoothness_loss"] == 0.0
            modes[variant] = {
                "objective": values["objective"],
                "normal_coefficient": values["normal_coefficient"],
            }
        output[mode] = modes
    return output


def historical_l2_equivalence() -> dict[str, Any]:
    mesh = ns.ph.build_mesh(20, 10)
    output = {}
    for mode in sorted(ns.am.MODES):
        q = smooth_controls(mesh, mode)
        for smooth_weight in (0.0, 1.0):
            scales = ns.ns.normalization(mesh, 0.05)
            _, _, new_gradient, new_values = ns.evaluate(
                mesh,
                q,
                mode,
                0.05,
                "l2",
                0.0,
                smooth_weight,
                scales,
                np.zeros(mesh.nfree),
            )
            _, _, old_gradient, old_values = ns.gs.study.evaluate(
                mesh, q, mode, 0.05, smooth_weight, np.zeros(mesh.nfree)
            )
            assert np.isclose(
                new_values["objective"], old_values["objective"], rtol=1e-12, atol=1e-14
            )
            error = float(np.max(np.abs(new_gradient - old_gradient)))
            assert error <= 1e-12, (mode, smooth_weight, error)
            output[f"{mode}/smooth-{int(smooth_weight)}"] = {
                "gradient_max_abs_error": error
            }
    return output


def main(cfg: Config) -> None:
    report = {
        "thresholds": {
            "direct_relative_derivative": 2e-8,
            "implicit_relative_derivative": 3e-5,
            "l2_historical_gradient_abs": 1e-12,
        },
        "curve_normal_loss": normal_curve_checks(),
        "mapped_normal_loss": mapped_normal_check(),
        "regularizer_pullback": regularizer_pullback_checks(),
        "full_implicit_objectives": implicit_objectives(),
        "neutral_initialization": neutral_state_checks(),
        "historical_l2_equivalence": historical_l2_equivalence(),
        "source_sha256": {
            str(path.resolve()): sha256(path)
            for path in [
                *ns.numerical_sources(),
                Path(__file__),
                Path(__file__).with_name("experiment.py"),
            ]
        },
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
