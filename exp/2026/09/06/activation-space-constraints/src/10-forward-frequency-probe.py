"""Forward probe of how tetrahedral activation frequency reaches the surface."""

# ruff: noqa: PLR0915

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, override

import matplotlib as mpl

mpl.use("Agg")

import activation_models as activation
import block_physics as block
import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps
import torch
from liblaf.cherries import core, plugins, profiles
from scipy.ndimage import gaussian_filter

from liblaf import cherries

logger = logging.getLogger(__name__)


class ProfileLocalComet(profiles.Profile):
    """Normal remote logging with read-only Git provenance."""

    @override
    def init(self) -> core.Run:
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    output_dir: Path = cherries.output("10-frequency", mkdir=True)
    nx: int = 48
    ny: int = 10
    mean_log_contraction: float = 0.15
    modulation_rms_values: tuple[float, ...] = (0.015, 0.03)
    wave_numbers: tuple[int, ...] = (1, 4)
    highpass_lengths: tuple[float, ...] = (0.03, 0.06, 0.12)


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    mean = float(np.average(values, weights=weights))
    rms = float(np.sqrt(np.average(values * values, weights=weights)))
    centered_rms = float(np.sqrt(np.average(np.square(values - mean), weights=weights)))
    return {"mean": mean, "rms": rms, "centered_rms": centered_rms}


def highpass(field: np.ndarray, ell: float, dx: float) -> np.ndarray:
    return field - gaussian_filter(field, sigma=ell / dx, mode="reflect")


def surface_metrics(
    field: np.ndarray,
    weights: np.ndarray,
    dx: float,
    highpass_lengths: tuple[float, ...],
) -> dict[str, float]:
    metrics = {
        f"response/{key}": value
        for key, value in weighted_stats(field, weights).items()
    }
    gx = np.gradient(field, dx, axis=1)
    gz = np.gradient(field, dx, axis=0)
    metrics["slope_rms"] = float(
        np.sqrt(np.average(gx * gx + gz * gz, weights=weights))
    )
    lap = (
        field[1:-1, 2:]
        + field[1:-1, :-2]
        + field[2:, 1:-1]
        + field[:-2, 1:-1]
        - 4.0 * field[1:-1, 1:-1]
    ) / dx**2
    metrics["laplacian_rms"] = float(
        np.sqrt(np.average(lap * lap, weights=weights[1:-1, 1:-1]))
    )
    for ell in highpass_lengths:
        hp = highpass(field, ell, dx)
        metrics[f"highpass_rms/ell_{ell:g}"] = float(
            np.sqrt(np.average(hp * hp, weights=weights))
        )
    return metrics


def make_pattern(
    centers: np.ndarray, volumes: np.ndarray, wave_number: int
) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.cos(2.0 * np.pi * wave_number * centers[:, 0]) * np.cos(
        2.0 * np.pi * wave_number * centers[:, 2]
    )
    centered = raw - np.average(raw, weights=volumes)
    scale = np.sqrt(np.average(centered * centered, weights=volumes))
    assert scale > 0.0
    pattern = centered / scale
    stats = weighted_stats(pattern, volumes)
    assert abs(stats["mean"]) < 1.0e-12
    assert math.isclose(stats["rms"], 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12)
    return pattern, stats


def activation_values(
    a: np.ndarray, fibers: torch.Tensor
) -> tuple[torch.Tensor, np.ndarray]:
    q = torch.as_tensor(a[:, None])
    ainv, _ = activation.matrices(q, "F", fibers)
    return activation.packed(ainv), ainv.detach().cpu().numpy()


def plot_summary(rows: list[dict[str, Any]], output: Path) -> None:
    selected = [row for row in rows if row["case"] != "uniform"]
    labels = [row["case"] for row in selected]
    x = np.arange(len(selected))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    for location, key, title in (
        ("top", "highpass_rms/ell_0.06", "Top surface"),
        ("interface", "highpass_rms/ell_0.06", "Muscle upper interface"),
    ):
        axis = axes[0] if location == "top" else axes[1]
        values = [row[location][key] for row in selected]
        axis.bar(x, values, color="#4477aa")
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_ylabel("high-pass normal response RMS")
        axis.set_title(f"{title}, smoothing length 0.06 L")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_maps(
    maps: dict[str, np.ndarray], output: Path, *, ell: float, dx: float
) -> None:
    names = sorted(maps)
    limit = max(float(np.max(np.abs(highpass(maps[name], ell, dx)))) for name in names)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.5), constrained_layout=True)
    image = None
    for axis, name in zip(axes.ravel(), names, strict=True):
        image = axis.imshow(
            highpass(maps[name], ell, dx),
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title(name)
        axis.set_xlabel("x / L")
        axis.set_ylabel("z / L")
    assert image is not None
    fig.colorbar(image, ax=axes, label="high-pass top normal response")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def experiment(cfg: Config) -> None:
    activation.validate()
    block.configure()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    physics = block.Physics(nx=cfg.nx, ny=cfg.ny)
    assert len(physics.top) == (cfg.nx + 1) ** 2
    dx = 1.0 / cfg.nx
    top_weights = physics.weights.reshape(cfg.nx + 1, cfg.nx + 1)
    interface = np.flatnonzero(np.isclose(physics.points[:, 1], 0.06))
    assert len(interface) == len(physics.top)
    fibers = torch.zeros((len(physics.ids), 3))
    fibers[:, 0] = 1.0
    zero_seed = np.zeros_like(physics.points)

    uniform_a = np.full(len(physics.ids), cfg.mean_log_contraction)
    uniform_packed, uniform_ainv = activation_values(uniform_a, fibers)
    start = time.perf_counter()
    uniform_u_t = physics.solve(uniform_packed, seed=zero_seed)
    uniform_elapsed = time.perf_counter() - start
    uniform_u = uniform_u_t.detach().cpu().numpy()
    uniform_forward = dict(physics.last_forward)

    rows: list[dict[str, Any]] = []
    uniform_det = physics.determinants(uniform_u, uniform_ainv)
    rows.append(
        {
            "case": "uniform",
            "wave_number": 0,
            "modulation_rms": 0.0,
            "activation": weighted_stats(uniform_a, physics.volumes),
            "continuation_elapsed_s": uniform_elapsed,
            "continuation_forward": uniform_forward,
            "reset_elapsed_s": uniform_elapsed,
            "reset_forward": uniform_forward,
            "branch_top_rms": 0.0,
            "branch_over_signal": 0.0,
            "determinants": {
                "min_det_f": float(uniform_det[0].min()),
                "min_det_ainv": float(uniform_det[1].min()),
                "min_det_g": float(uniform_det[2].min()),
            },
            "top": surface_metrics(
                np.zeros((cfg.nx + 1, cfg.nx + 1)),
                top_weights,
                dx,
                cfg.highpass_lengths,
            ),
            "interface": surface_metrics(
                np.zeros((cfg.nx + 1, cfg.nx + 1)),
                top_weights,
                dx,
                cfg.highpass_lengths,
            ),
        }
    )
    case_dir = cfg.output_dir / "uniform"
    case_dir.mkdir()
    physics.save_mesh(case_dir / "continuation.vtu", uniform_u, uniform_ainv)

    response_maps: dict[str, np.ndarray] = {}
    for b in cfg.modulation_rms_values:
        for wave_number in cfg.wave_numbers:
            name = f"k{wave_number:02d}-b{b:.3f}".replace(".", "p")
            pattern, pattern_stats = make_pattern(
                physics.centers, physics.volumes, wave_number
            )
            a = uniform_a + b * pattern
            assert np.all(a >= 0.0)
            packed, ainv = activation_values(a, fibers)

            start = time.perf_counter()
            continuation_t = physics.solve(packed, seed=uniform_u)
            continuation_elapsed = time.perf_counter() - start
            continuation = continuation_t.detach().cpu().numpy()
            continuation_forward = dict(physics.last_forward)
            start = time.perf_counter()
            reset_t = physics.solve(packed, seed=zero_seed)
            reset_elapsed = time.perf_counter() - start
            reset = reset_t.detach().cpu().numpy()
            reset_forward = dict(physics.last_forward)

            top_response = (
                continuation[physics.top, 1] - uniform_u[physics.top, 1]
            ).reshape(cfg.nx + 1, cfg.nx + 1)
            interface_response = (
                continuation[interface, 1] - uniform_u[interface, 1]
            ).reshape(cfg.nx + 1, cfg.nx + 1)
            branch = (reset[physics.top, 1] - continuation[physics.top, 1]).reshape(
                cfg.nx + 1, cfg.nx + 1
            )
            branch_rms = float(
                np.sqrt(np.average(branch * branch, weights=top_weights))
            )
            signal_rms = float(
                np.sqrt(np.average(top_response * top_response, weights=top_weights))
            )
            det = physics.determinants(continuation, ainv)
            row = {
                "case": name,
                "wave_number": wave_number,
                "modulation_rms": b,
                "pattern": pattern_stats,
                "activation": weighted_stats(a, physics.volumes),
                "activation_min": float(a.min()),
                "activation_max": float(a.max()),
                "continuation_elapsed_s": continuation_elapsed,
                "continuation_forward": continuation_forward,
                "reset_elapsed_s": reset_elapsed,
                "reset_forward": reset_forward,
                "branch_top_rms": branch_rms,
                "branch_over_signal": branch_rms / max(signal_rms, 1.0e-30),
                "determinants": {
                    "min_det_f": float(det[0].min()),
                    "min_det_ainv": float(det[1].min()),
                    "min_det_g": float(det[2].min()),
                },
                "top": surface_metrics(
                    top_response, top_weights, dx, cfg.highpass_lengths
                ),
                "interface": surface_metrics(
                    interface_response, top_weights, dx, cfg.highpass_lengths
                ),
            }
            rows.append(row)
            cherries.set_step(len(rows) - 1)
            cherries.log_metrics(
                {
                    f"{name}/top_highpass_ell006": row["top"]["highpass_rms/ell_0.06"],
                    f"{name}/interface_highpass_ell006": row["interface"][
                        "highpass_rms/ell_0.06"
                    ],
                    f"{name}/branch_over_signal": row["branch_over_signal"],
                    f"{name}/min_det_f": row["determinants"]["min_det_f"],
                }
            )
            case_dir = cfg.output_dir / name
            case_dir.mkdir()
            np.savez_compressed(
                case_dir / "fields.npz",
                activation=a,
                pattern=pattern,
                top_response=top_response,
                interface_response=interface_response,
                reset_minus_continuation=branch,
            )
            physics.save_mesh(case_dir / "continuation.vtu", continuation, ainv)
            response_maps[name] = top_response
            logger.info(
                "%s: top HP(.06)=%.6g branch/signal=%.3g min detF=%.6g",
                name,
                row["top"]["highpass_rms/ell_0.06"],
                row["branch_over_signal"],
                row["determinants"]["min_det_f"],
            )

    summary = {
        "schema_version": 1,
        "design": "forward-frequency-probe-isochoric-fiber-active-strain",
        "physics": {
            "nx": cfg.nx,
            "ny": cfg.ny,
            "n_points": len(physics.points),
            "n_tets": len(physics.tets),
            "n_active_tets": len(physics.ids),
            "fixed": "bottom only; top and sides free",
            "elasticity": "production StableNeoHookeanActive; G=F@A_inv",
            "fiber": [1.0, 0.0, 0.0],
            "mean_log_contraction": cfg.mean_log_contraction,
            "activation_map": "A_inv=exp(a)P+exp(-a/2)(I-P)",
            "mesh_points_sha256": block.array_hash(physics.points),
            "mesh_tets_sha256": block.array_hash(physics.tets),
            "active_ids_sha256": block.array_hash(physics.ids),
        },
        "analysis": {
            "normal_direction": "world y",
            "surface_weighting": "exact lumped reference triangle area",
            "highpass": "field - Gaussian(field), reflect boundary",
            "highpass_lengths": list(cfg.highpass_lengths),
            "interface_y": 0.06,
        },
        "solve_count": physics.solve_count,
        "cases": rows,
    }
    write_json(cfg.output_dir / "summary.json", summary)
    plot_summary(rows, cfg.output_dir / "frequency-response.png")
    plot_maps(
        response_maps,
        cfg.output_dir / "top-highpass-response-maps.png",
        ell=0.06,
        dx=dx,
    )
    logger.info(
        "Wrote %s with %d equilibrium solves", cfg.output_dir, physics.solve_count
    )


if __name__ == "__main__":
    profile: str | type[profiles.Profile]
    profile = "debug" if os.getenv("DEBUG") else ProfileLocalComet
    cherries.main(experiment, profile=profile)
