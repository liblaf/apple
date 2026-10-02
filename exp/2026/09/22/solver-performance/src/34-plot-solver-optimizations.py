"""Plot completed fixed-state adjoint and fixed-proposal forward replays.

This postprocessor deliberately fails when any requested receipt is absent or
incomplete.  It does not rerun a solve, infer a missing variant, or treat a
physical failure as a timing success.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from liblaf import cherries

EXPERIMENT = Path(__file__).resolve().parent.parent


class Config(cherries.BaseConfig):
    cpu_baseline: Path = EXPERIMENT / "data/adjoint-tolerance-baseline20-001"
    cpu_zero: Path = EXPERIMENT / "data/adjoint-tolerance-zero19-001"
    gpu_baseline: Path = EXPERIMENT / "data/adjoint-tolerance-baseline20-gpu-002"
    gpu_zero: Path = EXPERIMENT / "data/adjoint-tolerance-zero19-gpu-002"
    forward_easy: Path = EXPERIMENT / "data/forward-optimizations-easy-002"
    forward_hard: Path = EXPERIMENT / "data/forward-optimizations-hard-001"
    output_dir: Path = EXPERIMENT / "data/solver-optimization-visuals-001"


@dataclass(frozen=True)
class AdjointSeries:
    label: str
    group: str
    backend: str
    checkpoint_sha256: str
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ForwardVariant:
    difficulty: str
    variant: str
    seconds: float
    physical_success: bool
    agreement_pass: bool
    error: str | None


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    assert path.is_file(), f"required artifact is missing: {path}"
    return json.loads(path.read_text())


def require_adjoint(
    path: Path,
    *,
    label: str,
    group: str,
    receipt_arm: str,
    backend: str,
) -> AdjointSeries:
    summary_path = path / "summary.json"
    summary = read_json(summary_path)
    assert summary["schema"] == "smile-fixed-state-adjoint-tolerance-sweep-v1"
    assert summary["success"] is True, f"incomplete adjoint sweep: {path}"
    assert summary["reference_rtol"] == 1e-8
    assert len(summary["arms"]) == 1, summary["arms"].keys()
    received_arm, arm_data = next(iter(summary["arms"].items()))
    rows = tuple(arm_data["rows"])
    assert len(rows) == len(summary["tolerances"])
    assert tuple(row["rtol_requested"] for row in rows) == tuple(summary["tolerances"])
    assert all(row["success"] is True for row in rows)
    assert all(row["no_primal"]["forward_count"] == 0 for row in rows)
    assert all(row["adjoint"]["success"] is True for row in rows)
    checkpoint_sha256 = rows[0]["checkpoint"]["sha256"]
    assert all(row["checkpoint"]["sha256"] == checkpoint_sha256 for row in rows)
    assert received_arm == receipt_arm, (received_arm, receipt_arm)
    return AdjointSeries(label, group, backend, checkpoint_sha256, rows)


def require_forward(
    path: Path, *, difficulty: str
) -> tuple[list[ForwardVariant], dict[str, str]]:
    summary_path = path / "summary.json"
    results_path = path / "results.json"
    protocol_path = path / "protocol.json"
    summary = read_json(summary_path)
    results = read_json(results_path)
    protocol = read_json(protocol_path)
    assert summary["schema"] == "fixed-adam-proposal-forward-optimizations-v1"
    assert protocol["schema"] == summary["schema"]
    assert summary["results"] == results
    assert results, f"forward receipt has no variants: {path}"
    variants = []
    for row in results:
        assert isinstance(row["success"], bool)
        assert np.isfinite(row["seconds"])
        assert row["seconds"] >= 0
        agreement = row.get("agreement", {})
        variants.append(
            ForwardVariant(
                difficulty=difficulty,
                variant=row["variant"],
                seconds=float(row["seconds"]),
                physical_success=row["success"],
                agreement_pass=agreement.get("pass") is True,
                error=row.get("failure"),
            )
        )
    return variants, {
        "summary.json": sha256(summary_path),
        "results.json": sha256(results_path),
        "protocol.json": sha256(protocol_path),
    }


def comparison(row: dict[str, Any], key: str) -> float:
    value = row["comparison_to_reference"][key]["relative_l2"]
    assert value is not None
    assert value >= 0
    return float(value)


def plot_adjoint(
    ax: plt.Axes, series: list[AdjointSeries], value: str, ylabel: str
) -> None:
    markers = {"CPU": "o", "GPU contact": "s"}
    colors = {"baseline20": "#1f77b4", "zero19": "#d62728"}
    for item in series:
        x = np.asarray([row["rtol_requested"] for row in item.rows])
        if value == "seconds":
            y = np.asarray([row["owned_adjoint_seconds"] for row in item.rows])
        else:
            y = np.asarray([comparison(row, value) for row in item.rows])
            # Exact reference equality is meaningful.  A log axis needs a
            # display floor; preserve the unmodified values in summary.json.
            y = np.maximum(y, 1e-18)
        ax.plot(
            x,
            y,
            color=colors[item.group],
            marker=markers[item.backend],
            linestyle="-" if len(item.rows) == 5 else "--",
            label=item.label,
        )
    ax.set_xscale("log")
    if value != "seconds":
        ax.set_yscale("log")
        ax.text(
            0.02, 0.04, "exact zero shown at 1e-18", transform=ax.transAxes, fontsize=8
        )
    ax.set_xlabel("requested adjoint relative tolerance")
    ax.set_ylabel(ylabel)
    ax.grid(visible=True, which="both", alpha=0.25)


def plot_forward(ax: plt.Axes, variants: list[ForwardVariant]) -> None:
    colors = {
        "reset_cpu": "#4c78a8",
        "reuse_cpu": "#72b7b2",
        "reset_gpu": "#f58518",
        "reuse_gpu": "#eeca3b",
    }
    labels = [f"{row.difficulty}\n{row.variant.replace('_', ' ')}" for row in variants]
    bars = ax.bar(
        np.arange(len(variants)),
        [row.seconds for row in variants],
        color=[colors.get(row.variant, "#999999") for row in variants],
    )
    for bar, row in zip(bars, variants, strict=True):
        if not (row.physical_success and row.agreement_pass):
            if row.physical_success:
                status = "agreement failed"
            elif row.agreement_pass:
                status = "physical failed"
            else:
                status = "physical +\nagreement failed"
            bar.set_hatch("//")
            bar.set_edgecolor("#b00020")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                status,
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7,
                color="#b00020",
            )
    ax.set_xticks(np.arange(len(variants)), labels, rotation=28, ha="right")
    ax.set_ylabel("forward wall time (s)")
    ax.set_ylim(top=max(row.seconds for row in variants) * 1.22)
    ax.set_title(
        "Fixed projected-Adam proposal; hatched = failed physical or agreement"
    )
    ax.grid(visible=True, axis="y", alpha=0.25)


def main(cfg: Config) -> None:
    assert not cfg.output_dir.exists(), cfg.output_dir
    series = [
        require_adjoint(
            cfg.cpu_baseline,
            label="baseline20 / CPU",
            group="baseline20",
            receipt_arm="comparison-step-00020",
            backend="CPU",
        ),
        require_adjoint(
            cfg.gpu_baseline,
            label="baseline20 / GPU contact",
            group="baseline20",
            receipt_arm="comparison-step-00020",
            backend="GPU contact",
        ),
        require_adjoint(
            cfg.cpu_zero,
            label="zero19 / CPU",
            group="zero19",
            receipt_arm="latest",
            backend="CPU",
        ),
        require_adjoint(
            cfg.gpu_zero,
            label="zero19 / GPU contact",
            group="zero19",
            receipt_arm="latest",
            backend="GPU contact",
        ),
    ]
    for arm in ("baseline20", "zero19"):
        cpu, gpu = (item for item in series if item.group == arm)
        assert cpu.checkpoint_sha256 == gpu.checkpoint_sha256, arm
    easy, easy_hashes = require_forward(cfg.forward_easy, difficulty="easy")
    hard, hard_hashes = require_forward(cfg.forward_hard, difficulty="hard")
    cfg.output_dir.mkdir(parents=True)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    plot_adjoint(axes[0, 0], series, "seconds", "owned adjoint time (s)")
    plot_adjoint(
        axes[0, 1], series, "data_gradient_q", "relative data-q gradient error"
    )
    plot_adjoint(
        axes[1, 0],
        series,
        "next_projected_adam_q_update",
        "relative next Adam q-update error",
    )
    plot_forward(axes[1, 1], [*easy, *hard])
    axes[0, 0].set_title("Adjoint cost")
    axes[0, 1].set_title("Data-q sensitivity to tolerance")
    axes[1, 0].set_title("Projected Adam q-update sensitivity")
    axes[0, 0].legend(fontsize=8)
    png = cfg.output_dir / "solver-optimizations.png"
    figure.savefig(png, dpi=220)
    plt.close(figure)
    series_paths = {
        "baseline20 / CPU": cfg.cpu_baseline,
        "baseline20 / GPU contact": cfg.gpu_baseline,
        "zero19 / CPU": cfg.cpu_zero,
        "zero19 / GPU contact": cfg.gpu_zero,
    }
    source_hashes = {
        "script": sha256(Path(__file__)),
        **{
            f"{item.label}/summary.json": sha256(
                series_paths[item.label] / "summary.json"
            )
            for item in series
        },
        **{f"easy/{key}": value for key, value in easy_hashes.items()},
        **{f"hard/{key}": value for key, value in hard_hashes.items()},
    }
    summary = {
        "schema": "solver-optimization-visuals-v1",
        "success": True,
        "scope": "postprocessing only; no primal or adjoint solve was run",
        "source_hashes": source_hashes,
        "adjoint": {
            item.label: [
                {
                    "rtol": row["rtol_requested"],
                    "owned_adjoint_seconds": row["owned_adjoint_seconds"],
                    "data_q_relative_l2": comparison(row, "data_gradient_q"),
                    "q_update_relative_l2": comparison(
                        row, "next_projected_adam_q_update"
                    ),
                }
                for row in item.rows
            ]
            for item in series
        },
        "forward": [row.__dict__ for row in [*easy, *hard]],
        "figures": {"solver_optimizations": png.name},
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    cherries.log_metrics(
        {
            "adjoint_series": len(series),
            "forward_variants": len(easy) + len(hard),
        }
    )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main)
