# Copyright (c) 2026 liblaf
# ruff: noqa: C901, EM101, EM102, I001, PLR0912, TRY003
"""Render receipt-derived small-recovery diagnostics without numerical work."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps

from experiment_profile import ProfileCometNoCommit
from liblaf import cherries

ROOT = Path(__file__).resolve().parent.parent
COMPLETED = False


class Config(cherries.BaseConfig):
    """Immutable recovery receipt and an empty diagnostic output directory."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    input_summary: Path = ROOT / "data/70-small-recovery-v4/summary.json"
    output_dir: Path = ROOT / "data/77-optimizer-diagnostics"


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    """Describe one immutable input or generated output."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, value: Any) -> None:
    """Write strict JSON atomically."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def read_trace(path: Path, label: str) -> list[dict[str, float]]:
    """Read the exact real-state fields needed for display."""
    required = {
        "step",
        "raw_displacement_rms_fixture_unit",
        "projected_gradient_mapping_rms",
    }
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or required - set(reader.fieldnames):
            raise ValueError(f"trace schema differs: {label}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"trace is empty: {label}")
    output: list[dict[str, float]] = []
    for index, row in enumerate(rows):
        parsed = {name: float(row[name]) for name in required}
        if not all(np.isfinite(value) for value in parsed.values()):
            raise ValueError(f"trace is nonfinite: {label}:{index}")
        if int(parsed["step"]) != index:
            raise ValueError(f"trace step differs: {label}:{index}")
        output.append(parsed)
    return output


def relative_error(rows: list[dict[str, float]], target_motion: float) -> np.ndarray:
    """Return the declared recovery error divided by target motion."""
    if target_motion <= 0 or not np.isfinite(target_motion):
        raise ValueError("target motion must be finite and positive")
    return np.array(
        [row["raw_displacement_rms_fixture_unit"] / target_motion for row in rows]
    )


def gradient_ratio(rows: list[dict[str, float]]) -> np.ndarray:
    """Normalize projected-gradient mapping by its actual initial state."""
    initial = rows[0]["projected_gradient_mapping_rms"]
    if initial <= 0:
        raise ValueError("initial projected-gradient mapping must be positive")
    return np.array([row["projected_gradient_mapping_rms"] / initial for row in rows])


def save(figure: plt.Figure, output: Path, stem: str) -> dict[str, dict[str, Any]]:
    """Write an inspectable PNG and PDF pair."""
    paths = {"png": output / f"{stem}.png", "pdf": output / f"{stem}.pdf"}
    figure.savefig(paths["png"], dpi=180)
    figure.savefig(paths["pdf"])
    plt.close(figure)
    return {name: record(path) for name, path in paths.items()}


def screen_figure(
    output: Path,
    arms: list[tuple[str, str, list[dict[str, float]]]],
    target_motion: float,
) -> dict[str, dict[str, Any]]:
    """Plot three comparable 0..256 screens without compressing recovery."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for _identifier, label, rows in arms:
        steps = [row["step"] for row in rows]
        axes[0].plot(steps, relative_error(rows, target_motion), label=label)
        axes[1].plot(steps, gradient_ratio(rows), label=label)
    axes[0].set(
        yscale="log",
        xlabel="Actual optimizer step",
        ylabel="Relative displacement error / target motion",
        title="256-update screen: displacement error",
    )
    axes[1].set(
        yscale="log",
        xlabel="Actual optimizer step",
        ylabel="Projected-gradient mapping / step-0 mapping",
        title="256-update screen: projected-gradient ratio",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Small PSD recovery screens: each curve uses saved actual states"
        "\nCandidates' first projected-stress step: 2x baseline",
        fontsize=14,
    )
    return save(figure, output, "screen-256-diagnostics")


def recovery_figure(
    output: Path,
    rows: list[dict[str, float]],
    target_motion: float,
    recalibration_step: int,
    before_lr: float,
    after_lr: float,
) -> dict[str, dict[str, Any]]:
    """Plot the whole recovery with the one permitted rate change visible."""
    steps = np.array([row["step"] for row in rows])
    error = relative_error(rows, target_motion)
    pg = gradient_ratio(rows)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    axes[0].plot(
        steps, error, color="#1769aa", label="saved relative displacement error"
    )
    axes[0].axhline(
        1e-3, color="#d95f02", linestyle="--", label="acceptance gate = 1e-3"
    )
    axes[1].plot(steps, pg, color="#1769aa", label="saved projected-gradient ratio")
    for axis in axes:
        axis.axvline(recalibration_step, color="#7b3294", linestyle="--", linewidth=1.5)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Actual optimizer step")
        axis.set_yscale("log")
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Relative displacement error / target motion")
    axes[1].set_ylabel("Projected-gradient mapping / step-0 mapping")
    axes[0].set_title("Recovery error and acceptance gate")
    axes[1].set_title("Recovery projected-gradient mapping")
    axes[0].scatter([steps[-1]], [error[-1]], color="#1b9e77", zorder=3)
    axes[0].annotate(
        f"actual step {int(steps[-1])}\n{error[-1]:.9f}",
        xy=(steps[-1], error[-1]),
        xytext=(-92, 12),
        textcoords="offset points",
        color="#1b9e77",
        fontsize=8,
    )
    figure.suptitle(
        "Known-reachable PSD recovery: one state-preserving LR recalibration"
        f"\nstep {recalibration_step}: LR {before_lr:.8f} -> {after_lr:.8f}",
        color="#7b3294",
        fontsize=14,
    )
    return save(figure, output, "recovery-full-diagnostics")


def main(cfg: Config) -> None:
    """Render only hash-bound receipt data and preserve every source artifact."""
    global COMPLETED  # noqa: PLW0603
    input_summary = cfg.input_summary.resolve()
    summary = json.loads(input_summary.read_text())
    if summary.get("status") != "passed":
        raise ValueError("small recovery receipt is not passed")
    output = cfg.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cherries.log_input(input_summary)
    arms: list[tuple[str, str, list[dict[str, float]]]] = []
    input_records: dict[str, dict[str, Any]] = {"summary": record(input_summary)}
    for arm in summary.get("arms", []):
        trace = Path(arm["trace"]["path"])
        if record(trace) != arm["trace"]:
            raise ValueError(f"screen trace hash differs: {arm.get('id')}")
        epsilon = float(arm["epsilon"])
        learning_rate = float(arm["learning_rate"])
        label = f"epsilon={epsilon:.0e}, LR={learning_rate:.6g}"
        arms.append((arm["id"], label, read_trace(trace, arm["id"])))
        input_records[arm["id"]] = record(trace)
    if len(arms) != 3 or {name for name, _label, _rows in arms} != {
        "baseline-eps-1e-2",
        "candidate-eps-1e-4",
        "candidate-eps-1e-6",
    }:
        raise ValueError("small screen arms differ")
    recovery = summary["recovery"]
    trace = Path(recovery["trace"]["path"])
    if record(trace) != recovery["trace"]:
        raise ValueError("recovery trace hash differs")
    rows = read_trace(trace, "recovery-candidate-eps-1e-6")
    if (
        int(rows[-1]["step"]) != recovery["final"]["step"]
        or len(rows) != int(rows[-1]["step"]) + 1
    ):
        raise ValueError("recovery trace final witness differs")
    initial = float(recovery["initial"]["raw_displacement_rms_fixture_unit"])
    target_motion = float(summary["generated_target"]["target_motion_rms_fixture_unit"])
    if not np.isclose(initial, target_motion, rtol=0.0, atol=1e-15):
        raise ValueError("recovery initial target-motion normalization differs")
    phase = recovery["phases"]["initial_fixed_phase"]
    recalibration = recovery["recalibrations"]
    if len(recalibration) != 1 or recalibration[0]["step"] != 4096:
        raise ValueError("expected one recalibration at step 4096")
    if (
        phase["learning_rate"] != recovery["learning_rate"]
        or recalibration[0]["previous_learning_rate"] != phase["learning_rate"]
    ):
        raise ValueError("pre-recalibration learning rate differs")
    after_lr = float(recalibration[0]["new_learning_rate"])
    if recovery["phases"]["post_recalibration_phase"]["learning_rate"] != after_lr:
        raise ValueError("post-recalibration learning rate differs")
    input_records["recovery_trace"] = record(trace)
    figures = {
        "screen": screen_figure(output, arms, target_motion),
        "recovery": recovery_figure(
            output, rows, target_motion, 4096, float(phase["learning_rate"]), after_lr
        ),
    }
    result = {
        "schema_version": 1,
        "status": "completed_receipt_render",
        "scope": "receipt-derived diagnostic plots only; no solve, optimizer, or source-data mutation",
        "source": record(Path(__file__)),
        "inputs": input_records,
        "normalization": {
            "relative_displacement_error": "saved raw displacement RMS / saved generated target motion RMS",
            "projected_gradient_ratio": "saved projected-gradient mapping RMS / saved step-0 mapping RMS",
        },
        "recalibration": {
            "step": 4096,
            "previous_learning_rate": phase["learning_rate"],
            "new_learning_rate": after_lr,
            "preserved_state": recalibration[0]["preserved_state"],
        },
        "final_actual_witness": {
            "step": int(rows[-1]["step"]),
            "relative_displacement_error": float(
                relative_error(rows, target_motion)[-1]
            ),
            "acceptance_gate": 1e-3,
        },
        "figures": figures,
    }
    write_json(output / "summary.json", result)
    for value in figures.values():
        for item in value.values():
            cherries.log_output(Path(item["path"]))
    cherries.log_output(output / "summary.json")
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
