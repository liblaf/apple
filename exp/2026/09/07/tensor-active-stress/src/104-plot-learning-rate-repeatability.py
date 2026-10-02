# ruff: noqa: C901, EM101, EM102, PLR0915, TRY003
"""Plot only the step-512 learning-rate repeatability evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

ROOT = Path(__file__).resolve().parent.parent
COMPLETED = False


class Config(cherries.BaseConfig):
    summary: Path = ROOT / "data/100-repeatability/summary.json"
    output_dir: Path = cherries.output(
        "104-learning-rate-repeatability-plots", mkdir=True
    )


def record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main(cfg: Config) -> None:
    """Render the two declared noise ratios without recomputing any measurement."""
    global COMPLETED  # noqa: PLW0603
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"output directory must be empty: {out}")
    summary = json.loads(cfg.summary.read_text(encoding="utf-8"))
    if summary.get("mode") != "learning_rate_repeatability_aggregate":
        raise ValueError("wrong aggregate mode")
    if summary.get("status") not in {"passed", "failed"}:
        raise ValueError("aggregate status is not final")
    protocol = Path(summary["source101_protocol"]["path"])
    if record(protocol)["sha256"] != summary["source101_protocol"]["sha256"]:
        raise ValueError("protocol hash differs")
    checkpoints = summary.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {"512"}:
        raise ValueError("only checkpoint 512 is permitted")
    arms = checkpoints["512"].get("noise", {}).get("arms", {})
    if set(arms) != {"baseline", "selected"}:
        raise ValueError("expected baseline and larger-rate arms")
    rows = [
        {
            "step": 512,
            "arm": arm,
            "noise_over_own_update_percent": 100
            * float(values["noise_over_own_sample0_update"]),
            "noise_over_direction_difference_percent": 100
            * float(values["noise_over_baseline_candidate_signal"]),
        }
        for arm in ("baseline", "selected")
        for values in (arms[arm],)
    ]
    labels = ["LR 0.3\nbaseline", "LR 0.6\nlarger rate"]
    colors = ["#2864a5", "#cb6738"]
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 4.4), layout="constrained")
    panels = (
        ("noise_over_own_update_percent", 1.0, "Noise / arm update", "1% limit"),
        (
            "noise_over_direction_difference_percent",
            10.0,
            "Noise / LR 0.3 vs LR 0.6 difference",
            "10% limit",
        ),
    )
    for axis, (key, threshold, title, caption) in zip(axes, panels, strict=True):
        values = np.asarray([row[key] for row in rows])
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"invalid {key}")
        axis.bar(np.arange(2), values, color=colors, width=0.64)
        axis.axhline(
            threshold, color="#262626", linestyle="--", linewidth=1.1, label=caption
        )
        axis.set(xticks=np.arange(2), xticklabels=labels, ylabel="Percent", title=title)
        axis.set_ylim(0, max(threshold * 1.22, float(values.max()) * 1.25))
        for index, value in enumerate(values):
            axis.annotate(
                f"{value:.3g}%",
                (index, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        axis.legend(frameon=False, loc="upper right")
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    figure.suptitle(
        "Step-512 repeatability for the two learning-rate probes", fontsize=14
    )
    artifacts = []
    for extension in ("png", "pdf"):
        artifact = out / f"learning-rate-repeatability.{extension}"
        figure.savefig(artifact, dpi=220, facecolor="white")
        artifacts.append(record(artifact))
        cherries.log_output(artifact)
    plt.close(figure)
    values_path = out / "plot-values.json"
    values_path.write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    sources = out / "sources"
    sources.mkdir()
    shutil.copyfile(Path(__file__), sources / Path(__file__).name)
    shutil.copyfile(protocol, sources / protocol.name)
    result = {
        "status": "completed_postprocessing",
        "input": record(cfg.summary),
        "protocol": record(protocol),
        "source": record(sources / Path(__file__).name),
        "values": record(values_path),
        "artifacts": artifacts,
        "source_gate_passed": summary["gate_passed"],
    }
    result_path = out / "summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    cherries.log_output(values_path)
    cherries.log_output(result_path)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
