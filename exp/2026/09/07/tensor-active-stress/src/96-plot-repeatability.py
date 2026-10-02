"""Plot measured full-field optimizer noise against the frozen thresholds."""

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
    summary: Path = ROOT / "data/90-repeatability/summary.json"
    output_dir: Path = cherries.output("96-repeatability-plots", mkdir=True)


def record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main(cfg: Config) -> None:
    global COMPLETED  # noqa: PLW0603
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir())
    summary = json.loads(cfg.summary.read_text())
    assert summary["mode"] == "repeatability_aggregate"
    assert summary["status"] in {"passed", "failed"}
    protocol = Path(summary["source91_protocol"]["path"])
    assert record(protocol)["sha256"] == summary["source91_protocol"]["sha256"]
    colors = ["#2864a5", "#cb6738", "#2864a5", "#cb6738"]
    labels = ["64\nbaseline", "64\ncandidate", "256\nbaseline", "256\ncandidate"]
    rows = []
    for step in ("64", "256"):
        noise = summary["checkpoints"][step]["noise"]
        for arm in ("baseline", "candidate"):
            values = noise["arms"][arm]
            rows.append(
                {
                    "step": int(step),
                    "arm": arm,
                    "noise_over_own_update_percent": 100
                    * values["noise_over_own_sample0_update"],
                    "noise_over_direction_difference_percent": 100
                    * values["noise_over_baseline_candidate_signal"],
                }
            )
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), layout="constrained")
    panels = [
        ("noise_over_own_update_percent", 1.0, "Noise / arm update", "1% limit"),
        (
            "noise_over_direction_difference_percent",
            10.0,
            "Noise / baseline-candidate difference",
            "10% limit",
        ),
    ]
    for axis, (key, threshold, title, caption) in zip(axes, panels, strict=True):
        values = np.asarray([row[key] for row in rows])
        assert np.isfinite(values).all()
        assert (values >= 0).all()
        axis.bar(np.arange(4), values, color=colors, width=0.64)
        axis.axhline(
            threshold, color="#262626", linestyle="--", linewidth=1.1, label=caption
        )
        axis.set(xticks=np.arange(4), xticklabels=labels, ylabel="Percent", title=title)
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
        "Repeatability of the projected physical tensor update", fontsize=14
    )
    artifacts = []
    for extension in ("png", "pdf"):
        path = out / f"repeatability-noise.{extension}"
        figure.savefig(path, dpi=220, facecolor="white")
        artifacts.append(record(path))
        cherries.log_output(path)
    plt.close(figure)
    (out / "plot-values.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
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
        "values": record(out / "plot-values.json"),
        "artifacts": artifacts,
        "definition": "Maximum of all three pairwise full-field projected physical tensor differences. Each arm keeps its sample-0 calibrated settings and exact saved Adam state.",
        "source_gate_passed": summary["gate_passed"],
    }
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    cherries.log_output(out / "summary.json")
    cherries.log_output(out / "plot-values.json")
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
