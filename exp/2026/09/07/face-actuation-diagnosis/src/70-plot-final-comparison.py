"""Render a readable final-comparison figure from frozen data44 summary metrics only."""

# ruff: noqa: EM101, TRY003

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY = ROOT / "data" / "44-final-surface-comparison" / "summary.json"
DEFAULT_OUTPUT = ROOT / "data" / "70-final-comparison-figure"
EXPECTED_IDS = (
    "historical-saved-no-skin",
    "manual-c50-no-skin-baseline",
    "manual-c50-selected-muscle-lame-x10",
    "manual-c50-fat-lame-x0.1",
    "manual-c50-aponeurosis-lame-x0.1",
    "current-raw6-smooth-no-skin",
    "current-region5-no-floor",
    "current-raw6-no-skin",
    "historical-adam-raw6",
    "historical-adam-raw6-s",
)
DISPLAY_NAMES = {
    "historical-saved-no-skin": "Historical saved no-skin",
    "manual-c50-no-skin-baseline": "Manual c50 baseline",
    "manual-c50-selected-muscle-lame-x10": "Manual muscle Lamé x10",
    "manual-c50-fat-lame-x0.1": "Manual fat Lamé x0.1",
    "manual-c50-aponeurosis-lame-x0.1": "Manual aponeurosis Lamé x0.1",
    "current-raw6-smooth-no-skin": "Current Raw6-S · step 9",
    "current-region5-no-floor": "Current Region5 · step 40",
    "current-raw6-no-skin": "Current Raw6 · step 48",
    "historical-adam-raw6": "Matched Raw6 · selected best local 50",
    "historical-adam-raw6-s": "Matched Raw6-S · selected best local 64",
}
CATEGORY_COLORS = {
    "historical_saved": "#5b7282",
    "manual": "#6d8d68",
    "manual_material": "#8b7760",
    "current_inverse": "#a76359",
    "historical_adam": "#7864a0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_rows(summary_path: Path) -> list[dict[str, Any]]:
    document = json.loads(summary_path.read_text())
    rows = document.get("comparison_table")
    if (
        not isinstance(rows, list)
        or tuple(row.get("id") for row in rows) != EXPECTED_IDS
    ):
        raise ValueError("frozen data44 comparison-table order differs")
    return rows


def values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    if key == "fit":
        return np.asarray([float(row["fit_rms_mm"]) for row in rows])
    if key == "motion":
        return np.asarray([float(row["motion_rms_mm"]) for row in rows])
    return np.asarray(
        [
            float(
                row["highpass_rms_mm"]["5mm"]["normal_residual_highpass"]["mouth_10mm"]
            )
            for row in rows
        ]
    )


def draw(rows: list[dict[str, Any]], destination: Path) -> None:
    metrics = (
        ("fit", "Area-weighted fit RMS (mm)"),
        ("motion", "Surface motion RMS (mm)"),
        ("mouth_hp", "Mouth residual high-pass RMS, 5 mm (mm)"),
    )
    y = np.arange(len(rows))
    labels = [DISPLAY_NAMES[row["id"]] for row in rows]
    colors = [CATEGORY_COLORS[row["category"]] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.2), sharey=True)
    fig.subplots_adjust(left=0.19, right=0.99, top=0.9, bottom=0.12, wspace=0.04)
    for axis, (key, title) in zip(axes, metrics, strict=True):
        metric = values(rows, key)
        maximum = max(metric) * 1.22
        axis.barh(y, metric, color=colors, height=0.62)
        axis.set_xlim(0, maximum)
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
        axis.grid(axis="x", alpha=0.22)
        axis.set_axisbelow(True)
        for position, value in zip(y, metric, strict=True):
            axis.text(
                value + maximum * 0.018,
                position,
                f"{value:.3f}",
                va="center",
                fontsize=9,
            )
    axes[0].set_yticks(y, labels, fontsize=10)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Frozen ten-case surface comparison",
        x=0.01,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "Matched Raw6 and Raw6-S are selected-best early continuation states after a shared reset; "
        "they are not convergence or smoothing-success claims. Metrics are frozen data44 values.",
        fontsize=9,
    )
    fig.savefig(destination.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary, output = args.summary.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = load_rows(summary)
    with tempfile.TemporaryDirectory(
        prefix="plot-final-comparison-", dir=ROOT
    ) as temporary:
        stage = Path(temporary) / output.name
        stage.mkdir()
        figure_base = stage / "final-comparison"
        draw(rows, figure_base)
        helper = Path(__file__).resolve()
        manifest = {
            "schema_version": 1,
            "scope": "Presentation-only figure from frozen data44 summary; no numerical solve or audit mutation.",
            "source_summary": record(summary),
            "helper": record(helper),
            "metrics": [
                "area-weighted fit RMS (mm)",
                "surface motion RMS (mm)",
                "mouth normal-residual high-pass RMS at 5 mm (mm)",
            ],
            "case_order": list(EXPECTED_IDS),
            "output_files": {},
        }
        for path in (figure_base.with_suffix(".png"), figure_base.with_suffix(".pdf")):
            manifest["output_files"][path.name] = record(path)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stage), output)


if __name__ == "__main__":
    main()
