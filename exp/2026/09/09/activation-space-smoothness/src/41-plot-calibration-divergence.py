# ruff: noqa: EM101, EM102, TRY003
"""Plot the audited calibration/primary learned-axis trajectory divergence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

GROUP = Path(__file__).resolve().parents[1]
OUTPUT = GROUP / "data/18-calibration-main-divergence-audit"
INPUT = OUTPUT / "trace-comparison.csv"
AUDIT = OUTPUT / "summary.json"
FIGURE = OUTPUT / "calibration-vs-primary-first16.png"
RECEIPT = OUTPUT / "figure-receipt.json"
SNAPSHOT = OUTPUT / "sources/41-plot-calibration-divergence.py"
COLORS = {"calibration": "#687787", "primary": "#b64c35"}
LABELS = {
    "calibration": "Selected calibration pilot",
    "primary": "Primary Axis-off",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def main() -> None:
    for path in (FIGURE, RECEIPT, SNAPSHOT):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    with INPUT.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(17)):
        raise ValueError(f"expected exact audited steps 0..16, got {steps}")

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = (
        (axes[0, 0], "fit_rms_mm", "Fit RMS (mm)", "Target fit", "linear"),
        (axes[0, 1], "smoothness_C", "S(C)", "C variation", "symlog"),
        (
            axes[1, 0],
            "gradient_rms",
            "Fitting-gradient RMS",
            "Adjoint fitting gradient",
            "log",
        ),
        (
            axes[1, 1],
            "inverted_all_cells",
            "Inverted tetrahedra",
            "Mechanical diagnostic",
            "linear",
        ),
    )
    for axis, key, ylabel, title, scale in panels:
        for execution in ("calibration", "primary"):
            values = [float(row[f"{execution}_{key}"]) for row in rows]
            axis.plot(
                steps,
                values,
                color=COLORS[execution],
                label=LABELS[execution],
                linewidth=2.0,
                marker="o",
                markersize=3.0,
            )
        if scale == "symlog":
            positive = [
                float(row[f"{execution}_{key}"])
                for row in rows
                for execution in ("calibration", "primary")
                if float(row[f"{execution}_{key}"]) > 0.0
            ]
            axis.set_yscale("symlog", linthresh=0.5 * min(positive))
        elif scale == "log":
            axis.set_yscale("log")
        axis.set(
            xlabel="Adam updates",
            ylabel=ylabel,
            title=title,
            xlim=(0, 16),
        )
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Same seed, initialization, settings, and shared sources; separate executions",
        fontsize=13,
    )
    figure.savefig(FIGURE, dpi=220)
    plt.close(figure)

    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_bytes(Path(__file__).read_bytes())
    SNAPSHOT.chmod(0o444)
    if digest(SNAPSHOT) != digest(Path(__file__)):
        raise ValueError("saved plotting-source snapshot differs from live source")
    receipt = {
        "status": "completed_cpu_only_audit_figure",
        "scope": (
            "Exact rows 0 through 16 from the saved read-only divergence audit; "
            "no mechanics solve, optimizer update, interpolation, or smoothing"
        ),
        "comparison": (
            "Selected fit-only calibration lr-3 and primary Axis-off use the same "
            "seed, initialization, settings, and shared sources but are separate "
            "GPU executions"
        ),
        "panels": [
            "fit_rms_mm",
            "smoothness_C",
            "gradient_rms",
            "inverted_all_cells",
        ],
        "inputs": {
            "trace_comparison": record(INPUT),
            "audit_summary": record(AUDIT),
        },
        "output": record(FIGURE),
        "source": {
            "snapshot": record(SNAPSHOT),
            "live_at_generation": record(Path(__file__)),
        },
        "command": ("python src/41-plot-calibration-divergence.py"),
    }
    write_json(RECEIPT, receipt)


if __name__ == "__main__":
    main()
