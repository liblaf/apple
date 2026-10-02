"""CPU-only plots and face plates for the saved unrestricted L2 fit."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pydantic_settings as ps
from experiment import Profile
from run_support import receipt, write_json

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    source: Path = Path("41-l2-unrestricted-inverse-v2")
    output: Path = Path("42-l2-unrestricted-figures-v2")
    render: bool = True
    error_limit_mm: float = 10.0


def _display():
    path = Path(__file__).with_name("50-render-stage.py")
    spec = importlib.util.spec_from_file_location("single_fit_display", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _number(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    assert value not in (None, ""), key
    return float(value)


def plot(rows: list[dict[str, str]], out: Path) -> dict[str, int]:
    display = _display()
    steps = np.array([_number(row, "step") for row in rows])
    rms = np.array([_number(row, "fit_rms_mm") for row in rows])
    figure, axes = display.plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    figure.patch.set_facecolor(display.BACKGROUND)
    axes[0].plot(steps, rms, color="#1878b5", marker="o", label="position RMS")
    axes[0].set_ylabel("mm")
    axes[0].set_title("Unrestricted active-stress L2 fit")
    axes[0].grid(alpha=0.25)
    for key, label, color in (
        ("objective", "objective", "#263d52"),
        ("position_contribution", "position", "#1878b5"),
        ("normal_contribution", "normal", "#d56820"),
        ("regularizer_contribution", "smoothness", "#6d4b9a"),
    ):
        if key in rows[0]:
            axes[1].plot(
                steps,
                [_number(row, key) for row in rows],
                marker="o",
                label=label,
                color=color,
            )
    approximate = np.array(
        [row.get("solver_valid", "True").lower() != "true" for row in rows]
    )
    for axis in axes:
        axis.scatter(
            steps[approximate],
            np.interp(steps[approximate], steps, axis.lines[0].get_ydata()),
            marker="x",
            color="#b23a48",
            label="approximate solve" if axis is axes[0] else None,
        )
    axes[1].set_xlabel("attempted step")
    axes[1].set_ylabel("dimensionless contribution")
    axes[1].grid(alpha=0.25)
    axes[0].legend()
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(out / "plot.png", dpi=180, facecolor=display.BACKGROUND)
    display.plt.close(figure)
    return {"trace_rows": len(rows), "approximate_rows": int(approximate.sum())}


def render(source: Path, out: Path, error_limit_mm: float) -> list[str]:
    display = _display()
    stage = source / "l2-symmetric6"
    with (
        np.load(source / "mesh.npz", allow_pickle=False) as mesh,
        np.load(stage / "last.npz", allow_pickle=False) as state,
    ):
        rest = mesh["rest_points"]
        skin_ids = mesh["skin_ids"]
        triangles = mesh["triangles"]
        target = rest[skin_ids] + mesh["target_displacement_skin"]
        fit = rest[skin_ids] + state["u"][skin_ids]
    fit_mesh = display.poly(fit, triangles)
    fit_mesh.point_data["PositionErrorMM"] = 1000 * np.linalg.norm(fit - target, axis=1)
    target_mesh = display.poly(target, triangles)
    cameras = {
        item["id"]: item["camera"]
        for item in json.loads(display.CAMERA_RECEIPT.read_text())["views"]
    }
    views = {
        "full": dict(cameras["side-context"]),
        "mouth": dict(cameras["region1-mouth-corner"]),
    }
    views["full"]["parallel_scale"] *= 1.12
    files: list[str] = []
    for name, camera in views.items():
        paths = [out / f"{name}-{kind}.png" for kind in ("target", "fit", "error")]
        display.snapshot(target_mesh, camera, paths[0])
        display.snapshot(fit_mesh, camera, paths[1])
        display.snapshot(fit_mesh, camera, paths[2], error=True, limit=error_limit_mm)
        plate = out / f"{name}-face.png"
        display.plate(
            paths,
            ["Target", "Fit", "Position error"],
            f"Unrestricted L2 | {name}",
            plate,
        )
        files.extend(str(path.name) for path in (*paths, plate))
    return files


def main(cfg: Config) -> None:
    assert cfg.error_limit_mm > 0
    source = cherries.input(cfg.source)
    stage = source / "l2-symmetric6"
    trace = list(csv.DictReader((stage / "trace.csv").open()))
    assert trace
    summary = json.loads((stage / "summary.json").read_text())
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    counts = plot(trace, out)
    rendered = render(source, out, cfg.error_limit_mm) if cfg.render else []
    metadata = {
        "source": receipt(source / "protocol.json"),
        "stage_summary": receipt(stage / "summary.json"),
        "last_state": receipt(stage / "last.npz"),
        "plot": receipt(out / "plot.png"),
        "rendered": rendered,
        "counts": counts,
        "stage": summary,
    }
    write_json(out / "summary.json", metadata)
    cherries.log_metrics(
        {
            **counts,
            **{
                key: value
                for key, value in summary.items()
                if isinstance(value, (bool, int, float, str))
            },
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
