"""Plot the completed 20-step loss-normalization pilot endpoints."""

# ruff: noqa: EM101, EM102, TRY003, TRY004

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import comet_ml
import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

GROUP = Path(__file__).resolve().parents[1]
BRANCHES = (
    ("l2", "Pure L2", "#4d4a46"),
    ("gradient", "Pure gradient", "#3b6fa1"),
    ("beta-0.25", "Mixed β=0.25", "#d27b21"),
    ("beta-1", "Mixed β=1", "#b33a3a"),
    ("beta-4", "Mixed β=4", "#6a4c93"),
)


class Comet(plugins.Comet):
    @core.impl
    def start(self) -> None:
        experiment = comet_ml.start(
            project_name=self.run.project_name,
            experiment_config=comet_ml.ExperimentConfig(
                disabled=self.disabled,
                name=self.run.run_name,
                tags=self.run.tags,
                log_env_details=False,
                log_git_patch=False,
                auto_log_co2=False,
            ),
        )
        self.run.log_other("cherries/comet/url", experiment.url)


class ProfileCometNoCommit(profiles.Profile):
    def init(self) -> core.Run:
        run = core.run
        run.plugins.register(Comet(run=run, disabled=os.environ.get("DEBUG") == "1"))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Logging(run=run))
        run.plugins.register(plugins.Local(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    pilot_dir: Path = GROUP / "data/08-loss-pilot"
    selection: Path = GROUP / "data/09-loss-selection/selection.json"
    output_dir: Path = GROUP / "data/18-pilot-figures"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _value(metrics: dict[str, Any], names: tuple[str, ...]) -> float:
    for name in names:
        if name in metrics:
            value = float(metrics[name])
            if np.isfinite(value):
                return value
    raise ValueError(f"metrics lack one of {names}")


def _selected_branch(selection: dict[str, Any]) -> str:
    for name in ("selected_branch", "selected_candidate", "selected_name"):
        value = selection.get(name)
        if isinstance(value, str):
            return value
    beta = selection.get("selected_beta")
    if beta is not None:
        return f"beta-{float(beta):g}"
    raise ValueError("selection lacks selected branch or beta")


def main(cfg: Config) -> None:
    pilot_dir = cherries.input(cfg.pilot_dir.resolve())
    selection_path = cherries.input(cfg.selection.resolve())
    output = cherries.output(cfg.output_dir.resolve(), mkdir=True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selection = _json(selection_path)
    selected = _selected_branch(selection)
    records: list[dict[str, Any]] = []
    for identifier, label, color in BRANCHES:
        summary = _json(pilot_dir / identifier / "summary.json")
        if int(summary.get("last_step", -1)) != 20:
            raise ValueError(f"pilot {identifier} is not a 20-step endpoint")
        metrics = summary.get("last_metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"pilot {identifier} lacks last_metrics")
        records.append(
            {
                "id": identifier,
                "label": label,
                "color": color,
                "selected": identifier == selected,
                "status": str(summary.get("status", "status unavailable")),
                "position_rms_mm": _value(metrics, ("fit_rms_mm", "position_rms_mm")),
                "surface_gradient_rms": _value(
                    metrics, ("surface_gradient_rms", "gradient_rms")
                ),
                "roughness": _value(
                    metrics, ("normalized_tensor_roughness", "activation_smoothness")
                ),
            }
        )
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), constrained_layout=True)
    for item in records:
        axes[0].scatter(
            item["position_rms_mm"],
            item["surface_gradient_rms"],
            color=item["color"],
            marker="*" if item["selected"] else "o",
            s=180 if item["selected"] else 85,
            edgecolor="black" if item["selected"] else "none",
            linewidth=0.8,
            zorder=3,
        )
        axes[0].annotate(
            item["label"],
            (item["position_rms_mm"], item["surface_gradient_rms"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )
    axes[0].set_title("20-step pilot endpoint tradeoff")
    axes[0].set_xlabel("Position residual RMS (mm)")
    axes[0].set_ylabel("Surface-gradient residual RMS (dimensionless)")
    axes[0].grid(alpha=0.25)
    names = [item["label"] for item in records]
    bars = axes[1].bar(
        np.arange(len(records)),
        [item["roughness"] for item in records],
        color=[item["color"] for item in records],
        edgecolor=["black" if item["selected"] else "none" for item in records],
        linewidth=1.2,
    )
    for bar, item in zip(bars, records, strict=True):
        if item["selected"]:
            bar.set_hatch("//")
    axes[1].set_title("Normalized tensor roughness at step 20")
    axes[1].set_ylabel("Dimensionless")
    axes[1].set_xticks(np.arange(len(records)), names, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Pilot endpoints only; not convergence evidence", fontsize=14)
    image = output / "pilot-tradeoff.png"
    figure.savefig(image, dpi=180, facecolor="white")
    plt.close(figure)
    source_copy = output / "sources" / Path(__file__).name
    source_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), source_copy)
    receipt = {
        "status": "completed_20_step_pilot_rendering",
        "scope": "Read-only visualization of saved 20-step pilot endpoints; no convergence claim or solver call.",
        "selection": selection,
        "selected_branch": selected,
        "records": records,
        "source_receipt": {
            "executed": str(Path(__file__).resolve()),
            "snapshot": str(source_copy.resolve()),
            "same_bytes": source_copy.read_bytes() == Path(__file__).read_bytes(),
        },
        "runtime_receipt": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "debug": os.environ.get("DEBUG") == "1",
        },
    }
    summary = output / "summary.json"
    summary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    cherries.log_output(image)
    cherries.log_output(summary)
    cherries.log_metrics(
        {f"{item['id']}/position_rms_mm": item["position_rms_mm"] for item in records}
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
