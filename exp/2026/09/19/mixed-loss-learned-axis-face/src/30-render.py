"""Render matched learned-axis contraction-only fits with smoothness off and on."""

# ruff: noqa: C901, EM101, EM102, PLR0915, TRY003, TRY004

from __future__ import annotations

import csv
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
import pyvista as pv
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
CAMERA_RECEIPT = (
    ROOT / "exp/2026/09/08/physical-volume-closeups/data/20-regions/summary.json"
)
BACKGROUND = "#f4f2ed"
WINDOW = (1600, 1600)
SHAPE_COLOR = "#aeb7ba"


class Comet(plugins.Comet):
    """Comet recorder that avoids slow environment and Git-patch collection."""

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
    comparison_dir: Path = GROUP / "data/20-comparison"
    output_dir: Path = GROUP / "data/30-figures"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _mesh(points: np.ndarray, triangles: np.ndarray) -> pv.PolyData:
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("invalid skin triangle array")
    faces = np.column_stack(
        (np.full(len(triangles), 3, dtype=np.int64), triangles)
    ).ravel()
    return pv.PolyData(np.asarray(points, dtype=np.float64), faces)


def _set_camera(plotter: pv.Plotter, camera: dict[str, Any]) -> None:
    plotter.enable_parallel_projection()
    plotter.camera.position = camera["position"]
    plotter.camera.focal_point = camera["focal_point"]
    plotter.camera.up = camera["view_up"]
    plotter.camera.parallel_scale = camera["parallel_scale"]
    plotter.set_background(BACKGROUND)
    plotter.reset_camera_clipping_range()


def _add_lights(plotter: pv.Plotter, camera: dict[str, Any]) -> None:
    focus = np.asarray(camera["focal_point"], dtype=np.float64)
    backward = np.asarray(camera["position"], dtype=np.float64) - focus
    backward /= np.linalg.norm(backward)
    right = np.cross(np.asarray(camera["view_up"], dtype=np.float64), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    for position, intensity in (
        (focus + 0.3 * (0.72 * right + 0.35 * up + 0.60 * backward), 0.85),
        (focus + 0.3 * backward, 0.20),
    ):
        plotter.add_light(
            pv.Light(
                position=position,
                focal_point=focus,
                intensity=intensity,
                light_type="scene light",
                positional=False,
            ),
            only_active=True,
        )


def _render(mesh: pv.PolyData, camera: dict[str, Any], title: str, path: Path) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW, lighting="none")
    actor = plotter.add_mesh(
        mesh,
        color=SHAPE_COLOR,
        smooth_shading=False,
        ambient=0.20,
        diffuse=0.80,
        specular=0.0,
    )
    if actor.GetProperty().GetInterpolation() != 0:
        raise AssertionError("comparison requires flat shading")
    _add_lights(plotter, camera)
    _set_camera(plotter, camera)
    label = plotter.add_text(title, position="upper_left", color="black", font_size=14)
    label.GetTextProperty().SetBackgroundColor(244 / 255, 242 / 255, 237 / 255)
    label.GetTextProperty().SetBackgroundOpacity(0.9)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(path)
    plotter.close()


def _plate(images: list[Path], title: str, path: Path) -> None:
    figure, axes = plt.subplots(
        1, len(images), figsize=(4 * len(images), 4.2), constrained_layout=True
    )
    figure.patch.set_facecolor(BACKGROUND)
    for axis, image in zip(axes, images, strict=True):
        axis.imshow(plt.imread(image))
        axis.set_axis_off()
    figure.suptitle(title, fontsize=15)
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "step" not in rows[0]:
        raise ValueError(f"trace lacks step: {path}")
    result: dict[str, np.ndarray] = {}
    for key in rows[0]:
        try:
            column = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        except ValueError:
            continue
        if np.isfinite(column).all():
            result[key] = column
    return result


def _column(trace: dict[str, np.ndarray], names: tuple[str, ...]) -> str | None:
    known = {key.casefold(): key for key in trace}
    return next(
        (known[name.casefold()] for name in names if name.casefold() in known), None
    )


def _progress(traces: dict[str, dict[str, np.ndarray]], path: Path) -> dict[str, str]:
    specs = {
        "Position residual RMS": ("mm", ("fit_rms_mm", "position_rms_mm"), False),
        "Surface-gradient residual RMS": (
            "dimensionless",
            ("surface_gradient_rms", "gradient_rms"),
            False,
        ),
        "Motion RMS": ("mm", ("motion_rms_mm",), False),
        "Normalized tensor roughness": (
            "dimensionless",
            ("normalized_tensor_roughness", "activation_smoothness"),
            False,
        ),
        "Projected gradient norm": ("log scale", ("projected_gradient_rms",), True),
        "Inverted tetrahedra": ("count", ("inverted_all_cells",), False),
    }
    selected: dict[str, tuple[str, dict[str, str | None], bool]] = {
        label: (
            unit,
            {method: _column(trace, names) for method, trace in traces.items()},
            log,
        )
        for label, (unit, names, log) in specs.items()
    }
    if len(selected) != 6 or len(traces) != 2 or "off" not in traces:
        raise ValueError(
            "progress plot requires exactly off and one smoothness-on branch"
        )
    missing = {
        label: [method for method, column in columns.items() if column is None]
        for label, (_, columns, _) in selected.items()
        if any(column is None for column in columns.values())
    }
    if missing:
        raise ValueError(f"progress trace lacks required metrics: {missing}")
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    colors = {"off": "#4d4a46"}
    for axis, (label, (unit, columns, log_scale)) in zip(
        axes.flat, selected.items(), strict=True
    ):
        for method, column in columns.items():
            assert column is not None
            axis.plot(
                traces[method]["step"],
                traces[method][column],
                label=f"smoothness {method}",
                color=colors.get(method, "#b33a3a"),
            )
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(f"{label} ({unit})")
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
    return {
        label: next(column for column in columns.values() if column is not None)
        for label, (_, columns, _) in selected.items()
    }


def _last(path: Path, npoints: int) -> tuple[np.ndarray, int]:
    with np.load(path, allow_pickle=False) as saved:
        required = {"u", "step"}
        if required - set(saved.files):
            raise ValueError(f"last state lacks {sorted(required - set(saved.files))}")
        u, step = np.asarray(saved["u"], dtype=np.float64), int(saved["step"])
        valid = bool(saved["solver_valid"]) if "solver_valid" in saved.files else True
    if u.shape != (npoints, 3) or not np.isfinite(u).all() or not valid:
        raise ValueError(f"invalid last state: {path}")
    return u, step


def main(cfg: Config) -> None:
    source = cherries.input(cfg.comparison_dir.resolve())
    output = cherries.output(cfg.output_dir.resolve(), mkdir=True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    stage_label = "pilot | " if "pilot" in str(source).casefold() else ""
    with np.load(source / "mesh.npz", allow_pickle=False) as saved:
        required = {
            "rest_points",
            "skin_ids",
            "triangles",
            "target_displacement_skin",
            "skin_vertex_weights",
            "initial_u",
        }
        if required - set(saved.files):
            raise ValueError(f"mesh lacks {sorted(required - set(saved.files))}")
        rest = np.asarray(saved["rest_points"], dtype=np.float64)
        skin_ids = np.asarray(saved["skin_ids"], dtype=np.int64)
        triangles = np.asarray(saved["triangles"], dtype=np.int64)
        target_u = np.asarray(saved["target_displacement_skin"], dtype=np.float64)
        weights = np.asarray(saved["skin_vertex_weights"], dtype=np.float64)
        initial_u = np.asarray(saved["initial_u"], dtype=np.float64)
    nskin = len(skin_ids)
    if (
        rest.shape[1:] != (3,)
        or len(np.unique(skin_ids)) != nskin
        or target_u.shape != (nskin, 3)
        or initial_u.shape != rest.shape
        or weights.shape != (nskin,)
        or triangles.min() < 0
        or triangles.max() >= nskin
    ):
        raise ValueError("invalid comparison mesh mapping")
    reference = rest[skin_ids]
    target = reference + target_u
    cameras = {view["id"]: view for view in _json(CAMERA_RECEIPT)["views"]}
    base = cameras["side-context"]
    full_view = {
        **base,
        "camera": {
            **base["camera"],
            "parallel_scale": 1.12 * base["camera"]["parallel_scale"],
        },
        "label": f"{base['label']} (12% wider shared framing)",
    }
    states: dict[str, tuple[str, np.ndarray, int | None]] = {
        "target": ("Target surface", target_u, None),
        "reference": ("Neutral (shared initial state)", initial_u[skin_ids], None),
    }
    summaries, traces = {}, {}
    protocol = _json(source / "protocol.json")
    beta = float(protocol["selected_beta"])
    branches = (
        ("mixed-off", "off", "Mixed L2 + gradient, smoothness off"),
        ("mixed-on", "on", "Mixed L2 + gradient, smoothness on"),
    )
    for folder_name, trace_name, label in branches:
        folder = source / folder_name
        summary = _json(folder / "summary.json")
        u, step = _last(folder / "last.npz", len(rest))
        metrics = summary.get("last_metrics")
        if not isinstance(metrics, dict) or "inverted_all_cells" not in metrics:
            raise ValueError(f"{folder_name} summary lacks final inversion count")
        count = int(metrics["inverted_all_cells"])
        status = str(summary.get("status", "status unavailable"))
        status_label = "budget stop" if "budget" in status else status.replace("_", " ")
        states[folder_name] = (
            f"{label}\nβ = {beta:g} | {stage_label}step {step} | {status_label}\n{count} inverted {'tet' if count == 1 else 'tets'}",
            u[skin_ids],
            step,
        )
        summaries[folder_name] = summary
        traces[trace_name] = _load_trace(folder / "trace.csv")
    meshes, metrics = {}, {}
    for identifier, (title, displacement, step) in states.items():
        points = reference + displacement
        error = 1000 * np.linalg.norm(points - target, axis=1)
        mesh = _mesh(points, triangles)
        mesh.point_data["GlobalPointId"] = skin_ids
        mesh.point_data["targetpos"] = target
        mesh.point_data["refpoints"] = reference
        mesh.point_data["u"] = displacement
        mesh.point_data["PositionErrorMM"] = error
        mesh.field_data["StateId"] = np.asarray([identifier])
        mesh.field_data["TargetRepresentation"] = np.asarray(
            ["surface-only correspondence field"]
        )
        file = output / "skins" / f"{identifier}.vtp"
        file.parent.mkdir(parents=True, exist_ok=True)
        mesh.save(file, binary=True)
        meshes[identifier] = mesh
        metrics[identifier] = {
            "title": title,
            "step": step,
            "position_rms_mm_weighted": float(
                np.sqrt(np.average(error**2, weights=weights))
            ),
            "position_max_mm": float(error.max()),
            "skin": str(file.relative_to(output)),
        }
    images = []
    for identifier in ("target", "reference", "mixed-off", "mixed-on"):
        image = output / "full-face" / f"{identifier}.png"
        _render(meshes[identifier], full_view["camera"], states[identifier][0], image)
        images.append(image)
    _plate(
        images,
        "Matched learned-axis mixed L2 + gradient comparison",
        output / "full-face-overview.png",
    )
    detail_views = [
        cameras["region1-mouth-corner"],
        cameras["region2-lateral-cheek"],
        cameras["region3-lower-cheek"],
    ]
    for detail in detail_views:
        images = []
        for identifier in ("target", "mixed-off", "mixed-on"):
            image = output / "details" / detail["id"] / f"{identifier}.png"
            _render(meshes[identifier], detail["camera"], states[identifier][0], image)
            images.append(image)
        _plate(
            images,
            detail["label"],
            output / "details" / f"{detail['id']}-overview.png",
        )
    trace_columns = _progress(traces, output / "optimization-progress.png")
    source_copy = output / "sources" / Path(__file__).name
    source_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), source_copy)
    summary = {
        "status": "completed_surface_rendering",
        "scope": "Rendering saved corresponding skin states only; no target-volume construction or solver call.",
        "target": "surface-only displacement field on supplied skin correspondence",
        "states": metrics,
        "optimizer_summaries": summaries,
        "protocol": protocol,
        "render": {
            "full_view": full_view,
            "detail_views": detail_views,
            "camera_policy": "fixed parallel camera shared by every state",
            "lighting": "shared oblique key and fill lights",
            "shading": "flat",
            "deformation_scale": 1.0,
        },
        "trace_columns_rendered": trace_columns,
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
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    for path in output.rglob("*.png"):
        cherries.log_output(path)
    cherries.log_output(output / "summary.json")
    cherries.log_metrics(
        {
            "render/skin_vertices": nskin,
            "render/skin_triangles": len(triangles),
            **{
                f"{name}/position_rms_mm": values["position_rms_mm_weighted"]
                for name, values in metrics.items()
            },
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
