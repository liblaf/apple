"""Render matched surface-only target, neutral, L2, and gradient-only face fits.

The target exists only on the skin correspondence.  This renderer never
constructs a target volume or presents one as a feasible volumetric state.
"""

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003, TRY004

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
ERROR_CMAP = "viridis"


class Comet(plugins.Comet):
    """Comet recorder without slow environment or Git patch collection."""

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
    """Local/Comet evidence, with neither Git commits nor debug uploads."""

    def init(self) -> core.Run:
        run = core.run
        run.plugins.register(Comet(run=run, disabled=os.environ.get("DEBUG") == "1"))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Logging(run=run))
        run.plugins.register(plugins.Local(run=run))
        return run


class Config(cherries.BaseConfig):
    """Completed optimizer states and a Cherries-managed render destination."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison_dir: Path = cherries.input("10-comparison")
    output_dir: Path = cherries.output("30-figures", mkdir=True)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _faces(triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("skin triangles must have shape (n, 3)")
    return np.column_stack(
        (np.full(len(triangles), 3, dtype=np.int64), triangles)
    ).ravel()


def _polydata(points: np.ndarray, triangles: np.ndarray) -> pv.PolyData:
    return pv.PolyData(np.asarray(points, dtype=np.float64), _faces(triangles))


def _add_lights(plotter: pv.Plotter, camera: dict[str, Any]) -> None:
    """Use the established oblique-face lighting construction."""
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


def _set_camera(plotter: pv.Plotter, camera: dict[str, Any]) -> None:
    plotter.enable_parallel_projection()
    plotter.camera.position = camera["position"]
    plotter.camera.focal_point = camera["focal_point"]
    plotter.camera.up = camera["view_up"]
    plotter.camera.parallel_scale = camera["parallel_scale"]
    plotter.set_background(BACKGROUND)
    plotter.reset_camera_clipping_range()


def _render_shape(
    mesh: pv.PolyData, camera: dict[str, Any], title: str, path: Path
) -> None:
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
        raise AssertionError("face comparison requires flat shading")
    _add_lights(plotter, camera)
    _set_camera(plotter, camera)
    label = plotter.add_text(title, position="upper_left", color="black", font_size=14)
    label.GetTextProperty().SetBackgroundColor(244 / 255, 242 / 255, 237 / 255)
    label.GetTextProperty().SetBackgroundOpacity(0.9)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(path)
    plotter.close()


def _render_error(
    mesh: pv.PolyData, camera: dict[str, Any], limit_mm: float, title: str, path: Path
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW, lighting="none")
    actor = plotter.add_mesh(
        mesh,
        scalars="PositionErrorMM",
        preference="point",
        cmap=ERROR_CMAP,
        clim=(0.0, limit_mm),
        n_colors=256,
        smooth_shading=False,
        ambient=0.20,
        diffuse=0.80,
        specular=0.0,
        scalar_bar_args={
            "title": "Corresponding-vertex\nposition error (mm)",
            "vertical": True,
            "color": "black",
            "position_x": 0.80,
            "position_y": 0.10,
            "width": 0.08,
            "height": 0.45,
            "title_font_size": 15,
            "label_font_size": 13,
            "fmt": "%.3g",
            "background_color": BACKGROUND,
        },
    )
    if actor.GetProperty().GetInterpolation() != 0:
        raise AssertionError("error maps require flat shading")
    _add_lights(plotter, camera)
    _set_camera(plotter, camera)
    label = plotter.add_text(title, position="upper_left", color="black", font_size=14)
    label.GetTextProperty().SetBackgroundColor(244 / 255, 242 / 255, 237 / 255)
    label.GetTextProperty().SetBackgroundOpacity(0.9)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(path)
    plotter.close()


def _gradient_rms(
    values: np.ndarray, points: np.ndarray, triangles: np.ndarray
) -> float:
    vertices = points[triangles]
    twice_area_normal = np.cross(
        vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
    )
    twice_area_squared = np.sum(twice_area_normal**2, axis=1)
    if not np.all(twice_area_squared > 0.0):
        raise ValueError("reference skin contains degenerate triangles")
    gradients = (
        np.stack(
            (
                np.cross(twice_area_normal, vertices[:, 2] - vertices[:, 1]),
                np.cross(twice_area_normal, vertices[:, 0] - vertices[:, 2]),
                np.cross(twice_area_normal, vertices[:, 1] - vertices[:, 0]),
            ),
            axis=1,
        )
        / twice_area_squared[:, None, None]
    )
    gradient = np.einsum("tvi,tvj->tij", values[triangles], gradients)
    areas = 0.5 * np.sqrt(twice_area_squared)
    return float(
        np.sqrt(np.sum(areas * np.sum(gradient**2, axis=(1, 2))) / np.sum(areas))
    )


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "step" not in rows[0]:
        raise ValueError(f"trace lacks a step column: {path}")
    values: dict[str, np.ndarray] = {}
    for key in rows[0]:
        try:
            column = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        except ValueError:
            continue
        if np.isfinite(column).all():
            values[key] = column
    if np.any(np.diff(values["step"]) < 0):
        raise ValueError(f"trace steps must not decrease: {path}")
    return values


def _find_column(
    trace: dict[str, np.ndarray], candidates: tuple[str, ...]
) -> str | None:
    lowered = {key.casefold(): key for key in trace}
    for candidate in candidates:
        if candidate.casefold() in lowered:
            return lowered[candidate.casefold()]
    return None


def _render_progress(
    traces: dict[str, dict[str, np.ndarray]], path: Path
) -> dict[str, str]:
    metrics = {
        "Position residual RMS": (
            "mm",
            (
                "position_rms_mm",
                "fit_rms_mm",
                "position_rms",
                "fit_rms",
            ),
        ),
        "Surface-gradient residual RMS": (
            "dimensionless",
            (
                "surface_gradient_rms",
                "gradient_rms",
                "surface_grad_rms",
            ),
        ),
        "Primary high-pass residual normal RMS": (
            "mm",
            (
                "primary_highpass_residual_normal_rms_mm",
                "primary_union_normal_residual_highpass_5mm_rms_mm",
            ),
        ),
        "Low-frequency target projection": (
            "dimensionless",
            (
                "primary_low_frequency_normal_target_projection",
                "primary_union_low_frequency_normal_target_projection",
            ),
        ),
        "Motion RMS": ("mm", ("motion_rms_mm",)),
    }
    available = {
        label: (
            unit,
            {
                method: _find_column(trace, candidates)
                for method, trace in traces.items()
            },
        )
        for label, (unit, candidates) in metrics.items()
    }
    available = {
        label: record for label, record in available.items() if any(record[1].values())
    }
    if not available:
        return {}
    figure, axes = plt.subplots(
        1, len(available), figsize=(5.3 * len(available), 4.2), squeeze=False
    )
    colors = {"l2": "#4d4a46", "gradient": "#b33a3a"}
    for axis, (label, (unit, columns)) in zip(axes[0], available.items(), strict=True):
        for method, column in columns.items():
            if column:
                axis.plot(
                    traces[method]["step"],
                    traces[method][column],
                    label=method,
                    color=colors[method],
                )
        axis.set_title(f"{label} ({unit})")
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
    return {
        label: next(column for column in columns.values() if column)
        for label, (_, columns) in available.items()
    }


def _plate(images: list[Path], title: str, path: Path) -> None:
    figure, axes = plt.subplots(
        1, len(images), figsize=(4.0 * len(images), 4.2), constrained_layout=True
    )
    figure.patch.set_facecolor(BACKGROUND)
    for axis, image in zip(axes, images, strict=True):
        axis.imshow(plt.imread(image))
        axis.set_axis_off()
    figure.suptitle(title, fontsize=15)
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _state_u(path: Path, npoints: int) -> tuple[np.ndarray, int]:
    with np.load(path, allow_pickle=False) as saved:
        if {"u", "step"} - set(saved.files):
            raise ValueError(f"{path} lacks u or step")
        u, step = np.asarray(saved["u"], dtype=np.float64), int(saved["step"])
        valid = bool(saved["solver_valid"]) if "solver_valid" in saved.files else True
    if u.shape != (npoints, 3) or not np.isfinite(u).all() or not valid:
        raise ValueError(f"invalid saved state: {path}")
    return u, step


def main(cfg: Config) -> None:
    source = cfg.comparison_dir
    output = cfg.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    mesh_path = source / "mesh.npz"
    with np.load(mesh_path, allow_pickle=False) as saved:
        required = {
            "rest_points",
            "skin_ids",
            "triangles",
            "target_displacement_skin",
            "skin_vertex_weights",
            "initial_u",
        }
        absent = required - set(saved.files)
        if absent:
            raise ValueError(f"mesh input lacks {sorted(absent)}")
        rest = np.asarray(saved["rest_points"], dtype=np.float64)
        skin_ids = np.asarray(saved["skin_ids"], dtype=np.int64)
        triangles = np.asarray(saved["triangles"], dtype=np.int64)
        target_u = np.asarray(saved["target_displacement_skin"], dtype=np.float64)
        weights = np.asarray(saved["skin_vertex_weights"], dtype=np.float64)
        initial_u = np.asarray(saved["initial_u"], dtype=np.float64)
    if (
        rest.ndim != 2
        or rest.shape[1] != 3
        or len(np.unique(skin_ids)) != len(skin_ids)
    ):
        raise ValueError("invalid unique ordered skin GlobalPointId mapping")
    if np.any(skin_ids < 0) or np.any(skin_ids >= len(rest)):
        raise ValueError("skin ids are outside rest_points")
    nskin = len(skin_ids)
    if target_u.shape != (nskin, 3) or initial_u.shape != rest.shape:
        raise ValueError("target/initial displacement dimensions disagree")
    if (
        triangles.min() < 0
        or triangles.max() >= nskin
        or not np.isfinite(target_u).all()
    ):
        raise ValueError("invalid surface target")
    if weights.ndim != 1 or np.any(weights <= 0.0) or not np.isfinite(weights).all():
        raise ValueError("skin weights must be positive and finite")

    camera_doc = _json(CAMERA_RECEIPT)
    camera_by_id = {item["id"]: item for item in camera_doc["views"]}
    source_full_view = camera_by_id["side-context"]
    full_view = {
        **source_full_view,
        "camera": {
            **source_full_view["camera"],
            "parallel_scale": 1.12 * source_full_view["camera"]["parallel_scale"],
        },
        "label": f"{source_full_view['label']} (12% wider shared framing)",
    }
    detail_views = [
        camera_by_id[key] for key in ("region1-mouth-corner", "region2-lateral-cheek")
    ]
    reference_points = rest[skin_ids]
    states: dict[str, tuple[str, np.ndarray, int | None]] = {
        "reference": ("Neutral (shared initial state)", np.zeros((nskin, 3)), None),
        "target": ("Target surface", target_u, None),
        "initial": ("Neutral shared initialization", initial_u[skin_ids], None),
    }
    summaries: dict[str, dict[str, Any]] = {}
    traces: dict[str, dict[str, np.ndarray]] = {}
    for method in ("l2", "gradient"):
        folder = source / method
        summary = _json(folder / "summary.json")
        u, step = _state_u(folder / "last.npz", len(rest))
        last_metrics = summary.get("last_metrics")
        if (
            not isinstance(last_metrics, dict)
            or "inverted_all_cells" not in last_metrics
        ):
            raise ValueError(f"{method} summary lacks final inversion count")
        inversions = int(last_metrics["inverted_all_cells"])
        noun = "tet" if inversions == 1 else "tets"
        method_title = "Surface L2" if method == "l2" else "Surface-gradient only"
        states[method] = (
            f"{method_title} | step {step} | {inversions} inverted {noun}",
            u[skin_ids],
            step,
        )
        summaries[method] = summary
        traces[method] = _load_trace(folder / "trace.csv")

    target_points = reference_points + target_u
    skin_meshes: dict[str, pv.PolyData] = {}
    state_metrics: dict[str, Any] = {}
    for identifier, (title, displacement, step) in states.items():
        points = reference_points + displacement
        error = 1000.0 * np.linalg.norm(points - target_points, axis=1)
        mesh = _polydata(points, triangles)
        mesh.point_data["GlobalPointId"] = skin_ids
        mesh.point_data["targetpos"] = target_points
        mesh.point_data["refpoints"] = reference_points
        mesh.point_data["u"] = displacement
        mesh.point_data["PositionErrorMM"] = error
        mesh.field_data["StateId"] = np.asarray([identifier])
        mesh.field_data["TargetRepresentation"] = np.asarray(
            ["surface-only correspondence field"]
        )
        skin_path = output / "skins" / f"{identifier}.vtp"
        skin_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.save(skin_path, binary=True)
        skin_meshes[identifier] = mesh
        state_metrics[identifier] = {
            "title": title,
            "step": step,
            "position_rms_mm_weighted": float(
                np.sqrt(np.average(error**2, weights=weights))
            ),
            "position_max_mm": float(error.max()),
            "surface_gradient_error_rms": _gradient_rms(
                points - target_points, reference_points, triangles
            ),
            "surface_displacement_gradient_rms": _gradient_rms(
                displacement, reference_points, triangles
            ),
            "skin": str(skin_path.relative_to(output)),
        }

    error_limit = max(
        state_metrics[identifier]["position_max_mm"]
        for identifier in ("initial", "l2", "gradient")
    )
    if error_limit <= 0.0:
        raise ValueError("all surface errors are zero")
    all_shape_images: list[Path] = []
    for identifier in ("target", "reference", "l2", "gradient"):
        image = output / "full-face" / f"{identifier}.png"
        _render_shape(
            skin_meshes[identifier], full_view["camera"], states[identifier][0], image
        )
        all_shape_images.append(image)
    _plate(
        all_shape_images,
        "Matched surface correspondence: target and fitted states",
        output / "full-face-overview.png",
    )
    error_images: list[Path] = []
    for identifier in ("initial", "l2", "gradient"):
        image = output / "errors" / f"{identifier}.png"
        _render_error(
            skin_meshes[identifier],
            full_view["camera"],
            error_limit,
            states[identifier][0],
            image,
        )
        error_images.append(image)
    _plate(
        error_images,
        "Common-scale corresponding-vertex position error",
        output / "position-error-overview.png",
    )
    for detail in detail_views:
        images: list[Path] = []
        for identifier in ("target", "reference", "l2", "gradient"):
            image = output / "details" / detail["id"] / f"{identifier}.png"
            _render_shape(
                skin_meshes[identifier], detail["camera"], states[identifier][0], image
            )
            images.append(image)
        _plate(
            images, detail["label"], output / "details" / f"{detail['id']}-overview.png"
        )
    trace_columns = _render_progress(traces, output / "optimization-progress.png")
    source_copy = output / "sources" / Path(__file__).name
    source_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), source_copy)
    summary = {
        "status": "completed_surface_rendering",
        "scope": "Rendering and direct corresponding-surface diagnostics only; no target-volume construction or solver call.",
        "target": "target is a surface-only displacement field on the supplied skin correspondence",
        "input_mesh": str(mesh_path),
        "states": state_metrics,
        "optimizer_summaries": summaries,
        "render": {
            "full_view": full_view,
            "detail_views": detail_views,
            "camera_policy": "fixed parallel cameras shared by every state",
            "lighting": "shared established oblique-face key and fill lights",
            "shading": "flat",
            "deformation_scale": 1.0,
            "error_colormap": ERROR_CMAP,
            "common_error_limits_mm": [0.0, error_limit],
            "error_definition": "1000 * norm(position - targetpos), evaluated at matching GlobalPointId vertices",
            "gradient_error_definition": "area-normalized fixed-reference piecewise-linear surface gradient RMS",
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
            "render/common_position_error_limit_mm": error_limit,
            **{
                f"{identifier}/position_rms_mm": values["position_rms_mm_weighted"]
                for identifier, values in state_metrics.items()
            },
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
