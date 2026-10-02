"""Render saved neutral-start 3-D face L2 versus surface-normal fits."""

# ruff: noqa: EM102, PLR0915, TRY003, TRY004

from __future__ import annotations

import csv
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment import Profile

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
RECEIPT = ROOT / "exp/2026/09/08/physical-volume-closeups/data/20-regions/summary.json"
LOG = logging.getLogger(__name__)
BACKGROUND, SHAPE_COLOR, WINDOW = "#f4f2ed", "#aeb7ba", (1600, 1600)
BRANCHES = ("smooth-off-l2", "smooth-off-normal", "smooth-on-l2", "smooth-on-normal")
LABELS = {
    "smooth-off-l2": "L2 | smooth off",
    "smooth-off-normal": "L2 + normal | smooth off",
    "smooth-on-l2": "L2 | smooth on",
    "smooth-on-normal": "L2 + normal | smooth on",
}
COLORS = {
    "smooth-off-l2": "#4d4a46",
    "smooth-off-normal": "#c05621",
    "smooth-on-l2": "#1769aa",
    "smooth-on-normal": "#9c2c2c",
}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison_dir: Path = GROUP / "data/10-comparison"
    output_dir: Path = GROUP / "data/30-figures"
    shared_step: int | None = None


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _poly(points: np.ndarray, triangles: np.ndarray) -> pv.PolyData:
    faces = (
        np.column_stack((np.full(len(triangles), 3), triangles))
        .astype(np.int64)
        .ravel()
    )
    return pv.PolyData(points, faces)


def _camera(plotter: pv.Plotter, camera: dict[str, Any]) -> None:
    plotter.enable_parallel_projection()
    plotter.set_background(BACKGROUND)
    plotter.camera.position = camera["position"]
    plotter.camera.focal_point = camera["focal_point"]
    plotter.camera.up = camera["view_up"]
    plotter.camera.parallel_scale = camera["parallel_scale"]
    plotter.reset_camera_clipping_range()


def _lights(plotter: pv.Plotter, camera: dict[str, Any]) -> None:
    focus = np.asarray(camera["focal_point"])
    back = np.asarray(camera["position"]) - focus
    back /= np.linalg.norm(back)
    right = np.cross(np.asarray(camera["view_up"]), back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    for position, intensity in (
        (focus + 0.3 * (0.72 * right + 0.35 * up + 0.60 * back), 0.85),
        (focus + 0.3 * back, 0.2),
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


def _snapshot(
    mesh: pv.PolyData,
    camera: dict[str, Any],
    title: str,
    path: Path,
    *,
    error: bool = False,
    limit: float = 1,
) -> None:
    p = pv.Plotter(off_screen=True, window_size=WINDOW, lighting="none")
    args = {"smooth_shading": False, "ambient": 0.2, "diffuse": 0.8, "specular": 0}
    if error:
        actor = p.add_mesh(
            mesh,
            scalars="PositionErrorMM",
            cmap="viridis",
            clim=(0, limit),
            scalar_bar_args={
                "title": "Error (mm)",
                "vertical": True,
                "color": "black",
                "position_x": 0.76,
                "position_y": 0.1,
                "width": 0.11,
                "height": 0.50,
                "n_labels": 4,
                "fmt": "%.1f",
                "title_font_size": 26,
                "label_font_size": 24,
            },
            **args,
        )
    else:
        actor = p.add_mesh(mesh, color=SHAPE_COLOR, **args)
    assert actor.GetProperty().GetInterpolation() == 0
    _lights(p, camera)
    _camera(p, camera)
    text = p.add_text(title, position="upper_left", color="black", font_size=26)
    text.GetTextProperty().SetBackgroundColor(244 / 255, 242 / 255, 237 / 255)
    text.GetTextProperty().SetBackgroundOpacity(0.9)
    p.screenshot(path)
    p.close()


def _trace(folder: Path) -> dict[str, np.ndarray]:
    with (folder / "trace.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "step" not in rows[0]:
        raise ValueError(f"bad trace: {folder}")
    return {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}


def _step(folder: Path) -> int:
    trace = _trace(folder)
    return int(trace["step"][-1])


def _saved_steps(folder: Path) -> set[int]:
    result = {
        int(path.stem.removeprefix("step-")) for path in folder.glob("step-*.npz")
    }
    if not result:
        result.add(_step(folder))
    return result


def _inversions(folder: Path, step: int) -> int:
    """Read inversion count from the trace row that produced the rendered state."""
    trace = _trace(folder)
    index = np.flatnonzero(trace["step"] == step)
    if len(index) != 1:
        raise ValueError(f"no unique trace state {step}: {folder}")
    for key in ("inverted_all_cells", "inverted_cells"):
        if key in trace:
            return int(trace[key][index[0]])
    raise ValueError(f"trace lacks inversion count: {folder}")


def _last(folder: Path, n: int, step: int) -> np.ndarray:
    candidate = folder / f"step-{step:04d}.npz"
    path = candidate if candidate.is_file() else folder / "last.npz"
    with np.load(path, allow_pickle=False) as saved:
        assert int(saved["step"]) == step
        assert bool(saved["solver_valid"])
        u = np.asarray(saved["u"], dtype=float)
    if u.shape != (n, 3):
        raise ValueError(f"invalid state: {path}")
    return u


def _plate(images: list[Path], title: str, path: Path, shape: tuple[int, int]) -> None:
    fig, axes = plt.subplots(
        *shape, figsize=(4 * shape[1], 4.2 * shape[0]), squeeze=False
    )
    fig.patch.set_facecolor(BACKGROUND)
    for axis, image in zip(axes.flat, images, strict=True):
        axis.imshow(plt.imread(image))
        axis.set_axis_off()
    fig.suptitle(title, y=0.99, fontsize=15)
    fig.subplots_adjust(
        left=0.01, bottom=0.02, right=0.99, top=0.96, wspace=0.01, hspace=0.01
    )
    fig.savefig(path, dpi=180, facecolor=BACKGROUND)
    plt.close(fig)


def _curves(traces: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    specs = (
        ("Own objective / initial", "objective", True, False),
        ("Position RMS (mm)", "fit_rms_mm", False, False),
        ("Normal angle RMS (deg)", "normal_angle_rms_deg", False, False),
        ("Activation variation R", "activation_smoothness", False, False),
        (
            "Target-relative 5 mm high-pass residual (mm)",
            "primary_union_normal_residual_highpass_5mm_rms_mm",
            False,
            False,
        ),
        ("Physical gradient RMS / initial", "physical_gradient_rms", True, True),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, (title, key, normalized, log) in zip(axes.flat, specs, strict=True):
        for branch, trace in traces.items():
            if key not in trace:
                raise ValueError(f"missing {key}: {branch}")
            y = trace[key] / trace[key][0] if normalized else trace[key]
            axis.plot(trace["step"], y, color=COLORS[branch], label=LABELS[branch])
        if log:
            axis.set_yscale("log")
        axis.set(title=title, xlabel="Adam update")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Saved neutral-start histories; fixed budget is not convergence", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(cfg: Config) -> None:
    LOG.info("Rendering saved face states only; no inverse solves are run.")
    source = cherries.input(cfg.comparison_dir.resolve())
    output = cherries.output(cfg.output_dir.resolve(), mkdir=True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source / "mesh.npz", allow_pickle=False) as x:
        rest = np.asarray(x["rest_points"], float)
        skin = np.asarray(x["skin_ids"], int)
        triangles = np.asarray(x["triangles"], int)
        target_u = np.asarray(x["target_displacement_skin"], float)
        initial = np.asarray(x["initial_u"], float)
    reference, target = rest[skin], rest[skin] + target_u
    folders = {branch: source / branch for branch in BRANCHES}
    steps = {branch: _step(folder) for branch, folder in folders.items()}
    shared_steps = set.intersection(
        *(_saved_steps(folder) for folder in folders.values())
    )
    if not shared_steps:
        raise ValueError(f"no shared saved state: {steps}")
    shared = max(shared_steps) if cfg.shared_step is None else cfg.shared_step
    if shared not in shared_steps:
        raise ValueError(
            f"requested shared step {shared} is unavailable: {sorted(shared_steps)}"
        )
    cameras = {item["id"]: item["camera"] for item in _json(RECEIPT)["views"]}
    full = {
        **cameras["side-context"],
        "parallel_scale": 1.12 * cameras["side-context"]["parallel_scale"],
    }
    mouth = cameras["region1-mouth-corner"]
    states = {
        "target": (target_u, "Target surface"),
        "neutral": (initial[skin], "Neutral"),
        **{
            branch: (
                _last(folder, len(rest), shared)[skin],
                f"{LABELS[branch]}\nstep {shared} | {_inversions(folder, shared)} inverted tets",
            )
            for branch, folder in folders.items()
        },
    }
    meshes = {}
    errors = []
    for key, (u, _title) in states.items():
        points = reference + u
        mesh = _poly(points, triangles)
        error = 1000 * np.linalg.norm(points - target, axis=1)
        mesh.point_data["PositionErrorMM"] = error
        mesh.point_data["GlobalPointId"] = skin
        mesh.point_data["targetpos"] = target
        mesh.point_data["refpoints"] = reference
        mesh.point_data["u"] = u
        meshes[key] = mesh
        errors.extend(error)
    limit = float(np.percentile(errors, 99))
    for view, camera in (("full", full), ("mouth", mouth)):
        paths = []
        for key, (_u, title) in states.items():
            path = output / f"{view}-{key}.png"
            _snapshot(meshes[key], camera, title, path)
            paths.append(path)
        _plate(
            [paths[0], paths[2], paths[3], paths[0], paths[4], paths[5]],
            f"{view.capitalize()} view: shared target, L2, and L2 + normal columns",
            output / f"{view}-comparison.png",
            (2, 3),
        )
    error_paths = []
    for key in BRANCHES:
        path = output / f"error-{key}.png"
        _snapshot(
            meshes[key],
            full,
            f"{LABELS[key]} | step {shared}",
            path,
            error=True,
            limit=limit,
        )
        error_paths.append(path)
    _plate(
        error_paths,
        "Corresponding-vertex error maps: common 99th-percentile scale",
        output / "position-error-maps.png",
        (2, 2),
    )
    traces = {branch: _trace(folder) for branch, folder in folders.items()}
    _curves(traces, output / "loss-and-gradient-histories.png")
    shutil.copy2(Path(__file__), output / "source-30-render.py")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "requested_shared_step": cfg.shared_step,
                "shared_step": shared,
                "endpoint_steps": steps,
                "error_limit_mm_99th_percentile": limit,
                "target": "surface correspondence only; no target volume constructed",
                "python": sys.version,
            },
            indent=2,
        )
        + "\n"
    )
    for path in output.iterdir():
        cherries.log_output(path)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
