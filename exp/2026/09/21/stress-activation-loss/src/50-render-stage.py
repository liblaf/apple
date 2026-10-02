"""Render saved active-reference-stress and surface-fit views for one stage."""
# ruff: noqa: PLR0915

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment import Profile
from run_support import write_json

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

GROUP = Path(__file__).parents[1]
CAMERA_RECEIPT = (
    GROUP.parents[2]
    / "09"
    / "08"
    / "physical-volume-closeups"
    / "data"
    / "20-regions"
    / "summary.json"
)
BACKGROUND, POSITIVE, NEGATIVE = "#f4f2ed", "#cf5c16", "#1675b8"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    source: Path
    stage: str
    site: Path = Path("site")
    output: Path = Path("data/50-figures")
    error_limit_mm: float = 10.0


def receipt(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def poly(points: np.ndarray, triangles: np.ndarray) -> pv.PolyData:
    faces = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    return pv.PolyData(points, faces)


def camera(plotter: pv.Plotter, value: dict) -> None:
    plotter.enable_parallel_projection()
    plotter.set_background(BACKGROUND)
    plotter.camera.position = value["position"]
    plotter.camera.focal_point = value["focal_point"]
    plotter.camera.up = value["view_up"]
    plotter.camera.parallel_scale = value["parallel_scale"]
    plotter.reset_camera_clipping_range()


def snapshot(
    mesh: pv.PolyData,
    value: dict,
    path: Path,
    *,
    error: bool = False,
    limit: float = 10,
) -> None:
    plotter = pv.Plotter(
        off_screen=True, window_size=(1400, 1400), lighting="three lights"
    )
    if error:
        plotter.add_mesh(
            mesh,
            scalars="PositionErrorMM",
            cmap="viridis",
            clim=(0, limit),
            smooth_shading=False,
            scalar_bar_args={"title": "mm", "n_labels": 4, "fmt": "%.1f"},
        )
    else:
        plotter.add_mesh(mesh, color="#aeb7ba", smooth_shading=False)
    camera(plotter, value)
    plotter.screenshot(path)
    plotter.close()


def sample(centers: np.ndarray, value: dict, spacing: float) -> np.ndarray:
    back = np.asarray(value["position"]) - value["focal_point"]
    back /= np.linalg.norm(back)
    right = np.cross(value["view_up"], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    centered = centers - value["focal_point"]
    xy = np.column_stack((centered @ right, centered @ up))
    inside = np.max(np.abs(xy), axis=1) < 0.94 * value["parallel_scale"]
    ids = np.flatnonzero(inside)
    grid = np.floor(xy[ids] / spacing).astype(np.int64)
    depth = centered[ids] @ back
    order = np.lexsort((-depth, grid[:, 0], grid[:, 1]))
    _, first = np.unique(grid[order], axis=0, return_index=True)
    chosen = np.sort(ids[order[first]])
    assert len(chosen) > 20
    return chosen


def glyphs(
    centers: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray, limit: float
) -> pv.PolyData:
    values = eigenvalues.ravel()
    directions = np.swapaxes(eigenvectors, 1, 2).reshape(-1, 3)
    centers = np.repeat(centers, 3, axis=0)
    half = 0.0015 * np.minimum(np.abs(values) / limit, 1)
    points = np.empty((2 * len(values), 3))
    points[0::2], points[1::2] = (
        centers - half[:, None] * directions,
        centers + half[:, None] * directions,
    )
    lines = np.column_stack(
        (
            np.full(len(values), 2),
            2 * np.arange(len(values)),
            2 * np.arange(len(values)) + 1,
        )
    ).ravel()
    output = pv.PolyData(points, lines=lines)
    output.cell_data["PrincipalQKPa"] = values
    return output


def q_snapshot(
    surface: pv.PolyData,
    skin: pv.PolyData,
    centers: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    value: dict,
    path: Path,
    *,
    axes: bool,
    magnitude_limit: float,
    principal_limit: float,
) -> None:
    plotter = pv.Plotter(
        off_screen=True, window_size=(1400, 1400), lighting="three lights"
    )
    plotter.add_mesh(skin, color="#89949b", opacity=0.08, smooth_shading=False)
    if not axes:
        plotter.add_mesh(
            surface,
            scalars="ReferenceQMagnitudeKPa",
            preference="cell",
            cmap="magma",
            clim=(0, magnitude_limit),
            smooth_shading=False,
            lighting=False,
            scalar_bar_args={"title": "|Q| (kPa)", "n_labels": 4, "fmt": "%.1f"},
        )
    else:
        plotter.add_mesh(surface, color="#9da7af", opacity=0.07, smooth_shading=False)
        if principal_limit > 0:
            chosen = sample(
                centers, value, 0.002 if value["parallel_scale"] > 0.05 else 0.0008
            )
            lines = glyphs(
                centers[chosen],
                eigenvalues[chosen],
                eigenvectors[chosen],
                principal_limit,
            )
            signed = lines.cell_data["PrincipalQKPa"]
            for positive, color in ((True, POSITIVE), (False, NEGATIVE)):
                mask = signed > 0 if positive else signed < 0
                if np.any(mask):
                    plotter.add_mesh(
                        lines.extract_cells(mask),
                        color=color,
                        line_width=2,
                        lighting=False,
                    )
    camera(plotter, value)
    plotter.screenshot(path)
    plotter.close()


def plate(paths: list[Path], titles: list[str], title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5.3))
    fig.patch.set_facecolor(BACKGROUND)
    for axis, image, label in zip(axes, paths, titles, strict=True):
        axis.imshow(plt.imread(image))
        axis.set_axis_off()
        axis.set_title(label, fontsize=12)
    fig.suptitle(title, fontsize=16, y=0.99)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.90, wspace=0.01)
    fig.savefig(path, dpi=180, facecolor=BACKGROUND)
    plt.close(fig)


def update_status(path: Path, stage: str, images: dict[str, str]) -> None:
    state = json.loads(path.read_text())
    target = next(item for item in state["stages"] if item["id"] == stage)
    target["images"] = images
    state["updated_at"] = datetime.datetime.now().astimezone().isoformat()
    write_json(path, state)


def main(cfg: Config) -> None:
    source = cherries.input(cfg.source)
    stage_dir = source / cfg.stage
    stage_summary = json.loads((stage_dir / "summary.json").read_text())
    budget = int(stage_summary["budget"])
    stage_status = str(stage_summary["status"])
    with (
        np.load(source / "mesh.npz", allow_pickle=False) as mesh,
        np.load(stage_dir / "last.npz", allow_pickle=False) as state,
    ):
        rest = np.asarray(mesh["rest_points"], float)
        tets = np.asarray(mesh["tets"], int)
        active = np.asarray(mesh["active_ids"], int)
        skin_ids = np.asarray(mesh["skin_ids"], int)
        triangles = np.asarray(mesh["triangles"], int)
        target_u = np.asarray(mesh["target_displacement_skin"], float)
        u = np.asarray(state["u"], float)
        qhat = np.asarray(state["Qhat"], float)
        step = int(state["step"])
        stress_reference = float(state["stress_reference_MPa"])
        mode = str(state["mode"])
    assert u.shape == rest.shape
    assert qhat.shape == (len(active), 3, 3)
    result_root = cfg.output / cfg.stage
    asset_root = cfg.site / "assets" / cfg.stage
    result_root.mkdir(parents=True, exist_ok=False)
    asset_root.mkdir(parents=True, exist_ok=False)
    target = rest[skin_ids] + target_u
    result = rest[skin_ids] + u[skin_ids]
    fit = poly(result, triangles)
    fit.point_data["PositionErrorMM"] = 1000 * np.linalg.norm(result - target, axis=1)
    target_mesh = poly(target, triangles)
    cameras = {
        x["id"]: x["camera"] for x in json.loads(CAMERA_RECEIPT.read_text())["views"]
    }
    views = {
        "full": dict(cameras["side-context"]),
        "mouth": dict(cameras["region1-mouth-corner"]),
    }
    views["full"]["parallel_scale"] *= 1.12
    for name, value in views.items():
        target_path, fit_path, error_path = (
            asset_root / f"{name}-{item}.png" for item in ("target", "fit", "error")
        )
        snapshot(target_mesh, value, target_path)
        snapshot(fit, value, fit_path)
        snapshot(fit, value, error_path, error=True, limit=cfg.error_limit_mm)
        plate(
            [target_path, fit_path, error_path],
            ["Target", "Stage result", "Position error"],
            f"{cfg.stage} | {stage_status} | {mode} | update {step} / budget {budget}",
            asset_root / f"{name}-face.png",
        )
    points = rest + u
    cells = np.column_stack((np.full(len(active), 4), tets[active])).ravel()
    volume = pv.UnstructuredGrid(cells, np.full(len(active), pv.CellType.TETRA), points)
    q_kpa = 1000 * stress_reference * qhat
    magnitude = np.linalg.norm(q_kpa, axis=(1, 2))
    values, vectors = np.linalg.eigh(q_kpa)
    volume.cell_data["ReferenceQMagnitudeKPa"] = magnitude
    volume.cell_data["PrincipalQKPa"] = values
    volume.cell_data["GlobalCellId"] = active
    surface = volume.extract_surface(algorithm="dataset_surface")
    surface.save(result_root / "active-q-surface.vtp")
    raw = result_root / "active-q.vtu"
    volume.save(raw)
    if raw.stat().st_size > 100 * 1024**2:
        raw.unlink()
    skin = poly(result, triangles)
    centers = points[tets[active]].mean(axis=1)
    all_zero = bool(not np.any(magnitude))
    declared_limit = 1000 * stress_reference
    limits = {
        "magnitude_kPa": float(np.quantile(magnitude, 0.99)),
        "principal_kPa": float(np.quantile(np.abs(values), 0.99)),
    }
    if all_zero:
        limits = {"magnitude_kPa": declared_limit, "principal_kPa": declared_limit}
    assert all(np.isfinite(value) and value > 0 for value in limits.values())
    for name, value in views.items():
        magnitude_path, axes_path = (
            asset_root / f"{name}-q-magnitude.png",
            asset_root / f"{name}-q-axes.png",
        )
        q_snapshot(
            surface,
            skin,
            centers,
            values,
            vectors,
            value,
            magnitude_path,
            axes=False,
            magnitude_limit=limits["magnitude_kPa"],
            principal_limit=limits["principal_kPa"],
        )
        q_snapshot(
            surface,
            skin,
            centers,
            values,
            vectors,
            value,
            axes_path,
            axes=True,
            magnitude_limit=limits["magnitude_kPa"],
            principal_limit=limits["principal_kPa"],
        )
        plate(
            [magnitude_path, axes_path],
            ["Reference Q magnitude", "Reference Q principal axes"],
            f"{cfg.stage} | {stage_status} | update {step} / budget {budget} | Q in kPa; "
            + ("all-zero Q; no axes" if all_zero else "orange positive, blue negative"),
            asset_root / f"{name}-activation-q.png",
        )
    images = {
        "face comparison": f"assets/{cfg.stage}/full-face.png",
        "mouth comparison": f"assets/{cfg.stage}/mouth-face.png",
        "reference Q": f"assets/{cfg.stage}/full-activation-q.png",
        "mouth reference Q": f"assets/{cfg.stage}/mouth-activation-q.png",
    }
    update_status(cfg.site / "status.json", cfg.stage, images)
    summary = {
        "passed": True,
        "stage": cfg.stage,
        "mode": mode,
        "stage_status": stage_status,
        "step": step,
        "budget": budget,
        "error_limit_mm": cfg.error_limit_mm,
        "reference_stress_MPa": stress_reference,
        "reference_Q": "Q = stress_reference_MPa * Qhat; magnitude and eigenpairs are reference-stress quantities, displayed on the current deformed active-tet geometry.",
        "limits": limits,
        "all_zero_Q": all_zero,
        "display_limit_source": "declared stress reference kPa"
        if all_zero
        else "stage 99th percentile",
        "source": {
            "mesh": receipt(source / "mesh.npz"),
            "state": receipt(stage_dir / "last.npz"),
            "camera": receipt(CAMERA_RECEIPT),
            "renderer": receipt(Path(__file__)),
        },
        "images": images,
        "active_q_vtu_written": raw.exists(),
    }
    write_json(result_root / "summary.json", summary)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
