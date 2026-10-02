"""Build a shared-scale atlas from saved active-stress chain states only."""
# ruff: noqa: PLR0915, RUF001

from __future__ import annotations

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
BACKGROUND = "#f4f2ed"
MODES = ("symmetric6", "psd6", "rankone_fixed", "rankone_learned")
LOSSES = ("l2", "normal")


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    source: Path
    site: Path = Path("site")
    output: Path = Path("data/70-comparison")
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


def face_map(mesh: pv.PolyData, value: dict, path: Path, limit: float) -> None:
    plotter = pv.Plotter(
        off_screen=True, window_size=(1000, 1000), lighting="three lights"
    )
    plotter.add_mesh(
        mesh,
        scalars="PositionErrorMM",
        cmap="viridis",
        clim=(0, limit),
        smooth_shading=False,
        show_scalar_bar=False,
    )
    camera(plotter, value)
    plotter.screenshot(path)
    plotter.close()


def q_map(
    surface: pv.PolyData, skin: pv.PolyData, value: dict, path: Path, limit: float
) -> None:
    plotter = pv.Plotter(
        off_screen=True, window_size=(1000, 1000), lighting="three lights"
    )
    plotter.add_mesh(skin, color="#89949b", opacity=0.08, smooth_shading=False)
    plotter.add_mesh(
        surface,
        scalars="ReferenceQMagnitudeKPa",
        preference="cell",
        cmap="magma",
        clim=(0, limit),
        smooth_shading=False,
        lighting=False,
        show_scalar_bar=False,
    )
    camera(plotter, value)
    plotter.screenshot(path)
    plotter.close()


def atlas(
    paths: dict[str, Path],
    labels: dict[str, str],
    title: str,
    cmap: str,
    limit: float,
    path: Path,
) -> None:
    figure, axes = plt.subplots(4, 2, figsize=(10, 18))
    figure.patch.set_facecolor(BACKGROUND)
    for row, mode in enumerate(MODES):
        for column, loss in enumerate(LOSSES):
            stage = f"{loss}-{mode}"
            axis = axes[row, column]
            axis.set_axis_off()
            axis.set_title(labels[stage], fontsize=9)
            if stage in paths:
                axis.imshow(plt.imread(paths[stage]))
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No accepted saved state",
                    ha="center",
                    va="center",
                    color="#5f6b76",
                    transform=axis.transAxes,
                )
    figure.suptitle(title, fontsize=15, y=0.995)
    figure.subplots_adjust(
        left=0.02, right=0.89, top=0.965, bottom=0.02, hspace=0.11, wspace=0.02
    )
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, limit), cmap=cmap),
        cax=figure.add_axes((0.91, 0.37, 0.022, 0.28)),
        ticks=np.linspace(0, limit, 5),
        format="%.1f",
    )
    colorbar.ax.set_title("mm" if cmap == "viridis" else "kPa", fontsize=11, pad=8)
    figure.savefig(path, dpi=190, facecolor=BACKGROUND)
    plt.close(figure)


def update_status(path: Path, images: dict[str, str]) -> None:
    state = json.loads(path.read_text())
    state["comparison_images"] = images
    write_json(path, state)


def main(cfg: Config) -> None:
    source = cherries.input(cfg.source)
    summaries = {
        f"{loss}-{mode}": json.loads(
            (source / f"{loss}-{mode}" / "summary.json").read_text()
        )
        for loss in LOSSES
        for mode in MODES
        if (source / f"{loss}-{mode}" / "summary.json").is_file()
    }
    states = {
        stage: source / stage / "last.npz"
        for stage, summary in summaries.items()
        if summary.get("last_step") is not None
        and (source / stage / "last.npz").is_file()
    }
    out = cfg.output
    assets = cfg.site / "assets" / "comparison"
    out.mkdir(parents=True, exist_ok=False)
    if not states:
        update_status(cfg.site / "status.json", {})
        write_json(
            out / "summary.json",
            {
                "passed": False,
                "no_accepted_states": True,
                "message": "No accepted saved states; no atlas images were emitted.",
                "available_stages": [],
            },
        )
        return
    assets.mkdir(parents=True, exist_ok=False)
    with np.load(source / "mesh.npz", allow_pickle=False) as mesh:
        rest = np.asarray(mesh["rest_points"], float)
        tets = np.asarray(mesh["tets"], int)
        active = np.asarray(mesh["active_ids"], int)
        skin_ids = np.asarray(mesh["skin_ids"], int)
        triangles = np.asarray(mesh["triangles"], int)
        target = rest[skin_ids] + np.asarray(mesh["target_displacement_skin"], float)
    cameras = {
        item["id"]: item["camera"]
        for item in json.loads(CAMERA_RECEIPT.read_text())["views"]
    }
    view = {
        **cameras["side-context"],
        "parallel_scale": 1.12 * cameras["side-context"]["parallel_scale"],
    }
    face_paths: dict[str, Path] = {}
    q_paths: dict[str, Path] = {}
    q_values: dict[str, np.ndarray] = {}
    declared_limits: dict[str, float] = {}
    prepared: dict[str, tuple[pv.UnstructuredGrid, pv.PolyData, pv.PolyData]] = {}
    labels = {}
    for stage, summary in summaries.items():
        labels[stage] = f"{stage} | {summary.get('status', 'unknown')}"
        if stage not in states:
            continue
        with np.load(states[stage], allow_pickle=False) as saved:
            u = np.asarray(saved["u"], float)
            qhat = np.asarray(saved["Qhat"], float)
            step = int(saved["step"])
            reference = float(saved["stress_reference_MPa"])
        assert u.shape == rest.shape
        assert qhat.shape == (len(active), 3, 3)
        labels[stage] = f"{stage} | step {step} | {summary['status']}"
        result = rest[skin_ids] + u[skin_ids]
        face = poly(result, triangles)
        face.point_data["PositionErrorMM"] = 1000 * np.linalg.norm(
            result - target, axis=1
        )
        face_paths[stage] = out / f"{stage}-position-error.png"
        face_map(face, view, face_paths[stage], cfg.error_limit_mm)
        points = rest + u
        cells = np.column_stack((np.full(len(active), 4), tets[active])).ravel()
        volume = pv.UnstructuredGrid(
            cells, np.full(len(active), pv.CellType.TETRA), points
        )
        magnitude = np.linalg.norm(1000 * reference * qhat, axis=(1, 2))
        volume.cell_data["ReferenceQMagnitudeKPa"] = magnitude
        q_values[stage] = magnitude
        declared_limits[stage] = 1000 * reference
        prepared[stage] = (
            volume,
            volume.extract_surface(algorithm="dataset_surface"),
            poly(result, triangles),
        )
    q_limit = float(np.quantile(np.concatenate(list(q_values.values())), 0.99))
    all_zero = bool(q_limit == 0)
    if all_zero:
        q_limit = max(declared_limits.values())
    assert np.isfinite(q_limit)
    assert q_limit > 0
    for stage, (_volume, surface, skin) in prepared.items():
        q_paths[stage] = out / f"{stage}-reference-q.png"
        q_map(surface, skin, view, q_paths[stage], q_limit)
    face_atlas = assets / "position-error-atlas.png"
    q_atlas = assets / "reference-q-atlas.png"
    atlas(
        face_paths,
        labels,
        f"Position error atlas | shared 0–{cfg.error_limit_mm:g} mm scale",
        "viridis",
        cfg.error_limit_mm,
        face_atlas,
    )
    atlas(
        q_paths,
        labels,
        (
            f"Reference Q magnitude atlas | declared reference cap {q_limit:.3g} kPa; all Q zero"
            if all_zero
            else f"Reference Q magnitude atlas | pooled 99th cap {q_limit:.3g} kPa"
        ),
        "magma",
        q_limit,
        q_atlas,
    )
    images = {
        "position error atlas": "assets/comparison/position-error-atlas.png",
        "reference Q atlas": "assets/comparison/reference-q-atlas.png",
    }
    update_status(cfg.site / "status.json", images)
    write_json(
        out / "summary.json",
        {
            "passed": True,
            "available_stages": sorted(states),
            "labels": labels,
            "position_error_scale_mm": [0, cfg.error_limit_mm],
            "reference_Q_magnitude_kPa_pooled_99th": None if all_zero else q_limit,
            "reference_Q_display_limit_kPa": q_limit,
            "reference_Q_display_limit_source": "declared stress reference kPa"
            if all_zero
            else "pooled 99th percentile",
            "all_available_Q_zero": all_zero,
            "reference_Q_clipping": {
                stage: int(np.count_nonzero(values > q_limit))
                for stage, values in q_values.items()
            },
            "inputs": {
                "mesh": receipt(source / "mesh.npz"),
                "camera": receipt(CAMERA_RECEIPT),
                "renderer": receipt(Path(__file__)),
            },
            "images": images,
        },
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
