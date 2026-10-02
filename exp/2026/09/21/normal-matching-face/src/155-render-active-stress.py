"""Matched current-configuration active-stress views for the selected face fits."""

# ruff: noqa: SLF001

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment import Profile

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

SPEC = importlib.util.spec_from_file_location(
    "face_stress_render_base", Path(__file__).with_name("30-render.py")
)
assert SPEC
assert SPEC.loader
old = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(old)
BRANCHES = ("smooth-off-normal", "smooth-on-normal")
BACKGROUND = "#f4f2ed"
POSITIVE, NEGATIVE = "#cf5c16", "#1675b8"
MAX_LINE_M = 0.004


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    source: Path = Path("150-active-stress")
    comparison: Path = Path("132-reference-continuation")
    output: Path = Path("155-active-stress-figures")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def receipt(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def shared_sample(centers: np.ndarray, valid: np.ndarray, camera: dict, spacing: float):
    """One nearest-camera reference centroid per projected square, shared by fits."""
    back = np.asarray(camera["position"]) - camera["focal_point"]
    back /= np.linalg.norm(back)
    right = np.cross(camera["view_up"], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    centered = centers - camera["focal_point"]
    xy = np.column_stack((centered @ right, centered @ up))
    inside = np.max(np.abs(xy), axis=1) < camera["parallel_scale"] * 0.94
    ids = np.flatnonzero(valid & inside)
    grid = np.floor(xy[ids] / spacing).astype(np.int64)
    depth = centered[ids] @ back
    order = np.lexsort((-depth, grid[:, 0], grid[:, 1]))
    _, first = np.unique(grid[order], axis=0, return_index=True)
    selected = np.sort(ids[order[first]])
    assert len(selected) > 20
    return selected


def axes_mesh(field: dict, selected: np.ndarray, limit: float) -> pv.PolyData:
    values = 1000 * field["eigenvalues_MPa"][selected]
    directions = np.swapaxes(field["eigenvectors"][selected], 1, 2).reshape(-1, 3)
    centers = np.repeat(field["deformed_centers"][selected], 3, axis=0)
    values = values.ravel()
    half = 0.5 * MAX_LINE_M * np.minimum(np.abs(values) / limit, 1)
    points = np.empty((2 * len(values), 3))
    points[0::2] = centers - half[:, None] * directions
    points[1::2] = centers + half[:, None] * directions
    lines = np.column_stack(
        (
            np.full(len(values), 2),
            2 * np.arange(len(values)),
            2 * np.arange(len(values)) + 1,
        )
    ).ravel()
    mesh = pv.PolyData(points, lines=lines)
    mesh.cell_data["PrincipalStressKPa"] = values
    mesh.cell_data["GlobalCellId"] = np.repeat(field["active_ids"][selected], 3)
    mesh.cell_data["PrincipalIndexAscending"] = np.tile(np.arange(3), len(selected))
    mesh.cell_data["DisplayLengthM"] = 2 * half
    return mesh


def render(
    surface: pv.PolyData,
    skin: pv.PolyData,
    field: dict,
    sample: np.ndarray,
    camera: dict,
    mode: str,
    limits: dict,
    path: Path,
) -> None:
    p = pv.Plotter(off_screen=True, window_size=(1500, 1500), lighting="three lights")
    p.set_background(BACKGROUND)
    p.add_mesh(skin, color="#89949b", opacity=0.055, smooth_shading=False)
    invalid = ~field["valid"]
    if mode == "magnitude":
        p.add_mesh(
            surface,
            scalars="MagnitudeKPa",
            preference="cell",
            cmap="magma",
            clim=(0, limits["magnitude_kPa"]),
            nan_color="#00c4d4",
            smooth_shading=False,
            lighting=False,
            show_scalar_bar=False,
        )
    else:
        p.add_mesh(surface, color="#9da7af", opacity=0.08, show_scalar_bar=False)
        glyphs = axes_mesh(field, sample, limits["principal_kPa"])
        for positive, color in ((True, POSITIVE), (False, NEGATIVE)):
            values = glyphs.cell_data["PrincipalStressKPa"]
            mask = values > 0 if positive else values < 0
            if np.any(mask):
                p.add_mesh(
                    glyphs.extract_cells(mask),
                    color=color,
                    line_width=2.1,
                    render_lines_as_tubes=False,
                    lighting=False,
                )
        glyphs.save(path.with_suffix(".vtp"))
    if np.any(invalid):
        p.add_points(
            field["deformed_centers"][invalid],
            color="#00c4d4",
            point_size=16,
            render_points_as_spheres=True,
        )
    old._camera(p, camera)
    p.screenshot(path)
    p.close()


def plate(
    paths: list[Path],
    view: str,
    limits: dict,
    sample_count: int,
    validity_caption: str,
    path: Path,
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(13, 14.4))
    fig.patch.set_facecolor(BACKGROUND)
    for index, (ax, file) in enumerate(zip(axs.flat, paths, strict=True)):
        ax.imshow(plt.imread(file))
        ax.set_axis_off()
        branch = "Smoothness off" if index % 2 == 0 else "Smoothness on"
        quantity = "stress magnitude" if index < 2 else "principal stress axes"
        ax.set_title(f"{branch} | {quantity}", fontsize=15, pad=10)
    fig.suptitle(
        f"Activation-induced Cauchy stress | {'Full face' if view == 'full' else 'Mouth close-up'} | update 200",
        fontsize=19,
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.015, right=0.905, bottom=0.08, top=0.94, wspace=0.025, hspace=0.12
    )
    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(0, limits["magnitude_kPa"]), cmap="magma"
        ),
        cax=fig.add_axes((0.925, 0.56, 0.018, 0.32)),
        ticks=np.linspace(0, limits["magnitude_kPa"], 5),
        format="%.1f",
    )
    colorbar.ax.set_title("kPa", fontsize=13, pad=10)
    colorbar.ax.tick_params(labelsize=12)
    fig.text(
        0.04, 0.054, "Orange: positive principal stress", color=POSITIVE, fontsize=12
    )
    fig.text(
        0.50, 0.054, "Blue: negative principal stress", color=NEGATIVE, fontsize=12
    )
    fig.text(
        0.04,
        0.033,
        f"Axes: 4 mm = {limits['principal_kPa']:.1f} kPa; {sample_count:,} matched cells. Color and length capped at pooled 99th percentiles.",
        fontsize=10.5,
    )
    fig.text(
        0.04,
        0.014,
        validity_caption,
        fontsize=10.5,
    )
    fig.savefig(path, dpi=190, facecolor=BACKGROUND)
    plt.close(fig)


def main(cfg: Config) -> None:
    source = cherries.input(cfg.source)
    comparison = cherries.input(cfg.comparison)
    summary_path = source / "summary.json"
    summary = read(summary_path)
    assert summary["passed"] is True
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    fields = {}
    for branch in BRANCHES:
        with np.load(source / branch / "stress-field.npz", allow_pickle=False) as z:
            fields[branch] = {k: z[k] for k in z.files}
    a, b = (fields[name] for name in BRANCHES)
    assert np.array_equal(a["active_ids"], b["active_ids"])
    assert np.array_equal(a["rest_centers"], b["rest_centers"])
    magnitudes = np.concatenate(
        [f["magnitude_kPa"][f["valid"]] for f in fields.values()]
    )
    eigenvalues = np.concatenate(
        [
            1000 * np.abs(f["eigenvalues_MPa"][f["valid"]]).ravel()
            for f in fields.values()
        ]
    )
    limits = {
        "magnitude_kPa": float(np.quantile(magnitudes, 0.99)),
        "principal_kPa": float(np.quantile(eigenvalues, 0.99)),
    }
    assert all(np.isfinite(v) and v > 0 for v in limits.values())
    with np.load(comparison / "mesh.npz", allow_pickle=False) as mesh:
        rest, skin_ids, triangles = (
            mesh["rest_points"],
            mesh["skin_ids"],
            mesh["triangles"],
        )
    cameras = {v["id"]: v["camera"] for v in read(old.RECEIPT)["views"]}
    receipts = {
        "extraction": receipt(summary_path),
        "camera": receipt(old.RECEIPT),
        "renderer": receipt(Path(__file__)),
    }
    result = {
        "passed": True,
        "limits": limits,
        "scales": "Pooled per-cell 99th percentiles, shared between both branches; raw exports are not clipped.",
        "glyph_selection": "One frontmost reference centroid per projected bin, using the same global cell IDs in both fits. This is a sparse reference-selected illustration, not exact current-state occlusion.",
        "views": {},
        "branches": {},
        "inputs": receipts,
    }
    surfaces, skins = {}, {}
    for branch, f in fields.items():
        f["total_inversions"] = int(
            read(comparison / branch / "summary.json")["last_metrics"][
                "inverted_all_cells"
            ]
        )
        volume = pv.read(source / branch / "stress.vtu")
        assert np.array_equal(volume.cell_data["GlobalCellId"], f["active_ids"])
        volume.cell_data["MagnitudeKPa"] = f["magnitude_kPa"]
        surfaces[branch] = volume.extract_surface(algorithm="dataset_surface")
        with np.load(comparison / branch / "last.npz", allow_pickle=False) as z:
            assert int(z["step"]) == 200
            skins[branch] = old._poly((rest + z["u"])[skin_ids], triangles)
        skins[branch].save(out / f"{branch}-skin.vtp")
        surfaces[branch].save(out / f"{branch}-stress-surface.vtp")
        valid = f["valid"]
        result["branches"][branch] = {
            "magnitude_clipped_cells": int(
                np.count_nonzero(f["magnitude_kPa"][valid] > limits["magnitude_kPa"])
            ),
            "principal_clipped_modes": int(
                np.count_nonzero(
                    1000 * np.abs(f["eigenvalues_MPa"][valid]) > limits["principal_kPa"]
                )
            ),
            "magnitude_max_kPa": float(np.max(f["magnitude_kPa"][valid])),
            "field": receipt(source / branch / "stress-field.npz"),
            "mesh": receipt(source / branch / "stress.vtu"),
        }
    for view, camera_id, spacing in (
        ("full", "side-context", 0.002),
        ("mouth", "region1-mouth-corner", 0.0008),
    ):
        camera = dict(cameras[camera_id])
        if view == "full":
            camera["parallel_scale"] *= 1.1
        sample = shared_sample(
            a["rest_centers"], a["valid"] & b["valid"], camera, spacing
        )
        np.save(out / f"{view}-sample-global-ids.npy", a["active_ids"][sample])
        result["views"][view] = {
            "camera": camera,
            "spacing_m": spacing,
            "sample_cells": len(sample),
        }
        images = []
        for mode in ("magnitude", "axes"):
            for branch in BRANCHES:
                path = out / f"{view}-{mode}-{branch}.png"
                render(
                    surfaces[branch],
                    skins[branch],
                    fields[branch],
                    sample,
                    camera,
                    mode,
                    limits,
                    path,
                )
                images.append(path)
        plate(
            images,
            view,
            limits,
            len(sample),
            "Cyan: "
            + "/".join(
                str(int(np.count_nonzero(~fields[b]["valid"]))) for b in BRANCHES
            )
            + " inverted active cells excluded (off/on); "
            + "/".join(str(fields[b]["total_inversions"]) for b in BRANCHES)
            + " inverted tets overall. Raw exports are unclipped.",
            out / f"{view}-active-stress.png",
        )
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
