"""Render the exact frozen skin priors used by the diagnostic forward solve."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs, array_sha256

from liblaf import cherries


class Config(cherries.BaseConfig):
    input_dir: Path = GROUP / "data/simple-skin-forward-inputs-001"
    output_dir: Path = GROUP / "data/prescribed-skin-field-visuals-001"


def render(skin: pv.PolyData, field: str, cmap: str, view: str) -> np.ndarray:
    plot = pv.Plotter(off_screen=True, window_size=(900, 1100))
    plot.set_background("white")
    plot.enable_anti_aliasing("ssaa")
    plot.add_mesh(
        skin,
        scalars=field,
        preference="cell",
        cmap=cmap,
        clim=(float(skin[field].min()), float(skin[field].max())),
        lighting=False,
        show_scalar_bar=False,
    )
    center = np.asarray(skin.center)
    direction = np.array([0, 0, 1]) if view == "front" else np.array([1, 0, 0])
    plot.camera_position = (center + direction, center, (0, 1, 0))
    plot.enable_parallel_projection()
    bounds = skin.bounds
    width = bounds[1] - bounds[0] if view == "front" else bounds[5] - bounds[4]
    plot.camera.parallel_scale = 1.04 * max(
        (bounds[3] - bounds[2]) / 2, width / (2 * 900 / 1100)
    )
    result = plot.screenshot(return_img=True)
    plot.close()
    return result


def figure(
    path: Path,
    panels: list[tuple[np.ndarray, str, str, tuple[float, float], str]],
) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(12, 8.5), layout="constrained")
    for ax, (pixels, title, cmap, limits, unit) in zip(axes, panels, strict=True):
        ax.imshow(pixels)
        ax.axis("off")
        ax.set_title(title, fontsize=16, fontweight="semibold", pad=10)
        scale = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(*limits), cmap=cmap)
        bar = fig.colorbar(
            scale, ax=ax, orientation="horizontal", fraction=0.045, pad=0.01
        )
        bar.set_label(unit, fontsize=13)
        bar.ax.tick_params(labelsize=11)
    fig.suptitle(
        "Prescribed skin fields | exact forward inputs\n"
        "Literature inverse fits + manual regional transfer; not a measured subject map",
        fontsize=14,
    )
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    manifest_path = cfg.input_dir / "skin-field-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    field_path = cfg.input_dir / "skin-field.npz"
    assert sha256(field_path) == manifest["artifact"]["sha256"]
    for key, filename in (("npz", "inputs.npz"), ("manifest", "manifest.json")):
        assert (
            sha256(cfg.input_dir / "prepared" / filename)
            == manifest["prepared_inputs"][f"{key}_sha256"]
        )
    prepared = PreparedInputs.load(
        cfg.input_dir / "prepared/inputs.npz", cfg.input_dir / "prepared/manifest.json"
    )
    assert sha256(prepared.skin_path) == manifest["skin_reference"]["sha256"]
    skin = pv.read(prepared.skin_path)
    with np.load(field_path) as archive:
        fields = {key: archive[key] for key in archive.files}
    assert set(fields) == set(manifest["arrays"])
    for key, value in fields.items():
        assert array_sha256(value) == manifest["arrays"][key]["sha256"]
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    local_faces = np.asarray(skin.faces).reshape(-1, 4)
    assert np.all(local_faces[:, 0] == 3)
    assert np.array_equal(ids[local_faces[:, 1:]], fields["skin_triangles"])
    baseline = fields["baseline_N_per_m"]
    assert np.array_equal(baseline[:, 0, 0], baseline[:, 1, 1])
    assert np.all(baseline[:, 0, 1] == 0)
    assert np.all(baseline[:, 1, 0] == 0)
    skin.cell_data["E_kPa"] = 1000 * fields["E_mpa"]
    skin.cell_data["N0_N_per_m"] = baseline[:, 0, 0]
    skin.cell_data["mean_prestress_kPa"] = baseline[:, 0, 0] / fields["h_m"] / 1000
    skin.cell_data["thickness_m"] = fields["h_m"]
    skin.save(cfg.output_dir / "prescribed-skin-fields.vtp")
    panels = {}
    ranges = {}
    for field, cmap, title, unit in (
        ("E_kPa", "viridis", "Skin stiffness E", "Derived modulus (kPa)"),
        (
            "N0_N_per_m",
            "magma",
            "Skin prestress N0",
            "Isotropic membrane stress resultant (N/m)",
        ),
    ):
        limits = (float(skin[field].min()), float(skin[field].max()))
        ranges[field] = limits
        for view in ("front", "profile"):
            panels[field, view] = (
                render(skin, field, cmap, view),
                f"{title} | {view}\n{limits[0]:.1f} to {limits[1]:.1f}",
                cmap,
                limits,
                unit,
            )
    assets = []
    for filename, selected, caption in (
        (
            "00-skin-fields-overview.png",
            [("E_kPa", "front"), ("N0_N_per_m", "front")],
            "Exact prescribed inputs: stiffness 127.4-257.9 kPa and isotropic skin prestress resultant 30.4-80.8 N/m. At the assumed 1 mm thickness, the latter equals 30.4-80.8 kPa mean 3D stress numerically. These fields are priors, not optimized results.",
        ),
        (
            "01-skin-stiffness-front-profile.png",
            [("E_kPa", "front"), ("E_kPa", "profile")],
            "Front and side views use the same stiffness scale. Values derive from Flynn's regional Ogden fits and a smooth 20 mm Gaussian blend of manual bilateral anchors. Flat lighting preserves the scalar colors.",
        ),
        (
            "02-skin-prestress-front-profile.png",
            [("N0_N_per_m", "front"), ("N0_N_per_m", "profile")],
            "Front and side views use the same prestress scale. The two fitted directional stresses are averaged into an isotropic reference membrane stress. Nose, lips, eyelids and scalp values are extrapolated.",
        ),
    ):
        path = cfg.output_dir / filename
        figure(path, [panels[key] for key in selected])
        assets.append(
            {"filename": filename, "caption": caption, "sha256": sha256(path)}
        )
    write_json(
        cfg.output_dir / "summary.json",
        {
            "schema": "joint-prescribed-skin-field-visuals-v1",
            "input_field_sha256": sha256(field_path),
            "input_manifest_sha256": sha256(manifest_path),
            "triangles": skin.n_cells,
            "points": skin.n_points,
            "ranges": ranges,
            "field_construction": manifest["field_construction"],
            "source": manifest["source"],
            "assets": assets,
            "interpretation": "Exact frozen prescribed fields; no physics parameters or simulation states were changed.",
        },
    )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
