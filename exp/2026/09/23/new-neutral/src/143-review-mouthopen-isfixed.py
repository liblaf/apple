"""Render a saved corrected-neutral MouthOpen inverse endpoint without solving."""

# ruff: noqa: PLR0915

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "rigid_review", GROUP / "src/131-review-rigid-inverse.py"
)
assert SPEC is not None
assert SPEC.loader is not None
review = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = review
SPEC.loader.exec_module(review)

RIGID_COLORS = {"cranium": "#e5d8bb", "mandible": "#cdb787", "eyes": "#91c9da"}


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/inverse-mouthopen-isfixed-003"
    neutral_dir: Path = GROUP / "data/forward-isfixed-001"
    output_dir: Path = GROUP / "data/review-mouthopen-isfixed-002"
    parent_run_dir: Path = GROUP / "data/inverse-mouthopen-isfixed-002"


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verified(item: dict[str, Any]) -> Path:
    path = Path(item["path"])
    assert record(path) == item, path
    return path


def faces(triangles: np.ndarray) -> np.ndarray:
    assert triangles.ndim == 2
    assert triangles.shape[1] == 3
    assert triangles.min() >= 0
    return np.column_stack((np.full(len(triangles), 3), triangles)).ravel()


def surface(points: np.ndarray, ids: np.ndarray, triangles: np.ndarray) -> pv.PolyData:
    assert points.shape == (len(ids), 3)
    assert len(ids) == len(np.unique(ids))
    assert triangles.max() < len(ids)
    mesh = pv.PolyData(points, faces(triangles))
    mesh.point_data["GlobalPointId"] = ids
    return mesh


def camera(meshes: list[pv.DataSet], view: str) -> tuple[list, float]:
    points = np.concatenate([np.asarray(mesh.points) for mesh in meshes])
    low, high = points.min(axis=0), points.max(axis=0)
    center, span = (low + high) / 2, float(np.max(high - low))
    direction = np.asarray((0, 0, 2.7) if view == "front" else (2.7, 0, 0))
    return [list(center + span * direction), list(center), [0, 1, 0]], 0.60 * span


def add_anatomy(plot: pv.Plotter, anatomy: dict[str, pv.PolyData]) -> None:
    for name, mesh in anatomy.items():
        plot.add_mesh(
            mesh,
            color=RIGID_COLORS[name],
            smooth_shading=True,
            opacity=0.82,
        )


def render_target_fit(
    output: Path,
    target: pv.PolyData,
    fit_boundary: pv.PolyData,
    anatomy: dict[str, pv.PolyData],
    *,
    view: str,
) -> str:
    position, scale = camera([target, fit_boundary, *anatomy.values()], view)
    plot = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2000, 1000))
    plot.set_background("#f7f7f5")
    for column, (title, mesh, color) in enumerate(
        (
            ("Transferred MouthOpen target skin", target, "#737b86"),
            ("Saved inverse FEM fit full tetmesh boundary", fit_boundary, "#c75f42"),
        )
    ):
        plot.subplot(0, column)
        if column == 1:
            add_anatomy(plot, anatomy)
        plot.add_mesh(mesh, color=color, smooth_shading=True)
        plot.add_text(
            f"{title}\n{view} · identical true scale",
            position="upper_left",
            font_size=15,
            color="#202124",
        )
        plot.camera_position = position
        plot.camera.parallel_projection = True
        plot.camera.parallel_scale = scale
        plot.reset_camera_clipping_range()
    name = f"target-vs-fit-{view}.png"
    plot.show(screenshot=output / name, auto_close=True)
    return name


def render_tetmesh(
    output: Path,
    neutral: pv.PolyData,
    fit: pv.PolyData,
    neutral_anatomy: dict[str, pv.PolyData],
    fit_anatomy: dict[str, pv.PolyData],
    *,
    view: str,
    edges: bool = False,
) -> str:
    position, scale = camera(
        [neutral, fit, *neutral_anatomy.values(), *fit_anatomy.values()], view
    )
    plot = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2000, 1000))
    plot.set_background("#f7f7f5")
    for column, (title, mesh, anatomy, color) in enumerate(
        (
            ("Corrected neutral tetmesh boundary", neutral, neutral_anatomy, "#737b86"),
            ("Saved inverse fit tetmesh boundary", fit, fit_anatomy, "#c75f42"),
        )
    ):
        plot.subplot(0, column)
        add_anatomy(plot, anatomy)
        plot.add_mesh(
            mesh,
            color=color,
            smooth_shading=not edges,
            show_edges=edges,
            edge_color="#3c3734",
            line_width=0.45,
            ambient=0.25,
            diffuse=0.7,
        )
        plot.add_text(
            f"{title}\ncomplete boundary{' with edges' if edges else ''} · {view} · true scale",
            position="upper_left",
            font_size=15,
            color="#202124",
        )
        plot.camera_position = position
        plot.camera.parallel_projection = True
        plot.camera.parallel_scale = scale
        plot.reset_camera_clipping_range()
    name = f"neutral-vs-fit-tetmesh-{'edges-' if edges else ''}{view}.png"
    plot.show(screenshot=output / name, auto_close=True)
    return name


def render_error(
    output: Path, fit: pv.PolyData, target: pv.PolyData, *, view: str
) -> str:
    error_mm = np.linalg.norm(
        np.asarray(fit.points) - np.asarray(target.points), axis=1
    )
    error_mm *= 1000
    plot = pv.Plotter(off_screen=True, window_size=(1200, 1000))
    plot.set_background("#f7f7f5")
    plot.add_mesh(
        fit,
        scalars=error_mm,
        cmap="magma",
        clim=(0, float(np.quantile(error_mm, 0.99))),
        scalar_bar_args={"title": "target error (mm)"},
        smooth_shading=True,
    )
    position, scale = camera([fit], view)
    plot.add_text(
        f"Transferred MouthOpen target error on saved FEM skin\n{view} · 99th percentile color scale",
        position="upper_left",
        font_size=15,
        color="#202124",
    )
    plot.camera_position = position
    plot.camera.parallel_projection = True
    plot.camera.parallel_scale = scale
    plot.reset_camera_clipping_range()
    name = f"fit-error-{view}.png"
    plot.show(screenshot=output / name, auto_close=True)
    return name


def curves(
    output: Path, rows: list[dict], error_mm: np.ndarray, parent_rows: list[dict]
) -> str:
    combined_rows = list(parent_rows)
    if parent_rows:
        parent_last = parent_rows[-1]["iteration"]
        continuation = [
            {**row, "iteration": parent_last + row["iteration"]} for row in rows
        ]
        if continuation and continuation[0]["iteration"] == parent_last:
            continuation = continuation[1:]
        combined_rows.extend(continuation)
    else:
        combined_rows = rows
    assert combined_rows
    iterations = np.asarray([row["iteration"] for row in combined_rows])
    fit_rms = np.asarray([row["fit_rms_mm"] for row in combined_rows])
    force = np.asarray([row["force_norm_n"] for row in combined_rows])
    threshold = np.asarray([row["force_threshold_n"] for row in combined_rows])
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    axes[0].plot(iterations, fit_rms, color="#c75f42", marker="o")
    axes[0].set(
        xlabel="Accepted optimizer iteration", ylabel="Fit RMS (mm)", title="Fit"
    )
    axes[1].semilogy(iterations, force, color="#087d81", marker="o", label="force")
    axes[1].semilogy(iterations, threshold, color="#555", ls=":", label="threshold")
    axes[1].set(
        xlabel="Accepted optimizer iteration",
        ylabel="Residual (N)",
        title="Forward force",
    )
    axes[1].legend()
    ordered = np.sort(error_mm)
    axes[2].plot(ordered, np.linspace(0, 1, len(ordered)), color="#6a4c93")
    axes[2].set(
        xlabel="Target error (mm)",
        ylabel="Cumulative skin fraction",
        title="Saved fit error",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    name = "fit-error-force-curves.png"
    figure.savefig(output / name, dpi=180)
    plt.close(figure)
    return name


def main(cfg: Config) -> None:
    run, neutral_dir, output, parent_run = (
        cfg.run_dir.resolve(),
        cfg.neutral_dir.resolve(),
        cfg.output_dir.resolve(),
        cfg.parent_run_dir.resolve(),
    )
    assert not output.exists(), output
    paths = {
        "protocol": run / "protocol.json",
        "summary": run / "summary.json",
        "progress": run / "progress.jsonl",
        "endpoint": run / "endpoint.npz",
    }
    # A saved endpoint is required: this renderer never replaces an unfinished fit
    # with its neutral initialization.
    assert all(path.is_file() for path in paths.values()), paths
    inputs = {name: record(path) for name, path in paths.items()}
    parent_paths = {
        "protocol": parent_run / "protocol.json",
        "summary": parent_run / "summary.json",
        "progress": parent_run / "progress.jsonl",
        "endpoint": parent_run / "endpoint.npz",
        "checkpoint": parent_run / "checkpoint.pt",
    }
    assert all(path.is_file() for path in parent_paths.values()), parent_paths
    parent_inputs = {name: record(path) for name, path in parent_paths.items()}
    parent_summary = json.loads(parent_paths["summary"].read_text())
    assert parent_summary["endpoint"] == parent_inputs["endpoint"]
    assert parent_summary["status"] == "pose_step_exhausted"
    protocol, summary = (
        json.loads(paths[name].read_text()) for name in ("protocol", "summary")
    )
    assert protocol["schema"] == "new-neutral-mouthopen-rigid6-inverse-v1"
    assert protocol["expression_name"] == "MouthOpen"
    assert summary["endpoint"] == inputs["endpoint"]
    rendering_path = verified(protocol["rendering"]["archive"])
    blendshape_path = verified(protocol["sources"]["blendshapes"])
    neutral_endpoint_path = verified(protocol["sources"]["neutral_endpoint"])
    neutral_summary_path = verified(protocol["sources"]["neutral_summary"])
    inputs.update(
        rendering=record(rendering_path),
        blendshapes=record(blendshape_path),
        neutral_endpoint=record(neutral_endpoint_path),
        neutral_summary=record(neutral_summary_path),
    )
    neutral_protocol = json.loads((neutral_dir / "protocol.json").read_text())
    assert record(neutral_dir / "summary.json") == inputs["neutral_summary"]
    reference_path = verified(
        neutral_protocol["reference_configuration"]["constitutive_volume"]
    )
    coverage = neutral_protocol["coverage"]
    assert coverage["coverage"]["soft_cranium"]
    assert coverage["coverage"]["soft_mandible"]
    assert coverage["coverage"]["soft_eyes"]
    assert coverage["coverage"]["soft_soft"] is False
    assert coverage["coverage"]["rigid_rigid"] is False
    inputs["corrected_reference_volume"] = record(reference_path)
    volume = pv.read(reference_path)
    assert isinstance(volume, pv.UnstructuredGrid)
    assert np.all(volume.celltypes == pv.CellType.TETRA)
    with np.load(rendering_path, allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    required = {
        "full_reference_points_m",
        "skin_global_ids",
        "skin_triangles",
        "cranium_global_ids",
        "cranium_triangles",
        "mandible_global_ids",
        "mandible_triangles",
        "eye_global_ids",
        "eye_triangles",
    }
    assert set(fields) == required
    full_reference = fields["full_reference_points_m"]
    assert np.array_equal(full_reference[: volume.n_points], volume.points)
    with np.load(paths["endpoint"], allow_pickle=False) as archive:
        displacement = archive["displacement_m"]
        pose = archive["pose_rad_m"]
    with np.load(neutral_endpoint_path, allow_pickle=False) as archive:
        neutral_displacement = archive["displacement_m"]
    assert displacement.shape == full_reference.shape
    assert np.isfinite(displacement).all()
    assert neutral_displacement.shape == volume.points.shape
    assert pose.shape == (6,)
    assert np.isfinite(pose).all()
    final = summary["final"]
    assert np.array_equal(pose, np.asarray(final["pose_rad_m"]))
    rows = [json.loads(line) for line in paths["progress"].read_text().splitlines()]
    assert rows
    assert rows[-1]["iteration"] == final["iteration"]
    parent_rows = [
        json.loads(line) for line in parent_paths["progress"].read_text().splitlines()
    ]
    assert parent_rows
    assert parent_rows[-1]["iteration"] == parent_summary["final"]["iteration"]
    with np.load(blendshape_path, allow_pickle=False) as archive:
        index = [str(name) for name in archive["expression_names"]].index("MouthOpen")
        skin_ids, skin_triangles = archive["skin_global_ids"], archive["skin_triangles"]
        target = surface(archive["target_points_m"][index], skin_ids, skin_triangles)
    assert np.array_equal(skin_ids, fields["skin_global_ids"])
    fit_skin = surface(
        full_reference[skin_ids] + displacement[skin_ids], skin_ids, skin_triangles
    )
    neutral_volume = volume.copy(deep=True)
    neutral_volume.points = volume.points + neutral_displacement
    fit_volume = volume.copy(deep=True)
    fit_volume.points = volume.points + displacement[: volume.n_points]
    neutral_boundary, fit_boundary = (
        mesh.extract_surface(algorithm=None) for mesh in (neutral_volume, fit_volume)
    )
    assert np.array_equal(neutral_boundary.faces, fit_boundary.faces)
    neutral_full_displacement = np.zeros_like(full_reference)
    neutral_full_displacement[: volume.n_points] = neutral_displacement
    anatomy = {}
    for key, label in (
        ("cranium", "cranium"),
        ("mandible", "mandible"),
        ("eye", "eyes"),
    ):
        ids, triangles = fields[f"{key}_global_ids"], fields[f"{key}_triangles"]
        anatomy[label] = (
            surface(
                full_reference[ids] + neutral_full_displacement[ids], ids, triangles
            ),
            surface(full_reference[ids] + displacement[ids], ids, triangles),
        )
    neutral_anatomy = {name: pair[0] for name, pair in anatomy.items()}
    fit_anatomy = {name: pair[1] for name, pair in anatomy.items()}
    error_mm = (
        np.linalg.norm(np.asarray(fit_skin.points) - np.asarray(target.points), axis=1)
        * 1000
    )
    output.mkdir(parents=True)
    shutil.copy2(__file__, output / Path(__file__).name)
    fit_skin.save(output / "fit-skin.vtp")
    target.save(output / "mouthopen-transferred-skin-target.vtp")
    neutral_boundary.save(output / "neutral-full-tetmesh-boundary.vtp")
    fit_boundary.save(output / "fit-full-tetmesh-boundary.vtp")
    images = [
        *(
            render_target_fit(output, target, fit_boundary, fit_anatomy, view=view)
            for view in ("front", "side")
        ),
        render_tetmesh(
            output,
            neutral_boundary,
            fit_boundary,
            neutral_anatomy,
            fit_anatomy,
            view="front",
            edges=True,
        ),
        *(
            render_tetmesh(
                output,
                neutral_boundary,
                fit_boundary,
                neutral_anatomy,
                fit_anatomy,
                view=view,
            )
            for view in ("front", "side")
        ),
        render_error(output, fit_skin, target, view="front"),
        curves(output, rows, error_mm, parent_rows),
    ]
    status = {
        "saved_status": summary["status"],
        "forward_converged": bool(final["forward_converged"]),
        "contact_valid": bool(final["contact_valid"]),
        "forward_valid": bool(final["valid_forward"]),
        "inverse_converged": bool(summary["inverse_converged"]),
        "inverted_tetrahedra": int(final["geometry"]["inverted_tetrahedra"]),
        "detF_min": float(final["geometry"]["detF_min"]),
    }
    receipt = {
        "schema": "corrected-neutral-mouthopen-saved-review-v1",
        "inputs": inputs,
        "parent_002_binding": {
            "run": parent_inputs,
            "status": parent_summary["status"],
            "accepted_optimizer_iterations": parent_summary["final"]["iteration"],
            "endpoint_hash": parent_inputs["endpoint"]["sha256"],
        },
        "status": status,
        "geometry": {
            "fem_vertices": volume.n_points,
            "tetrahedra": volume.n_cells,
            "boundary_triangles": fit_boundary.n_cells,
        },
        "target": {
            "name": "MouthOpen",
            "expression_index": index,
            "scope": "Transferred skin-only kinematic target; no volumetric target was invented.",
            "vertices": target.n_points,
            "triangles": target.n_cells,
        },
        "fit_error_mm": {
            "unweighted_rms": float(
                np.sqrt(np.mean(error_mm**2)),
            ),
            "mean": float(error_mm.mean()),
            "max": float(error_mm.max()),
            "p95": float(np.quantile(error_mm, 0.95)),
        },
        "material_and_contact": {
            "active_strain": protocol["materials"]["formulation"],
            "skin_prestretch": protocol["materials"]["skin"],
            "ipc_policy": protocol["ipc_policy"],
            "ipc_stiffness_mpa": protocol["ipc_stiffness_mpa"],
            "collision_scope": "Soft tissue against complete cranium, mandible, and fixed eyes. Soft-soft and rigid-rigid contact are disabled. Rendering performs no collision query.",
        },
        "optimizer_fit_rms_mm": {
            "area_weighted_initial": float(summary["initial"]["fit_rms_mm"]),
            "area_weighted_final": float(final["fit_rms_mm"]),
        },
        "saved_pose_rad_m": pose.tolist(),
        "curve_lineage": {
            "parent_accepted_optimizer_iterations": parent_rows[-1]["iteration"],
            "continuation_accepted_optimizer_iterations": final["iteration"],
            "total_accepted_optimizer_iterations": parent_rows[-1]["iteration"]
            + final["iteration"],
        },
        "solver_rerun": False,
        "assets": {
            path.name: record(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    image_html = "".join(
        f'<figure><img src="{name}" alt="Saved MouthOpen fit review"></figure>'
        for name in images
    )
    (output / "index.html").write_text(
        f"""<!doctype html><meta charset=\"utf-8\"><title>MouthOpen from corrected neutral</title><style>body{{font:16px/1.5 system-ui;max-width:1500px;margin:2rem auto;padding:0 1rem}}img{{max-width:100%}}.status{{padding:1rem;background:#fff1c7}}details{{margin:1rem 0;padding:0.8rem;background:#f0f0ee}}</style><p><a href=\"../\">Back to corrected neutral</a></p><h1>MouthOpen fit from corrected neutral</h1><p class=\"status\"><strong>Limited fit after {receipt["curve_lineage"]["total_accepted_optimizer_iterations"]} accepted updates; inverse optimization did not converge.</strong> The saved forward endpoint is converged, contact-valid, and forward-valid.</p><p>Area-weighted optimizer fit RMS: {receipt["optimizer_fit_rms_mm"]["area_weighted_initial"]:.4f} → {receipt["optimizer_fit_rms_mm"]["area_weighted_final"]:.4f} mm. The surface error below is unweighted: RMS {receipt["fit_error_mm"]["unweighted_rms"]:.4f} mm, p95 {receipt["fit_error_mm"]["p95"]:.4f} mm, maximum {receipt["fit_error_mm"]["max"]:.4f} mm.</p><p>Geometry: {status["inverted_tetrahedra"]:,} inverted tetrahedra; min detF {status["detF_min"]:.6g}. This page is a saved-state rendering and does not re-evaluate physics.</p><p>The transferred MouthOpen target is a skin-only kinematic target. The smooth full-tetmesh views compare the corrected neutral and the saved fitted FEM volume; the final front view adds element edges. Active strain uses the recorded multiplicative formulation with recorded skin prestretch. Collision covers soft tissue against complete cranium, mandible, and fixed eyes; soft-soft and rigid-rigid contact are disabled.</p>{image_html}<details><summary>Provenance</summary><p>Parent 002 endpoint SHA-256: {receipt["parent_002_binding"]["endpoint_hash"]}. Full input and asset hashes are in <a href=\"receipt.json\">receipt.json</a>.</p></details><p><a href=\"mouthopen-transferred-skin-target.vtp\">Transferred target skin</a> · <a href=\"neutral-full-tetmesh-boundary.vtp\">Neutral full boundary</a> · <a href=\"fit-full-tetmesh-boundary.vtp\">Fit full boundary</a></p>"""
    )
    receipt["assets"]["index.html"] = record(output / "index.html")
    write_json(output / "receipt.json", receipt)
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "review/fit_rms_mm": receipt["fit_error_mm"]["unweighted_rms"],
            "review/forward_valid": float(status["forward_valid"]),
            "review/inverse_converged": float(status["inverse_converged"]),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
