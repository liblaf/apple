"""Render an audited collision-off expression endpoint without solving."""

from __future__ import annotations

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
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
sys.path[:0] = [str(GROUP / "src"), str(JOINT)]
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402

RIGID_COLORS = {"cranium": "#e5d8bb", "mandible": "#cdb787", "eyes": "#91c9da"}


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/inverse-mouthopen-collision-off"
    neutral_dir: Path = GROUP / "data/forward-isfixed-001"
    output_dir: Path = GROUP / "data/review-mouthopen-collision-off"
    parent_run_dir: Path | None = None


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verified(item: dict[str, str]) -> Path:
    path = Path(item["path"])
    candidates = [path]
    if "data" in path.parts:
        candidates.append(
            GROUP / "data" / Path(*path.parts[path.parts.index("data") + 1 :])
        )
    for candidate in candidates:
        if candidate.is_file() and sha256(candidate) == item["sha256"]:
            return candidate
    raise AssertionError(item)


def progress(run: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (run / "progress.jsonl").read_text().splitlines()
        if line
    ]
    assert rows
    return rows


def lineage(run: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return oldest-first, de-duplicated rows and verified parent receipts."""
    protocol = json.loads((run / "protocol.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    rows = progress(run)
    assert rows[-1]["iteration"] == summary["final"]["iteration"]
    initialization = protocol["initialization"]
    continued = bool(initialization["optimizer_state"]["continued"])
    if not continued:
        return [dict(row, lineage_run=run.name) for row in rows], []
    checkpoint = verified(initialization["checkpoint"])
    endpoint = verified(initialization["endpoint"])
    parent = checkpoint.parent
    assert parent / "protocol.json" != run / "protocol.json"
    assert sha256(parent / "checkpoint.pt") == sha256(checkpoint)
    assert sha256(parent / "endpoint.npz") == sha256(endpoint)
    parent_protocol = json.loads((parent / "protocol.json").read_text())
    assert parent_protocol["expression_name"] == protocol["expression_name"]
    parent_summary = json.loads((parent / "summary.json").read_text())
    assert parent_summary["endpoint"]["sha256"] == sha256(parent / "endpoint.npz")
    parent_rows, parents = lineage(parent)
    parent_last, child_initial = parent_rows[-1], rows[0]
    np.testing.assert_allclose(
        parent_last["fit_rms_mm"], child_initial["fit_rms_mm"], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        parent_last["force_norm_n"],
        child_initial["force_norm_n"],
        rtol=1e-8,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        parent_last["pose_rad_m"], child_initial["pose_rad_m"], rtol=0, atol=1e-15
    )
    assert parent_last["optimizer_steps"] == child_initial["optimizer_steps"]
    binding = {
        "run_dir": str(parent.resolve()),
        "protocol": record(parent / "protocol.json"),
        "summary": record(parent / "summary.json"),
        "progress": record(parent / "progress.jsonl"),
        "endpoint": record(parent / "endpoint.npz"),
        "checkpoint": record(parent / "checkpoint.pt"),
    }
    current = [dict(row, lineage_run=run.name) for row in rows[1:]]
    return parent_rows + current, [*parents, binding]


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
        plot.add_mesh(mesh, color=RIGID_COLORS[name], smooth_shading=True, opacity=0.82)


def target_fit_image(
    output: Path,
    target: pv.PolyData,
    fit: pv.PolyData,
    anatomy: dict[str, pv.PolyData],
    view: str,
    expression: str,
) -> str:
    position, scale = camera([target, fit, *anatomy.values()], view)
    plot = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2000, 1000))
    plot.set_background("#f7f7f5")
    for column, (title, mesh, color) in enumerate(
        (
            (f"Transferred {expression} target skin", target, "#737b86"),
            ("Saved inverse FEM fit full tetmesh boundary", fit, "#c75f42"),
        )
    ):
        plot.subplot(0, column)
        if column:
            add_anatomy(plot, anatomy)
        plot.add_mesh(mesh, color=color, smooth_shading=True)
        plot.add_text(
            f"{title}\n{view} · identical true scale",
            position="upper_left",
            font_size=15,
            color="#202124",
        )
        (
            plot.camera_position,
            plot.camera.parallel_projection,
            plot.camera.parallel_scale,
        ) = position, True, scale
        plot.reset_camera_clipping_range()
    name = f"target-vs-fit-{view}.png"
    plot.show(screenshot=output / name, auto_close=True)
    return name


def boundary_image(
    output: Path,
    neutral: pv.PolyData,
    fit: pv.PolyData,
    neutral_anatomy: dict[str, pv.PolyData],
    fit_anatomy: dict[str, pv.PolyData],
    view: str,
) -> str:
    position, scale = camera(
        [neutral, fit, *neutral_anatomy.values(), *fit_anatomy.values()], view
    )
    plot = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2000, 1000))
    plot.set_background("#f7f7f5")
    for column, (title, mesh, anatomy, color) in enumerate(
        (
            (
                "Corrected neutral full tetmesh boundary",
                neutral,
                neutral_anatomy,
                "#737b86",
            ),
            (
                "Saved collision-off fit full tetmesh boundary",
                fit,
                fit_anatomy,
                "#c75f42",
            ),
        )
    ):
        plot.subplot(0, column)
        add_anatomy(plot, anatomy)
        plot.add_mesh(mesh, color=color, smooth_shading=True)
        plot.add_text(
            f"{title}\n{view} · true scale",
            position="upper_left",
            font_size=15,
            color="#202124",
        )
        (
            plot.camera_position,
            plot.camera.parallel_projection,
            plot.camera.parallel_scale,
        ) = position, True, scale
        plot.reset_camera_clipping_range()
    name = f"neutral-vs-fit-tetmesh-{view}.png"
    plot.show(screenshot=output / name, auto_close=True)
    return name


def error_image(
    output: Path, fit: pv.PolyData, target: pv.PolyData, view: str, expression: str
) -> str:
    error_mm = (
        np.linalg.norm(np.asarray(fit.points) - np.asarray(target.points), axis=1)
        * 1000
    )
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
        f"Transferred {expression} target error on saved FEM skin\n{view} · 99th percentile color scale",
        position="upper_left",
        font_size=15,
        color="#202124",
    )
    (
        plot.camera_position,
        plot.camera.parallel_projection,
        plot.camera.parallel_scale,
    ) = position, True, scale
    plot.reset_camera_clipping_range()
    name = f"fit-error-{view}.png"
    plot.show(screenshot=output / name, auto_close=True)
    return name


def curves(output: Path, rows: list[dict[str, Any]], error_mm: np.ndarray) -> str:
    iterations = np.asarray([row["optimizer_steps"]["q"] for row in rows])
    fit_rms = np.asarray([row["fit_rms_mm"] for row in rows])
    force = np.asarray([row["force_norm_n"] for row in rows])
    threshold = np.asarray([row["force_threshold_n"] for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    axes[0].plot(iterations, fit_rms, color="#c75f42", marker="o")
    axes[0].set(xlabel="Global q optimizer step", ylabel="Fit RMS (mm)", title="Fit")
    axes[1].semilogy(iterations, force, color="#087d81", marker="o", label="force")
    axes[1].semilogy(iterations, threshold, color="#555", ls=":", label="threshold")
    axes[1].set(
        xlabel="Global q optimizer step", ylabel="Residual (N)", title="Forward force"
    )
    axes[1].legend()
    axes[2].plot(np.sort(error_mm), np.linspace(0, 1, len(error_mm)), color="#6a4c93")
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


def main(cfg: Config) -> None:  # noqa: PLR0915
    run, neutral_dir, output = (
        cfg.run_dir.resolve(),
        cfg.neutral_dir.resolve(),
        cfg.output_dir.resolve(),
    )
    assert not output.exists(), output
    paths = {
        name: run / name
        for name in ("protocol.json", "summary.json", "progress.jsonl", "endpoint.npz")
    }
    assert all(path.is_file() for path in paths.values()), paths
    protocol, summary = (
        json.loads(paths[name].read_text())
        for name in ("protocol.json", "summary.json")
    )
    assert protocol["schema"] == "corrected-neutral-collision-off-rigid6-inverse-v1"
    assert protocol["collision_enabled"] is False
    expression = str(protocol["expression_name"])
    assert summary["endpoint"]["sha256"] == sha256(paths["endpoint.npz"])
    assert summary["status"] != "running"
    audit_path = run / "independent-audit.json"
    audit = json.loads(audit_path.read_text())
    assert audit["inputs"]["endpoint"]["sha256"] == sha256(paths["endpoint.npz"])
    assert audit["valid_forward"]
    rendering_path = verified(protocol["rendering"]["archive"])
    blendshape_path = verified(protocol["sources"]["blendshapes"])
    neutral_endpoint = verified(protocol["sources"]["neutral_endpoint"])
    neutral_summary = verified(protocol["sources"]["neutral_summary"])
    neutral_protocol = json.loads((neutral_dir / "protocol.json").read_text())
    assert sha256(neutral_dir / "summary.json") == sha256(neutral_summary)
    reference_path = verified(
        neutral_protocol["reference_configuration"]["constitutive_volume"]
    )
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
    with np.load(paths["endpoint.npz"], allow_pickle=False) as archive:
        displacement, pose = archive["displacement_m"], archive["pose_rad_m"]
    with np.load(neutral_endpoint, allow_pickle=False) as archive:
        neutral_displacement = archive["displacement_m"]
    assert displacement.shape == full_reference.shape
    assert neutral_displacement.shape == volume.points.shape
    assert pose.shape == (6,)
    final = summary["final"]
    np.testing.assert_array_equal(pose, np.asarray(final["pose_rad_m"]))
    current_rows = progress(run)
    assert current_rows[-1]["iteration"] == final["iteration"]
    rows, parent_binding = lineage(run)
    assert rows[-1]["lineage_run"] == run.name
    total_q_steps = int(rows[-1]["optimizer_steps"]["q"])
    total_pose_steps = int(rows[-1]["optimizer_steps"]["pose"])
    with np.load(blendshape_path, allow_pickle=False) as archive:
        index = [str(name) for name in archive["expression_names"]].index(expression)
        skin_ids, triangles = archive["skin_global_ids"], archive["skin_triangles"]
        target = surface(archive["target_points_m"][index], skin_ids, triangles)
    assert np.array_equal(skin_ids, fields["skin_global_ids"])
    fit_skin = surface(
        full_reference[skin_ids] + displacement[skin_ids], skin_ids, triangles
    )
    neutral_volume, fit_volume = volume.copy(deep=True), volume.copy(deep=True)
    neutral_volume.points, fit_volume.points = (
        volume.points + neutral_displacement,
        volume.points + displacement[: volume.n_points],
    )
    neutral_boundary, fit_boundary = (
        mesh.extract_surface(algorithm=None) for mesh in (neutral_volume, fit_volume)
    )
    assert np.array_equal(neutral_boundary.faces, fit_boundary.faces)
    neutral_full_displacement = np.zeros_like(full_reference)
    neutral_full_displacement[: volume.n_points] = neutral_displacement
    anatomy: dict[str, tuple[pv.PolyData, pv.PolyData]] = {}
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
    neutral_anatomy, fit_anatomy = (
        {name: pair[0] for name, pair in anatomy.items()},
        {name: pair[1] for name, pair in anatomy.items()},
    )
    error_mm = (
        np.linalg.norm(np.asarray(fit_skin.points) - np.asarray(target.points), axis=1)
        * 1000
    )
    output.mkdir(parents=True)
    shutil.copy2(__file__, output / Path(__file__).name)
    shutil.copy2(audit_path, output / "independent-audit.json")
    fit_skin.save(output / "fit-skin.vtp")
    target.save(output / f"{expression.lower()}-transferred-skin-target.vtp")
    neutral_boundary.save(output / "neutral-full-tetmesh-boundary.vtp")
    fit_boundary.save(output / "fit-full-tetmesh-boundary.vtp")
    for name, mesh in neutral_anatomy.items():
        mesh.save(output / f"neutral-{name}.vtp")
    for name, mesh in fit_anatomy.items():
        mesh.save(output / f"fit-{name}.vtp")
    with (output / "optimizer-lineage.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    images = [
        *(
            target_fit_image(
                output, target, fit_boundary, fit_anatomy, view, expression
            )
            for view in ("front", "side")
        ),
        *(
            boundary_image(
                output,
                neutral_boundary,
                fit_boundary,
                neutral_anatomy,
                fit_anatomy,
                view,
            )
            for view in ("front", "side")
        ),
        error_image(output, fit_skin, target, "front", expression),
        curves(output, rows, error_mm),
    ]
    images_html = "".join(
        f'<figure><img src="{name}" alt="Saved {expression} collision-off review"></figure>'
        for name in images
    )
    (output / "index.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>{expression} collision-off fit</title><style>body{{font:16px/1.5 system-ui;max-width:1500px;margin:2rem auto;padding:0 1rem}}img{{max-width:100%}}.status{{padding:1rem;background:#fff1c7}}details{{margin:1rem 0;padding:.8rem;background:#f0f0ee}}</style><h1>{expression} · collision-off coupled jaw and tissue</h1><p class="status"><strong>{total_q_steps} global q updates and {total_pose_steps} global pose updates · {summary["status"]}. Inverse convergence: {summary["inverse_converged"]}.</strong> The independent audit rebuilt the collision-disabled model and confirms the saved force and inversion contracts.</p><p>Area-weighted fit RMS: {rows[0]["fit_rms_mm"]:.4f} → {final["fit_rms_mm"]:.4f} mm. Jaw rotation: {final["pose_rotation_degrees"]:.4f}°; translation magnitude: {final["pose_translation_mm"]:.4f} mm. Residual force: {final["force_norm_n"]:.6g} N; tolerance: {final["force_threshold_n"]:.6g} N.</p><p>{protocol["tetrahedron_policy"]["excluded_tetrahedra"]:,} fully fixed tetrahedra are excluded from bulk mechanics. The full original tetmesh boundary, cranium, mandible, and eyes are shown for visual context. Contact and intersection results are not acceptance gates because collision was disabled throughout this run.</p><p>Among {final["geometry"]["retained_tetrahedra"]:,} retained mechanical tetrahedra, {final["geometry"]["inverted_tetrahedra"]:,} are inverted; their rest-volume fraction is {100 * final["geometry"]["inverted_rest_volume_fraction"]:.6g}%; minimum det(F) is {final["geometry"]["detF_min"]:.6g}. The declared allowance is {protocol["inversion_policy"]["maximum_inverted_tetrahedra"]} cells and {100 * protocol["inversion_policy"]["maximum_inverted_rest_volume_fraction"]:.4g}% rest volume.</p><p>The target is transferred skin geometry. The unweighted error map has RMS {np.sqrt(np.mean(error_mm**2)):.4f} mm, p95 {np.quantile(error_mm, 0.95):.4f} mm, and maximum {error_mm.max():.4f} mm.</p>{images_html}<details><summary>Verification and provenance</summary><p><a href="independent-audit.json">Independent endpoint audit</a> · <a href="optimizer-lineage.jsonl">Verified optimizer lineage</a></p></details>"""
    )
    receipt = {
        "schema": "collision-off-expression-saved-review-v1",
        "expression_name": expression,
        "inputs": {
            "protocol": record(paths["protocol.json"]),
            "summary": record(paths["summary.json"]),
            "endpoint": record(paths["endpoint.npz"]),
            "progress": record(paths["progress.jsonl"]),
            "independent_audit": record(audit_path),
            "rendering": record(rendering_path),
            "blendshapes": record(blendshape_path),
            "neutral_endpoint": record(neutral_endpoint),
            "neutral_summary": record(neutral_summary),
            "reference": record(reference_path),
        },
        "status": {
            "saved_status": summary["status"],
            "forward_converged": bool(final["forward_converged"]),
            "forward_valid": bool(final["valid_forward"]),
            "inverse_converged": bool(summary["inverse_converged"]),
        },
        "collision": {"enabled": False, "acceptance_gate": False},
        "geometry": {
            "fem_vertices": volume.n_points,
            "tetrahedra": volume.n_cells,
            "boundary_triangles": fit_boundary.n_cells,
        },
        "target": {
            "name": expression,
            "expression_index": index,
            "scope": "Transferred skin-only kinematic target; no volumetric target was invented.",
            "vertices": target.n_points,
            "triangles": target.n_cells,
        },
        "fit_error_mm": {
            "unweighted_rms": float(np.sqrt(np.mean(error_mm**2))),
            "mean": float(error_mm.mean()),
            "max": float(error_mm.max()),
            "p95": float(np.quantile(error_mm, 0.95)),
        },
        "optimizer_fit_rms_mm": {
            "area_weighted_initial": float(rows[0]["fit_rms_mm"]),
            "area_weighted_final": float(final["fit_rms_mm"]),
        },
        "optimizer_lineage": {
            "parent_runs": parent_binding,
            "rows": len(rows),
            "global_q_steps": total_q_steps,
            "global_pose_steps": total_pose_steps,
            "duplicate_continuation_initial_rows_removed": len(parent_binding),
        },
        "saved_pose_rad_m": pose.tolist(),
        "tetrahedron_policy": protocol["tetrahedron_policy"],
        "inversion_policy": protocol["inversion_policy"],
        "retained_geometry": final["geometry"],
        "solver_rerun": False,
        "assets": {
            path.name: record(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    write_json(output / "receipt.json", receipt)
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "review/fit_rms_mm": receipt["fit_error_mm"]["unweighted_rms"],
            "review/forward_valid": float(final["valid_forward"]),
            "review/inverse_converged": float(summary["inverse_converged"]),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
