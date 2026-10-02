"""Render a saved rigid MouthOpen inverse endpoint without solving physics."""
# ruff: noqa: PLR0915, PT018

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
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/inverse-mouthopen-rigid-002"
    review_dir: Path = GROUP / "data/review-repaired-reference-005"
    overwrite: bool = False


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verified(item: dict[str, Any]) -> Path:
    path = Path(item["path"])
    assert record(path) == item
    return path


def surface(
    points: np.ndarray, ids: np.ndarray, triangles: np.ndarray, displacement: np.ndarray
) -> pv.PolyData:
    assert points.ndim == 2 and points.shape[1] == 3
    assert len(points) == len(ids) == len(np.unique(ids))
    assert triangles.ndim == 2 and triangles.shape[1] == 3
    assert triangles.min() >= 0 and triangles.max() < len(ids)
    assert ids.max() < len(displacement)
    faces = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    return pv.PolyData(points + displacement[ids], faces)


def camera(meshes: list[pv.PolyData], view: str) -> tuple[list, float]:
    points = np.concatenate([mesh.points for mesh in meshes])
    low, high = points.min(0), points.max(0)
    center = (low + high) / 2
    span = float(max(high - low))
    direction = np.array([0, 0, 2.7]) if view == "front" else np.array([2.7, 0, 0])
    return [list(center + span * direction), list(center), [0.0, 1.0, 0.0]], 0.60 * span


def render(
    output: Path,
    target: pv.PolyData,
    fit: pv.PolyData,
    anatomy: dict[str, pv.PolyData],
    status: str,
    view: str,
) -> str:
    position, scale = camera([target, fit, *anatomy.values()], view)
    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1800, 900))
    plotter.set_background("#f7f7f5")
    for column, (title, skin, color) in enumerate(
        (
            ("MouthOpen target", target, "#737b86"),
            (f"Rigid inverse fit · {status}", fit, "#c75f42"),
        )
    ):
        plotter.subplot(0, column)
        plotter.add_text(
            f"{title}\n{view} · identical true scale",
            position="upper_left",
            font_size=15,
            color="#202124",
        )
        if column == 1:
            for name, mesh in anatomy.items():
                plotter.add_mesh(
                    mesh,
                    color="#9fc5e8" if name == "eyes" else "#e8dfc8",
                    smooth_shading=True,
                    opacity=0.82,
                )
        plotter.add_mesh(skin, color=color, smooth_shading=True)
        plotter.camera_position = position
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = scale
        plotter.reset_camera_clipping_range()
    name = f"target-vs-rigid-inverse-{view}.png"
    plotter.show(screenshot=output / name, auto_close=True)
    return name


def curves(output: Path, rows: list[dict], *, trial: bool) -> str:
    iteration = np.asarray([row["iteration"] for row in rows])
    fit = np.asarray([row["fit_rms_mm"] for row in rows])
    force = np.asarray([row["force_norm_n"] for row in rows])
    threshold = np.asarray([row["force_threshold_n"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(iteration, fit, color="#c75f42", marker="o")
    label = "Saved initial state / trial" if trial else "Saved state"
    axes[0].set(xlabel=label, ylabel="Fit RMS (mm)", title="Target fit")
    axes[1].semilogy(iteration, force, color="#087d81", marker="o", label="force norm")
    axes[1].semilogy(
        iteration, threshold, color="#555", ls=":", marker="x", label="threshold"
    )
    axes[1].set(xlabel=label, ylabel="Force norm (N)", title="Forward residual")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    name = "fit-force-curves.png"
    figure.savefig(output / name, dpi=180)
    plt.close(figure)
    return name


def main(cfg: Config) -> None:
    run, review = cfg.run_dir.resolve(), cfg.review_dir.resolve()
    output = review / "rigid-inverse"
    if output.exists():
        assert cfg.overwrite
        shutil.rmtree(output)
    protocol_path, summary_path, progress_path, endpoint_path = (
        run / "protocol.json",
        run / "summary.json",
        run / "progress.jsonl",
        run / "endpoint.npz",
    )
    snapshot_inputs = {
        "protocol": record(protocol_path),
        "summary": record(summary_path),
        "progress": record(progress_path),
        "endpoint": record(endpoint_path),
    }
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads(summary_path.read_text())
    assert protocol["schema"] == "new-neutral-mouthopen-rigid6-inverse-v1"
    assert protocol["expression_name"] == "MouthOpen"
    rendering_path = verified(protocol["rendering"]["archive"])
    blendshape_path = verified(protocol["sources"]["blendshapes"])
    snapshot_inputs["rendering"] = record(rendering_path)
    snapshot_inputs["blendshapes"] = record(blendshape_path)
    assert summary["endpoint"] == snapshot_inputs["endpoint"]
    with np.load(endpoint_path, allow_pickle=False) as data:
        assert {
            "displacement_m",
            "activation_inv",
            "active_cell_ids",
            "pose_rad_m",
        } <= set(data.files)
        displacement = data["displacement_m"]
        activation = data["activation_inv"]
        active_ids = data["active_cell_ids"]
        pose = data["pose_rad_m"]
    with np.load(rendering_path, allow_pickle=False) as data:
        fields = {name: data[name] for name in data.files}
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
    assert (
        displacement.shape == fields["full_reference_points_m"].shape
        and np.isfinite(displacement).all()
    )
    assert activation.shape == (len(active_ids), 6) and np.isfinite(activation).all()
    assert pose.shape == (6,) and np.isfinite(pose).all()
    rows = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert rows and rows[-1]["iteration"] == summary["final"]["iteration"]
    final = summary["final"]
    initial = summary["initial"]
    limits = protocol["parameterization"].get("adjacent_proposed_increment", {})
    assert np.array_equal(pose, np.asarray(final["pose_rad_m"]))
    with np.load(blendshape_path, allow_pickle=False) as data:
        index = [str(name) for name in data["expression_names"]].index("MouthOpen")
        assert np.array_equal(data["skin_global_ids"], fields["skin_global_ids"])
        target = pv.PolyData(
            data["target_points_m"][index],
            np.column_stack(
                (np.full(len(fields["skin_triangles"]), 3), fields["skin_triangles"])
            ).ravel(),
        )
    fit = surface(
        fields["full_reference_points_m"][fields["skin_global_ids"]],
        fields["skin_global_ids"],
        fields["skin_triangles"],
        displacement,
    )
    anatomy = {
        name: surface(
            fields["full_reference_points_m"][fields[f"{name}_global_ids"]],
            fields[f"{name}_global_ids"],
            fields[f"{name}_triangles"],
            displacement,
        )
        for name in ("cranium", "mandible", "eye")
    }
    anatomy["eyes"] = anatomy.pop("eye")
    output.mkdir(parents=True)
    endpoint_copy = output / "endpoint.npz"
    shutil.copy2(endpoint_path, endpoint_copy)
    images = [
        render(output, target, fit, anatomy, str(summary["status"]), view)
        for view in ("front", "side")
    ]
    trial_status = str(summary["status"]).startswith("forward_verified_trial")
    curve = curves(output, rows, trial=trial_status)
    for name, item in snapshot_inputs.items():
        assert record(Path(item["path"])) == item, name
    write_json(output / "snapshot-summary.json", summary)
    (output / "live-summary.json").symlink_to(summary_path)
    image_records = {name: record(output / name) for name in [*images, curve]}
    snapshot = {
        "schema": "saved-rigid-mouthopen-inverse-review-v1",
        "snapshot": snapshot_inputs,
        "summary": summary,
        "endpoint_pose_rad_m": pose.tolist(),
        "download_endpoint": record(endpoint_copy),
        "active_tetrahedra": len(active_ids),
        "images": image_records,
        "scope": "Saved endpoint review only; no forward or inverse physics was evaluated.",
        "live_metadata": {
            "path": "live-summary.json",
            "meaning": "Fetched independently by the webpage and not used to render the saved images.",
        },
    }
    write_json(output / "receipt.json", snapshot)
    figures = "".join(
        f'<figure><img src="{name}" alt="MouthOpen target and rigid inverse fit with saved anatomy"></figure>'
        for name in images
    )
    (output / "index.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>Rigid MouthOpen inverse</title><style>body{{font:16px/1.5 system-ui;max-width:1500px;margin:2rem auto;padding:0 1rem}}img{{max-width:100%}}.status{{padding:1rem;background:#fff1c7}}</style><p><a href="../">Back to neutral review</a></p><h1>Rigid MouthOpen inverse fit</h1><p class="status">Saved snapshot status: {summary["status"]}. Numerical forward convergence: {final["forward_converged"]}; inverse convergence: {final["inverse_converged"]}; contact valid: {final["contact_valid"]}.</p><p>Geometry is reported separately: {final["geometry"]["inverted_tetrahedra"]} inverted tetrahedra, min detF {final["geometry"]["detF_min"]:.6g}.</p><p><a href="endpoint.npz" download>Download pinned endpoint.npz</a></p><p>Fit RMS: {initial["fit_rms_mm"]:.4f} → {final["fit_rms_mm"]:.4f} mm across {summary.get("accepted_optimizer_iterations", final["iteration"])} accepted optimizer iterations. {"The final row is a forward-verified trial, not a completed optimizer iteration." if trial_status else ""} Force: {final["force_norm_n"]:.4g} N (threshold {final["force_threshold_n"]:.4g} N). Active strain RMS/max: {final["activation_rms"]:.4g} / {final["activation_max_abs"]:.4g}.</p><p>Saved jaw pose: rotation magnitude {np.degrees(np.linalg.norm(pose[:3])):.3f}°, translation magnitude {np.linalg.norm(pose[3:]) * 1000:.3f} mm. Adjacent proposed limits: {limits.get("rotation_degrees", "unrecorded")}° and {limits.get("translation_m", "unrecorded")} m.</p>{figures}<figure><img src="{curve}" alt="Saved fit and force curves"></figure><h2>Live metadata</h2><pre id="live">Loading independently fetched summary…</pre><p>The images and curves are pinned to <a href="receipt.json">receipt.json</a> and <a href="snapshot-summary.json">snapshot-summary.json</a>. The block below fetches <a href="live-summary.json">live-summary.json</a> separately and does not change the snapshot.</p><script>async function tick(){{try{{let s=await fetch('live-summary.json?'+Date.now()).then(r=>r.json());document.querySelector('#live').textContent=JSON.stringify({{status:s.status,final_iteration:s.final?.iteration,inverse_converged:s.inverse_converged}},null,2)}}catch(e){{document.querySelector('#live').textContent='Live summary unavailable: '+e}}}}tick();setInterval(tick,10000)</script>"""
    )
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "rigid_inverse/final_fit_rms_mm": final["fit_rms_mm"],
            "rigid_inverse/final_force_n": final["force_norm_n"],
            "rigid_inverse/rotation_degrees": float(
                np.degrees(np.linalg.norm(pose[:3]))
            ),
            "rigid_inverse/translation_mm": float(np.linalg.norm(pose[3:]) * 1000),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
