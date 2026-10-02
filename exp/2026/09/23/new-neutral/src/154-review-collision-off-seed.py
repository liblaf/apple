"""Render full original tetrahedral boundary for the initializer comparison."""

# ruff: noqa: C901, E402, PLR0912, PLR0915
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint, sha256, write_json


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/collision-off-seed-comparison-001"
    output_dir: Path = GROUP / "data/review-collision-off-seed-001"
    source_log: Path = GROUP / "tmp/152-test-mouthopen-collision-off-seed-001.log"


def bound(item: dict) -> Path:
    path = Path(item["path"])
    assert sha256(path) == item["sha256"]
    return path


def main(cfg: Config) -> None:
    run = cfg.run_dir.resolve()
    output = cfg.output_dir.resolve()
    assert not output.exists()
    output.mkdir(parents=True)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "finished_comparison"
    protocol = json.loads((run / "protocol.json").read_text())
    source_protocol_path = bound(protocol["source_protocol"])
    source = source_protocol_path.parent
    source_protocol = json.loads(source_protocol_path.read_text())
    reference = bound(source_protocol["sources"]["reference_repair"])
    volume = pv.read(reference.parent / "repaired-reference-volume.vtu")
    volume.point_data["OriginalTetVertexId"] = np.arange(volume.n_points)
    boundary = volume.extract_surface(algorithm=None)
    boundary_ids = np.asarray(boundary.point_data["OriginalTetVertexId"])
    with np.load(source / "rendering.npz") as saved:
        fields = {key: saved[key].copy() for key in saved.files}
    full_reference = fields["full_reference_points_m"]
    np.testing.assert_array_equal(volume.points, full_reference[: volume.n_points])
    with np.load(source / "endpoint.npz") as saved:
        old_u = saved["displacement_m"].copy()
    points = full_reference + old_u
    low, high = points.min(axis=0), points.max(axis=0)
    center, span = (low + high) / 2, float(np.max(high - low))
    pictures = []
    for row in summary["results"]:
        if row["method"] != "collision_off_pushout":
            continue
        trial = run / f"alpha-{row['alpha']:g}" / row["method"]
        stages = [("Audited starting fit", old_u)]
        for filename, label in (
            ("collision-off.pt", "Collision-off estimate"),
            ("saved-estimate.pt", "Saved approximate estimate"),
            ("projection-latest.pt", "Push-out iterate (diagnostic)"),
        ):
            path = trial / "seed" / filename
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                stages.append((label, saved["u_full"].numpy()))
        endpoint = trial / "endpoint.npz"
        if endpoint.exists():
            with np.load(endpoint) as saved:
                stages.append(
                    ("Collision-on correction", saved["displacement_m"].copy())
                )
        for view, direction in (("front", (0, 0, 2.7)), ("side", (2.7, 0, 0))):
            plot = pv.Plotter(
                off_screen=True,
                shape=(1, len(stages)),
                window_size=(700 * len(stages), 900),
            )
            plot.set_background("#f7f7f5")
            for index, (label, u) in enumerate(stages):
                plot.subplot(0, index)
                mesh = boundary.copy()
                mesh.points = (full_reference + u)[boundary_ids]
                plot.add_mesh(
                    mesh,
                    color="#c87960" if index else "#9da9b3",
                    smooth_shading=True,
                    opacity=0.8,
                )
                for key, color in (
                    ("cranium", "#dfd3ba"),
                    ("mandible", "#cab485"),
                    ("eye", "#91c9da"),
                ):
                    ids = fields[f"{key}_global_ids"]
                    faces = fields[f"{key}_triangles"]
                    anatomy = pv.PolyData(
                        (full_reference + u)[ids],
                        np.column_stack((np.full(len(faces), 3), faces)).ravel(),
                    )
                    plot.add_mesh(anatomy, color=color, smooth_shading=True)
                plot.add_text(
                    f"{label}\nFull tet boundary + bones / eyes\n{view}; diagnostic comparison",
                    position="upper_left",
                    font_size=12,
                    color="#202124",
                )
                plot.camera_position = [
                    list(center + span * np.asarray(direction)),
                    list(center),
                    [0, 1, 0],
                ]
                plot.camera.parallel_projection = True
                plot.camera.parallel_scale = 0.60 * span
                plot.reset_camera_clipping_range()
            filename = f"alpha-{row['alpha']:g}-{view}.png"
            plot.show(screenshot=output / filename, auto_close=True)
            pictures.append(filename)
    rows = []
    for row in summary["results"]:
        seed_path = (
            run / f"alpha-{row['alpha']:g}" / row["method"] / "seed" / "summary.json"
        )
        seed = json.loads(seed_path.read_text()) if seed_path.exists() else {}
        off = seed.get("collision_off", {})
        if "estimate_evaluation" in seed:
            off = {
                "grad_norm": seed["estimate_evaluation"]["true_free_force_norm"],
                "retained_geometry": seed["estimate_retained_geometry"],
            }
        projection = seed.get("projection", {})
        geometry = row.get(
            "seed_geometry",
            seed.get("retained_geometry", off.get("retained_geometry", {})),
        )
        if projection.get("trace") and "seed_geometry" not in row:
            geometry = projection["trace"][-1]["retained_geometry"]
        ccd = row.get(
            "old_to_seed_motion",
            row.get("failure", {})
            .get("receipt", {})
            .get("motion_audit", {})
            .get("old_to_candidate", {}),
        )
        rows.append(
            {
                "method": row["method"],
                "alpha": row["alpha"],
                "rotation_increment_deg": row["rotation_increment_deg"],
                "translation_increment_mm": row["translation_increment_mm"],
                "seconds": row["total_seconds"],
                "prior_estimate_seconds": row.get("prior_estimate_seconds", 0),
                "stage": row["stage"],
                "failure": row.get("failure", {}).get("message"),
                "collision_off_force_n": None
                if "grad_norm" not in off
                else off["grad_norm"] * 1e6,
                "collision_off_inversions": off.get("retained_geometry", {}).get(
                    "inverted_tetrahedra"
                ),
                "seed_inversions": geometry.get("inverted_tetrahedra"),
                "projection_iterations": max(0, len(projection.get("trace", [])) - 1),
                "ccd_fraction": ccd.get("ccd_fraction"),
                "valid_endpoint": row.get("endpoint", {}).get("valid_endpoint", False),
                "admitted_continuation": row["admitted_continuation"],
                "endpoint": row.get("endpoint"),
            }
        )
    logged_forces: dict[float, list[float]] = {}
    current_alpha = None
    for line in cfg.source_log.read_text().splitlines():
        start = re.search(r"Trial (\w+) alpha ([\d.]+),", line)
        if start:
            current_alpha = (
                float(start[2]) if start[1] == "collision_off_pushout" else None
            )
        match = re.search(r"Newton \d+ force ([\deE.+-]+),", line)
        if match and current_alpha is not None:
            logged_forces.setdefault(current_alpha, []).append(float(match[1]) * 1e6)
        if "Trial ended:" in line:
            current_alpha = None
    figure, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for row in summary["results"]:
        if row["method"] != "collision_off_pushout":
            continue
        path = run / f"alpha-{row['alpha']:g}" / row["method"] / "seed" / "summary.json"
        seed = json.loads(path.read_text())
        trace = seed.get("collision_off", {}).get("newton", {}).get("trace", [])
        forces = [float(point["force"]) * 1e6 for point in trace if "force" in point]
        if not forces:
            forces = list(logged_forces.get(row["alpha"], []))
        evaluation = seed.get("estimate_evaluation", {})
        if "true_free_force_norm" in evaluation:
            forces.append(float(evaluation["true_free_force_norm"]) * 1e6)
        if forces:
            ax.semilogy(
                np.arange(len(forces)),
                forces,
                label=f"Collision-off, fraction {row['alpha']:g}",
            )
    ax.axhline(0.01, color="#333333", linestyle="--", label="0.01 N force gate")
    ax.set(
        xlabel="Newton iteration (before update; last point is saved-state evaluation)",
        ylabel="Free force norm (N)",
        title="Collision-off force histories; contact correction is a separate solve",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    figure.savefig(output / "force.png", dpi=160)
    plt.close(figure)
    write_json(
        output / "comparison.json",
        {
            "rows": rows,
            "initial": summary["initial"],
            "source": str(run),
            "full_original_tets": volume.n_cells,
            "boundary_triangles": boundary.n_cells,
        },
    )
    table = "".join(
        f"<tr><td>{html.escape(r['method'])}</td><td>{r['rotation_increment_deg']:.3f} / {r['translation_increment_mm']:.3f}</td><td>{r['seconds']:.1f}</td><td>{r['collision_off_inversions']}</td><td>{r['seed_inversions']}</td><td>{r['ccd_fraction']}</td><td>{html.escape(r['failure'] or r['stage'])}</td></tr>"
        for r in rows
    )
    images = "".join(
        f'<figure><img src="{name}" style="width:100%"><figcaption>{name}</figcaption></figure>'
        for name in pictures
    )
    (
        output / "index.html"
    ).write_text(f"""<!doctype html><meta charset="utf-8"><title>Jaw initializer comparison</title>
<style>body{{font:16px system-ui;margin:2em auto;max-width:1600px;color:#202124}}table{{border-collapse:collapse}}td,th{{padding:10px;border:1px solid #ccc}}img{{max-width:100%}}</style>
<h1>Collision-off tissue estimate and push-out</h1><p>Both methods start from audited run005, update 61. Active strain and exact skin pre-strain are fixed. Bounds: 1 degree and 1 mm; second test uses one quarter of that motion.</p>
<p>Collision-off and pushed shapes are diagnostic initializers. Acceptance requires collision-on force ≤0.01 N, contact feasibility, at most 100 inverted retained tets and rest-volume fraction ≤1e-4, plus the original motion CCD gate. No diagnostic result replaces the inverse fit.</p>
<table><tr><th>Initializer</th><th>Rotation ° / translation mm</th><th>Seconds</th><th>Off inversions</th><th>Seed inversions</th><th>CCD fraction</th><th>Outcome</th></tr>{table}</table>
<p>Single ordered trial timings include initialization overhead and unrelated GPU contention. When using saved approximate estimates, the table shows additional repair/correction time; the larger collision-off attempt exhausted its 600-second budget; the smaller attempt converged in 191 seconds. Prior solve times are recorded separately in comparison.json. All original tetrahedra remain in the display boundary; fully fixed tetrahedra are excluded only from mechanics.</p>
{images}<img src="force.png"><p><a href="comparison.json">Numerical comparison</a> · <a href="../mouthopen-continuation/">Live inverse fit</a></p>""")
    cherries.log_asset(output / "index.html")
    cherries.log_asset(output / "comparison.json")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
