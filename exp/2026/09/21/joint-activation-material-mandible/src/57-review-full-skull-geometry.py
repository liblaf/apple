"""Compare complete source bones against reference and saved initialization states."""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs, _collision_geometry

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    geometry_audit: Path = (
        GROUP / "data/full-skull-initialization-audit-001/summary.json"
    )
    checkpoint: Path = (
        GROUP
        / "data/neutral-convergence-025-contact-spatial80-metric-bfgs-002/terminal.pt"
    )
    candidate: Path = (
        GROUP / "data/full-skull-initialization-candidate-001/candidate.npz"
    )
    repair_summary: Path | None = None
    prepared_dir: Path = GROUP / "data/prepared"
    render: bool = True
    output_dir: Path = GROUP / "data/full-skull-geometry-review-001"


def poly(points: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    return pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)))


def main(cfg: Config) -> None:  # noqa: PLR0915
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    spec = importlib.util.spec_from_file_location(
        "source_bone_review",
        Path(__file__).with_name("17-audit-source-bone-contact.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    audit = json.loads(cfg.geometry_audit.read_text())
    path = Path(audit["geometry"]["path"])
    assert sha256(path) == audit["geometry"]["sha256"]
    with np.load(path) as archive:
        a = {key: archive[key] for key in archive.files}
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    volume = pv.read(prepared.volume_path)
    points = a["fem_reference_points_m"]
    assert np.array_equal(points, volume.points)
    tets = np.asarray(volume.cells).reshape(-1, 5)[:, 1:]
    dm = np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
    det_reference = np.linalg.det(dm)
    old = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    assert old["stage"] == "neutral"
    states = {
        "Reference": np.zeros_like(points),
        f"Legacy neutral u{old['update']}": old["primal"]["neutral"].numpy(),
        "Geometric candidate": np.load(cfg.candidate)["initial_displacement_m"],
    }
    repair = None
    if cfg.repair_summary is not None:
        repair = json.loads(cfg.repair_summary.read_text())
        assert repair["schema"] == "joint-full-skull-initialization-repair-v1"
        assert repair["success"] is True
        assert repair["geometry_sha256"] == sha256(path)
        repaired_path = Path(repair["candidate"]["path"])
        assert repair["candidate"]["sha256"] == sha256(repaired_path)
        with np.load(repaired_path) as archive:
            states["Repaired initialization"] = archive["initial_displacement_m"]
    bones = {
        name: poly(a[f"{name}_points_m"], a[f"{name}_faces"])
        for name in ("cranium", "mandible")
    }
    rows = []
    for label, u in states.items():
        x = points + u
        ds = np.transpose(x[tets[:, 1:]] - x[tets[:, :1]], (0, 2, 1))
        determinant = np.linalg.det(ds) / det_reference
        surface = poly(x[a["soft_global_ids"]], a["soft_faces"])
        row = {
            "state": label,
            "detF_min": float(determinant.min()),
            "detF_max": float(determinant.max()),
            "inverted_tetrahedra": int(np.sum(determinant <= 0)),
            "full_skull_admitted": False,
            "contacts": {},
        }
        for name, bone in bones.items():
            pairs, _, _, _ = _collision_geometry(surface, bone)
            signed = geometry.signed_clearance(surface.points, bone)
            row["contacts"][name] = {
                "raw_triangle_intersection_pairs": len(pairs),
                "minimum_soft_node_signed_distance_mm": float(signed.min() * 1000),
                "strictly_inside_soft_nodes": int(np.sum(signed < 0)),
            }
        rows.append(row)
        if label == "Repaired initialization":
            assert row["inverted_tetrahedra"] == 0
            assert row["detF_min"] >= 0.25
            assert row["detF_max"] <= 2
            assert all(
                value["raw_triangle_intersection_pairs"] == 0
                for value in row["contacts"].values()
            )
        write_json(cfg.output_dir / "states.json", rows)
        LOG.info("Full-source state audit: %s", row)
    assets = []
    if cfg.render:
        for view in ("front", "side"):
            plot = geometry.plotter(
                f"Complete registered skull — {view}\nAll source triangles retained; unchanged coordinates"
            )
            plot.add_mesh(bones["cranium"], color="#decaa5", smooth_shading=True)
            plot.add_mesh(bones["mandible"], color="#589aa3", smooth_shading=True)
            filename = f"01-complete-skull-{view}.png"
            geometry.save(
                plot,
                cfg.output_dir / filename,
                geometry.camera(bones["cranium"].merge(bones["mandible"]), view),
            )
            assets.append(
                {
                    "filename": filename,
                    "caption": "Complete registered source cranium (35,162 triangles) and mandible (18,948 triangles). No source triangle exclusions or coordinate changes.",
                }
            )
        figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        labels = [row["state"].replace(" ", "\n", 1) for row in rows]
        x = np.arange(len(rows))
        for offset, name, color in (
            (-0.18, "cranium", "#ae8346"),
            (0.18, "mandible", "#448c97"),
        ):
            axes[0].bar(
                x + offset,
                [
                    row["contacts"][name]["raw_triangle_intersection_pairs"]
                    for row in rows
                ],
                0.36,
                label=name.title(),
                color=color,
            )
            axes[1].bar(
                x + offset,
                [
                    row["contacts"][name]["minimum_soft_node_signed_distance_mm"]
                    for row in rows
                ],
                0.36,
                label=name.title(),
                color=color,
            )
        for ax in axes:
            ax.set_xticks(x, labels)
            ax.spines[["top", "right"]].set_visible(False)
            ax.axhline(0, color="#777777", linewidth=0.8)
        axes[0].set_ylabel("Raw soft-bone triangle intersection pairs")
        axes[0].legend()
        axes[1].set_ylabel("Minimum soft-node signed distance (mm)")
        figure.suptitle(
            "Full-source skull initialization audit — geometry only", fontsize=15
        )
        figure.supxlabel(
            "Candidate001 rejected: 13 inversions. Repaired seed passes geometry checks; equilibrium remains unvalidated."
            if repair
            else "Candidate001 clears both bones but inverts 13 tetrahedra: rejected. Legacy checkpoint is superseded.",
            fontsize=10,
        )
        filename = "02-initialization-comparison.png"
        figure.savefig(cfg.output_dir / filename, dpi=170)
        plt.close(figure)
        assets.append(
            {
                "filename": filename,
                "caption": "Measured triangle intersections and signed nodal clearances against complete source bones. These are initialization diagnostics, not equilibrium or optimization results.",
            }
        )
    write_json(
        cfg.output_dir / "summary.json",
        {
            "schema": "joint-full-skull-geometry-review-v1",
            "geometry_audit_sha256": sha256(cfg.geometry_audit),
            "geometry_sha256": sha256(path),
            "legacy_checkpoint_sha256": sha256(cfg.checkpoint),
            "candidate_sha256": sha256(cfg.candidate),
            "repair_summary_sha256": sha256(cfg.repair_summary)
            if cfg.repair_summary
            else None,
            "repair": repair,
            "states": rows,
            "assets": assets,
            "full_skull_contact_admitted": False,
        },
    )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
