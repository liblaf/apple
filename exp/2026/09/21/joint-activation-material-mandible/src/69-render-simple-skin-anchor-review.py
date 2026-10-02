"""Render manual Flynn-region anchor candidates on the prepared skin mesh."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from scipy.spatial import cKDTree

from liblaf import cherries


class Config(cherries.BaseConfig):
    manifest: Path = GROUP / "data/prepared/manifest.json"
    landmarks: Path = (
        GROUP.parents[4] / "../melon/exp/2026/05/27/head/data/22-skin.landmarks.json"
    )
    output_dir: Path = cherries.output("simple-skin-anchor-review", mkdir=True)


REGIONAL_VALUES = {
    "CC": {"young_mpa": 0.204010, "prestress_n_per_m": 80.60},
    "NE": {"young_mpa": 0.258318, "prestress_n_per_m": 80.85},
    "NL": {"young_mpa": 0.102701, "prestress_n_per_m": 20.05},
    "FH": {"young_mpa": 0.151199, "prestress_n_per_m": 30.40},
    "CJ": {"young_mpa": 0.196160, "prestress_n_per_m": 78.35},
    "ZYG": {"young_mpa": 0.210727, "prestress_n_per_m": 61.35},
}

# Manually selected from front and side projections of the prepared reference
# mesh. These are review targets, not locations published by Flynn et al.
TARGETS_M = {
    "FH_C": (1.40705, 2.265, 0.085),
    "ZYG_R": (1.373, 2.205, 0.078),
    "ZYG_L": (1.44110, 2.205, 0.078),
    "CC_R": (1.374, 2.182, 0.082),
    "CC_L": (1.44010, 2.182, 0.082),
    "NL_R": (1.379, 2.168, 0.088),
    "NL_L": (1.43510, 2.168, 0.088),
    "NE_R": (1.351, 2.181, 0.038),
    "NE_L": (1.46310, 2.181, 0.038),
    "CJ_R": (1.373, 2.138, 0.057),
    "CJ_L": (1.44110, 2.138, 0.057),
}

COLORS = {
    "FH": "#7e57c2",
    "ZYG": "#00897b",
    "CC": "#1976d2",
    "NL": "#d81b60",
    "NE": "#fb8c00",
    "CJ": "#6d4c41",
}


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = archive_sources(cfg.output_dir)
    manifest = json.loads(cfg.manifest.read_text())
    skin_path = Path(manifest["sources"]["skin"]["path"])
    assert sha256(skin_path) == manifest["sources"]["skin"]["sha256"]
    assert cfg.landmarks.is_file(), cfg.landmarks

    skin = pv.read(skin_path)
    points = np.asarray(skin.points, dtype=np.float64)
    assert len(points) == 15_299
    assert np.isfinite(points).all()
    global_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    source_ids = np.asarray(skin.point_data["SourcePointId"], dtype=np.int64)
    tree = cKDTree(points)
    anchor_rows = []
    for name, target in TARGETS_M.items():
        distance, point_id = tree.query(np.asarray(target))
        region = name.split("_")[0]
        anchor_rows.append(
            {
                "name": name,
                "region": region,
                "side": name.split("_")[1],
                "target_m": list(target),
                "point_id": int(point_id),
                "global_point_id": int(global_ids[point_id]),
                "source_point_id": int(source_ids[point_id]),
                "point_m": points[point_id].tolist(),
                "target_snap_distance_mm": float(1e3 * distance),
                **REGIONAL_VALUES[region],
            }
        )

    center_x = 1.40705
    extents = np.column_stack((points.min(axis=0), points.max(axis=0)))
    summary = {
        "schema": "joint-simple-skin-anchor-review-v1",
        "success": True,
        "status": "manual geometric candidates for review; not registered measurements",
        "skin_path": str(skin_path),
        "skin_sha256": sha256(skin_path),
        "landmarks_path": str(cfg.landmarks.resolve()),
        "landmarks_sha256": sha256(cfg.landmarks),
        "skin_points": len(points),
        "world_axes": {
            "x": "left-right; x below midline is assumed subject right from the front view",
            "y": "superior-inferior; +Y is superior",
            "z": "posterior-anterior; +Z is anterior",
            "midline_x_m": center_x,
        },
        "bounds_m": {
            axis: extents[index].tolist() for index, axis in enumerate(("x", "y", "z"))
        },
        "selection": {
            "method": "manual target on front/side projections, then Euclidean nearest prepared-skin point",
            "right_left_policy": "paired manual targets reflected approximately about x=1.40705 m, then independently snapped",
            "anatomical_scope": {
                "FH": "central forehead above the brows",
                "ZYG": "lateral malar prominence below and lateral to the orbit",
                "CC": "central cheek between zygoma, nose, mouth, and masseter",
                "NL": "cheek immediately lateral to the oral commissure; not vermilion",
                "NE": "lateral parotideomasseteric cheek anterior to the ear",
                "CJ": "lateral lower cheek over the mandibular body/jowl",
            },
            "limitations": [
                "Flynn Figure 2 probe centers are not registered to this subject",
                "the six targets are manual anatomical correspondences for review",
                "bilateral mirroring is a modeling assumption",
                "no tangent material direction is inferred",
            ],
        },
        "anchors": anchor_rows,
        "figure": "anchors-front-side.png",
        "implementation_sha256": {
            key: value
            for key, value in provenance["sources"].items()
            if Path(key).name == "69-render-simple-skin-anchor-review.py"
        },
    }
    write_json(cfg.output_dir / "anchors.json", summary)

    xyz_mm = 1e3 * (points - np.array((center_x, 2.2, 0.0)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
    axes[0].scatter(
        xyz_mm[:, 0],
        xyz_mm[:, 1],
        c=points[:, 2],
        cmap="Greys_r",
        s=1.7,
        alpha=0.58,
        linewidths=0,
    )
    axes[1].scatter(
        xyz_mm[:, 2],
        xyz_mm[:, 1],
        color="#c5c8cc",
        s=1.7,
        alpha=0.45,
        linewidths=0,
    )

    for row in anchor_rows:
        point = 1e3 * (np.asarray(row["point_m"]) - np.array((center_x, 2.2, 0.0)))
        color = COLORS[row["region"]]
        axes[0].scatter(
            point[0], point[1], s=54, color=color, edgecolor="white", zorder=4
        )
        side_offset = 3.0 if row["side"] == "L" else -3.0
        axes[0].annotate(
            row["name"],
            (point[0], point[1]),
            xytext=(5 if point[0] >= 0 else -5, side_offset),
            textcoords="offset points",
            ha="left" if point[0] >= 0 else "right",
            fontsize=8,
            color=color,
            weight="bold",
        )
        if row["side"] in {"R", "C"}:
            axes[1].scatter(
                point[2], point[1], s=54, color=color, edgecolor="white", zorder=4
            )
            axes[1].annotate(
                row["name"],
                (point[2], point[1]),
                xytext=(5, 1),
                textcoords="offset points",
                fontsize=8,
                color=color,
                weight="bold",
            )

    axes[0].set_title("Front: candidate regional anchors", loc="left")
    axes[0].set_xlabel(
        "world X relative to midline (mm)\nviewer left = assumed subject right"
    )
    axes[0].set_ylabel("world Y relative to 2.2 m (mm); superior up")
    axes[0].set_aspect("equal")
    axes[1].set_title("Right-side candidates in profile", loc="left")
    axes[1].set_xlabel("world Z (mm); anterior right")
    axes[1].set_ylabel("world Y relative to 2.2 m (mm); superior up")
    axes[1].set_aspect("equal")
    for axis in axes:
        axis.grid(alpha=0.16)
    fig.suptitle(
        "Flynn-region transfer anchors — manual geometric review only",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(cfg.output_dir / "anchors-front-side.png", dpi=180)
    plt.close(fig)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
