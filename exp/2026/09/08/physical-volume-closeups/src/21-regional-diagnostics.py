"""Measure saved target-space residuals and plot unsmoothed surface sections."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import pyvista as pv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[6]
PREVIEW = GROUP / "data/19-region-preview-v2"
FIXTURE = (
    ROOT
    / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture/volume.vtu"
)
BASELINE = (
    Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
    / "exp/2026/09/08/physical-volume-baseline"
)
FINAL = BASELINE / "data/20-baseline/final.npz"

ROI_BOXES = {
    "right_nose_to_mouth": ((1.416, 1.450), (2.165, 2.195), (0.065, 0.110)),
    "right_lateral_cheek": ((1.449, 1.475), (2.150, 2.185), (0.000, 0.065)),
    "right_lower_cheek_jaw": ((1.440, 1.475), (2.125, 2.155), (0.005, 0.065)),
    "right_mouth_corner": ((1.430, 1.455), (2.150, 2.176), (0.060, 0.105)),
}
SECTION_Y = (2.170, 2.180, 2.190)
SECTION_X = (1.414, 1.460)
SECTION_Z_MIN = 0.040
COLORS = {"rest": "#87929f", "corrected": "#c53d3d", "target": "#1b78a5"}


class Config(cherries.BaseConfig):
    output_dir: Path = cherries.output("21-diagnostics", mkdir=True)


def record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def array_digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def roi_mask(points: np.ndarray, box: tuple[tuple[float, float], ...]) -> np.ndarray:
    return np.all(
        [
            (points[:, axis] >= limits[0]) & (points[:, axis] <= limits[1])
            for axis, limits in enumerate(box)
        ],
        axis=0,
    )


def triangle_segments(
    points: np.ndarray, triangles: np.ndarray, y: float
) -> np.ndarray:
    """Return the exact linear edge intersections of triangles with y = constant."""
    segments: list[np.ndarray] = []
    for triangle in points[triangles]:
        signed = triangle[:, 1] - y
        hits: list[np.ndarray] = []
        for left, right in ((0, 1), (1, 2), (2, 0)):
            a, b = triangle[left], triangle[right]
            sa, sb = signed[left], signed[right]
            if sa == 0.0 and sb == 0.0:
                # A coplanar edge is not a unique transverse section.
                continue
            if sa == 0.0:
                hits.append(a)
            elif sb == 0.0:
                hits.append(b)
            elif (sa < 0.0) != (sb < 0.0):
                hits.append(a + (-sa / (sb - sa)) * (b - a))
        unique: list[np.ndarray] = []
        for hit in hits:
            if not any(np.allclose(hit, old, rtol=0, atol=1e-13) for old in unique):
                unique.append(hit)
        if len(unique) == 2:
            segment = np.stack(unique)
            if (
                segment[:, 0].max() >= SECTION_X[0]
                and segment[:, 0].min() <= SECTION_X[1]
                and segment[:, 2].max() >= SECTION_Z_MIN
            ):
                segments.append(segment)
    return np.asarray(segments, dtype=np.float64)


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), out
    input_paths = [FIXTURE, FINAL, *(PREVIEW / f"{name}-skin.vtp" for name in COLORS)]
    inputs = {str(path): record(path) for path in input_paths}

    fixture = pv.read(FIXTURE)
    rest_points = np.asarray(fixture.points, dtype=np.float64)
    target_u = np.asarray(fixture.point_data["Smile"], dtype=np.float64)
    top = np.asarray(fixture.point_data["IsFace"], dtype=bool) & np.isfinite(
        target_u
    ).all(axis=1)
    assert int(top.sum()) == 15_302
    with np.load(FINAL, allow_pickle=False) as saved:
        assert int(saved["step"]) == 200 and bool(saved["solver_valid"])
        assert np.array_equal(saved["rest_points"], rest_points)
        corrected_u = np.asarray(saved["u"], dtype=np.float64)
    residual = corrected_u[top] - target_u[top]
    squared = np.sum(residual**2, axis=1)
    global_sum = float(squared.sum())
    global_rms_m = float(np.sqrt(squared.mean()))

    rois: dict[str, dict[str, object]] = {}
    target_points = rest_points[top] + target_u[top]
    for name, box in ROI_BOXES.items():
        selected = roi_mask(target_points, box)
        assert selected.any(), (name, box)
        rois[name] = {
            "selection_space": "target deformed coordinates",
            "box_xyz_m": {"x": box[0], "y": box[1], "z": box[2]},
            "count": int(selected.sum()),
            "fit_rms_mm": float(1000 * np.sqrt(squared[selected].mean())),
            "objective_loss_share": float(squared[selected].sum() / global_sum),
            "target_displacement_rms_mm": float(
                1000 * np.sqrt(np.sum(target_u[top][selected] ** 2, axis=1).mean())
            ),
        }

    surfaces: dict[str, pv.PolyData] = {
        name: pv.read(PREVIEW / f"{name}-skin.vtp") for name in COLORS
    }
    first = surfaces["rest"]
    ids = np.asarray(first.point_data["VolumePointIndex"], dtype=np.int64)
    faces = np.asarray(first.faces, dtype=np.int64).reshape(-1, 4)
    assert np.all(faces[:, 0] == 3)
    triangles = faces[:, 1:]
    for name, surface in surfaces.items():
        assert surface.n_points == 15_299 and surface.n_cells == 29_899
        assert np.array_equal(
            np.asarray(surface.point_data["VolumePointIndex"], dtype=np.int64), ids
        ), name
        assert np.array_equal(surface.faces, first.faces), name
    assert np.array_equal(surfaces["rest"].points, rest_points[ids])
    assert np.allclose(
        surfaces["corrected"].points,
        rest_points[ids] + corrected_u[ids],
        rtol=0,
        atol=3e-15,
    )
    assert np.allclose(
        surfaces["target"].points, rest_points[ids] + target_u[ids], rtol=0, atol=3e-15
    )

    section_rows: list[dict[str, object]] = []
    figure, axes = plt.subplots(
        len(SECTION_Y), 1, figsize=(8, 9), constrained_layout=True
    )
    for axis, y in zip(axes, SECTION_Y, strict=True):
        for name, surface in surfaces.items():
            segments = triangle_segments(np.asarray(surface.points), triangles, y)
            for index, segment in enumerate(segments):
                axis.plot(
                    segment[:, 0],
                    segment[:, 2],
                    color=COLORS[name],
                    linewidth=0.65,
                    label=name if index == 0 else None,
                )
                for endpoint, point in enumerate(segment):
                    section_rows.append(
                        {
                            "y_m": y,
                            "surface": name,
                            "segment": index,
                            "endpoint": endpoint,
                            "x_m": point[0],
                            "z_m": point[2],
                        }
                    )
        axis.set_title(f"Exact horizontal triangle sections at y = {y:.3f} m")
        axis.set_xlim(*SECTION_X)
        axis.set_ylim(SECTION_Z_MIN, 0.115)
        axis.set_aspect("equal", adjustable="box")
        axis.set_ylabel("z (m)")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("x (m)")
    axes[0].legend(loc="upper left", ncol=3, frameon=False)
    figure.suptitle(
        "Right nasolabial surface sections | exact triangle intersections, no smoothing"
    )
    figure.savefig(out / "nasolabial-horizontal-sections.png", dpi=220)
    plt.close(figure)
    with (out / "nasolabial-horizontal-segments.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("y_m", "surface", "segment", "endpoint", "x_m", "z_m")
        )
        writer.writeheader()
        writer.writerows(section_rows)

    summary = {
        "status": "completed_cpu_saved_state_diagnostics",
        "scope": "No forward solve, adjoint solve, optimizer update, geometry smoothing, interpolation, or fold-depth scalar.",
        "inputs": inputs,
        "source": record(Path(__file__)),
        "native_objective": {
            "target": "Smile",
            "selector": "IsFace & finite(Smile)",
            "count": int(top.sum()),
            "definition": "uniform Euclidean vector RMS; inverse loss is uniform Cartesian component MSE multiplied by 1e6",
            "vector_rms_mm": 1000 * global_rms_m,
            "coordinate_mse_m2": float(np.mean(residual**2)),
            "objective_mm2": float(np.mean(residual**2) * 1e6),
        },
        "exploratory_target_space_rois": {
            "overlap_allowed": True,
            "registration_limit": "Boxes are exploratory target-space coordinate ranges, not exact registration to rendered red boxes.",
            "rois": rois,
        },
        "surface_topology": {
            "point_count": int(first.n_points),
            "triangle_count": int(first.n_cells),
            "point_ids_sha256": array_digest(ids),
            "faces_sha256": array_digest(first.faces),
            "same_ids_and_connectivity": True,
        },
        "sections": {
            "planes_y_m": SECTION_Y,
            "x_range_m": SECTION_X,
            "z_min_m": SECTION_Z_MIN,
            "method": "Per-triangle affine intersections with fixed y planes; plotted x-z segments retain native triangles and are not joined or smoothed.",
            "segment_count": len(section_rows) // 2,
        },
        "outputs": {},
    }
    for path in (
        out / "nasolabial-horizontal-sections.png",
        out / "nasolabial-horizontal-segments.csv",
    ):
        summary["outputs"][path.name] = record(path)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    cherries.log_metrics(
        {
            "native_fit_rms_mm": 1000 * global_rms_m,
            **{
                f"roi/{name}/fit_rms_mm": values["fit_rms_mm"]
                for name, values in rois.items()
            },
        }
    )
    for path in out.iterdir():
        cherries.log_output(path)


if __name__ == "__main__":
    cherries.main(main, profile="debug" if os.environ.get("DEBUG") else None)
