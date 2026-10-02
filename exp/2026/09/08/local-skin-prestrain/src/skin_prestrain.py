"""Build a bounded, material-fixed local skin-contraction field.

The regions are selected once in frozen *target* coordinates, then carried to the
fixture's rest skin through ``GlobalPointId``.  Distances and the taper are
computed on rest-skin edges, so the emitted per-triangle field does not change
as a solve moves the skin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[6]
DEFAULT_FIXTURE = (
    ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
)
DEFAULT_CLOSEUPS = (
    ROOT / "exp/2026/09/08/physical-volume-closeups/data/21-diagnostics/summary.json"
)
DEFAULT_OUTPUT = ROOT / "exp/2026/09/08/local-skin-prestrain/data/10-prestrain-field"
MARKED_REGIONS = (
    "right_mouth_corner",
    "right_lateral_cheek",
    "right_lower_cheek_jaw",
)
PROTECTED_REGION = "right_nose_to_mouth"


def _sha256_array(value: np.ndarray) -> str:
    """Hash dtype, shape, and bytes to make a topology receipt unambiguous."""
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _box_mask(points: np.ndarray, box: tuple[tuple[float, float], ...]) -> np.ndarray:
    return np.logical_and.reduce(
        tuple(
            (points[:, axis] >= lower) & (points[:, axis] <= upper)
            for axis, (lower, upper) in enumerate(box)
        )
    )


def _smoothstep01(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _triangle_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0, p1, p2 = (
        points[triangles[:, 0]],
        points[triangles[:, 1]],
        points[triangles[:, 2]],
    )
    area = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1) / 2.0
    if not np.all(np.isfinite(area)) or np.any(area <= 0.0):
        raise ValueError("rest skin has non-finite or degenerate triangles")
    return area


def _rest_edge_graph(points: np.ndarray, triangles: np.ndarray) -> sp.csr_matrix:
    edges = np.concatenate(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)])
    )
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    graph = sp.coo_matrix(
        (
            np.concatenate((lengths, lengths)),
            (
                np.concatenate((edges[:, 0], edges[:, 1])),
                np.concatenate((edges[:, 1], edges[:, 0])),
            ),
        ),
        shape=(len(points), len(points)),
    ).tocsr()
    if graph.nnz == 0 or np.any(graph.data <= 0.0):
        raise ValueError("rest skin edge graph is invalid")
    return graph


@dataclass(frozen=True)
class SkinPrestrain:
    """A fixed scalar support and the Koiter packed inverse activation it encodes."""

    vertex_weight: np.ndarray
    w: np.ndarray
    c: np.ndarray
    activation_inv: np.ndarray
    triangles: np.ndarray
    point_ids: np.ndarray
    triangle_area: np.ndarray
    provenance: dict[str, Any]

    def write(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "skin-prestrain.npz",
            vertex_weight=self.vertex_weight,
            w=self.w,
            c=self.c,
            activation_inv=self.activation_inv,
            triangles=self.triangles,
            point_ids=self.point_ids,
            triangle_area=self.triangle_area,
        )
        (output_dir / "summary.json").write_text(
            json.dumps(self.provenance, indent=2, sort_keys=True) + "\n"
        )
        self._write_map(output_dir / "rest-skin-prestrain-front-xy.png")
        self._write_oblique(output_dir / "rest-skin-prestrain-oblique.png")

    def _write_map(self, path: Path) -> None:
        # The frontal material map uses rest x-y coordinates.
        # Coordinates are saved in provenance only through the fixture digest.
        fig, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
        # Point coordinates are intentionally not part of this object; use a
        # stable triangulation supplied by build_skin_prestrain in provenance.
        # The companion VTP provides the spatial rendering representation.
        mesh_path = Path(self.provenance["outputs"]["rest_skin_vtp"])
        mesh = pv.read(mesh_path)
        points = np.asarray(mesh.points)
        faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
        image = axis.tripcolor(
            points[:, 0],
            points[:, 1],
            faces,
            self.vertex_weight,
            shading="gouraud",
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        axis.set(
            xlabel="rest x (m)",
            ylabel="rest y (m)",
            title="Material-fixed local skin contraction support",
        )
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis, label="w")
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _write_oblique(self, path: Path) -> None:
        """Use the verified closeup side-context camera; gray is no support."""
        mesh = pv.read(Path(self.provenance["outputs"]["rest_skin_vtp"]))
        supported = mesh.threshold(
            (np.finfo(float).eps, 1.0), scalars="prestrain_weight", preference="cell"
        )
        camera = {
            "position": (1.6277167704612385, 2.212135838523062, 0.19295607473209175),
            "focal_point": (1.425, 2.202, 0.047),
            "view_up": (0.0, 1.0, 0.0),
            "parallel_scale": 0.105,
        }
        focus = np.asarray(camera["focal_point"])
        backward = np.asarray(camera["position"]) - focus
        backward /= np.linalg.norm(backward)
        right = np.cross(np.asarray(camera["view_up"]), backward)
        right /= np.linalg.norm(right)
        up = np.cross(backward, right)
        plotter = pv.Plotter(off_screen=True, window_size=(1000, 1000), lighting="none")
        plotter.add_mesh(
            mesh,
            color="#c8c8c3",
            smooth_shading=False,
            ambient=0.20,
            diffuse=0.80,
            specular=0.0,
        )
        plotter.add_mesh(
            supported,
            scalars="prestrain_weight",
            preference="cell",
            cmap="magma",
            clim=(0.0, 1.0),
            smooth_shading=False,
            scalar_bar_args={"title": "material weight w"},
        )
        for position, intensity in (
            (focus + 0.3 * (0.72 * right + 0.35 * up + 0.60 * backward), 0.85),
            (focus + 0.3 * backward, 0.20),
        ):
            plotter.add_light(
                pv.Light(
                    position=position,
                    focal_point=focus,
                    intensity=intensity,
                    light_type="scene light",
                    positional=False,
                ),
                only_active=True,
            )
        plotter.enable_parallel_projection()
        plotter.camera.position = camera["position"]
        plotter.camera.focal_point = camera["focal_point"]
        plotter.camera.up = camera["view_up"]
        plotter.camera.parallel_scale = camera["parallel_scale"]
        plotter.set_background("#242c36")
        plotter.reset_camera_clipping_range()
        plotter.screenshot(path)
        plotter.close()


def build_skin_prestrain(
    fixture: Path = DEFAULT_FIXTURE,
    closeups_summary: Path = DEFAULT_CLOSEUPS,
    taper_m: float = 0.005,
    max_contraction: float = 0.01,
) -> SkinPrestrain:
    """Build the local field from frozen target-space boxes on the fixture.

    ``c = max_contraction * w`` contracts both material axes.  The Koiter
    packed inverse activation is ``[1/(1-c)-1, 1/(1-c)-1, 0]`` per triangle.
    """
    if not (np.isfinite(taper_m) and taper_m > 0.0):
        raise ValueError("taper_m must be finite and positive")
    if not (np.isfinite(max_contraction) and 0.0 <= max_contraction < 1.0):
        raise ValueError("max_contraction must be finite in [0, 1)")
    fixture = Path(fixture)
    if fixture.is_dir():
        fixture = fixture / "volume.vtu"
    fixture = fixture.resolve()
    skin_path = fixture.with_name("skin.vtp").resolve()
    closeups_summary = Path(closeups_summary).resolve()

    volume = pv.read(fixture)
    skin = pv.read(skin_path)
    rest_volume = np.asarray(volume.points, dtype=np.float64)
    target_u = np.asarray(volume.point_data["Smile"], dtype=np.float64)
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    if np.any(ids < 0) or np.any(ids >= len(rest_volume)):
        raise ValueError("skin GlobalPointId lies outside the fixture volume")
    rest = np.asarray(skin.points, dtype=np.float64)
    if not np.array_equal(rest, rest_volume[ids]):
        raise ValueError(
            "skin rest coordinates no longer exactly map to fixture GlobalPointId"
        )
    faces = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)
    if np.any(faces[:, 0] != 3):
        raise ValueError("skin must contain triangles only")
    triangles = faces[:, 1:]
    target = rest + target_u[ids]
    if not np.isfinite(target).all():
        raise ValueError("target skin displacement contains non-finite values")

    payload = json.loads(closeups_summary.read_text())
    rois = payload["exploratory_target_space_rois"]["rois"]
    boxes: dict[str, tuple[tuple[float, float], ...]] = {}
    masks: dict[str, np.ndarray] = {}
    for name, record in rois.items():
        box = tuple(
            tuple(float(value) for value in record["box_xyz_m"][axis])
            for axis in ("x", "y", "z")
        )
        mask = _box_mask(target, box)
        if int(mask.sum()) != int(record["count"]):
            raise ValueError(
                f"frozen target ROI {name} has {mask.sum()} skin points; expected {record['count']}"
            )
        boxes[name], masks[name] = box, mask
    marked = np.logical_or.reduce(tuple(masks[name] for name in MARKED_REGIONS))
    protected = masks[PROTECTED_REGION]
    if not marked.any() or not protected.any():
        raise ValueError("marked or protected target-space ROI has empty skin support")

    graph = _rest_edge_graph(rest, triangles)
    outside = ~marked
    if not outside.any():
        raise ValueError("marked ROI unexpectedly covers every skin vertex")
    d_outside = dijkstra(
        graph, directed=False, indices=np.flatnonzero(outside), min_only=True
    )
    d_protected = dijkstra(
        graph, directed=False, indices=np.flatnonzero(protected), min_only=True
    )
    if not np.isfinite(d_outside).all() or not np.isfinite(d_protected).all():
        raise ValueError("rest skin graph is disconnected from a required ROI")
    # A hard marked support, with a smooth 5 mm interior rest-geodesic ramp.
    marked_taper = marked * _smoothstep01(d_outside / taper_m)
    protection_taper = _smoothstep01(d_protected / taper_m)
    vertex_weight = marked_taper * protection_taper
    # Koiter receives one constant value per cell.  Averaging the marked ramp
    # reduces cell-to-cell jumps; the minimum protection factor prevents any
    # cell incident to the protected NLF vertices from being prestrained.
    w = marked_taper[triangles].mean(axis=1) * protection_taper[triangles].min(axis=1)
    c = max_contraction * w
    activation_inv = np.zeros((len(triangles), 3), dtype=np.float64)
    activation_inv[:, :2] = (1.0 / (1.0 - c) - 1.0)[:, None]
    triangle_area = _triangle_areas(rest, triangles)
    if not (np.all(np.isfinite(w)) and np.all((w >= 0.0) & (w <= 1.0))):
        raise AssertionError("weight escaped [0, 1]")
    if not (
        np.all((c >= 0.0) & (c <= max_contraction))
        and np.all(activation_inv[:, 2] == 0.0)
    ):
        raise AssertionError("invalid contraction or Koiter packed shear")
    weighted = lambda field: float(np.sum(triangle_area * field) / triangle_area.sum())
    nonzero = w[w > 0.0]
    provenance: dict[str, Any] = {
        "status": "completed_cpu_material_fixed_field",
        "scope": "No forward solve, adjoint solve, optimizer update, rest-coordinate change, or target geometry modification.",
        "inputs": {
            "fixture_volume": _file_record(fixture),
            "fixture_skin": _file_record(skin_path),
            "closeups_summary": _file_record(closeups_summary),
        },
        "selection": {
            "space": "frozen target deformed coordinates: rest_skin + Smile[GlobalPointId]",
            "marked_regions": list(MARKED_REGIONS),
            "protected_region": PROTECTED_REGION,
            "boxes_xyz_m": {
                name: {
                    axis: boxes[name][index]
                    for index, axis in enumerate(("x", "y", "z"))
                }
                for name in rois
            },
            "skin_point_counts": {
                name: int(mask.sum()) for name, mask in masks.items()
            },
        },
        "field": {
            "definition": "w_vertex = marked * smoothstep(d_outside/0.005) * smoothstep(d_protected/0.005); w_triangle = mean(marked vertex taper) * min(protection vertex taper); c = 0.01*w",
            "taper_m": taper_m,
            "max_contraction": max_contraction,
            "support_vertex_count": int(np.count_nonzero(vertex_weight)),
            "support_triangle_count": int(np.count_nonzero(w)),
            "outside_marked_vertex_count": int(np.count_nonzero(~marked)),
            "protected_vertex_count": int(protected.sum()),
            "protected_vertices_exact_zero": bool(
                np.all(vertex_weight[protected] == 0.0)
            ),
            "max_w": float(w.max()),
            "max_c": float(c.max()),
            "area_weighted_mean_w": weighted(w),
            "w_quantiles_nonzero": {
                str(q): float(np.quantile(nonzero, q))
                for q in (0.0, 0.25, 0.5, 0.75, 1.0)
            },
        },
        "koiter_activation_inv": {
            "shape": list(activation_inv.shape),
            "packing": "[Ainv_00 - 1, Ainv_11 - 1, Ainv_01]",
            "definition": "[1/(1-c)-1, 1/(1-c)-1, 0]",
            "identity_outside_support": True,
            "max_abs": float(np.abs(activation_inv).max()),
        },
        "topology": {
            "point_count": len(rest),
            "triangle_count": len(triangles),
            "point_ids_sha256": _sha256_array(ids),
            "triangles_sha256": _sha256_array(triangles),
            "rest_points_sha256": _sha256_array(rest),
            "triangle_area_sha256": _sha256_array(triangle_area),
        },
        "outputs": {
            "rest_skin_vtp": str((DEFAULT_OUTPUT / "rest-skin-prestrain.vtp").resolve())
        },
    }
    return SkinPrestrain(
        vertex_weight, w, c, activation_inv, triangles, ids, triangle_area, provenance
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--closeups-summary", type=Path, default=DEFAULT_CLOSEUPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--taper-m", type=float, default=0.005)
    parser.add_argument("--max-contraction", type=float, default=0.01)
    args = parser.parse_args()
    field = build_skin_prestrain(
        args.fixture, args.closeups_summary, args.taper_m, args.max_contraction
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    skin = pv.read(
        (args.fixture / "skin.vtp")
        if args.fixture.is_dir()
        else args.fixture.with_name("skin.vtp")
    )
    skin.point_data["prestrain_weight"] = field.vertex_weight
    skin.cell_data["prestrain_weight"] = field.w
    skin.cell_data["contraction"] = field.c
    skin.cell_data["activation_inv"] = field.activation_inv
    skin.save(args.output_dir / "rest-skin-prestrain.vtp")
    # The VTP path is intentionally output-relative in the portable summary.
    field.provenance["outputs"]["rest_skin_vtp"] = str(
        (args.output_dir / "rest-skin-prestrain.vtp").resolve()
    )
    field.write(args.output_dir)


if __name__ == "__main__":
    main()
