# Copyright (c) 2026 liblaf
"""Rebase the pruned no-skin fixture onto the repaired bulk reference."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
MOUTH = ROOT / "exp/2026/09/29/mouthopen-activation"
NEUTRAL = ROOT / "exp/2026/09/23/new-neutral"
HISTORICAL = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402
from stress_physics import active_graph  # noqa: E402


class Config(cherries.BaseConfig):
    output: Path = Path("50-fixed-reference")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def tet_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    volumes = np.empty(len(tets), dtype=np.float64)
    for start in range(0, len(tets), 100_000):
        cells = tets[start : start + 100_000]
        dm = points[cells[:, 1:]] - points[cells[:, :1]]
        volumes[start : start + len(cells)] = np.linalg.det(dm) / 6.0
    assert np.isfinite(volumes).all()
    assert np.all(volumes > 0), (
        "repaired reference contains nonpositive rest tetrahedra"
    )
    return volumes


def changed(a: np.ndarray, b: np.ndarray) -> dict:
    norms = np.linalg.norm(a - b, axis=1)
    return {
        "changed_vertices": int(np.count_nonzero(norms)),
        "maximum_m": float(norms.max()),
        "rms_m": float(np.sqrt(np.mean(norms**2))),
        "p95_m": float(np.quantile(norms, 0.95)),
    }


def main(cfg: Config) -> None:  # noqa: PLR0915
    out_dir = GROUP / "data" / cfg.output
    assert not out_dir.exists(), out_dir
    source = {
        "script": Path(__file__),
        "historical_volume": HISTORICAL / "volume.vtu",
        "pruned_volume": MOUTH / "data/30-pruned-fixture/volume.vtu",
        "pruned_skin": MOUTH / "data/30-pruned-fixture/skin.vtp",
        "pruned_mapping": MOUTH / "data/30-pruned-fixture/mapping.npz",
        "repaired_reference": NEUTRAL
        / "data/reference-clearance-002/reference-clearance.npz",
        "repaired_full_volume": NEUTRAL
        / "data/reference-clearance-002/repaired-reference-volume.vtu",
        "parent_mesh": MOUTH / "data/91-smile-mouthopen-transition-003/mesh.npz",
        "parent_endpoints": MOUTH
        / "data/91-smile-mouthopen-transition-003/endpoints.npz",
    }
    paths = {
        key: cherries.input(path) for key, path in source.items() if key != "script"
    }
    old = pv.read(paths["historical_volume"])
    volume = pv.read(paths["pruned_volume"])
    skin = pv.read(paths["pruned_skin"])
    repaired_full = pv.read(paths["repaired_full_volume"])
    old_tets = np.asarray(old.cells).reshape(-1, 5)[:, 1:]
    new_tets = np.asarray(volume.cells).reshape(-1, 5)[:, 1:]
    np.testing.assert_array_equal(old.cells, repaired_full.cells)
    np.testing.assert_array_equal(old.celltypes, repaired_full.celltypes)
    for field in ("IsFixed", "GroupId"):
        np.testing.assert_array_equal(
            old.point_data[field], repaired_full.point_data[field]
        )
    with np.load(paths["pruned_mapping"], allow_pickle=False) as mapping:
        new_to_old_point = mapping["new_to_original_point"].copy()
        new_to_old_cell = mapping["new_to_original_cell"].copy()
        old_to_new_point = mapping["original_to_new_point"].copy()
    assert len(new_to_old_point) == volume.n_points
    assert len(new_to_old_cell) == volume.n_cells
    np.testing.assert_array_equal(new_to_old_point[new_tets], old_tets[new_to_old_cell])
    np.testing.assert_array_equal(
        old_to_new_point[new_to_old_point], np.arange(volume.n_points)
    )
    np.testing.assert_array_equal(volume.points, old.points[new_to_old_point])
    np.testing.assert_array_equal(
        volume.point_data["OriginalPointId"], new_to_old_point
    )
    np.testing.assert_array_equal(volume.cell_data["OriginalCellId"], new_to_old_cell)
    np.testing.assert_array_equal(
        volume.point_data["IsFixed"], old.point_data["IsFixed"][new_to_old_point]
    )
    with np.load(paths["repaired_reference"], allow_pickle=False) as ref:
        source_reference = ref["reference_points_m"].copy()
        repaired_points = ref["repaired_points_m"].copy()
        np.testing.assert_array_equal(
            repaired_points - source_reference, ref["displacement_m"]
        )
    np.testing.assert_array_equal(repaired_full.points, repaired_points)
    fixed = np.asarray(volume.point_data["IsFixed"], dtype=bool)
    np.testing.assert_array_equal(
        source_reference[new_to_old_point][fixed], volume.points[fixed]
    )
    np.testing.assert_array_equal(
        repaired_points[new_to_old_point][fixed],
        source_reference[new_to_old_point][fixed],
    )
    np.testing.assert_array_equal(
        volume.point_data["FixedMask"], np.repeat(fixed[:, None], 3, axis=1)
    )
    assert not np.any(fixed[new_tets].all(axis=1))
    new_points = repaired_points[new_to_old_point]
    original_volumes = tet_volumes(np.asarray(volume.points), new_tets)
    new_volumes = tet_volumes(new_points, new_tets)
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    assert np.unique(skin_ids).size == len(skin_ids) == 15_299
    np.testing.assert_array_equal(skin.points, volume.points[skin_ids])
    triangles = np.asarray(skin.faces).reshape(-1, 4)[:, 1:]
    with np.load(paths["parent_mesh"], allow_pickle=False) as mesh:
        mesh_arrays = {key: mesh[key].copy() for key in mesh.files}
    with np.load(paths["parent_endpoints"], allow_pickle=False) as endpoints:
        np.testing.assert_array_equal(
            endpoints["new_to_original_point"], new_to_old_point
        )
        np.testing.assert_array_equal(
            endpoints["new_to_original_cell"], new_to_old_cell
        )
        assert (
            endpoints["S_smile"].shape
            == endpoints["S_mouthopen"].shape
            == (len(mesh_arrays["active_ids"]), 3, 3)
        )
    np.testing.assert_array_equal(mesh_arrays["rest_points"], volume.points)
    np.testing.assert_array_equal(mesh_arrays["tets"], new_tets)
    np.testing.assert_array_equal(mesh_arrays["skin_ids"], skin_ids)
    np.testing.assert_array_equal(mesh_arrays["triangles"], triangles)
    active_ids = np.flatnonzero(volume.cell_data["ActivationMask"])
    np.testing.assert_array_equal(mesh_arrays["active_ids"], active_ids)
    fraction = np.asarray(volume.cell_data["MuscleFraction"], dtype=np.float64)
    region = np.asarray(volume.cell_data["ActivationControlId"], dtype=np.int64)[
        active_ids
    ]
    graph_i, graph_j, graph_w = active_graph(
        new_points, new_tets, active_ids, region, fraction
    )
    np.testing.assert_array_equal(mesh_arrays["edge_i"], graph_i)
    np.testing.assert_array_equal(mesh_arrays["edge_j"], graph_j)
    old_active_weights = original_volumes[active_ids] * fraction[active_ids]
    old_active_weights /= old_active_weights.sum()
    np.testing.assert_allclose(
        mesh_arrays["active_volume_weights"], old_active_weights, rtol=1e-12, atol=0
    )
    mesh_arrays["rest_points"] = new_points
    new_active_weights = new_volumes[active_ids] * fraction[active_ids]
    mesh_arrays["active_volume_weights"] = new_active_weights / new_active_weights.sum()
    mesh_arrays["edge_weight"] = graph_w
    volume.points = new_points
    volume.cell_data["Volume"] = new_volumes
    skin.points = new_points[skin_ids]
    a, b, c = skin.points[triangles].transpose(1, 0, 2)
    skin.cell_data["RestArea"] = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    out_volume = cherries.output(cfg.output / "fixture/volume.vtu", mkdir=True)
    out_skin = cherries.output(cfg.output / "fixture/skin.vtp")
    out_mapping = cherries.output(cfg.output / "fixture/mapping.npz")
    out_mesh = cherries.output(cfg.output / "mesh.npz")
    out_endpoints = cherries.output(cfg.output / "endpoints.npz")
    volume.save(out_volume)
    skin.save(out_skin)
    shutil.copy2(paths["pruned_mapping"], out_mapping)
    np.savez_compressed(out_mesh, **mesh_arrays)
    shutil.copy2(paths["parent_endpoints"], out_endpoints)
    reread = pv.read(out_volume)
    np.testing.assert_array_equal(reread.points, new_points)
    np.testing.assert_array_equal(
        reread.cells, np.asarray(pv.read(paths["pruned_volume"]).cells)
    )
    np.testing.assert_array_equal(
        reread.point_data["FixedMask"], np.repeat(fixed[:, None], 3, axis=1)
    )
    assert sha256(out_mapping) == sha256(paths["pruned_mapping"])
    assert sha256(out_endpoints) == sha256(paths["parent_endpoints"])
    summary = {
        "schema": "pruned-repaired-reference-transfer-v1",
        "status": "prepared",
        "topology_mapping": "historical and repaired full cell connectivity identical; pruned cells map by OriginalCellId and vertices by OriginalPointId",
        "activation_transfer": "S_smile and S_mouthopen copied byte-for-byte by unchanged pruned active cell identity",
        "reference_distinction": "repaired source reference differs from historical coordinates on free vertices; fixed coordinates agree exactly",
        "counts": {
            "points": volume.n_points,
            "tets": volume.n_cells,
            "skin_points": skin.n_points,
            "active_cells": len(active_ids),
            "fixed_points": int(fixed.sum()),
        },
        "geometry": {
            "historical_to_repaired_source_reference": changed(
                np.asarray(old.points)[new_to_old_point],
                source_reference[new_to_old_point],
            ),
            "historical_to_final_repaired": changed(
                np.asarray(old.points)[new_to_old_point], new_points
            ),
            "source_reference_to_final_repaired": changed(
                source_reference[new_to_old_point], new_points
            ),
            "minimum_original_tet_volume_m3": float(original_volumes.min()),
            "minimum_repaired_tet_volume_m3": float(new_volumes.min()),
            "total_original_tet_volume_m3": float(original_volumes.sum()),
            "total_repaired_tet_volume_m3": float(new_volumes.sum()),
        },
        "inputs": {key: receipt(path) for key, path in source.items()},
        "outputs": {
            "fixture/volume.vtu": receipt(out_volume),
            "fixture/skin.vtp": receipt(out_skin),
            "fixture/mapping.npz": receipt(out_mapping),
            "mesh.npz": receipt(out_mesh),
            "endpoints.npz": receipt(out_endpoints),
        },
    }
    summary_path = cherries.output(cfg.output / "summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report = cherries.output("../docs/50-fixed-reference.md", mkdir=True)
    report.write_text(
        "# Repaired reference transfer for pruned activation fixture\n\n"
        "The full historical and repaired meshes have identical tetrahedron connectivity. "
        "The pruned fixture maps to them by its saved original point and cell IDs. "
        "Every prescribed point has identical coordinates in the historical, repaired-source, and final repaired references.\n\n"
        f"The final reference has {volume.n_points:,} points, {volume.n_cells:,} positive-volume tetrahedra, "
        f"and {len(active_ids):,} active cells. Minimum rest volume: {new_volumes.min():.9g} m³. "
        "The Smile and MouthOpen activation arrays and the point/cell maps are byte-identical to their sources.\n\n"
        "The repaired source reference differs from the historical free-point reference. "
        "The output therefore records both coordinate differences in `summary.json`. "
        "The derived mesh updates bulk rest volumes, active dual-volume weights, graph weights, and skin rest areas. "
        "Skin IDs and observation weights are retained exactly. These are fixture preparation artifacts, not a new equilibrium solve.\n"
    )
    cherries.log_metrics(
        {
            "reference/minimum_tet_volume_m3": float(new_volumes.min()),
            "reference/points": volume.n_points,
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
