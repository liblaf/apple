# Copyright (c) 2026 liblaf
"""Remove fully prescribed tetrahedra from the historical MouthOpen fixture."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
HISTORICAL = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
PREPARATION = GROUP / "data/10-mandible"
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))

from experiment import Profile  # noqa: E402

FACE_PATTERN = np.asarray(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))


class Config(cherries.BaseConfig):
    output: Path = Path("30-pruned-fixture")


def receipt(path: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def active_graph_counts(
    tets: np.ndarray,
    active: np.ndarray,
    control: np.ndarray,
    keep_cells: np.ndarray,
) -> dict[str, int]:
    """Count same-control face neighbors before and after pruning."""
    ids = np.flatnonzero(active)
    faces = np.sort(tets[ids][:, FACE_PATTERN].reshape(-1, 3), axis=1)
    owners = np.repeat(ids, 4)
    order = np.lexsort(faces.T[::-1])
    faces, owners = faces[order], owners[order]
    equal = np.flatnonzero(np.all(faces[1:] == faces[:-1], axis=1))
    assert not np.any(np.diff(equal) == 1)
    left, right = owners[equal], owners[equal + 1]
    same = control[left] == control[right]
    kept = keep_cells[left] & keep_cells[right]
    return {
        "same_control_face_edges_before": int(same.sum()),
        "same_control_face_edges_after": int(np.count_nonzero(same & kept)),
        "same_control_face_edges_removed": int(np.count_nonzero(same & ~kept)),
    }


def main(cfg: Config) -> None:  # noqa: PLR0915
    volume_path = cherries.input(HISTORICAL / "volume.vtu")
    skin_path = cherries.input(HISTORICAL / "skin.vtp")
    prepared_path = cherries.input(PREPARATION / "prepared.npz")
    pose_path = cherries.input(PREPARATION / "pose.json")
    audit_path = cherries.input(PREPARATION / "audit.json")
    output_dir = GROUP / "data" / cfg.output
    assert not output_dir.exists(), output_dir

    volume = pv.read(volume_path)
    skin = pv.read(skin_path)
    assert isinstance(volume, pv.UnstructuredGrid)
    assert isinstance(skin, pv.PolyData)
    assert set(np.unique(volume.celltypes)) == {int(pv.CellType.TETRA)}
    old_tets = np.asarray(volume.cells).reshape(-1, 5)[:, 1:].astype(np.int64)
    old_points = np.asarray(volume.points)
    fixed = np.asarray(volume.point_data["IsFixed"], dtype=bool)
    fixed_components = np.asarray(volume.point_data["FixedMask"], dtype=bool)
    np.testing.assert_array_equal(
        fixed_components, np.repeat(fixed[:, None], 3, axis=1)
    )
    pose_receipt = json.loads(pose_path.read_text())
    audit = json.loads(audit_path.read_text())
    assert pose_receipt["status"] == "blocked_by_prescribed_cell_inversions"
    assert audit["status"] == "blocked_by_prescribed_cell_inversions"
    with np.load(prepared_path, allow_pickle=False) as prepared:
        np.testing.assert_array_equal(old_points, prepared["X"])
        np.testing.assert_array_equal(old_tets, prepared["tets"])
        np.testing.assert_array_equal(fixed_components, prepared["fixed_mask"])
        old_skin_ids = np.asarray(prepared["skin_ids"], dtype=np.int64)
        old_allfixed = np.asarray(prepared["allfixed_cell_ids"], dtype=np.int64)
        jaw = np.asarray(prepared["jaw_mask"], dtype=bool)
        pose = np.asarray(prepared["pose"], dtype=np.float64)
        target_skin = np.asarray(prepared["target_skin"]).copy()
        triangles = np.asarray(prepared["triangles"], dtype=np.int64)
    np.testing.assert_array_equal(
        old_allfixed, np.flatnonzero(fixed[old_tets].all(axis=1))
    )
    np.testing.assert_array_equal(old_skin_ids, skin.point_data["GlobalPointId"])
    np.testing.assert_array_equal(
        triangles, np.asarray(skin.faces).reshape(-1, 4)[:, 1:]
    )
    np.testing.assert_array_equal(old_points[old_skin_ids], skin.points)
    np.testing.assert_array_equal(
        jaw, fixed & (np.asarray(volume.point_data["GroupId"]) == 28)
    )
    np.testing.assert_array_equal(
        pose, pose_receipt["full_rigid_chin_fit"]["pose_rad_m"]
    )
    assert len(old_allfixed) == 2249

    keep_cells = np.ones(volume.n_cells, dtype=bool)
    keep_cells[old_allfixed] = False
    kept_cell_ids = np.flatnonzero(keep_cells)
    kept_old_tets = old_tets[kept_cell_ids]
    used_points = np.zeros(volume.n_points, dtype=bool)
    used_points[kept_old_tets.ravel()] = True
    removed_point_ids = np.flatnonzero(~used_points)
    assert len(removed_point_ids) == 760
    assert np.all(fixed[removed_point_ids])
    assert np.all(used_points[~fixed])
    assert np.all(used_points[old_skin_ids])
    kept_point_ids = np.flatnonzero(used_points)
    point_old_to_new = np.full(volume.n_points, -1, dtype=np.int64)
    point_old_to_new[kept_point_ids] = np.arange(len(kept_point_ids))
    cell_old_to_new = np.full(volume.n_cells, -1, dtype=np.int64)
    cell_old_to_new[kept_cell_ids] = np.arange(len(kept_cell_ids))
    new_tets = point_old_to_new[kept_old_tets]
    assert np.all(new_tets >= 0)
    assert not np.any(fixed[kept_point_ids][new_tets].all(axis=1))

    active_old = np.asarray(volume.cell_data["ActivationMask"], dtype=bool)
    control_old = np.asarray(volume.cell_data["ActivationControlId"], dtype=np.int64)
    graph = active_graph_counts(old_tets, active_old, control_old, keep_cells)
    active_kept = active_old[kept_cell_ids]
    control_kept = control_old[kept_cell_ids].copy()
    region_old = np.unique(control_kept[active_kept])
    assert np.all(region_old >= 0)
    region_new = np.arange(len(region_old), dtype=np.int64)
    remapped = not np.array_equal(region_old, region_new)
    if remapped:
        control_kept[active_kept] = np.searchsorted(
            region_old, control_kept[active_kept]
        )
    np.testing.assert_array_equal(np.unique(control_kept[active_kept]), region_new)
    assert np.all(control_kept[~active_kept] == -1)

    cell_stream = np.column_stack((np.full(len(new_tets), 4), new_tets)).ravel()
    pruned = pv.UnstructuredGrid(
        cell_stream,
        np.full(len(new_tets), int(pv.CellType.TETRA), dtype=np.uint8),
        old_points[kept_point_ids].copy(),
    )
    for name in volume.point_data:
        pruned.point_data[name] = np.asarray(volume.point_data[name])[
            kept_point_ids
        ].copy()
    for name in volume.cell_data:
        pruned.cell_data[name] = np.asarray(volume.cell_data[name])[
            kept_cell_ids
        ].copy()
    for name in volume.field_data:
        pruned.field_data[name] = np.asarray(volume.field_data[name]).copy()
    pruned.point_data["OriginalPointId"] = kept_point_ids
    pruned.cell_data["OriginalCellId"] = kept_cell_ids
    pruned.point_data["GlobalPointId"] = np.arange(pruned.n_points, dtype=np.int64)
    if remapped:
        pruned.cell_data["ActivationControlId"] = control_kept
        for field in ("ActivationRegionMuscleId", "ActivationRegionName"):
            pruned.field_data[field] = np.asarray(volume.field_data[field])[
                region_old
            ].copy()

    new_skin = skin.copy(deep=True)
    new_skin.point_data["OriginalPointId"] = old_skin_ids
    new_skin.point_data["GlobalPointId"] = point_old_to_new[old_skin_ids]
    assert np.all(new_skin.point_data["GlobalPointId"] >= 0)
    np.testing.assert_array_equal(
        new_skin.points, pruned.points[new_skin.point_data["GlobalPointId"]]
    )
    np.testing.assert_array_equal(new_skin.faces, skin.faces)
    np.testing.assert_array_equal(
        target_skin,
        old_points[old_skin_ids]
        + np.asarray(volume.point_data["MouthOpen"])[old_skin_ids],
    )

    output_dir.mkdir(parents=True)
    volume_out = cherries.output(cfg.output / "volume.vtu")
    skin_out = cherries.output(cfg.output / "skin.vtp")
    mapping_out = cherries.output(cfg.output / "mapping.npz")
    summary_out = cherries.output(cfg.output / "summary.json")
    pruned.save(volume_out)
    new_skin.save(skin_out)
    np.savez_compressed(
        mapping_out,
        original_to_new_point=point_old_to_new,
        new_to_original_point=kept_point_ids,
        original_to_new_cell=cell_old_to_new,
        new_to_original_cell=kept_cell_ids,
        removed_original_point_ids=removed_point_ids,
        removed_original_cell_ids=old_allfixed,
        original_to_new_skin_point=point_old_to_new[old_skin_ids],
        old_active_control_ids=region_old,
        new_active_control_ids=region_new,
    )

    # Read the saved artifacts back; downstream code consumes these files.
    saved_volume = pv.read(volume_out)
    saved_skin = pv.read(skin_out)
    assert saved_volume.n_cells == len(kept_cell_ids)
    assert saved_volume.n_points == len(kept_point_ids)
    np.testing.assert_array_equal(
        saved_volume.point_data["OriginalPointId"], kept_point_ids
    )
    np.testing.assert_array_equal(
        saved_volume.cell_data["OriginalCellId"], kept_cell_ids
    )
    np.testing.assert_array_equal(
        saved_volume.point_data["GlobalPointId"], np.arange(len(kept_point_ids))
    )
    np.testing.assert_array_equal(
        saved_skin.point_data["GlobalPointId"], point_old_to_new[old_skin_ids]
    )
    np.testing.assert_array_equal(saved_skin.points, skin.points)
    np.testing.assert_array_equal(saved_skin.faces, skin.faces)
    np.testing.assert_array_equal(
        saved_volume.points[saved_skin.point_data["GlobalPointId"]], saved_skin.points
    )
    assert not np.any(
        np.asarray(saved_volume.point_data["IsFixed"], dtype=bool)[
            np.asarray(saved_volume.cells).reshape(-1, 5)[:, 1:]
        ].all(axis=1)
    )
    np.testing.assert_array_equal(
        np.asarray(saved_volume.point_data["FixedMask"], dtype=bool),
        np.repeat(
            np.asarray(saved_volume.point_data["IsFixed"], dtype=bool)[:, None],
            3,
            axis=1,
        ),
    )
    summary = {
        "schema": "mouthopen-pruned-historical-fixture-v1",
        "status": "topology_pruned_requires_forward_validation",
        "operation": "remove exactly every original tetrahedron with four IsFixed vertices, then compact only unused fixed vertices",
        "source": {
            "volume": receipt(volume_path),
            "skin": receipt(skin_path),
            "prepared": receipt(prepared_path),
            "pose": receipt(pose_path),
            "audit": receipt(audit_path),
            "script": receipt(Path(__file__)),
        },
        "counts": {
            "old_points": int(volume.n_points),
            "new_points": int(pruned.n_points),
            "removed_unused_fixed_points": len(removed_point_ids),
            "old_cells": int(volume.n_cells),
            "new_cells": int(pruned.n_cells),
            "removed_allfixed_cells": len(old_allfixed),
            "removed_active_cells": int(np.count_nonzero(active_old[old_allfixed])),
            "old_active_cells": int(active_old.sum()),
            "new_active_cells": int(active_kept.sum()),
            "old_skin_points": int(skin.n_points),
            "new_skin_points": int(saved_skin.n_points),
            "old_fixed_points": int(fixed.sum()),
            "new_fixed_points": int(
                np.count_nonzero(saved_volume.point_data["IsFixed"])
            ),
            "jaw_fixed_points_retained": int(np.count_nonzero(jaw[kept_point_ids])),
        },
        "cell_volume_m3": {
            "old_total": float(np.asarray(volume.cell_data["Volume"]).sum()),
            "removed": float(
                np.asarray(volume.cell_data["Volume"])[old_allfixed].sum()
            ),
            "new_total": float(np.asarray(pruned.cell_data["Volume"]).sum()),
        },
        "activation_graph": {
            **graph,
            "old_region_count": len(np.unique(control_old[active_old])),
            "new_region_count": len(region_old),
            "control_ids_remapped": remapped,
            "old_to_new_control_ids": dict(
                zip(region_old.tolist(), region_new.tolist(), strict=True)
            ),
            "must_rebuild_graph": True,
            "old_graph_edges_and_active_tet_indices_are_invalid_after_pruning": True,
        },
        "boundary": {
            "policy": "IsFixed is sole FEM clamp; IsFixed intersect Mandible receives jaw pose",
            "fixed_mask_matches_IsFixed_in_saved_fixture": True,
            "retained_allfixed_cells": 0,
            "jaw_group_id_from_GroupName": int(
                [str(x) for x in pruned.field_data["GroupName"]].index("Mandible")
            ),
        },
        "mapping": {
            "point_global_id_updated_to_new_index": True,
            "skin_global_id_updated_to_new_volume_index": True,
            "original_point_and_cell_ids_saved": True,
            "free_vertices_retained": True,
            "skin_geometry_and_connectivity_unchanged": True,
            "point_and_cell_fields_copied_except_explicit_id_and_optional_control_remap": True,
            "material_parameters_unchanged": True,
        },
        "outputs": {
            "volume": receipt(volume_out),
            "skin": receipt(skin_out),
            "mapping": receipt(mapping_out),
        },
        "caution": "Pruning changes the physical domain, active graph, and FEM boundary surface; no equilibrium, contact, or fit claim follows from this CPU build.",
    }
    assert np.isclose(
        summary["cell_volume_m3"]["old_total"] - summary["cell_volume_m3"]["removed"],
        summary["cell_volume_m3"]["new_total"],
        rtol=1e-12,
    )
    summary_out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    cherries.log_metrics(
        {
            "removed/cells": len(old_allfixed),
            "removed/active_cells": int(np.count_nonzero(active_old[old_allfixed])),
            "removed/points": len(removed_point_ids),
            "remaining/allfixed_cells": 0,
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
