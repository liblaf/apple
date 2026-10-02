"""Transfer all source blendshape displacements to the eye-inclusive neutral."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs, array_sha256
from joint_frozen_neutral import load_script

from liblaf import cherries


class Config(cherries.BaseConfig):
    parent_neutral: Path = GROUP / "data/frozen-neutral-004"
    eye_run: Path = GROUP / "data/eye-neutral-forward-002"
    eye_review: Path = GROUP / "data/eye-neutral-forward-review-006/summary.json"
    eyes_manifest: Path = GROUP / "data/rigid-eyes-001/manifest.json"
    output_dir: Path = cherries.output("expression-inputs-002", mkdir=True)


def record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def main(cfg: Config) -> None:  # noqa: PLR0915
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    parent_manifest = json.loads((cfg.parent_neutral / "manifest.json").read_text())
    parent_state = np.load(cfg.parent_neutral / "state.npz")
    run = json.loads((cfg.eye_run / "summary.json").read_text())
    review = json.loads(cfg.eye_review.read_text())
    assert parent_manifest["success"]
    assert run["success"]
    assert review["success"]
    assert review["run"]["summary"]["sha256"] == sha256(cfg.eye_run / "summary.json")
    checkpoint = Path(run["checkpoint"]["path"])
    assert sha256(checkpoint) == run["checkpoint"]["sha256"]
    with np.load(checkpoint, allow_pickle=False) as stored:
        neutral_displacement = np.asarray(stored["displacement_m"], dtype=np.float64)
    parent_prepared = PreparedInputs.load(
        Path(parent_manifest["sources"]["prepared_npz"]["path"]),
        Path(parent_manifest["sources"]["prepared_manifest"]["path"]),
    )
    volume = pv.read(parent_prepared.volume_path)
    skin = pv.read(parent_prepared.skin_path)
    reference = np.asarray(volume.points, dtype=np.float64)
    assert neutral_displacement.shape == reference.shape
    neutral_points = reference + neutral_displacement
    names = tuple(
        name
        for name, value in volume.point_data.items()
        if np.asarray(value).shape == (volume.n_points, 3)
        and name not in {"FixedValue", "FixedMask"}
    )
    assert len(names) == 36, names
    parent_names = tuple(parent_manifest["cohort"]["names"])
    assert all(name in names for name in parent_names)
    arrays = {key: np.asarray(parent_state[key]).copy() for key in parent_state.files}
    obs = arrays["observation_node_ids"]
    targets = np.stack(
        [np.asarray(volume.point_data[name], dtype=np.float64)[obs] for name in names]
    )
    assert np.array_equal(
        targets[[names.index(name) for name in parent_names]],
        arrays["target_displacement_m"],
    )
    arrays["target_displacement_m"] = targets
    arrays["expression_displacement_m"] = targets.copy()
    arrays["neutral_displacement_m"] = neutral_displacement
    arrays["neutral_points_m"] = neutral_points
    arrays["target_points_m"] = neutral_points[obs][None] + targets
    arrays["target_total_displacement_m"] = neutral_displacement[obs][None] + targets
    cells = np.asarray(volume.cells).reshape(-1, 5)[:, 1:]
    ds = neutral_points[cells[:, 1:]] - neutral_points[cells[:, :1]]
    volumes = np.linalg.det(ds) / 6
    arrays["active_effective_volume_m3"] = (
        volumes[arrays["active_cell_ids"]] * arrays["active_muscle_fraction"]
    )
    preparation = load_script("10-prepare-inputs.py")
    gi, gj, conductance, graph = preparation.build_graph(
        neutral_points,
        cells,
        arrays["active_cell_ids"],
        np.asarray(volume.cell_data["MuscleId"]),
        np.asarray(volume.cell_data["MuscleFraction"]),
    )
    assert np.array_equal(gi, arrays["graph_i"])
    assert np.array_equal(gj, arrays["graph_j"])
    arrays["graph_conductance_m"] = conductance
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    assert np.array_equal(skin_ids, obs)
    triangles = np.asarray(skin.faces).reshape(-1, 4)[:, 1:]
    xyz = neutral_points[skin_ids][triangles]
    areas = (
        np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        / 2
    )
    weights = np.zeros(len(obs))
    np.add.at(weights, triangles.ravel(), np.repeat(areas / 3, 3))
    arrays["observation_area_weights_m2"] = weights
    arrays["observation_weight_normalized"] = weights / weights.sum()
    state_path = cfg.output_dir / "state.npz"
    np.savez_compressed(state_path, **arrays)
    manifest = {
        "schema": "joint-eye-expression-inputs-v1",
        "success": True,
        "expression_names": list(names),
        "coordinate_contract": "x_target = X_constitutive + u_eye_neutral + d_original_expression; no FEM reference or material field is rebased",
        "activation": "six independent symmetric active-stress components per active muscle tetrahedron and expression",
        "parent_frozen_neutral": {
            "directory": str(cfg.parent_neutral.resolve()),
            "manifest_sha256": sha256(cfg.parent_neutral / "manifest.json"),
        },
        "sources": {
            "parent_manifest": record(cfg.parent_neutral / "manifest.json"),
            "parent_state": record(cfg.parent_neutral / "state.npz"),
            "eye_run_summary": record(cfg.eye_run / "summary.json"),
            "eye_run_protocol": record(cfg.eye_run / "protocol.json"),
            "eye_checkpoint": record(checkpoint),
            "eye_review": record(cfg.eye_review),
            "eyes_manifest": record(cfg.eyes_manifest),
            "source_expression_volume": record(parent_prepared.volume_path),
        },
        "artifacts": {"state.npz": record(state_path)},
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": array_sha256(value),
            }
            for key, value in arrays.items()
        },
        "neutral": {
            "force_threshold_code": run["force_threshold"],
            "force_norm_code": run["final_free_force_norm"],
            "eye_fixed": review["fixed_eye_displacement_max_m"] == 0.0,
            "soft_rigid_intersection_free": review["ipc_soft_rigid_intersections"]
            is False,
        },
        "regularization_geometry": {
            "graph": graph,
            "effective_volumes_recomputed": True,
            "observation_areas_recomputed": True,
        },
        "transfer_validation": {
            "all_targets": len(names),
            "parent_six_exactly_preserved": True,
            "max_target_coordinate_error_m": float(
                np.abs(
                    reference[obs][None]
                    + arrays["target_total_displacement_m"]
                    - arrays["target_points_m"]
                ).max()
            ),
            "active_tetrahedra": len(arrays["active_cell_ids"]),
        },
    }
    write_json(cfg.output_dir / "manifest.json", manifest)
    cherries.log_metrics(
        {
            "expressions": len(names),
            "active_tetrahedra": len(arrays["active_cell_ids"]),
            "transfer_error_m": manifest["transfer_validation"][
                "max_target_coordinate_error_m"
            ],
        }
    )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
