"""Estimate the unrestricted rigid MouthOpen pose from the saved chin patch."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from chin_rigid_pose import estimate_rigid_chin_pose, validate_rigid_pose_fit

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
sys.path.insert(0, str(JOINT))

from joint_common import ProfileJoint, write_json  # noqa: E402


class Config(cherries.BaseConfig):
    source_protocol: Path = GROUP / "data/inverse-mouthopen-003/protocol.json"
    output_dir: Path = GROUP / "data/chin-rigid-pose-001"


def _record(path: Path) -> dict[str, str]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


def _bound_record(value: dict) -> Path:
    path = Path(value["path"])
    assert path.is_file()
    assert _record(path)["sha256"] == value["sha256"]
    return path


def main(cfg: Config) -> None:
    assert not cfg.output_dir.exists(), cfg.output_dir
    protocol_path = cfg.source_protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    assert protocol["schema"] == "new-neutral-mouthopen-raw6-inverse-v1"
    blendshapes = _bound_record(protocol["sources"]["blendshapes"])
    endpoint = _bound_record(protocol["sources"]["neutral_endpoint"])
    repair = _bound_record(protocol["sources"]["reference_repair"])
    chin = _bound_record(protocol["initialization"]["chin_estimate"])
    old = json.loads(chin.read_text())
    assert old["schema"] == "chin-pose-seed-v1"
    patch = np.asarray(old["patch_local_ids"], dtype=np.int64)
    assert len(patch) == 27
    with np.load(blendshapes, allow_pickle=False) as data:
        names = [str(name) for name in data["expression_names"]]
        index = names.index(protocol["expression_name"])
        assert index == protocol["expression_index"]
        ids = np.asarray(data["skin_global_ids"], dtype=np.int64)
        triangles = np.asarray(data["skin_triangles"], dtype=np.int64)
        neutral = np.asarray(data["new_neutral_points_m"], dtype=np.float64)
        target = np.asarray(data["target_points_m"][index], dtype=np.float64)
    with np.load(endpoint, allow_pickle=False) as data:
        displacement = np.asarray(data["displacement_m"], dtype=np.float64)
    with np.load(repair, allow_pickle=False) as data:
        reference = np.asarray(data["repaired_points_m"], dtype=np.float64)
    np.testing.assert_array_equal(reference[ids] + displacement[ids], neutral)
    pivot = np.asarray(protocol["parameterization"]["jaw_pivot_m"], dtype=np.float64)
    estimate = estimate_rigid_chin_pose(neutral, target, triangles, patch, pivot)
    synthetic = validate_rigid_pose_fit()
    assert synthetic["noncommuting_rotation_forward_max_error_m"] < 1e-12
    assert synthetic["rotation_vector_error_rad"] < 1e-12
    assert synthetic["translation_error_m"] < 1e-12
    assert synthetic["det_rotation"] > 0.999999999999
    estimate["sources"] = {
        "source_protocol": _record(protocol_path),
        "blendshapes": _record(blendshapes),
        "neutral_endpoint": _record(endpoint),
        "reference_repair": _record(repair),
        "chin_patch_receipt": _record(chin),
    }
    estimate["synthetic_validation"] = synthetic
    estimate["unconstrained_fit"] = True
    estimate["seed_only"] = True
    cfg.output_dir.mkdir(parents=True)
    output = cfg.output_dir / "estimate.json"
    write_json(output, estimate)
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "weighted_rms_before_mm": estimate["weighted_rms_before_m"] * 1000,
            "weighted_rms_after_mm": estimate["weighted_rms_after_m"] * 1000,
            "rotation_degrees": estimate["fit_rotation_degrees"],
            "translation_norm_mm": estimate["translation_norm_m"] * 1000,
            "svd_condition": estimate["svd_condition"],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
