"""Audit local volume untangling on an existing reference-repair candidate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from reference_volume_repair import untangle_tetrahedra

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint, sha256  # noqa: E402


class Config(cherries.BaseConfig):
    candidate: Path = (
        GROUP / "data/reference-clearance-001/attempt-01/surface-projected.npz"
    )
    output_dir: Path = GROUP / "data/reference-untangle-001"


def main(cfg: Config) -> None:
    assert not cfg.output_dir.exists()
    cfg.output_dir.mkdir()
    manifest = json.loads(
        (
            ROOT
            / "exp/2026/09/21/joint-activation-material-mandible/data/frozen-neutral-004/manifest.json"
        ).read_text()
    )
    source = manifest["sources"]["constitutive_volume"]
    assert sha256(Path(source["path"])) == source["sha256"]
    mesh = pv.read(source["path"])
    tets = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    geometry_source = manifest["sources"]["geometry"]
    assert sha256(Path(geometry_source["path"])) == geometry_source["sha256"]
    with np.load(geometry_source["path"]) as geometry:
        fixed = geometry["fixed_global_ids"]
    with np.load(cfg.candidate) as archive:
        reference = archive["reference_points_m"]
        candidate = archive["repaired_points_m"]
    assert np.array_equal(reference, mesh.points)
    result, receipt = untangle_tetrahedra(reference, candidate, tets, fixed)
    np.savez_compressed(
        cfg.output_dir / "candidate.npz",
        reference_points_m=reference,
        repaired_points_m=result,
        displacement_m=result - reference,
    )
    receipt["input"] = {"path": str(cfg.candidate), "sha256": sha256(cfg.candidate)}
    (cfg.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    cherries.log_output(cfg.output_dir)
    cherries.log_metrics(receipt["final"])
    assert receipt["success"], receipt["final"]


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
