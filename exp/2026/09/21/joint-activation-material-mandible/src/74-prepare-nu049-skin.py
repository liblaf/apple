"""Create the user-requested Poisson-ratio variant without changing other inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import array_sha256

from liblaf import cherries


class Config(cherries.BaseConfig):
    source_dir: Path = GROUP / "data/simple-skin-forward-inputs-001"
    output_dir: Path = GROUP / "data/simple-skin-forward-nu049-inputs-001"
    poisson: float = 0.49


def main(cfg: Config) -> None:
    assert 0 < cfg.poisson < 0.5
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    source_manifest = cfg.source_dir / "skin-field-manifest.json"
    source_artifact = cfg.source_dir / "skin-field.npz"
    manifest = json.loads(source_manifest.read_text())
    assert sha256(source_artifact) == manifest["artifact"]["sha256"]
    with np.load(source_artifact, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, value in arrays.items():
        assert array_sha256(value) == manifest["arrays"][name]["sha256"]
    arrays["nu"][:] = cfg.poisson
    target = cfg.output_dir / "skin-field.npz"
    np.savez_compressed(target, **arrays)
    manifest["artifact"] = {"path": str(target.resolve()), "sha256": sha256(target)}
    manifest["arrays"]["nu"]["sha256"] = array_sha256(arrays["nu"])
    manifest["field_construction"]["skin_poisson_ratio"] = cfg.poisson
    manifest["field_construction"]["actual_field_ranges"]["nu"] = [cfg.poisson] * 2
    manifest["parent_variant"] = {
        "manifest_path": str(source_manifest.resolve()),
        "manifest_sha256": sha256(source_manifest),
        "artifact_path": str(source_artifact.resolve()),
        "artifact_sha256": sha256(source_artifact),
        "change": "Only Poisson ratio changes, by explicit user instruction; E, thickness, stress, triangle ordering and reference geometry remain identical.",
    }
    manifest["provenance"] = {
        "path": str((cfg.output_dir / "provenance.json").resolve()),
        "sha256": sha256(cfg.output_dir / "provenance.json"),
        "poisson_status": "user-selected common near-incompressibility assumption",
    }
    write_json(cfg.output_dir / "skin-field-manifest.json", manifest)
    with np.load(target, allow_pickle=False) as archive:
        for name, value in arrays.items():
            assert np.array_equal(archive[name], value)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
