"""Restore and verify the original experiment without modifying its source tree."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

GROUP = Path(__file__).resolve().parents[1]
REPO = GROUP.parents[4]
OLD = REPO / "exp/2026/09/21/stress-activation-loss"


def receipt(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    output = GROUP / "data/00-frozen-source"
    output.mkdir(parents=True, exist_ok=False)
    deployment = OLD / "tmp/remote-campaign/deployment.tar.gz"
    manifest_path = OLD / "tmp/remote-campaign/deployment-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with tarfile.open(deployment) as archive:
        archive.extractall(output, filter="data")
    apple = output / "apple"
    for name, record in manifest["files"].items():
        assert receipt(apple / name)["sha256"] == record["sha256"], name
    captured = OLD / "data/51-visualization-checkpoints-002/l2-normal"
    source_records = json.loads((captured / "source-freeze.json").read_text())
    # The deployment and actual loaded numerical source freeze must agree.
    for name, record in source_records.items():
        relative = Path(name).relative_to("sources")
        assert receipt(output / relative)["sha256"] == record["sha256"], name
    source_protocol = json.loads((captured / "protocol.json").read_text())
    fixture = (
        apple / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
    )
    for label, filename in (("volume", "volume.vtu"), ("skin", "skin.vtp")):
        assert (
            receipt(fixture / filename)["sha256"]
            == source_protocol["fixture"][label]["sha256"]
        )
    shared = GROUP / "data/10-sweep"
    shared.mkdir(parents=True, exist_ok=False)
    parent = shared / "parent-fixed-axis.npz"
    mesh = shared / "mesh.npz"
    shutil.copy2(captured / "l2-normal-rankone_fixed/last.npz", parent)
    shutil.copy2(captured / "mesh.npz", mesh)
    protocol = {
        "schema": "activation-smoothness-continuation-v1",
        "parent_checkpoint": receipt(parent),
        "mesh": receipt(mesh),
        "historical_protocol": receipt(captured / "protocol.json"),
        "frozen_source_root": str(apple.resolve()),
        "historical_source_freeze": source_records,
        "source_file_count": len(source_records),
        "deployment": receipt(deployment),
        "deployment_file_count": len(manifest["files"]),
        "activation_model": "strain",
        "mode": "rankone_learned",
        "multipliers": [1, 3, 10],
        "base_smooth_weight": 7.2e-7,
        "normal_weight": source_protocol["loss"]["normal_weight"],
        "steps": 200,
        "learning_rate": 0.05,
        "adam_eps": 1e-8,
        "fresh_adam": True,
        "initialization": "same saved S, fixed reference axes, and displacement seed; preserve historical zero-amplitude chart initialization",
        "selection": {
            "directional_roughness_max_relative_to_1x": 0.5,
            "fit_rms_max_relative_to_1x": 1.05,
            "normal_rms_max_relative_to_1x": 1.05,
            "tie_break": "smallest multiplier meeting all three proposed criteria",
            "physical_validity": "report independently; finite approximate continuation is not convergence certification",
        },
        "historical_runtime_note": "All new branches use one local RTX 4090 runtime; the historical source run used an RTX 5090 and different Python/Torch patch versions.",
    }
    (shared / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    shutil.copy2(captured / "protocol.json", shared / "historical-protocol.json")
    (output / "verification.json").write_text(
        json.dumps(
            {
                "deployment": receipt(deployment),
                "deployment_manifest": receipt(manifest_path),
                "deployment_files_verified": len(manifest["files"]),
                "numerical_sources_verified": len(source_records),
                "fixture_verified": True,
                "parent_checkpoint": receipt(parent),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "source_files": len(source_records),
                "deployment_files": len(manifest["files"]),
                "protocol": str(shared / "protocol.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
