"""Verify target geometry, comparison pixels, and changed published files."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from PIL import Image

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
OUTPUT = GROUP / "data/103-target-report-verification"


def main() -> None:
    assert not OUTPUT.exists(), OUTPUT
    OUTPUT.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location(
        "previous_report_verifier", GROUP / "src/99-verify-oblique-report.py"
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    data = GROUP / "data/101-oblique-target"
    summary = json.loads((data / "summary.json").read_text())
    baseline = json.loads((GROUP / "data/97-baseline-oblique/summary.json").read_text())
    assert summary["camera"] == baseline["camera"]
    assert summary["style"] == baseline["style"]
    for record in summary["sources"]:
        assert verifier.digest(Path(record["path"])) == record["sha256"], record["path"]
    fixture = (
        GROUP.parents[1] / "07/face-actuation-diagnosis/data/12-historical-fixture"
    )
    skin, volume = pv.read(fixture / "skin.vtp"), pv.read(fixture / "volume.vtu")
    target = pv.read(summary["target"]["mesh"]["path"])
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    np.testing.assert_array_equal(
        target.points, skin.points + volume.point_data["Smile"][ids]
    )
    np.testing.assert_array_equal(target.faces, skin.faces)
    np.testing.assert_array_equal(target.point_data["GlobalPointId"], ids)
    with Image.open(data / "target.png") as picture:
        assert picture.size == (1800, 1800)
        panels = {"target": np.array(picture.convert("RGB"))}
    for state in baseline["states"]:
        with Image.open(state["output"]["path"]) as picture:
            panels[state["id"]] = np.array(picture.convert("RGB"))
    expected_orders = {
        "old-vs-psd": ["target", "old", "psd"],
        "comparison": ["target", "old", "corrected", "psd"],
    }
    for name, order in expected_orders.items():
        assert summary["composites"][name]["order"] == order
        with Image.open(data / f"{name}.png") as picture:
            np.testing.assert_array_equal(
                np.array(picture.convert("RGB")),
                np.concatenate([panels[state] for state in order], axis=1),
            )
    site = GROUP / "site"
    current = json.loads((site / "manifest.json").read_text())
    prior = json.loads(
        (GROUP / "data/102-report-site-before-target/manifest.json").read_text()
    )
    previous_hashes = {
        record["site_path"]: record["sha256"]
        for record in [prior["index"], *prior["copied_files"]]
    }
    changed = [
        record["site_path"]
        for record in [current["index"], *current["copied_files"]]
        if previous_hashes.get(record["site_path"]) != record["sha256"]
    ]
    changed.append("manifest.json")
    local = []
    for relative in changed:
        receipt = verifier.stream_http_digest(relative)
        assert receipt["sha256"] == verifier.digest(site / relative), relative
        local.append(receipt)
    parser = verifier.PageParser()
    parser.feed((site / "index.html").read_text())
    for reference in parser.references:
        relative, fragment = verifier.relative_target(reference)
        assert (site / relative).is_file(), reference
        if fragment:
            assert fragment in parser.anchors, reference
    for name in expected_orders:
        assert f"assets/story/oblique-target/{name}.png" in parser.references
    peers = []
    for relative in (
        "index.html",
        "manifest.json",
        "assets/story/oblique-target/target.png",
        "assets/story/oblique-target/old-vs-psd.png",
        "assets/story/oblique-target/comparison.png",
        "downloads/story/oblique-target-summary.json",
        "downloads/activation-investigation.md",
    ):
        digest = verifier.peer_digest(relative)
        assert digest == verifier.digest(site / relative), relative
        peers.append({"path": relative, "sha256": digest})
    receipt = {
        "status": "passed",
        "target_points": target.n_points,
        "target_cells": target.n_cells,
        "target_geometry": "Exactly skin + Smile[GlobalPointId]; topology and point IDs identical",
        "camera_and_style": "Identical to baseline 97",
        "comparison_pixels": "Exact unscaled panels, target first, existing result pixels unchanged",
        "source_hashes_verified": len(summary["sources"]),
        "changed_files_verified": local,
        "unchanged_file_verification": "Prior full publication receipt data/99-oblique-report-verification/local-http.json",
        "peer_files_verified": peers,
        "service": verifier.verify_service(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    cherries.log_metrics(
        {
            "verification/changed_files": len(local),
            "verification/peer_files": len(peers),
            "verification/target_points": target.n_points,
        }
    )
    cherries.log_output(OUTPUT)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
