"""Independently verify the projected neighborhood and its published locator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
OUTPUT = GROUP / "data/109-muscle-location-verification"


def main() -> None:
    assert not OUTPUT.exists(), OUTPUT
    OUTPUT.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location(
        "report_verifier", GROUP / "src/99-verify-oblique-report.py"
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    directory = GROUP / "data/107-muscle-location"
    summary = json.loads((directory / "summary.json").read_text())
    for record in [*summary["sources"], *summary["outputs"]]:
        assert verifier.digest(Path(record["path"])) == record["sha256"], record["path"]
    volume = pv.read(
        GROUP.parents[1]
        / "07/face-actuation-diagnosis/data/12-historical-fixture/volume.vtu"
    )
    cells = np.asarray(volume.cells).reshape(-1, 5)[:, 1:]
    with np.load(directory / "projection.npz") as saved:
        ids = np.load(GROUP / "data/93-focused-muscle-patch/patch-source-cell-ids.npy")
        np.testing.assert_array_equal(saved["cell_ids"], ids)
        point_ids = np.unique(cells[ids])
        np.testing.assert_array_equal(saved["point_ids"], point_ids)
        points = volume.points[point_ids]
        np.testing.assert_array_equal(saved["rest_points"], points)
        camera = summary["camera"]
        focal = np.asarray(camera["focal_point"])
        backward = np.asarray(camera["position"]) - focal
        backward /= np.linalg.norm(backward)
        right = np.cross(camera["view_up"], backward)
        right /= np.linalg.norm(right)
        up = np.cross(backward, right)
        projected = 900 + np.column_stack(
            ((points - focal) @ right, (points - focal) @ up)
        ) * (900 / camera["parallel_scale"])
        np.testing.assert_allclose(
            saved["display_pixels"], projected, rtol=0, atol=1e-8
        )
        minimum = projected.min(axis=0) - 18
        maximum = projected.max(axis=0) + 18
        np.testing.assert_allclose(saved["padded_minimum"], minimum, rtol=0, atol=1e-8)
        np.testing.assert_allclose(saved["padded_maximum"], maximum, rtol=0, atol=1e-8)
        np.testing.assert_allclose(
            summary["box"]["minimum"], minimum, rtol=0, atol=1e-8
        )
        np.testing.assert_allclose(
            summary["box"]["maximum"], maximum, rtol=0, atol=1e-8
        )
        assert np.all(projected >= minimum) and np.all(projected <= maximum)
        maximum_error = float(np.max(np.abs(saved["display_pixels"] - projected)))
    site = GROUP / "site"
    current = json.loads((site / "manifest.json").read_text())
    prior = json.loads(
        (GROUP / "data/108-report-site-before-locator/manifest.json").read_text()
    )
    old_hashes = {
        record["site_path"]: record["sha256"]
        for record in [prior["index"], *prior["copied_files"]]
    }
    changed = [
        record["site_path"]
        for record in [current["index"], *current["copied_files"]]
        if old_hashes.get(record["site_path"]) != record["sha256"]
    ] + ["manifest.json"]
    local = []
    for relative in changed:
        record = verifier.stream_http_digest(relative)
        assert record["sha256"] == verifier.digest(site / relative), relative
        local.append(record)
    peers = []
    for relative in (
        "index.html",
        "manifest.json",
        "assets/story/muscle-location.png",
        "downloads/story/muscle-location-summary.json",
        "downloads/sources/107-render-muscle-location.py",
        "downloads/activation-investigation.md",
    ):
        digest = verifier.peer_digest(relative)
        assert digest == verifier.digest(site / relative), relative
        peers.append({"path": relative, "sha256": digest})
    receipt = {
        "status": "passed",
        "patch_cells": len(ids),
        "patch_vertices": len(point_ids),
        "patch_identity": "Exact section-3 rest-selected cell IDs",
        "projection": "Independent analytic orthographic projection matches VTK WorldToDisplay",
        "maximum_projection_error_pixels": maximum_error,
        "box_encloses_all_patch_vertices": True,
        "box_padding_pixels": 18,
        "source_and_output_hashes": "matched",
        "changed_local_files": local,
        "peer_files": peers,
        "service": verifier.verify_service(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    cherries.log_metrics(
        {
            "verification/patch_cells": len(ids),
            "verification/projection_error_pixels": maximum_error,
            "verification/changed_files": len(local),
            "verification/peer_files": len(peers),
        }
    )
    cherries.log_output(OUTPUT)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
