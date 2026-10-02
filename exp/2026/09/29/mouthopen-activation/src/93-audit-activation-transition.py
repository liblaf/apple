# Copyright (c) 2026 liblaf
"""CPU audit of a completed Smile-to-MouthOpen activation transition."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.append(str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa:E402


class Config(cherries.BaseConfig):
    output: Path = Path("93-transition-audit")
    transition: Path = GROUP / "data/91-smile-mouthopen-transition-003"
    fixture: Path = GROUP / "data/30-pruned-fixture"
    prepared: Path = GROUP / "data/10-mandible/prepared.npz"


def h(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(c: Config) -> None:  # noqa: C901, PLR0915
    out = cherries.output(c.output / "analysis.json", mkdir=True).parent
    assert not (out / "analysis.json").exists()
    s = json.loads((c.transition / "summary.json").read_text())
    assert s["status"] == "completed"
    assert not s["physical_validity_claim"]
    assert len(s["frames"]) == 121
    for r in s["inputs"].values():
        assert h(Path(r["path"])) == r["sha256"]
    if s["initial_seed"] is not None:
        for name in ("seed", "parent_summary"):
            receipt = s["initial_seed"][name]
            assert h(Path(receipt["path"])) == receipt["sha256"]
    for record in s["outputs"].values():
        assert h(Path(record["path"])) == record["sha256"]
    source_manifest = json.loads((c.transition / "source-manifest.json").read_text())
    for record in source_manifest.values():
        assert h(Path(record["path"])) == record["sha256"]
        assert h(Path(record["source"])) == record["sha256"]
    with np.load(c.transition / "endpoints.npz") as z:
        ss, sm, pose, pivot, source_rows, pointmap, cellmap = (
            z[k].copy()
            for k in (
                "S_smile",
                "S_mouthopen",
                "pose_mouthopen",
                "pivot",
                "smile_source_rows",
                "new_to_original_point",
                "new_to_original_cell",
            )
        )
    with np.load(c.transition / "mesh.npz") as z:
        points, tets, active, skin, weights = (
            z[k].copy()
            for k in (
                "rest_points",
                "tets",
                "active_ids",
                "skin_ids",
                "skin_vertex_weights",
            )
        )
    vol = pv.read(c.fixture / "volume.vtu")
    fixed = np.asarray(vol.point_data["IsFixed"], bool)
    names = list(vol.field_data["GroupName"])
    jaw = fixed & (np.asarray(vol.point_data["GroupId"]) == names.index("Mandible"))
    assert np.array_equal(points, vol.points)
    assert np.array_equal(tets, np.asarray(vol.cells).reshape(-1, 5)[:, 1:])
    assert np.array_equal(
        cellmap[active], np.asarray(vol.cell_data["OriginalCellId"])[active]
    )
    assert np.array_equal(pointmap, np.asarray(vol.point_data["OriginalPointId"]))
    smile_path = Path(s["inputs"]["smile"]["path"])
    smile_mesh_path = Path(s["inputs"]["smile_mesh"]["path"])
    mouth_path = Path(s["inputs"]["mouthopen"]["path"])
    with np.load(smile_path) as z:
        smile_full = z["S"].copy()
        smile_u = z["u"][pointmap].copy()
    with np.load(mouth_path) as z:
        np.testing.assert_allclose(sm, z["S"], rtol=0, atol=0)
        mouth_u = z["u"].copy()
    np.testing.assert_allclose(ss, smile_full[source_rows], rtol=0, atol=0)
    with np.load(smile_mesh_path) as z:
        original_active, original_tets = z["active_ids"], z["tets"]
        np.testing.assert_array_equal(points, z["rest_points"][pointmap])
    np.testing.assert_array_equal(original_active[source_rows], cellmap[active])
    np.testing.assert_array_equal(pointmap[tets], original_tets[cellmap])
    with np.load(c.prepared) as z:
        np.testing.assert_allclose(pose, z["pose"])
        np.testing.assert_array_equal(pivot, z["pivot"])

    def change_metrics(u: np.ndarray, reference: np.ndarray) -> dict:
        delta = u - reference
        return {
            "maximum_vertex_change_mm": float(
                1000 * np.linalg.norm(delta, axis=1).max()
            ),
            "skin_rms_change_mm": float(
                1000 * np.sqrt(np.sum(weights[:, None] * delta[skin] ** 2))
            ),
        }

    def boundary(a: float) -> np.ndarray:
        q = np.zeros_like(points)
        R = Rotation.from_rotvec((a * pose)[:3]).as_matrix()
        q[jaw] = (points[jaw] - pivot) @ R.T + pivot + (a * pose)[3:] - points[jaw]
        return q

    def det(u: np.ndarray) -> np.ndarray:
        r = np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
        x = points + u
        return np.linalg.det(
            np.transpose(x[tets[:, 1:]] - x[tets[:, :1]], (0, 2, 1))
        ) / np.linalg.det(r)

    for alpha in (0, 0.25, 0.5, 0.75, 1):
        assert np.linalg.eigvalsh((1 - alpha) * ss + alpha * sm).min() >= -1e-12
    frames = []
    previous = None
    jump = []
    indices = []
    endpoint_checks = {}
    force, inversions, intersections = [], [], []
    for row in s["frames"]:
        i, a = int(row["index"]), float(row["alpha"])
        indices.append(i)
        assert np.isclose(a, (1 - np.cos(np.pi * i / 120)) / 2, atol=1e-15, rtol=0)
        p = Path(row["checkpoint"]["path"])
        assert h(p) == row["checkpoint"]["sha256"]
        with np.load(p) as z:
            u, aa, pp = z["u"], float(z["alpha"]), z["pose"]
        assert np.isfinite(u).all()
        assert aa == a
        np.testing.assert_allclose(pp, a * pose)
        np.testing.assert_allclose(u[fixed], boundary(a)[fixed], atol=1e-12, rtol=0)
        d = row["diagnostics"]
        assert d["solver_valid"]
        assert d["accepted_force_norm"] <= 1e-10
        assert d["inverted_cells"] <= 1144
        force.append(float(d["accepted_force_norm"]))
        inversions.append(int(d["inverted_cells"]))
        intersections.append(bool(d["geometry"]["has_intersections"]))
        if previous is not None:
            jump.append(
                float(np.sqrt(np.mean(np.sum((u - previous) ** 2, axis=1))) * 1000)
            )
        if i in {0, 30, 60, 90, 120}:
            J = det(u)
            assert int((J <= 0).sum()) == d["inverted_cells"]
            np.testing.assert_allclose(J.min(), d["minimum_J"], atol=1e-9, rtol=1e-10)
            frames.append(
                {
                    "index": i,
                    "alpha": a,
                    "minimum_J": float(J.min()),
                    "inverted_cells": int((J <= 0).sum()),
                }
            )
        if i in {0, 120}:
            name = "Smile" if i == 0 else "MouthOpen"
            reference = smile_u if i == 0 else mouth_u
            drift = change_metrics(u, reference)
            claimed = s[
                "smile_reequilibration" if i == 0 else "mouthopen_endpoint_difference"
            ]
            for key, value in drift.items():
                np.testing.assert_allclose(value, claimed[key], atol=1e-12, rtol=1e-12)
            target = np.asarray(vol.point_data[name])[skin]
            assert np.isfinite(target).all()
            fit = float(
                1000 * np.sqrt(np.sum(weights[:, None] * (u[skin] - target) ** 2))
            )
            endpoint_checks[name] = {**drift, "fit_rms_mm": fit}
        previous = u
    assert indices == list(range(121))
    result = {
        "schema": "activation-transition-cpu-audit-v1",
        "frames": 121,
        "selected_detF": frames,
        "maximum_temporal_displacement_rms_mm": max(jump),
        "median_temporal_displacement_rms_mm": float(np.median(jump)),
        "force_norm": {"max": max(force), "median": float(np.median(force))},
        "inverted_cells": {"min": min(inversions), "max": max(inversions)},
        "self_intersection_frames": int(sum(intersections)),
        "input_summary_sha256": h(c.transition / "summary.json"),
        "mapping": s["mapping"],
        "smile_reequilibration": s["smile_reequilibration"],
        "mouthopen_replay": s["mouthopen_replay"],
        "mouthopen_endpoint_difference": s["mouthopen_endpoint_difference"],
        "independent_endpoint_checks": endpoint_checks,
        "verified_frozen_source_count": len(source_manifest),
        "physical_validity_claim": False,
        "note": "Frames are strict numerical equilibria on the pruned mesh; inversion and boundary-intersection diagnostics remain outside a mechanical-validity claim.",
    }
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    cherries.log_metrics({"frames": 121, "max_jump_mm": max(jump)})


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
