"""Independently audit saved fixed-activation contact states on CPU."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
from pathlib import Path

import ipctk
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402

LOG = logging.getLogger(__name__)
TERMINAL = {"completed", "blocked", "failed", "interrupted"}


class Config(cherries.BaseConfig):
    output: Path = Path("25-existing-contact-audit")
    run: Path = GROUP / "data/23-contact-existing-activation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_record(record: dict) -> None:
    path = Path(record["path"])
    assert path.is_file(), path
    assert sha256(path) == record["sha256"], path


def activation_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def detf_counts(
    points: np.ndarray, tets: np.ndarray, dm_inv: np.ndarray, u: np.ndarray
) -> tuple[int, float]:
    deformed = points + u
    inverted = 0
    minimum = math.inf
    for start in range(0, len(tets), 100_000):
        ids = tets[start : start + 100_000]
        moved = np.transpose(deformed[ids[:, 1:]] - deformed[ids[:, :1]], (0, 2, 1))
        j = np.linalg.det(moved @ dm_inv[start : start + len(ids)])
        assert np.isfinite(j).all()
        inverted += int(np.count_nonzero(j <= 0))
        minimum = min(minimum, float(j.min()))
    return inverted, minimum


def main(cfg: Config) -> None:  # noqa: C901, PLR0915
    out = cherries.output(cfg.output / "summary.json", mkdir=True)
    assert not out.exists()
    summary_path = cfg.run / "summary.json"
    source_manifest_path = cfg.run / "source-manifest.json"
    parent = json.loads(summary_path.read_text())
    assert parent["schema"] == "collision-activation-transition-v1"
    assert parent["status"] in TERMINAL, (
        "the numerical process has not written a terminal status"
    )
    inputs = parent["inputs"]
    for record in inputs.values():
        checked_record(record)
    source_manifest = json.loads(source_manifest_path.read_text())
    for record in source_manifest.values():
        checked_record(record)
        assert sha256(Path(record["source"])) == record["sha256"]
    assert any(
        Path(item["source"]).name == "23-contact-existing-activation.py"
        for item in source_manifest.values()
    )
    source = Path(parent["config"]["source"])
    parent91 = json.loads((source / "summary.json").read_text())
    assert parent91["status"] == "completed"
    for name in ("mesh.npz", "endpoints.npz"):
        assert sha256(cfg.run / name) == sha256(source / name)
        assert sha256(cfg.run / name) == parent91["outputs"][name]["sha256"]
    with np.load(cfg.run / "endpoints.npz") as z:
        mouth = z["S_mouthopen"].copy()
        smile = z["S_smile"].copy()
        full_pose = z["pose_mouthopen"].copy()
        pivot = z["pivot"].copy()
    with np.load(cfg.run / "mesh.npz") as z:
        points = z["rest_points"].copy()
        tets = z["tets"].copy()
        active_ids = z["active_ids"].copy()
    assert mouth.shape == smile.shape == (len(active_ids), 3, 3)
    volume = pv.read(Path(inputs["volume"]["path"]))
    np.testing.assert_array_equal(points, np.asarray(volume.points))
    np.testing.assert_array_equal(tets, np.asarray(volume.cells).reshape(-1, 5)[:, 1:])
    np.testing.assert_array_equal(
        active_ids, np.flatnonzero(np.asarray(volume.cell_data["ActivationMask"], bool))
    )
    dm_inv = np.empty((len(tets), 3, 3), dtype=np.float64)
    for start in range(0, len(tets), 100_000):
        ids = tets[start : start + 100_000]
        rest = np.transpose(points[ids[:, 1:]] - points[ids[:, :1]], (0, 2, 1))
        dm_inv[start : start + len(ids)] = np.linalg.inv(rest)
    fixed = np.asarray(volume.point_data["IsFixed"], bool)
    fixed_mask = np.asarray(volume.point_data["FixedMask"], bool)
    np.testing.assert_array_equal(fixed_mask.any(axis=1), fixed)
    names = [str(x) for x in np.asarray(volume.field_data["GroupName"]).ravel()]
    jaw = fixed & (np.asarray(volume.point_data["GroupId"]) == names.index("Mandible"))
    assert int(jaw.sum()) == 5989
    surface = volume.extract_surface(algorithm=None, pass_pointid=True)
    global_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    faces = np.asfortranarray(
        np.asarray(surface.faces).reshape(-1, 4)[:, 1:], dtype=np.int32
    )
    mesh = ipctk.CollisionMesh(
        np.asfortranarray(points[global_ids]), ipctk.edges(faces), faces
    )
    mesh.init_adjacencies()
    full_mesh = ipctk.CollisionMesh(
        np.asfortranarray(points[global_ids]), ipctk.edges(faces), faces
    )
    full_mesh.init_adjacencies()
    patches = np.where(fixed, 0, np.arange(len(fixed)) + 1).astype(np.int32)
    mesh.can_collide = ipctk.make_vertex_patches_filter(patches[global_ids])
    assert not mesh.can_collide(
        int(np.flatnonzero(fixed[global_ids])[0]),
        int(np.flatnonzero(fixed[global_ids])[-1]),
    )
    assert mesh.can_collide(
        int(np.flatnonzero(fixed[global_ids])[0]),
        int(np.flatnonzero(~fixed[global_ids])[0]),
    )
    broad = ipctk.LBVH()
    inversion_limit = parent["config"]["inversion_fraction_limit"] * len(tets)
    force_limit = parent["config"]["force_atol"]

    def boundary(alpha: float) -> np.ndarray:
        result = np.zeros_like(points)
        pose = alpha * full_pose
        result[jaw] = (
            (points[jaw] - pivot) @ Rotation.from_rotvec(pose[:3]).as_matrix().T
            + pivot
            + pose[3:]
            - points[jaw]
        )
        return result

    def inspect(row: dict, *, frame: bool) -> dict:
        record = row["checkpoint"]
        checked_record(record)
        with np.load(record["path"]) as z:
            u = z["u"].copy()
            alpha = float(z["alpha"])
            saved_pose = z["pose"].copy()
            fraction = float(z["beta"] if frame else z["fraction"])
        assert u.shape == points.shape
        assert np.isfinite(u).all()
        if frame:
            assert abs(alpha - float(row["alpha"])) < 1e-12
        np.testing.assert_allclose(saved_pose, alpha * full_pose, rtol=0, atol=1e-12)
        np.testing.assert_allclose(u[fixed], boundary(alpha)[fixed], rtol=0, atol=1e-12)
        if frame:
            beta = float(row["beta"])
            assert abs(fraction - beta) < 1e-15
            assert abs(alpha - (1 - beta)) < 1e-12
            activation = (1 - beta) * mouth + beta * smile
        else:
            stage = row["phase"]
            assert abs(fraction - float(row["fraction"])) < 1e-15
            if stage == "neutral":
                assert alpha == 0
                activation = np.zeros_like(mouth)
            elif stage == "initialization":
                assert abs(alpha - fraction) < 1e-12
                activation = fraction * mouth
            elif stage == "mouthopen":
                assert alpha == 1
                activation = mouth
            elif stage == "transition":
                assert abs(alpha - (1 - fraction)) < 1e-12
                activation = (1 - fraction) * mouth + fraction * smile
            else:
                raise AssertionError(stage)
        receipt = row["diagnostics"]
        assert receipt["solver_valid"] is True
        assert math.isfinite(receipt["accepted_force_norm"])
        assert receipt["accepted_force_norm"] <= force_limit
        assert receipt["contact"]["contact_valid"] is True
        assert receipt["contact"]["scoped_boundary_no_intersections"] is True
        assert receipt["inverted_cells"] <= inversion_limit
        inverted, minimum = detf_counts(points, tets, dm_inv, u)
        assert inverted == receipt["inverted_cells"]
        assert math.isclose(minimum, receipt["minimum_J"], rel_tol=1e-7, abs_tol=1e-7)
        scoped_positions = np.asfortranarray(points[global_ids] + u[global_ids])
        scoped_intersects = bool(ipctk.has_intersections(mesh, scoped_positions, broad))
        assert not scoped_intersects
        LOG.info(
            "Verified %s %s, force %.3g, inverted %d",
            "frame" if frame else "accepted",
            fraction,
            receipt["accepted_force_norm"],
            inverted,
        )
        return {
            "checkpoint_sha256": record["sha256"],
            "fraction": fraction,
            "alpha": alpha,
            "activation_sha256": activation_digest(activation),
            "activation_max_abs": float(np.max(np.abs(activation))),
            "free_force_norm": receipt["accepted_force_norm"],
            "recomputed_inverted_cells": inverted,
            "recomputed_minimum_J": minimum,
            "recomputed_scoped_intersections": scoped_intersects,
        }

    accepted = [inspect(row, frame=False) for row in parent["accepted"]]
    frames = [inspect(row, frame=True) for row in parent["frames"]]
    failed_state = None
    failed_path = cfg.run / "failed-solver-state.npz"
    if failed_path.exists():
        with np.load(failed_path) as z:
            failed_u = z["u"].copy()
        assert failed_u.shape == points.shape
        assert np.isfinite(failed_u).all()
        failed_alpha = (
            0.0 if parent.get("failure", {}).get("phase") == "neutral" else None
        )
        if failed_alpha is not None:
            np.testing.assert_allclose(
                failed_u[fixed], boundary(failed_alpha)[fixed], rtol=0, atol=1e-12
            )
        failed_inverted, failed_minimum = detf_counts(points, tets, dm_inv, failed_u)
        failed_positions = np.asfortranarray(points[global_ids] + failed_u[global_ids])
        failed_state = {
            "checkpoint_sha256": sha256(failed_path),
            "fixed_boundary_checked_at_alpha": failed_alpha,
            "fixed_boundary_max_abs_error": float(
                np.max(np.abs(failed_u[fixed] - boundary(failed_alpha)[fixed]))
            )
            if failed_alpha is not None
            else None,
            "recomputed_inverted_cells": failed_inverted,
            "recomputed_minimum_J": failed_minimum,
            "recomputed_scoped_intersections": bool(
                ipctk.has_intersections(mesh, failed_positions, broad)
            ),
            "recomputed_unfiltered_intersections": bool(
                ipctk.has_intersections(full_mesh, failed_positions, broad)
            ),
            "accepted_equilibrium": False,
            "force_norm_note": "No accepted free-force receipt is attached to this failed solver state.",
        }
    complete = parent["status"] == "completed"
    if complete:
        assert len(frames) == parent["config"]["frames"]
        assert frames[0]["fraction"] == 0
        assert frames[-1]["fraction"] == 1
    result = {
        "schema": "existing-activation-contact-independent-audit-v1",
        "numerical_status": parent["status"],
        "experiment_completed": complete,
        "audit_complete_for_saved_states": True,
        "zero_frames_is_completion": False,
        "inputs_verified": len(inputs),
        "frozen_source_records_verified": len(source_manifest),
        "endpoint_sha256": {
            "mouthopen_S": activation_digest(mouth),
            "smile_S": activation_digest(smile),
        },
        "mesh_identical_to_parent_and_fixture": True,
        "contact_scope": "all faces; only fixed-fixed vertex pairs exempt",
        "accepted_count": len(accepted),
        "frame_count": len(frames),
        "accepted": accepted,
        "frames": frames,
        "failed_solver_state": failed_state,
        "numerical_failure": parent.get("failure"),
        "source_summary_sha256": sha256(summary_path),
        "source_manifest_sha256": sha256(source_manifest_path),
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    LOG.info(
        "Wrote %s: status %s; %d accepted, %d frames",
        out,
        parent["status"],
        len(accepted),
        len(frames),
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
