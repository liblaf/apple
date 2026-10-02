# Copyright (c) 2026 liblaf
"""Independently audit saved repaired-reference contact states on CPU."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import ipctk
import numpy as np
import pyvista as pv
import torch
from scipy.spatial.transform import Rotation

from liblaf import cherries
from liblaf.apple.forward._problem import ForwardProblem

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402
from fixed_reference_contact import (  # noqa: E402
    attach_fixed_reference_contact,
    build_fixed_reference_contact,
)
from stress_physics import (  # noqa: E402
    FacePhysics,
    configure,
    strain_to_activation_inv,
)


class Config(cherries.BaseConfig):
    output: Path = Path("54-fixed-contact-audit")
    run: Path = GROUP / "data/52-fixed-activation-contact"
    fresh_force: bool = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_record(row: dict) -> None:
    path = Path(row["path"])
    assert path.is_file(), path
    assert sha256(path) == row["sha256"], path


def tensor_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def tet_j(points: np.ndarray, moved: np.ndarray, tets: np.ndarray) -> tuple[int, float]:
    inverted = 0
    minimum = math.inf
    for start in range(0, len(tets), 100_000):
        ids = tets[start : start + 100_000]
        rest = np.transpose(points[ids[:, 1:]] - points[ids[:, :1]], (0, 2, 1))
        current = np.transpose(moved[ids[:, 1:]] - moved[ids[:, :1]], (0, 2, 1))
        j = np.linalg.det(current) / np.linalg.det(rest)
        assert np.isfinite(j).all()
        inverted += int(np.count_nonzero(j <= 0))
        minimum = min(minimum, float(j.min()))
    return inverted, minimum


def main(cfg: Config) -> None:  # noqa: C901, PLR0915
    output = cherries.output(cfg.output / "summary.json", mkdir=True)
    assert not output.exists(), output
    run = cfg.run
    numerical = json.loads((run / "summary.json").read_text())
    assert numerical["schema"] == "fixed-reference-activation-transition-v1"
    assert numerical["status"] in {"completed", "blocked", "interrupted", "failed"}
    assert numerical.get("provenance_verified") is True
    for row in numerical["inputs"].values():
        check_record(row)
    frozen = json.loads((run / "source-manifest.json").read_text())
    for row in frozen.values():
        check_record(row)
        assert sha256(Path(row["source"])) == row["sha256"]
    assert any(
        Path(row["source"]).name == "52-run-fixed-activation-contact.py"
        for row in frozen.values()
    )
    assert any(
        Path(row["source"]).name == "fixed_reference_contact.py"
        for row in frozen.values()
    )
    for name in ("mesh.npz", "endpoints.npz"):
        assert (
            sha256(run / name)
            == numerical["inputs"]["mesh" if name == "mesh.npz" else "endpoints"][
                "sha256"
            ]
        )
    source = Path(numerical["config"]["source"])
    parent = json.loads((source / "summary.json").read_text())
    assert parent["status"] == "prepared"
    for name in ("mesh.npz", "endpoints.npz"):
        assert sha256(run / name) == parent["outputs"][name]["sha256"]
    with np.load(run / "mesh.npz", allow_pickle=False) as mesh:
        points = mesh["rest_points"].copy()
        tets = mesh["tets"].copy()
        active_ids = mesh["active_ids"].copy()
    with np.load(run / "endpoints.npz", allow_pickle=False) as endpoints:
        smile = endpoints["S_smile"].copy()
        mouth = endpoints["S_mouthopen"].copy()
        pose = endpoints["pose_mouthopen"].copy()
        pivot = endpoints["pivot"].copy()
    assert smile.shape == mouth.shape == (len(active_ids), 3, 3)
    volume = pv.read(Path(numerical["inputs"]["volume"]["path"]))
    np.testing.assert_array_equal(volume.points, points)
    np.testing.assert_array_equal(np.asarray(volume.cells).reshape(-1, 5)[:, 1:], tets)
    np.testing.assert_array_equal(
        np.flatnonzero(volume.cell_data["ActivationMask"]), active_ids
    )
    fixed = np.asarray(volume.point_data["IsFixed"], dtype=bool)
    np.testing.assert_array_equal(
        volume.point_data["FixedMask"], np.repeat(fixed[:, None], 3, axis=1)
    )
    group_names = [str(x) for x in np.asarray(volume.field_data["GroupName"]).ravel()]
    jaw = fixed & (
        np.asarray(volume.point_data["GroupId"]) == group_names.index("Mandible")
    )
    contact_surface = numerical["contact_surface"]
    for name, row in contact_surface["source"].items():
        check_record(row)
        assert numerical["inputs"][f"rigid_{name}"]["sha256"] == row["sha256"]
    ref = build_fixed_reference_contact(
        volume,
        stiffness_mpa=numerical["config"]["stiffness_mpa"],
        dhat_m=numerical["config"]["dhat_m"],
        minimum_distance_m=numerical["config"]["minimum_distance_m"],
        cranium_path=Path(contact_surface["source"]["cranium"]["path"]),
        mandible_path=Path(contact_surface["source"]["mandible"]["path"]),
        eyes_path=Path(contact_surface["source"]["eyes"]["path"]),
    )
    assert ref.receipt == contact_surface
    with np.load(run / "rigid-geometry.npz", allow_pickle=False) as geometry:
        np.testing.assert_array_equal(
            geometry["points"], ref.full_reference_points(points)
        )
        np.testing.assert_array_equal(
            geometry["contact_indices"], ref.contact.indices.numpy(force=True)
        )
        np.testing.assert_array_equal(
            geometry["rigid_mandible_mask"], ref.full_mandible_mask[len(points) :]
        )
    full_points = ref.full_reference_points(points)
    full_fixed = np.r_[fixed, np.ones(len(full_points) - len(points), bool)]
    full_jaw = ref.full_mandible_mask
    assert np.array_equal(full_jaw[: len(points)], jaw)
    assert np.all(full_fixed[full_jaw])
    scope = numerical["collision_scope"]
    force_limit = numerical["config"]["force_atol"]
    inverted_limit = numerical["config"]["inversion_fraction_limit"] * len(tets)
    broad = ipctk.LBVH()
    force_problem = None
    force_model = None
    force_physics = None
    force_contact = None
    if cfg.fresh_force:
        configure()
        force_physics = FacePhysics(
            Path(numerical["config"]["fixture"]),
            activation_model="strain",
            atol=force_limit,
        )
        np.testing.assert_array_equal(force_physics.points, points)
        np.testing.assert_array_equal(force_physics.tets, tets)
        np.testing.assert_array_equal(force_physics.ids, active_ids)
        gpu_reference = build_fixed_reference_contact(
            force_physics.mesh,
            stiffness_mpa=numerical["config"]["stiffness_mpa"],
            dhat_m=numerical["config"]["dhat_m"],
            minimum_distance_m=numerical["config"]["minimum_distance_m"],
            cranium_path=Path(contact_surface["source"]["cranium"]["path"]),
            mandible_path=Path(contact_surface["source"]["mandible"]["path"]),
            eyes_path=Path(contact_surface["source"]["eyes"]["path"]),
        )
        assert gpu_reference.receipt == contact_surface
        force_model = attach_fixed_reference_contact(
            force_physics.forward.model,
            gpu_reference,
            torch.zeros_like(torch.as_tensor(full_points)),
        )
        force_contact = gpu_reference.contact
        np.testing.assert_array_equal(
            force_model.dof_map.fixed_indices.numpy(force=True),
            np.flatnonzero(np.repeat(full_fixed, 3)),
        )
        force_problem = ForwardProblem(model=force_model)

    def boundary(alpha: float) -> np.ndarray:
        values = np.zeros_like(full_points)
        scaled = alpha * pose
        values[full_jaw] = (
            (full_points[full_jaw] - pivot)
            @ Rotation.from_rotvec(scaled[:3]).as_matrix().T
            + pivot
            + scaled[3:]
            - full_points[full_jaw]
        )
        return values

    def inspect(row: dict, *, frame: bool) -> dict:  # noqa: PLR0915
        check_record(row["checkpoint"])
        with np.load(row["checkpoint"]["path"], allow_pickle=False) as z:
            u = z["u"].copy()
            full_u = z["u_full"].copy()
            alpha = float(z["alpha"])
            value = float(z["beta"] if frame else z["fraction"])
            np.testing.assert_allclose(z["pose"], alpha * pose, rtol=0, atol=1e-12)
        assert u.shape == points.shape
        assert full_u.shape == full_points.shape
        assert np.isfinite(full_u).all()
        np.testing.assert_array_equal(full_u[: len(points)], u)
        np.testing.assert_allclose(
            full_u[full_fixed], boundary(alpha)[full_fixed], rtol=0, atol=1e-12
        )
        if frame:
            assert abs(value - row["beta"]) < 1e-15
            assert abs(alpha - row["alpha"]) < 1e-15
            assert abs(alpha - (1 - value)) < 1e-12
            activation = (1 - value) * mouth + value * smile
        else:
            assert abs(value - row["fraction"]) < 1e-15
            stage = row["phase"]
            if stage == "neutral":
                assert value == alpha == 0
                activation = np.zeros_like(mouth)
            elif stage == "initialization":
                assert abs(value - alpha) < 1e-12
                activation = value * mouth
            elif stage == "transition":
                assert abs(alpha - (1 - value)) < 1e-12
                activation = (1 - value) * mouth + value * smile
            else:
                raise AssertionError(stage)
        diagnostics = row["diagnostics"]
        assert diagnostics["solver_valid"] is True
        assert math.isfinite(diagnostics["accepted_force_norm"])
        assert diagnostics["accepted_force_norm"] <= force_limit
        assert diagnostics["contact"]["contact_valid"] is True
        assert diagnostics["contact"]["scoped_boundary_no_intersections"] is True
        assert diagnostics["inverted_cells"] <= inverted_limit
        count, min_j = tet_j(points, points + u, tets)
        assert count == diagnostics["inverted_cells"]
        assert math.isclose(min_j, diagnostics["minimum_J"], rel_tol=1e-7, abs_tol=1e-7)
        contact_points = np.asfortranarray(
            ref.contact.vertices.numpy(force=True)
            + full_u[ref.contact.indices.numpy(force=True)]
        )
        intersects = bool(
            ipctk.has_intersections(ref.contact.collision_mesh, contact_points, broad)
        )
        assert not intersects
        measured_force = None
        fresh_contact = None
        if cfg.fresh_force:
            assert force_physics is not None
            assert force_model is not None
            assert force_contact is not None
            assert force_problem is not None
            with torch.no_grad():
                activation_tensor = torch.as_tensor(activation)
                force_physics.materials["muscle"]["activation_inv"] = torch.zeros(
                    (force_physics.mesh.n_cells, 6)
                ).index_copy(
                    0,
                    force_physics.id_t,
                    strain_to_activation_inv(activation_tensor),
                )
                force_model.set_materials(force_physics.materials)
                force_model.dof_map.fixed_values = (
                    torch.as_tensor(boundary(alpha))
                    .flatten()[force_model.dof_map.fixed_indices]
                    .clone()
                )
                full_tensor = torch.as_tensor(full_u)
                state = force_model.State(
                    u=full_tensor,
                    collision=force_contact.state_at(full_tensor),
                )
                measured_force = float(
                    torch.linalg.vector_norm(force_problem.grad(state))
                )
                fresh_contact = force_contact.diagnostics(state.collision, state.u)
            assert math.isfinite(measured_force)
            assert measured_force <= force_limit
            assert fresh_contact["contact_numerically_valid"] is True
            assert (
                fresh_contact["active_contact_count"]
                == diagnostics["contact"]["active_contact_count"]
            )
            saved_gap = diagnostics["contact"]["minimum_active_distance_m"]
            new_gap = fresh_contact["minimum_active_distance_m"]
            assert (saved_gap is None) == (new_gap is None)
            if new_gap is not None:
                assert new_gap > numerical["config"]["minimum_distance_m"]
                assert math.isclose(new_gap, saved_gap, rel_tol=1e-7, abs_tol=1e-12)
            assert math.isclose(
                fresh_contact["barrier_energy"],
                diagnostics["contact"]["barrier_energy"],
                rel_tol=1e-7,
                abs_tol=1e-12,
            )
            assert math.isclose(
                measured_force,
                diagnostics["accepted_force_norm"],
                rel_tol=1e-5,
                abs_tol=max(1e-12, 1e-4 * force_limit),
            )
        return {
            "checkpoint_sha256": row["checkpoint"]["sha256"],
            "phase": "frame" if frame else row["phase"],
            "fraction": value,
            "alpha": alpha,
            "activation_sha256": tensor_digest(activation),
            "force_norm_recorded": diagnostics["accepted_force_norm"],
            "force_norm_fresh": measured_force,
            "fresh_contact": fresh_contact,
            "inverted_cells_independent": count,
            "minimum_J_independent": min_j,
            "scoped_contact_intersections_independent": intersects,
        }

    resume_receipt = None
    if "resume" in numerical:
        resume = numerical["resume"]
        check_record(resume["summary"])
        check_record(resume["checkpoint"])
        previous = json.loads(Path(resume["summary"]["path"]).read_text())
        assert previous["status"] == "blocked"
        assert previous["provenance_verified"] is True
        assert previous["accepted"]
        prior_last = previous["accepted"][-1]
        assert prior_last["checkpoint"]["sha256"] == resume["checkpoint"]["sha256"]
        assert prior_last["phase"] == resume["phase"]
        assert prior_last["fraction"] == resume["fraction"]
        assert numerical["accepted"]
        first = numerical["accepted"][0]
        assert first["phase"] == resume["phase"]
        assert first["fraction"] == resume["fraction"]
        assert first["diagnostics"] == prior_last["diagnostics"]
        with (
            np.load(resume["checkpoint"]["path"], allow_pickle=False) as old_state,
            np.load(first["checkpoint"]["path"], allow_pickle=False) as new_state,
        ):
            np.testing.assert_array_equal(new_state["u_full"], old_state["u_full"])
            np.testing.assert_array_equal(new_state["u"], old_state["u"])
            assert float(new_state["fraction"]) == float(old_state["fraction"])
            assert float(new_state["alpha"]) == float(old_state["alpha"])
            np.testing.assert_array_equal(new_state["pose"], old_state["pose"])
        resume_receipt = {
            "prior_summary_sha256": resume["summary"]["sha256"],
            "prior_checkpoint_sha256": resume["checkpoint"]["sha256"],
            "first_checkpoint_sha256": first["checkpoint"]["sha256"],
            "phase": resume["phase"],
            "fraction": resume["fraction"],
            "first_state_exactly_inherited": True,
        }
    accepted = [inspect(row, frame=False) for row in numerical["accepted"]]
    frames = [inspect(row, frame=True) for row in numerical["frames"]]
    if numerical["status"] == "completed":
        assert len(frames) == numerical["config"]["frames"]
        assert len(accepted) >= 2
        if resume_receipt is None:
            assert accepted[0]["phase"] == "neutral"
        assert accepted[-1]["fraction"] == 1.0
        assert frames[0]["fraction"] == 0.0
        assert frames[-1]["fraction"] == 1.0
    else:
        assert len(frames) < numerical["config"]["frames"]
    failed = None
    if (run / "failed-solver-state.npz").is_file():
        with np.load(run / "failed-solver-state.npz", allow_pickle=False) as z:
            full_u = z["u_full"].copy()
            np.testing.assert_array_equal(z["u"], full_u[: len(points)])
        assert full_u.shape == full_points.shape
        assert np.isfinite(full_u).all()
        count, min_j = tet_j(points, points + full_u[: len(points)], tets)
        contact_points = np.asfortranarray(
            ref.contact.vertices.numpy(force=True)
            + full_u[ref.contact.indices.numpy(force=True)]
        )
        failed = {
            "sha256": sha256(run / "failed-solver-state.npz"),
            "inverted_cells": count,
            "minimum_J": min_j,
            "scoped_contact_intersections": bool(
                ipctk.has_intersections(
                    ref.contact.collision_mesh, contact_points, broad
                )
            ),
            "accepted_equilibrium": False,
        }
    result = {
        "schema": "fixed-reference-contact-independent-audit-v1",
        "status": "verified_completed"
        if numerical["status"] == "completed"
        else "verified_incomplete",
        "numerical_status": numerical["status"],
        "numerical_summary_sha256": sha256(run / "summary.json"),
        "source_count": len(frozen),
        "input_count": len(numerical["inputs"]),
        "activation_endpoints_sha256": {
            "Smile": tensor_digest(smile),
            "MouthOpen": tensor_digest(mouth),
        },
        "activation_policy": "exact saved endpoint arrays, linear tensor blend; no activation optimization",
        "force_scope": "fresh GPU free-force residual at every saved accepted state and frame"
        if cfg.fresh_force
        else "recorded accepted free-force residuals only; independent GPU force recomputation not performed",
        "collision_scope": scope,
        "counts": {
            "accepted": len(accepted),
            "frames": len(frames),
            "expected_frames": numerical["config"]["frames"],
        },
        "accepted": accepted,
        "frames": frames,
        "failed_solver_state": failed,
        "resume": resume_receipt,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report = GROUP / "docs" / f"{cfg.output.name}.md"
    assert not report.exists(), report
    report.write_text(
        "# Independent fixed-reference contact audit\n\n"
        f"Numerical status: **{numerical['status']}**. Independently verified {len(accepted)} accepted checkpoints and {len(frames)} frames.\n\n"
        "For each saved state, this audit rechecks the saved full displacement, prescribed physical and appended rigid nodes, "
        "tetrahedron Jacobians, and scoped IPC surface intersections on CPU. It verifies every input and frozen source hash "
        "and confirms the Smile and MouthOpen activation arrays are byte-identical to the prepared source. "
        + (
            "Fresh GPU free-force residuals were recomputed for every saved state with the exact tensor blend, fixed boundary, and contact model.\n\n"
            if cfg.fresh_force
            else "Force thresholds are verified from the saved numerical receipts; a fresh GPU force evaluation is not part of this audit.\n\n"
        )
        + f"Result: `{result['status']}`. See `data/{cfg.output}/summary.json` for per-state hashes and measurements.\n"
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
