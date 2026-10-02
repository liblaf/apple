# Copyright (c) 2026 liblaf
"""Independently audit a matched Stage-3 MouthOpen-to-Smile run.

The numerical run always stores an extended visual geometry.  Only the
collision-on branch makes those appended rigid nodes part of its FEM/contact
state; the collision-off branch evaluates force on physical FEM nodes only.
"""

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
OLD_GROUP = ROOT / "exp/2026/09/30/mouthopen-smile-collisions"
sys.path.extend(
    (str(OLD_GROUP / "src"), str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
)

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
    output: Path = Path("30-collision-on-audit")
    run: Path = GROUP / "data/20-collision-on"
    source: Path = GROUP / "data/10-stage3"
    fresh_force: bool = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record_matches(row: dict) -> None:
    path = Path(row["path"])
    assert path.is_file(), path
    assert sha256(path) == row["sha256"], path


def tensor_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def tet_j(points: np.ndarray, moved: np.ndarray, tets: np.ndarray) -> tuple[int, float]:
    count, minimum = 0, math.inf
    for start in range(0, len(tets), 100_000):
        ids = tets[start : start + 100_000]
        rest = np.transpose(points[ids[:, 1:]] - points[ids[:, :1]], (0, 2, 1))
        current = np.transpose(moved[ids[:, 1:]] - moved[ids[:, :1]], (0, 2, 1))
        j = np.linalg.det(current) / np.linalg.det(rest)
        assert np.isfinite(j).all()
        count += int(np.count_nonzero(j <= 0))
        minimum = min(minimum, float(j.min()))
    return count, minimum


def cosine_beta(index: int, frames: int) -> float:
    return float((1 - np.cos(np.pi * index / (frames - 1))) / 2)


def main(cfg: Config) -> None:  # noqa: C901, PLR0912, PLR0915
    output = cherries.output(cfg.output / "summary.json", mkdir=True)
    assert not output.exists(), output
    numerical = json.loads((cfg.run / "summary.json").read_text())
    assert numerical["schema"] == "fixed-reference-activation-transition-v1"
    assert numerical["status"] in {"completed", "blocked", "interrupted", "failed"}
    numerical_finalization_interrupted = (
        numerical["status"] == "interrupted" and "provenance_verified" not in numerical
    )
    assert (
        numerical["provenance_verified"] is True or numerical_finalization_interrupted
    )
    assert isinstance(numerical["config"]["collision_enabled"], bool)
    collision_enabled = numerical["config"]["collision_enabled"]
    assert numerical["config"]["activation_stage"] == "rankone_fixed"
    assert Path(numerical["config"]["source"]).resolve() == cfg.source.resolve()
    for row in numerical["inputs"].values():
        record_matches(row)
    frozen = json.loads((cfg.run / "source-manifest.json").read_text())
    for row in frozen.values():
        record_matches(row)
        assert sha256(Path(row["source"])) == row["sha256"]
    frozen_names = {Path(row["source"]).name for row in frozen.values()}
    is_resume_variant = "21-resume-stage3-transition.py" in frozen_names
    assert ("20-run-stage3-transition.py" in frozen_names) != is_resume_variant
    assert any(
        name in frozen_names
        for name in ("20-run-stage3-transition.py", "21-resume-stage3-transition.py")
    )
    assert "fixed_reference_contact.py" in frozen_names
    if is_resume_variant:
        assert numerical["config"]["max_newton_steps"] == 80

    prepared = json.loads((cfg.source / "summary.json").read_text())
    assert prepared["status"] == "prepared"
    for row in prepared["inputs"].values():
        record_matches(row)
    for name in ("mesh.npz", "endpoints.npz"):
        assert sha256(cfg.source / name) == prepared["outputs"][name]["sha256"]
        assert (
            sha256(cfg.run / name)
            == numerical["inputs"]["mesh" if name == "mesh.npz" else "endpoints"][
                "sha256"
            ]
        )
    smile_stage3_path = Path(prepared["inputs"]["smile_stage3_checkpoint"]["path"])
    mouth_stage3_path = Path(prepared["inputs"]["mouthopen_stage3_checkpoint"]["path"])
    with (
        np.load(smile_stage3_path, allow_pickle=False) as smile_stage3,
        np.load(mouth_stage3_path, allow_pickle=False) as mouth_stage3,
    ):
        for checkpoint in (smile_stage3, mouth_stage3):
            assert str(checkpoint["mode"]) == "rankone_fixed"
            assert str(checkpoint["activation_model"]) == "strain"
        assert bool(smile_stage3["solver_valid"]) is False
        assert bool(mouth_stage3["solver_valid"]) is True
        stage3_smile_s = smile_stage3["S"].copy()
        stage3_mouth_s = mouth_stage3["S"].copy()
    with (
        np.load(cfg.source / "mesh.npz", allow_pickle=False) as source_mesh,
        np.load(cfg.source / "endpoints.npz", allow_pickle=False) as source_endpoints,
        np.load(cfg.run / "mesh.npz", allow_pickle=False) as mesh,
        np.load(cfg.run / "endpoints.npz", allow_pickle=False) as endpoints,
    ):
        for name in ("rest_points", "tets", "active_ids"):
            np.testing.assert_array_equal(mesh[name], source_mesh[name])
        for name in ("S_mouthopen", "S_smile", "pose_mouthopen", "pivot"):
            np.testing.assert_array_equal(endpoints[name], source_endpoints[name])
        points, tets, active_ids = (
            mesh["rest_points"].copy(),
            mesh["tets"].copy(),
            mesh["active_ids"].copy(),
        )
        mouth, smile = endpoints["S_mouthopen"].copy(), endpoints["S_smile"].copy()
        pose, pivot = endpoints["pose_mouthopen"].copy(), endpoints["pivot"].copy()
        source_rows = source_endpoints["smile_source_rows"].copy()
        point_map = source_endpoints["new_to_original_point"].copy()
        cell_map = source_endpoints["new_to_original_cell"].copy()
    with (
        np.load(
            Path(prepared["inputs"]["smile_checkpoint_mesh"]["path"]),
            allow_pickle=False,
        ) as smile_mesh,
        np.load(
            Path(prepared["inputs"]["mouthopen_checkpoint_mesh"]["path"]),
            allow_pickle=False,
        ) as mouth_mesh,
    ):
        np.testing.assert_array_equal(mouth_mesh["rest_points"], points)
        np.testing.assert_array_equal(mouth_mesh["tets"], tets)
        np.testing.assert_array_equal(mouth_mesh["active_ids"], active_ids)
        np.testing.assert_array_equal(point_map[tets], smile_mesh["tets"][cell_map])
        expected_rows = np.searchsorted(smile_mesh["active_ids"], cell_map[active_ids])
        np.testing.assert_array_equal(source_rows, expected_rows)
        np.testing.assert_array_equal(
            smile_mesh["active_ids"][source_rows], cell_map[active_ids]
        )
    np.testing.assert_array_equal(mouth, stage3_mouth_s)
    np.testing.assert_array_equal(smile, stage3_smile_s[source_rows])
    assert mouth.shape == smile.shape == (len(active_ids), 3, 3)

    volume = pv.read(Path(numerical["inputs"]["volume"]["path"]))
    np.testing.assert_array_equal(volume.points, points)
    np.testing.assert_array_equal(np.asarray(volume.cells).reshape(-1, 5)[:, 1:], tets)
    np.testing.assert_array_equal(
        np.flatnonzero(volume.cell_data["ActivationMask"]), active_ids
    )
    fixed = np.asarray(volume.point_data["IsFixed"], bool)
    groups = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).ravel()
    ]
    jaw = fixed & (np.asarray(volume.point_data["GroupId"]) == groups.index("Mandible"))
    contact_surface = numerical["contact_surface"]
    for name, row in contact_surface["source"].items():
        record_matches(row)
        assert numerical["inputs"][f"rigid_{name}"]["sha256"] == row["sha256"]
    reference = build_fixed_reference_contact(
        volume,
        stiffness_mpa=numerical["config"]["stiffness_mpa"],
        dhat_m=numerical["config"]["dhat_m"],
        minimum_distance_m=numerical["config"]["minimum_distance_m"],
        cranium_path=Path(contact_surface["source"]["cranium"]["path"]),
        mandible_path=Path(contact_surface["source"]["mandible"]["path"]),
        eyes_path=Path(contact_surface["source"]["eyes"]["path"]),
    )
    assert reference.receipt == contact_surface
    full_points = reference.full_reference_points(points)
    full_fixed = np.r_[fixed, np.ones(len(full_points) - len(points), bool)]
    full_jaw = reference.full_mandible_mask
    assert np.array_equal(full_jaw[: len(points)], jaw)
    force_limit = numerical["config"]["force_atol"]
    inversion_limit = numerical["config"]["inversion_fraction_limit"] * len(tets)

    def boundary(alpha: float) -> np.ndarray:
        result = np.zeros_like(full_points)
        scaled = alpha * pose
        result[full_jaw] = (
            (full_points[full_jaw] - pivot)
            @ Rotation.from_rotvec(scaled[:3]).as_matrix().T
            + pivot
            + scaled[3:]
            - full_points[full_jaw]
        )
        return result

    force_physics = force_model = force_contact = force_problem = None
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
        if collision_enabled:
            fresh_reference = build_fixed_reference_contact(
                force_physics.mesh,
                stiffness_mpa=numerical["config"]["stiffness_mpa"],
                dhat_m=numerical["config"]["dhat_m"],
                minimum_distance_m=numerical["config"]["minimum_distance_m"],
                cranium_path=Path(contact_surface["source"]["cranium"]["path"]),
                mandible_path=Path(contact_surface["source"]["mandible"]["path"]),
                eyes_path=Path(contact_surface["source"]["eyes"]["path"]),
            )
            assert fresh_reference.receipt == contact_surface
            force_model = attach_fixed_reference_contact(
                force_physics.forward.model,
                fresh_reference,
                torch.zeros_like(torch.as_tensor(full_points)),
            )
            force_contact = fresh_reference.contact
            np.testing.assert_array_equal(
                force_model.dof_map.fixed_indices.numpy(force=True),
                np.flatnonzero(np.repeat(full_fixed, 3)),
            )
        else:
            force_model = force_physics.forward.model
            np.testing.assert_array_equal(
                force_model.dof_map.fixed_indices.numpy(force=True),
                np.flatnonzero(np.repeat(fixed, 3)),
            )
        force_problem = ForwardProblem(model=force_model)

    def inspect(row: dict, *, frame: bool) -> dict:  # noqa: PLR0915
        record_matches(row["checkpoint"])
        with np.load(row["checkpoint"]["path"], allow_pickle=False) as saved:
            u, full_u = saved["u"].copy(), saved["u_full"].copy()
            alpha, beta = (
                float(saved["alpha"]),
                float(saved["beta"] if frame else saved["fraction"]),
            )
            np.testing.assert_allclose(saved["pose"], alpha * pose, rtol=0, atol=1e-12)
        assert u.shape == points.shape
        assert full_u.shape == full_points.shape
        assert np.isfinite(full_u).all()
        np.testing.assert_array_equal(full_u[: len(points)], u)
        np.testing.assert_allclose(
            full_u[full_fixed], boundary(alpha)[full_fixed], rtol=0, atol=1e-12
        )
        if frame:
            assert beta == row["beta"]
            assert alpha == row["alpha"]
        else:
            assert row["phase"] == "transition"
            assert beta == row["fraction"]
        np.testing.assert_allclose(alpha, 1 - beta, rtol=0, atol=2e-15)
        activation = (1 - beta) * mouth + beta * smile
        diagnostics = row["diagnostics"]
        assert diagnostics["solver_valid"] is True
        assert math.isfinite(diagnostics["accepted_force_norm"])
        assert diagnostics["accepted_force_norm"] <= force_limit
        count, minimum_j = tet_j(points, points + u, tets)
        assert count == diagnostics["inverted_cells"] <= inversion_limit
        assert math.isclose(
            minimum_j, diagnostics["minimum_J"], rel_tol=1e-7, abs_tol=1e-7
        )
        positions = np.asfortranarray(
            reference.contact.vertices.numpy(force=True)
            + full_u[reference.contact.indices.numpy(force=True)]
        )
        intersects = bool(
            ipctk.has_intersections(
                reference.contact.collision_mesh, positions, ipctk.LBVH()
            )
        )
        fresh_force = fresh_contact = None
        if collision_enabled:
            assert diagnostics["contact"]["contact_valid"] is True
            assert diagnostics["contact"]["scoped_boundary_no_intersections"] is True
            assert not intersects
        if cfg.fresh_force:
            assert force_physics is not None
            assert force_model is not None
            assert force_problem is not None
            with torch.no_grad():
                force_physics.materials["muscle"]["activation_inv"] = torch.zeros(
                    (force_physics.mesh.n_cells, 6)
                ).index_copy(
                    0,
                    force_physics.id_t,
                    strain_to_activation_inv(torch.as_tensor(activation)),
                )
                force_model.set_materials(force_physics.materials)
                values = boundary(alpha)
                fixed_values = (
                    torch.as_tensor(values)
                    .flatten()[force_model.dof_map.fixed_indices]
                    .clone()
                )
                force_model.dof_map.fixed_values = fixed_values
                if collision_enabled:
                    assert force_contact is not None
                    state = force_model.State(
                        u=torch.as_tensor(full_u),
                        collision=force_contact.state_at(torch.as_tensor(full_u)),
                    )
                else:
                    state = force_model.State(u=torch.as_tensor(u), collision=None)
                fresh_force = float(torch.linalg.vector_norm(force_problem.grad(state)))
                if collision_enabled:
                    fresh_contact = force_contact.diagnostics(state.collision, state.u)
            assert math.isfinite(fresh_force)
            assert fresh_force <= force_limit
            assert math.isclose(
                fresh_force,
                diagnostics["accepted_force_norm"],
                rel_tol=1e-5,
                abs_tol=max(1e-12, 1e-4 * force_limit),
            )
            if collision_enabled:
                assert fresh_contact is not None
                assert fresh_contact["contact_numerically_valid"] is True
                assert (
                    fresh_contact["active_contact_count"]
                    == diagnostics["contact"]["active_contact_count"]
                )
        return {
            "checkpoint_sha256": row["checkpoint"]["sha256"],
            "phase": "frame" if frame else row["phase"],
            "beta": beta,
            "alpha": alpha,
            "activation_sha256": tensor_digest(activation),
            "force_norm_recorded": diagnostics["accepted_force_norm"],
            "force_norm_fresh": fresh_force,
            "fresh_contact": fresh_contact,
            "inverted_cells_independent": count,
            "minimum_J_independent": minimum_j,
            "scoped_contact_intersections_independent": intersects,
        }

    resume_receipt = None
    assert not is_resume_variant or "resume" in numerical
    if "resume" in numerical:
        resume = numerical["resume"]
        record_matches(resume["summary"])
        record_matches(resume["checkpoint"])
        previous = json.loads(Path(resume["summary"]["path"]).read_text())
        assert previous["status"] == "blocked"
        assert previous["provenance_verified"] is True
        search_budget_change = resume.get("search_budget_change")
        if is_resume_variant:
            assert previous["failure"]["type"] == "TimeoutError"
            assert search_budget_change is not None
            assert search_budget_change["field"] == "max_newton_steps"
            assert (
                search_budget_change["prior"] == previous["config"]["max_newton_steps"]
            )
            assert (
                search_budget_change["current"]
                == numerical["config"]["max_newton_steps"]
                == 80
            )
            assert (search_budget_change["prior"], search_budget_change["current"]) in {
                (5000, 80),
                (80, 80),
            }
            assert search_budget_change["allowed_transitions"] == [[5000, 80], [80, 80]]
            assert search_budget_change["acceptance_gates_unchanged"] is True
        else:
            assert search_budget_change is None
        for key, value in numerical["config"].items():
            if key == "max_newton_steps" and is_resume_variant:
                continue
            if key not in {"output", "resume", "wall_seconds"}:
                assert previous["config"][key] == value, key
        for key, item in numerical["inputs"].items():
            if key != "resume_summary":
                assert previous["inputs"][key]["sha256"] == item["sha256"], key
        last = previous["accepted"][-1]
        assert last["checkpoint"]["sha256"] == resume["checkpoint"]["sha256"]
        first = numerical["accepted"][0]
        assert first["phase"] == last["phase"] == resume["phase"]
        assert first["fraction"] == last["fraction"] == resume["fraction"]
        assert first["diagnostics"] == last["diagnostics"]
        with (
            np.load(resume["checkpoint"]["path"], allow_pickle=False) as old_state,
            np.load(first["checkpoint"]["path"], allow_pickle=False) as new_state,
        ):
            np.testing.assert_array_equal(new_state["u"], old_state["u"])
            np.testing.assert_array_equal(new_state["u_full"], old_state["u_full"])
            np.testing.assert_array_equal(new_state["pose"], old_state["pose"])
        inherited_frames = 0
        if is_resume_variant:
            resumed_frames = {row["index"]: row for row in numerical["frames"]}
            for old_frame in previous["frames"]:
                if old_frame["beta"] < resume["fraction"] - 1e-15:
                    copied_frame = resumed_frames[old_frame["index"]]
                    assert (
                        copied_frame["checkpoint"]["sha256"]
                        == old_frame["checkpoint"]["sha256"]
                    )
                    inherited_frames += 1
        resume_receipt = {
            "prior_summary_sha256": resume["summary"]["sha256"],
            "prior_checkpoint_sha256": resume["checkpoint"]["sha256"],
            "first_checkpoint_sha256": first["checkpoint"]["sha256"],
            "first_state_exactly_inherited": True,
            "byte_exact_inherited_frames": inherited_frames
            if is_resume_variant
            else None,
            "search_budget_change": search_budget_change,
        }

    accepted = [inspect(row, frame=False) for row in numerical["accepted"]]
    assert accepted
    assert accepted[0]["phase"] == "transition"
    if resume_receipt is None:
        assert accepted[0]["beta"] == 0
        assert accepted[0]["alpha"] == 1
    frames = [inspect(row, frame=True) for row in numerical["frames"]]
    for index, row in enumerate(frames):
        assert row["beta"] == cosine_beta(index, numerical["config"]["frames"])
    if numerical["status"] == "completed":
        assert len(frames) == numerical["config"]["frames"]
        assert frames[0]["beta"] == 0
        assert frames[-1]["beta"] == 1
    else:
        assert len(frames) < numerical["config"]["frames"]
    result = {
        "schema": "stage3-matched-transition-independent-audit-v1",
        "status": "verified_completed"
        if numerical["status"] == "completed"
        else "verified_incomplete",
        "numerical_status": numerical["status"],
        "numerical_summary_sha256": sha256(cfg.run / "summary.json"),
        "source_summary": {
            "path": str((cfg.source / "summary.json").resolve()),
            "sha256": sha256(cfg.source / "summary.json"),
        },
        "provenance_verified": True,
        "numerical_finalization": {
            "interrupted": numerical_finalization_interrupted,
            "numerical_terminal_certification": "absent_due_to_interrupt"
            if numerical_finalization_interrupted
            else "present",
            "independent_provenance_verified": True,
        },
        "runner_source": "21-resume-stage3-transition.py"
        if is_resume_variant
        else "20-run-stage3-transition.py",
        "collision_enabled": collision_enabled,
        "activation_stage": "rankone_fixed",
        "stage3_checkpoints_sha256": {
            "Smile": sha256(smile_stage3_path),
            "MouthOpen": sha256(mouth_stage3_path),
        },
        "historical_solver_valid": {"Smile": False, "MouthOpen": True},
        "activation_endpoints_sha256": {
            "MouthOpen": tensor_digest(mouth),
            "Smile": tensor_digest(smile),
        },
        "force_scope": "fresh GPU free-force residuals"
        if cfg.fresh_force
        else "recorded force receipts",
        "contact_scope": numerical["collision_scope"],
        "counts": {
            "accepted": len(accepted),
            "frames": len(frames),
            "expected_frames": numerical["config"]["frames"],
        },
        "accepted": accepted,
        "frames": frames,
        "resume": resume_receipt,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
