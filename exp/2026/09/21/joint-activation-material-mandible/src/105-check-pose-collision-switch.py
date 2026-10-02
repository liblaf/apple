# ruff: noqa: ANN001
"""CPU checks for switching from collision-free pose fitting to joint contact."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json

from liblaf import cherries


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/pose-collision-switch-check-001"


def load_script(filename: str, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).with_name(filename)
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(runner, checks, directory, *, feasible=True, fail=False, target=-0.2):
    from types import SimpleNamespace

    fitter, calls = checks.fixture(runner, directory, target=target)
    fitter.cfg.pose_collision = False
    fitter.contact_enabled = True
    fitter.rigid_contact = object()
    fitter.runtime.forward = SimpleNamespace(
        model=SimpleNamespace(collision=fitter.rigid_contact),
        state=SimpleNamespace(collision=object()),
    )
    fitter.physics = SimpleNamespace(runtime=fitter.runtime)
    fitter.pose_contact_audit = lambda _state: {
        "geometric_seed_feasible": feasible,
        "soft_rigid_intersections": not feasible,
    }
    path = directory / "expressions/A/latest.pt"
    state = checks.saved(directory)
    state["metrics"].update(initial_fit_rms_mm=1.0, target_motion_rms_mm=1.0)
    state.update(optimizer={"stale": True}, history=[999.0], qualifying_consecutive=4)
    runner.atomic_torch(path, state)
    original_evaluate = fitter.evaluate
    records = []

    def evaluate(index, q, jaw, seed, seed_jaw):
        assert fitter.physics.runtime is fitter.runtime
        assert fitter.runtime.forward.model.collision is (
            fitter.rigid_contact if fitter.contact_enabled else None
        )
        records.append(
            {
                "enabled": fitter.contact_enabled,
                "seed": seed.clone(),
                "jaw": jaw.detach().clone(),
                "seed_jaw": seed_jaw.clone(),
                "activation": q.detach().clone(),
                "warm_adjoint_empty": not fitter.runtime.warm_adjoints,
            }
        )
        if fail and fitter.contact_enabled:
            message = "synthetic contact solve failure"
            raise runner.ForwardConvergenceError(message, receipt={"success": False})
        candidate = original_evaluate(index, q, jaw, seed, seed_jaw)
        candidate["metrics"].update(initial_fit_rms_mm=1.0, target_motion_rms_mm=1.0)
        if fitter.contact_enabled:
            candidate["displacement_m"] += 0.1
            candidate["gradient_q"] += 2.0
            candidate["gradient_jaw"] += 3.0
        return candidate

    fitter.evaluate = evaluate
    return fitter, records, calls


def install_stubs(runner, stack):
    from types import SimpleNamespace
    from unittest.mock import patch

    import joint_collision_off_pose

    switches = []

    def install(physics, enabled):
        old = physics.runtime
        assert old.forward.model.collision is not None
        assert not old.warm_adjoints
        old.forward.model.collision = old.forward.model.collision if enabled else None
        assert old.forward.state.collision is None
        physics.runtime = SimpleNamespace(forward=old.forward, warm_adjoints={})
        switches.append(enabled)
        return physics.runtime

    stack.enter_context(
        patch.object(
            runner,
            "install_expression_runtime",
            lambda physics: install(physics, enabled=True),
        )
    )
    stack.enter_context(
        patch.object(
            joint_collision_off_pose,
            "install_collision_off_pose_runtime",
            lambda physics: install(physics, enabled=False),
        )
    )
    return switches


def handoff_case(runner, checks, directory, *, feasible=True, fail=False):
    import json

    fitter, records, _ = fixture(
        runner, checks, directory, feasible=feasible, fail=fail
    )
    before = checks.saved(directory)
    path = directory / "expressions/A/latest.pt"
    digest = sha256(path)
    fitter.fit_step(0)
    after = checks.saved(directory)
    receipt = json.loads((path.parent / "contact-handoff.json").read_text())
    if not feasible or fail:
        assert sha256(path) == digest
        assert (
            fitter.status["expressions"]["A"]["status"] == "requires_contact_recovery"
        )
        assert not receipt["joint_started"]
        assert not (path.parent / "joint-initial.pt").exists()
        assert after["fit_stage"] == "pose_only"
        assert len(records) == int(fail)
    else:
        assert len(records) == 1
        record = records[0]
        assert record["enabled"]
        assert record["warm_adjoint_empty"]
        for key in ("jaw", "seed_jaw"):
            torch.testing.assert_close(
                record[key], before["jaw_normalized"], rtol=0, atol=0
            )
        torch.testing.assert_close(
            record["seed"], before["displacement_m"], rtol=0, atol=0
        )
        assert torch.count_nonzero(record["activation"]) == 0
        assert after["fit_stage"] == "joint"
        assert receipt["joint_started"]
        assert receipt["full_gradients_recomputed"]
        assert after["accepted_steps"] == before["accepted_steps"]
        assert not after["inverse_converged"]
        for key, delta in (
            ("gradient_q", 2.0),
            ("gradient_jaw", 3.0),
            ("displacement_m", 0.1),
        ):
            torch.testing.assert_close(
                after[key], before[key] + delta, atol=1e-14, rtol=0
            )
        assert "optimizer" not in after
        assert after["history"] == []
        assert after["qualifying_consecutive"] == 0
        final = torch.load(path.parent / "pose-only-final.pt", weights_only=False)
        checks.physical_equal(before, final)
    return {
        "feasible": feasible,
        "contact_forward_failure": fail,
        "fresh_contact_evaluations": len(records),
        "joint_started": receipt["joint_started"],
        "accepted_checkpoint_retained_on_failure": not feasible or fail,
        "same_parameters_fresh_contact_state_and_gradients": feasible and not fail,
    }


def trial_case(runner, checks, directory):
    fitter, records, _ = fixture(runner, checks, directory, target=0.12)
    fitter.fit_step(0)
    assert len(records) == 1
    assert not records[0]["enabled"]
    assert torch.count_nonzero(records[0]["activation"]) == 0
    assert checks.saved(directory)["fit_stage"] == "pose_only"
    assert checks.saved(directory)["accepted_steps"] == 1
    return {
        "actual_pose_trial_uses_collision_off_model": True,
        "activation_exactly_zero": True,
    }


def audit_case(runner, directory):
    from types import SimpleNamespace
    from unittest.mock import patch

    directory.mkdir()
    fitter = runner.Fitter.__new__(runner.Fitter)
    fitter.hinge_axis = torch.tensor([1.0, 0.0, 0.0])
    calls = []
    full = torch.arange(15, dtype=torch.float64).reshape(5, 3) * 1e-4
    fem = full[:2].clone()
    jaw = torch.tensor([0.25])

    def extend(seed, pose):
        torch.testing.assert_close(seed, fem, atol=0, rtol=0)
        torch.testing.assert_close(
            pose, runner.hinge_pose(jaw, fitter.hinge_axis), atol=0, rtol=0
        )
        calls.append("extend")
        return full

    class Collisions:
        def __len__(self) -> int:
            return 1

        def compute_minimum_distance(self, _mesh, _positions):
            return self.distance**2

    collisions = Collisions()
    collision = SimpleNamespace(
        vertices=torch.zeros((3, 3)),
        indices=torch.tensor([0, 3, 4]),
        collision_mesh=object(),
        min_distance=1e-8,
    )

    def state_at(value):
        torch.testing.assert_close(value, full, atol=0, rtol=0)
        calls.append("contact_state")
        return SimpleNamespace(collisions=collisions)

    collision.state_at = state_at
    fitter.rigid_contact = collision
    fitter.physics = SimpleNamespace(full_skull=SimpleNamespace(extend_seed=extend))
    state = {"displacement_m": fem, "jaw_normalized": jaw}
    reports = {}
    for name, intersects, gap, expected in (
        ("intersection", True, 2e-8, False),
        ("sub_buffer", False, 5e-9, False),
        ("feasible", False, 2e-8, True),
    ):
        collisions.distance = gap

        def intersection_check(mesh, positions, _broad_phase, intersects=intersects):
            torch.testing.assert_close(
                torch.as_tensor(positions), full[collision.indices], atol=0, rtol=0
            )
            assert mesh is collision.collision_mesh
            return intersects

        with patch.object(runner.ipctk, "has_intersections", intersection_check):
            report = fitter.pose_contact_audit(state)
        assert report["geometric_seed_feasible"] is expected
        reports[name] = report
    assert calls.count("extend") == 3
    assert calls.count("contact_state") == 2
    return {
        "full_geometry_extended_at_current_jaw": True,
        "appended_skull_and_eye_indices_audited": True,
        "cases": reports,
    }


def main(cfg: Config) -> None:
    from contextlib import ExitStack

    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    runner = load_script("93-fit-expressions.py", "collision_switch_fitter")
    checks = load_script("104-check-pose-first.py", "pose_fixture")
    paths = [
        Path(runner.__file__),
        Path(checks.__file__),
        Path(__file__),
        Path(__file__).with_name("joint_collision_off_pose.py"),
    ]
    report = {
        "success": False,
        "cpu_only": True,
        "scope": "actual fitter switching/handoff; mocked equilibrium and installer construction",
        "implementation_sha256": {str(path.resolve()): sha256(path) for path in paths},
    }
    write_json(cfg.output_dir / "summary.json", report)
    with ExitStack() as stack:
        switches = install_stubs(runner, stack)
        report["pose_trial"] = trial_case(runner, checks, cfg.output_dir / "pose-trial")
        report["infeasible_handoff"] = handoff_case(
            runner, checks, cfg.output_dir / "infeasible", feasible=False
        )
        report["contact_forward_failure"] = handoff_case(
            runner, checks, cfg.output_dir / "forward-failure", fail=True
        )
        report["feasible_handoff"] = handoff_case(
            runner, checks, cfg.output_dir / "feasible"
        )
        report["runtime_switch_sequence"] = switches
    report["geometric_audit"] = audit_case(runner, cfg.output_dir / "audit")
    for path, digest in report["implementation_sha256"].items():
        assert sha256(Path(path)) == digest
    report["success"] = True
    write_json(cfg.output_dir / "summary.json", report)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
