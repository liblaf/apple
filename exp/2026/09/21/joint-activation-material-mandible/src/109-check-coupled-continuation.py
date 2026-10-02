"""CPU checks of adaptive coupled seed continuation driver contracts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import joint_coupled_continuation as driver
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_equilibrium import ForwardConvergenceError

from liblaf import cherries


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/coupled-continuation-check-001"


def inputs() -> dict:
    return {
        "physics": object(),
        "old_stress": torch.eye(3).unsqueeze(0).requires_grad_(),
        "new_stress": (3 * torch.eye(3)).unsqueeze(0).requires_grad_(),
        "old_pose": torch.zeros(6, requires_grad=True),
        "new_pose": torch.tensor([0.02, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True),
        "seed": torch.zeros((2, 3), requires_grad=True),
        "axis": torch.tensor([1.0, 0.0, 0.0]),
        "pivot": torch.zeros(3),
        "linear_rtol": 1e-7,
    }


def snapshot(kwargs: dict) -> dict:
    return {
        key: value.detach().clone()
        for key, value in kwargs.items()
        if isinstance(value, torch.Tensor)
    }


def unchanged(kwargs: dict, original: dict) -> None:
    for key, value in original.items():
        torch.testing.assert_close(kwargs[key], value, atol=0, rtol=0)
        assert kwargs[key].grad is None


def adaptive_case() -> dict:
    kwargs = inputs()
    original = snapshot(kwargs)
    calls = []

    def predictor(_physics: object, **values: object) -> tuple[torch.Tensor, dict]:
        calls.append(snapshot(values))
        call = len(calls)
        if call in {1, 3}:
            receipt = (
                {"geometry": {"ccd_fraction": 0.5}}
                if call == 1
                else {"linear_result": "failed", "relative_residual": 0.1}
            )
            message = (
                "synthetic CCD rejection" if call == 1 else "synthetic linear failure"
            )
            raise ForwardConvergenceError(message, receipt=receipt)
        expected = {
            2: (0.0, 0.4, 0.4, 0.0),
            4: (0.4, 0.7, 0.5, 10.0),
            5: (0.7, 1.0, 1.0, 20.0),
        }[call]
        before, after, scale, seed_value = expected
        torch.testing.assert_close(
            values["old_pose"],
            torch.lerp(original["old_pose"], original["new_pose"], before),
            atol=1e-16,
            rtol=0,
        )
        torch.testing.assert_close(
            values["new_pose"],
            torch.lerp(original["old_pose"], original["new_pose"], after),
            atol=1e-16,
            rtol=0,
        )
        torch.testing.assert_close(
            values["old_stress"],
            torch.lerp(original["old_stress"], original["new_stress"], before),
            atol=1e-15,
            rtol=0,
        )
        torch.testing.assert_close(
            values["new_stress"],
            torch.lerp(original["old_stress"], original["new_stress"], after),
            atol=1e-15,
            rtol=0,
        )
        assert abs(values["residual_scale"] - scale) < 1e-15
        torch.testing.assert_close(
            values["seed"], torch.full((2, 3), seed_value), atol=0, rtol=0
        )
        assert not values["seed"].requires_grad
        return values["seed"] + 10, {
            "success": True,
            "geometry": {"admitted": True},
            "old_free_force_norm": 1e3,
        }

    with patch.object(driver, "predict_expression_seed", predictor):
        result, receipt = driver.prepare_expression_seed(
            **kwargs, equilibrate_substeps=False
        )
    assert receipt["success"]
    assert receipt["progress"] == 1.0
    assert receipt["accepted_substeps"] == 3
    assert len(calls) == 5
    assert [row["admitted"] for row in receipt["attempts"]] == [
        False,
        True,
        False,
        True,
        True,
    ]
    assert not receipt["intermediate_equilibrium_required"]
    assert receipt["final_strict_equilibrium_required"]
    torch.testing.assert_close(
        calls[-1]["new_pose"], original["new_pose"], atol=0, rtol=0
    )
    torch.testing.assert_close(
        calls[-1]["new_stress"], original["new_stress"], atol=0, rtol=0
    )
    torch.testing.assert_close(result, torch.full((2, 3), 30.0), atol=0, rtol=0)
    assert not result.requires_grad
    result.fill_(777)
    unchanged(kwargs, original)
    return {
        "progress_proposals": [row["progress_proposed"] for row in receipt["attempts"]],
        "residual_scales": [row["remaining_fraction"] for row in receipt["attempts"]],
        "accepted_substeps": 3,
        "recomputed_from_current_nonequilibrium_state": True,
        "exact_target_reached": True,
        "inputs_unmodified": True,
        "linear_failure_retried": True,
    }


def budget_case(*, accepted_limit: bool) -> dict:
    kwargs = inputs()
    original = snapshot(kwargs)
    calls = 0

    def predictor(_physics: object, **values: object) -> tuple[torch.Tensor, dict]:
        nonlocal calls
        calls += 1
        if accepted_limit and calls == 2:
            return values["seed"] + 1, {"success": True, "geometry": {"admitted": True}}
        message = "synthetic rejected substep"
        raise ForwardConvergenceError(
            message, receipt={"geometry": {"ccd_fraction": 0.5}}
        )

    with patch.object(driver, "predict_expression_seed", predictor):
        try:
            driver.prepare_expression_seed(
                **kwargs, max_steps=1, max_attempts=2, equilibrate_substeps=False
            )
        except ForwardConvergenceError as error:
            receipt = error.receipt
        else:
            message = (
                "Budget exhaustion incorrectly returned a partial replacement seed"
            )
            raise AssertionError(message)
    assert not receipt["success"]
    assert receipt["progress"] == (0.4 if accepted_limit else 0.0)
    assert receipt["accepted_substeps"] == int(accepted_limit)
    assert calls == 2
    unchanged(kwargs, original)
    return {
        "accepted_limit": accepted_limit,
        "progress": receipt["progress"],
        "attempts": calls,
        "raised_without_partial_return": True,
        "inputs_unmodified": True,
    }


def contract_case() -> dict:
    propagated = []
    for kind in (AssertionError, ValueError, RuntimeError):
        kwargs = inputs()
        original = snapshot(kwargs)
        marker = kind("synthetic unexpected contract failure")
        with patch.object(
            driver, "predict_expression_seed", side_effect=marker
        ) as predictor:
            try:
                driver.prepare_expression_seed(**kwargs, equilibrate_substeps=False)
            except kind as error:
                assert error is marker  # noqa: PT017 - verify unchanged exception identity
            else:
                message = "Unexpected contract failure was swallowed"
                raise AssertionError(message)
        assert predictor.call_count == 1
        unchanged(kwargs, original)
        propagated.append(kind.__name__)
    return {"unexpected_errors_propagate_immediately": propagated}


def strict_case(*, fail_first: bool = False, exhaust: bool = False) -> dict:  # noqa: PLR0915
    from types import SimpleNamespace

    kwargs = inputs()
    original = snapshot(kwargs)
    predictions = []
    corrections = []
    physics = SimpleNamespace(runtime=SimpleNamespace(last_forward={}))
    kwargs["physics"] = physics

    def predictor(_physics: object, **values: object) -> tuple[torch.Tensor, dict]:
        predictions.append(snapshot(values))
        call = len(predictions)
        if call == 1:
            message = "synthetic full-step CCD rejection"
            raise ForwardConvergenceError(
                message, receipt={"geometry": {"ccd_fraction": 0.5}}
            )
        before = (
            0.0
            if call == 2 or (fail_first and call == 3)
            else (0.2 if fail_first else 0.4)
        )
        expected_seed = 0.0 if before == 0 else 110.0
        torch.testing.assert_close(
            values["seed"], torch.full((2, 3), expected_seed), atol=0, rtol=0
        )
        for suffix, name in (("pose", "pose"), ("stress", "stress")):
            expected = torch.lerp(
                original[f"old_{name}"], original[f"new_{name}"], before
            )
            torch.testing.assert_close(
                values[f"old_{suffix}"], expected, atol=1e-15, rtol=0
            )
        return values["seed"] + 10, {"success": True, "geometry": {"admitted": True}}

    def solve(**values: object) -> torch.Tensor:
        corrections.append(snapshot(values))
        expected_progress = 0.2 if fail_first and len(corrections) == 2 else 0.4
        assert not torch.is_grad_enabled()
        assert values["key"] == "coupled-internal-corrector"
        assert float(values["skin_multiplier"]) == 1.0
        for key, expected in (
            (
                "pose",
                torch.lerp(
                    original["old_pose"], original["new_pose"], expected_progress
                ),
            ),
            (
                "active_stress",
                torch.lerp(
                    original["old_stress"], original["new_stress"], expected_progress
                ),
            ),
        ):
            torch.testing.assert_close(values[key], expected, atol=1e-15, rtol=0)
            assert not values[key].requires_grad
        torch.testing.assert_close(values["pose"], values["seed_pose"], atol=0, rtol=0)
        torch.testing.assert_close(
            values["seed"], torch.full((2, 3), 10.0), atol=0, rtol=0
        )
        if fail_first and len(corrections) == 1:
            values["seed"].fill_(999)
            physics.runtime.last_forward = {"success": False, "sentinel": 999}
            message = "synthetic strict corrector failure"
            raise ForwardConvergenceError(message, receipt=physics.runtime.last_forward)
        physics.runtime.last_forward = {"success": True, "sentinel": 7}
        return values["seed"] + 100

    physics.solve = solve
    with patch.object(driver, "predict_expression_seed", predictor):
        if exhaust:
            try:
                driver.prepare_expression_seed(**kwargs, max_steps=1, max_attempts=2)
            except ForwardConvergenceError as error:
                receipt = error.receipt
            else:
                message = "Failed internal corrector incorrectly returned a seed"
                raise AssertionError(message)
            assert not receipt["success"]
            assert receipt["progress"] == 0.0
            assert receipt["accepted_substeps"] == 0
            assert len(corrections) == 1
        else:
            result, receipt = driver.prepare_expression_seed(**kwargs)
            assert receipt["success"]
            assert receipt["progress"] == 1.0
            assert receipt["accepted_substeps"] == 2
            assert len(corrections) == (2 if fail_first else 1)
            torch.testing.assert_close(
                result, torch.full((2, 3), 120.0), atol=0, rtol=0
            )
            torch.testing.assert_close(
                predictions[-1]["new_pose"], original["new_pose"], atol=0, rtol=0
            )
            physics.runtime.last_forward["sentinel"] = 999
            admitted = [row for row in receipt["attempts"] if row["admitted"]]
            assert admitted[0]["details"]["internal_corrector"]["sentinel"] == 7
            assert "internal_corrector" not in admitted[-1]["details"]
    assert receipt["intermediate_equilibrium_required"]
    unchanged(kwargs, original)
    return {
        "fail_first_corrector": fail_first,
        "budget_exhausted": exhaust,
        "progress": receipt["progress"],
        "accepted_substeps": receipt["accepted_substeps"],
        "predictor_calls": len(predictions),
        "corrector_calls": len(corrections),
        "corrected_seed_used_by_next_predictor": not exhaust,
        "no_final_correction_inside_driver": not exhaust,
        "failed_corrector_does_not_advance_state": fail_first,
        "no_intermediate_autograd": True,
        "inputs_unmodified": True,
    }


def main(cfg: Config) -> None:
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    paths = [
        Path(__file__),
        Path(driver.__file__),
        Path(__file__).with_name("joint_coupled_seed.py"),
        Path(__file__).with_name("joint_coupled_predictor.py"),
    ]
    report = {
        "success": False,
        "cpu_only": True,
        "scope": "actual continuation driver with mocked predictor; no physical equilibrium claim",
        "implementation_sha256": {str(path.resolve()): sha256(path) for path in paths},
    }
    write_json(cfg.output_dir / "summary.json", report)
    report["adaptive_recompute"] = adaptive_case()
    report["accepted_substep_budget"] = budget_case(accepted_limit=True)
    report["attempt_budget"] = budget_case(accepted_limit=False)
    report["contract_failures"] = contract_case()
    report["strict_default"] = strict_case()
    report["strict_corrector_retry"] = strict_case(fail_first=True)
    report["strict_corrector_budget_failure"] = strict_case(
        fail_first=True, exhaust=True
    )
    for path, digest in report["implementation_sha256"].items():
        assert sha256(Path(path)) == digest
    report["success"] = True
    write_json(cfg.output_dir / "summary.json", report)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
