"""CPU regression for projected-Adam owned-state rollback and retry."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch


def assert_tree_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right), (left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            assert_tree_equal(left_item, right_item)
    else:
        assert left == right, (left, right)


def load_joint_pilot() -> Any:
    source = Path(__file__).with_name("30-joint-pilot.py")
    spec = importlib.util.spec_from_file_location("joint_pilot_transaction", source)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_retry(transaction: Any) -> dict[str, Any]:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    parameter.grad = torch.tensor([2.0], dtype=torch.float64)
    optimizer = torch.optim.Adam([parameter], lr=0.1, eps=1.0e-8)
    auxiliary = {
        "seed": 7,
        "warm": 11,
        "material": 13,
        "fixed": 17,
        "u": 19,
        "contact": "accepted",
        "last_receipt": "accepted",
        "forward": 0,
        "adjoint": 0,
    }
    attempts = {"count": 0}

    def restore() -> None:
        auxiliary["seed"] = 7
        auxiliary["warm"] = 11
        auxiliary["material"] = 13
        auxiliary["fixed"] = 17
        auxiliary["u"] = 19
        auxiliary["contact"] = "accepted"
        auxiliary["last_receipt"] = "accepted"

    def evaluate() -> dict[str, float]:
        attempts["count"] += 1
        auxiliary["forward"] += 1
        auxiliary["adjoint"] += 1
        auxiliary["seed"] = 100 + attempts["count"]
        auxiliary["warm"] = 200 + attempts["count"]
        auxiliary["material"] = 300 + attempts["count"]
        auxiliary["fixed"] = 400 + attempts["count"]
        auxiliary["u"] = 500 + attempts["count"]
        auxiliary["contact"] = f"rebuilt-{attempts['count']}"
        auxiliary["last_receipt"] = f"trial-{attempts['count']}"
        if attempts["count"] == 1:
            message = "synthetic CCD rejection"
            raise AssertionError(message)
        parameter.grad = 2 * parameter.detach().clone()
        return {"objective": float(parameter.detach().square())}

    def project() -> dict[str, float]:
        with torch.no_grad():
            parameter.clamp_(0.0, 2.0)
        return {"parameter": float(parameter.detach())}

    receipt = transaction(
        optimizer=optimizer,
        parameters=(parameter,),
        prepare_step=lambda: None,
        project=project,
        restore_auxiliary=restore,
        evaluate_trial=evaluate,
        cost_counters=lambda: (auxiliary["forward"], auxiliary["adjoint"]),
        accepted_objective=1.0,
        backtrack_factor=0.5,
        maximum_trials=3,
        armijo_coefficient=1.0e-4,
    )
    assert receipt["accepted"] is True
    assert [row["fraction"] for row in receipt["trials"]] == [1.0, 0.5]
    assert receipt["trials"][0]["error"] == "synthetic CCD rejection"
    assert receipt["trials"][0]["forward_solves"] == 1
    assert receipt["trials"][1]["adjoint_solves"] == 1
    assert torch.allclose(parameter, torch.tensor([0.95], dtype=torch.float64))
    assert int(optimizer.state[parameter]["step"]) == 1
    assert auxiliary["seed"] == 102
    assert auxiliary["warm"] == 202
    assert auxiliary["contact"] == "rebuilt-2"
    assert auxiliary["last_receipt"] == "trial-2"
    return {
        "receipt": receipt,
        "accepted_parameter": float(parameter.detach()),
        "optimizer_step": int(optimizer.state[parameter]["step"]),
        "accepted_auxiliary": auxiliary,
    }


def run_exhaustion(transaction: Any) -> dict[str, Any]:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    parameter.grad = torch.tensor([2.0], dtype=torch.float64)
    optimizer = torch.optim.Adam([parameter], lr=0.1, eps=1.0e-8)
    auxiliary = {
        "seed": 7,
        "warm": 11,
        "material": 13,
        "fixed": 17,
        "u": 19,
        "contact": "accepted",
        "last_receipt": "accepted",
        "forward": 0,
        "adjoint": 0,
    }

    def restore() -> None:
        auxiliary["seed"] = 7
        auxiliary["warm"] = 11
        auxiliary["material"] = 13
        auxiliary["fixed"] = 17
        auxiliary["u"] = 19
        auxiliary["contact"] = "accepted"
        auxiliary["last_receipt"] = "accepted"

    def reject() -> dict[str, float]:
        auxiliary["forward"] += 1
        auxiliary["adjoint"] += 1
        auxiliary["seed"] = -1
        auxiliary["warm"] = -2
        auxiliary["material"] = -3
        auxiliary["fixed"] = -4
        auxiliary["u"] = -5
        auxiliary["contact"] = "stale-rejected"
        auxiliary["last_receipt"] = "rejected"
        parameter.grad = torch.tensor([-999.0], dtype=torch.float64)
        message = "synthetic persistent contact rejection"
        raise AssertionError(message)

    def project() -> dict[str, float]:
        with torch.no_grad():
            parameter.clamp_(0.0, 2.0)
        return {"parameter": float(parameter.detach())}

    receipt = transaction(
        optimizer=optimizer,
        parameters=(parameter,),
        prepare_step=lambda: None,
        project=project,
        restore_auxiliary=restore,
        evaluate_trial=reject,
        cost_counters=lambda: (auxiliary["forward"], auxiliary["adjoint"]),
        accepted_objective=1.0,
        backtrack_factor=0.5,
        maximum_trials=2,
        armijo_coefficient=1.0e-4,
    )
    assert receipt["accepted"] is False
    assert torch.equal(parameter, torch.tensor([1.0], dtype=torch.float64))
    assert torch.equal(parameter.grad, torch.tensor([2.0], dtype=torch.float64))
    assert optimizer.state_dict()["state"] == {}
    assert auxiliary["seed"] == 7
    assert auxiliary["warm"] == 11
    assert auxiliary["material"] == 13
    assert auxiliary["fixed"] == 17
    assert auxiliary["u"] == 19
    assert auxiliary["contact"] == "accepted"
    assert auxiliary["last_receipt"] == "accepted"
    assert auxiliary["forward"] == 2
    assert auxiliary["adjoint"] == 2
    return {
        "receipt": receipt,
        "restored_parameter": float(parameter.detach()),
        "restored_gradient": float(parameter.grad.detach()),
        "restored_optimizer_state_entries": len(optimizer.state_dict()["state"]),
        "restored_auxiliary": auxiliary,
    }


def run_programmer_error(transaction: Any) -> dict[str, Any]:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    parameter.grad = torch.tensor([2.0], dtype=torch.float64)
    optimizer = torch.optim.Adam([parameter], lr=0.1, eps=1.0e-8)

    def programmer_error() -> dict[str, float]:
        message = "synthetic missing owned-state key"
        raise KeyError(message)

    propagated = False
    try:
        transaction(
            optimizer=optimizer,
            parameters=(parameter,),
            prepare_step=lambda: None,
            project=dict,
            restore_auxiliary=lambda: None,
            evaluate_trial=programmer_error,
            cost_counters=lambda: (0, 0),
            accepted_objective=1.0,
            backtrack_factor=0.5,
            maximum_trials=2,
            armijo_coefficient=1.0e-4,
        )
    except KeyError as error:
        propagated = True
        message = str(error)
    assert propagated
    return {"programmer_error_propagated": propagated, "error": message}


def run_budget_stop(transaction: Any) -> dict[str, Any]:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    parameter.grad = torch.tensor([2.0], dtype=torch.float64)
    optimizer = torch.optim.Adam([parameter], lr=0.1, eps=1.0e-8)
    restored = {"count": 0}

    def restore() -> None:
        restored["count"] += 1

    def unexpected_evaluation() -> dict[str, float]:
        message = "expired budget must stop before evaluating a trial"
        raise RuntimeError(message)

    receipt = transaction(
        optimizer=optimizer,
        parameters=(parameter,),
        prepare_step=lambda: None,
        project=lambda: {"parameter": float(parameter.detach())},
        restore_auxiliary=restore,
        evaluate_trial=unexpected_evaluation,
        cost_counters=lambda: (0, 0),
        accepted_objective=1.0,
        backtrack_factor=0.5,
        maximum_trials=2,
        armijo_coefficient=1.0e-4,
        deadline=time.perf_counter() - 1.0,
    )
    assert receipt["accepted"] is False
    assert receipt["budget_exhausted"] is True
    assert receipt["trials"] == []
    assert torch.equal(parameter, torch.tensor([1.0], dtype=torch.float64))
    assert torch.equal(parameter.grad, torch.tensor([2.0], dtype=torch.float64))
    assert optimizer.state_dict()["state"] == {}
    assert restored["count"] == 1
    return {"receipt": receipt, "restored": restored}


def run_implementation_admission(validate: Any) -> dict[str, Any]:
    expected = {"30-joint-pilot.py": "expected"}
    validate(expected, expected, artifact="synthetic matching artifact")
    mismatch_rejected = False
    try:
        validate(
            {"30-joint-pilot.py": "changed"},
            expected,
            artifact="synthetic changed artifact",
        )
    except AssertionError:
        mismatch_rejected = True
    assert mismatch_rejected
    return {"matching_admitted": True, "mismatch_rejected": mismatch_rejected}


def run_optimizer_preview(  # noqa: C901 - one end-to-end owned-state regression.
    preview: Any, *, freeze_shared: bool
) -> dict[str, Any]:
    dtype = torch.float64
    parameters = tuple(
        torch.nn.Parameter(value)
        for value in (
            torch.tensor([0.4, 0.6], dtype=dtype),
            torch.tensor([0.02, -0.02], dtype=dtype),
            torch.tensor([0.05, -0.05], dtype=dtype),
        )
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [parameters[0]], "lr": 0.1},
            {"params": [parameters[1]], "lr": 0.05},
            {"params": [parameters[2]], "lr": 0.02},
        ],
        eps=1.0e-8,
    )
    for warm_index in range(2):
        for parameter, scale in zip(parameters, (1.0, -0.5, 0.25), strict=True):
            parameter.grad = scale * parameter.detach() + 0.1 * (warm_index + 1)
        optimizer.step()
    with torch.no_grad():
        parameters[0].copy_(torch.tensor([0.98, 0.02], dtype=dtype))
        parameters[1].copy_(torch.tensor([0.09, -0.09], dtype=dtype))
        parameters[2].copy_(torch.tensor([0.19, -0.19], dtype=dtype))
    for parameter, gradient in zip(
        parameters,
        (
            torch.tensor([-2.0, 2.0], dtype=dtype),
            torch.tensor([-1.0, 1.0], dtype=dtype),
            torch.tensor([-0.5, 0.5], dtype=dtype),
        ),
        strict=True,
    ):
        parameter.grad = gradient.clone()

    accepted_parameters = tuple(parameter.detach().clone() for parameter in parameters)
    accepted_gradients = tuple(
        parameter.grad.detach().clone() for parameter in parameters
    )
    accepted_optimizer = deepcopy(optimizer.state_dict())
    assert all(int(optimizer.state[parameter]["step"]) == 2 for parameter in parameters)

    def prepare() -> None:
        if freeze_shared:
            parameters[2].grad = None

    def project() -> dict[str, Any]:
        with torch.no_grad():
            parameters[0].clamp_(0.0, 1.0)
            parameters[1].clamp_(-0.1, 0.1)
            parameters[2].clamp_(-0.2, 0.2)
        return {"freeze_shared": freeze_shared}

    receipt = preview(
        optimizer=optimizer,
        parameters=parameters,
        prepare_step=prepare,
        project=project,
    )
    for parameter, accepted in zip(parameters, accepted_parameters, strict=True):
        assert torch.equal(parameter, accepted)
    for parameter, accepted in zip(parameters, accepted_gradients, strict=True):
        assert torch.equal(parameter.grad, accepted)
    assert_tree_equal(optimizer.state_dict(), accepted_optimizer)

    reference_parameters = tuple(
        torch.nn.Parameter(value.detach().clone()) for value in accepted_parameters
    )
    reference_optimizer = torch.optim.Adam(
        [
            {"params": [reference_parameters[0]], "lr": 0.1},
            {"params": [reference_parameters[1]], "lr": 0.05},
            {"params": [reference_parameters[2]], "lr": 0.02},
        ],
        eps=1.0e-8,
    )
    reference_optimizer.load_state_dict(deepcopy(accepted_optimizer))
    for parameter, gradient in zip(
        reference_parameters, accepted_gradients, strict=True
    ):
        parameter.grad = gradient.clone()
    if freeze_shared:
        reference_parameters[2].grad = None
    reference_optimizer.step()
    with torch.no_grad():
        reference_parameters[0].clamp_(0.0, 1.0)
        reference_parameters[1].clamp_(-0.1, 0.1)
        reference_parameters[2].clamp_(-0.2, 0.2)
    expected_steps = tuple(
        parameter.detach() - accepted
        for parameter, accepted in zip(
            reference_parameters, accepted_parameters, strict=True
        )
    )
    for actual, expected in zip(receipt["steps"], expected_steps, strict=True):
        assert torch.allclose(actual, expected, rtol=0.0, atol=1.0e-15)
    if freeze_shared:
        assert torch.equal(receipt["steps"][2], torch.zeros_like(receipt["steps"][2]))
    else:
        assert bool((receipt["steps"][2] != 0.0).any())
    expected_directional = sum(
        float(torch.sum(gradient * step))
        for gradient, step in zip(accepted_gradients, expected_steps, strict=True)
    )
    assert abs(receipt["directional_derivative"] - expected_directional) <= 1.0e-15
    return {
        "freeze_shared": freeze_shared,
        "preview_matches_real_proposal": True,
        "full_state_restored": True,
        "nonzero_moment_step_before": 2,
        "proposal_steps": [step.tolist() for step in receipt["steps"]],
        "directional_derivative": receipt["directional_derivative"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joint_pilot = load_joint_pilot()
    transaction = joint_pilot.backtracked_projected_adam_step
    summary = {
        "schema": "joint-outer-transaction-validation-v1",
        "success": True,
        "retry": run_retry(transaction),
        "exhaustion": run_exhaustion(transaction),
        "programmer_error": run_programmer_error(transaction),
        "budget_stop": run_budget_stop(transaction),
        "implementation_admission": run_implementation_admission(
            joint_pilot.validate_implementation_sha256
        ),
        "optimizer_preview_control": run_optimizer_preview(
            joint_pilot.reversible_projected_optimizer_preview,
            freeze_shared=True,
        ),
        "optimizer_preview_joint": run_optimizer_preview(
            joint_pilot.reversible_projected_optimizer_preview,
            freeze_shared=False,
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
