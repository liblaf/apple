"""CPU-only provenance and Adam-resume gate for the 100-update Raw6 states."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import torch
from experiment import Profile

from liblaf import cherries

BRANCHES = (
    "smooth-off-l2",
    "smooth-off-normal",
    "smooth-on-l2",
    "smooth-on-normal",
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison_dir: Path = Path("10-comparison")
    old_audit_dir: Path = Path("20-verification")
    output: Path = Path("35-resume-checks")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict), path
    return value


def check_parent_provenance(protocol: dict[str, Any]) -> dict[str, int]:
    sources = protocol["sources"]
    assert isinstance(sources, dict)
    assert sources
    checked = 0
    for details in sources.values():
        assert isinstance(details, dict)
        snapshot = Path(details["snapshot"])
        current = Path(details["path"])
        expected = details["sha256"]
        assert snapshot.exists()
        assert current.exists()
        assert sha256(snapshot) == expected, snapshot
        assert sha256(current) == expected, current
        checked += 1
    fixture = protocol["fixture"]
    assert isinstance(fixture, dict)
    assert fixture
    for details in fixture.values():
        path = Path(details["path"])
        assert path.exists()
        assert sha256(path) == details["sha256"], path
    return {"sources_checked": checked, "fixtures_checked": len(fixture)}


def expected_adam(
    q: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    step: int,
    gradient: torch.Tensor,
    group: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    beta1, beta2 = group["betas"]
    next_step = step + 1
    next_m = m * beta1 + gradient * (1 - beta1)
    next_v = v * beta2 + gradient.square() * (1 - beta2)
    update = (next_m / (1 - beta1**next_step)) / (
        (next_v / (1 - beta2**next_step)).sqrt() + group["eps"]
    )
    return q - group["lr"] * update, next_m, next_v, next_step


def synthetic_gradient(shape: tuple[int, ...]) -> torch.Tensor:
    count = int(np.prod(shape))
    values = torch.arange(count, dtype=torch.float64)
    # Deterministic, nonconstant, and safely away from an all-zero update.
    return ((values.remainder(29) - 14) / 1000).reshape(shape)


def check_checkpoint(folder: Path, expected_step: int) -> dict[str, Any]:
    checkpoint_path = folder / "optimizer-latest.pt"
    last_path = folder / "last.npz"
    step_path = folder / f"step-{expected_step:04d}.npz"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["branch"] == folder.name
    assert int(checkpoint["step"]) == expected_step
    assert checkpoint["q"].device.type == "cpu"
    assert checkpoint["q"].dtype == torch.float64
    assert checkpoint["u"].dtype == np.float64
    with (
        np.load(last_path, allow_pickle=False) as last,
        np.load(step_path, allow_pickle=False) as saved_step,
    ):
        assert int(last["step"]) == expected_step
        assert int(saved_step["step"]) == expected_step
        assert bool(last["solver_valid"])
        assert bool(last["physical_volume_energy"])
        assert np.array_equal(last["q"], checkpoint["q"].numpy())
        assert np.array_equal(last["u"], checkpoint["u"])
        assert np.array_equal(saved_step["q"], last["q"])
        assert np.array_equal(saved_step["u"], last["u"])

    # The deepcopy is deliberate: optimizer.load_state_dict owns and mutates state.
    saved_optimizer = copy.deepcopy(checkpoint["optimizer"])
    original_state = checkpoint["optimizer"]["state"][0]
    original_m = original_state["exp_avg"].clone()
    original_v = original_state["exp_avg_sq"].clone()
    copied_state = saved_optimizer["state"][0]
    assert copied_state["exp_avg"].data_ptr() != original_state["exp_avg"].data_ptr()
    assert (
        copied_state["exp_avg_sq"].data_ptr() != original_state["exp_avg_sq"].data_ptr()
    )
    q = torch.nn.Parameter(checkpoint["q"].detach().clone())
    optimizer = torch.optim.Adam([q], lr=0.3, eps=0.01, betas=(0.9, 0.999))
    optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
    torch.testing.assert_close(
        original_state["exp_avg"], original_m, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        original_state["exp_avg_sq"], original_v, rtol=0.0, atol=0.0
    )
    state = optimizer.state[q]
    group = optimizer.param_groups[0]
    assert group["lr"] == 0.3
    assert group["eps"] == 0.01
    assert group["betas"] == (0.9, 0.999)
    assert group["weight_decay"] == 0
    assert not group["amsgrad"]
    assert not group["maximize"]
    assert int(state["step"].item()) == expected_step
    assert state["exp_avg"].shape == q.shape == (288235, 6)
    assert state["exp_avg_sq"].shape == q.shape
    assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float64

    gradient = synthetic_gradient(tuple(q.shape))
    expected_q, expected_m, expected_v, expected_t = expected_adam(
        q.detach().clone(),
        state["exp_avg"].detach().clone(),
        state["exp_avg_sq"].detach().clone(),
        expected_step,
        gradient,
        group,
    )
    q.grad = gradient.clone()
    optimizer.step()
    torch.testing.assert_close(q.detach(), expected_q, rtol=0.0, atol=2e-15)
    torch.testing.assert_close(state["exp_avg"], expected_m, rtol=0.0, atol=2e-18)
    torch.testing.assert_close(state["exp_avg_sq"], expected_v, rtol=0.0, atol=2e-20)
    assert int(state["step"].item()) == expected_t

    return {
        "checkpoint_sha256": sha256(checkpoint_path),
        "last_npz_sha256": sha256(last_path),
        "step_npz_sha256": sha256(step_path),
        "step": expected_step,
        "q_shape": list(q.shape),
        "u_shape": list(checkpoint["u"].shape),
        "adam": {
            "lr": group["lr"],
            "eps": group["eps"],
            "betas": list(group["betas"]),
            "step_before": expected_step,
            "step_after_synthetic_update": expected_t,
        },
        "synthetic_adam_formula": {
            "passed": True,
            "q_atol": 2e-15,
            "m_atol": 2e-18,
            "v_atol": 2e-20,
        },
    }


def main(cfg: Config) -> None:
    assert not torch.cuda.is_initialized(), "CPU preflight must not initialize CUDA"
    source = cherries.input(cfg.comparison_dir)
    old_audit = cherries.input(cfg.old_audit_dir)
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = source / "protocol.json"
    protocol = read_json(protocol_path)
    config = protocol["config"]
    expected_step = int(config["steps"])
    assert expected_step == 100
    assert tuple(config["branches"].split(",")) == BRANCHES
    provenance = check_parent_provenance(protocol)
    old_checks_path = old_audit / "checks.json"
    old_checks = read_json(old_checks_path)
    assert old_checks["passed"] is True
    assert old_checks["provenance"]["current_sources_checked"] > 0
    assert old_checks["provenance"]["fixture_receipts_checked"] > 0

    checks: dict[str, Any] = {
        "passed": False,
        "parent_protocol_sha256": sha256(protocol_path),
        "parent_provenance": provenance,
        "old20_audit": {
            "checks_json_sha256": sha256(old_checks_path),
            "current_20_verify_py_sha256": sha256(
                Path(__file__).parents[0] / "20-verify.py"
            ),
            "passed": old_checks["passed"],
            "provenance": old_checks["provenance"],
        },
        "branches": {},
    }
    for branch in BRANCHES:
        cherries.set_step(expected_step)
        checks["branches"][branch] = check_checkpoint(source / branch, expected_step)
        cherries.log_metric(f"{branch}/resume_preflight_passed", 1.0)
    checks["passed"] = True
    (output / "checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
