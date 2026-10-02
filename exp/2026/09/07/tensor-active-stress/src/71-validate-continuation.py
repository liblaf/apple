"""Validate saved-state contracts and constrained-gradient diagnostics on CPU."""

# ruff: noqa: PLR0915

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pydantic_settings as ps
import torch
from continuation_helpers import (
    BASE,
    load_checkpoint,
    projected_gradient_metrics,
    tensor_rms_mpa,
)
from experiment_profile import ProfileCometNoCommit
from tensor_controls import coordinates, project

from liblaf import cherries

HERE = Path(__file__).resolve().parent.parent
COMPLETED = False


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("71-continuation-validation", mkdir=True)


def main(cfg: Config) -> None:
    global COMPLETED  # noqa: PLW0603
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir())
    spec = importlib.util.spec_from_file_location(
        "tensor_continuation", HERE / "src/74-face-continuation.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    continuation_cfg = module.Config(_cli_parse_args=False)
    checkpoint = load_checkpoint(continuation_cfg.resume, continuation_cfg)
    assert checkpoint["step"] == 64
    eye = torch.eye(3).repeat(2, 1, 1)
    identity_coordinates = coordinates(eye)
    zero = torch.zeros_like(identity_coordinates)
    free_descent = projected_gradient_metrics(zero, -identity_coordinates, 10.0, 1.0)
    blocked_lower = projected_gradient_metrics(zero, identity_coordinates, 10.0, 1.0)
    blocked_upper = projected_gradient_metrics(
        10 * identity_coordinates, -identity_coordinates, 10.0, 1.0
    )
    assert abs(free_descent["projected_gradient_mapping_rms"] - 2**-0.5) < 1e-14
    assert blocked_lower["projected_gradient_mapping_rms"] == 0
    assert blocked_upper["projected_gradient_mapping_rms"] == 0
    physical_rms = tensor_rms_mpa(identity_coordinates, BASE.QREF)
    assert abs(physical_rms - 3**0.5 * BASE.QREF) < 1e-14

    # One deterministic update checks the actual full checkpoint without a face solve.
    original = checkpoint["q"].clone()
    opt_state = copy.deepcopy(checkpoint["optimizer"])
    q = torch.nn.Parameter(original.clone())
    optimizer = torch.optim.Adam([q], lr=0.3, eps=0.01)
    optimizer.load_state_dict(opt_state)
    gradient = original * 1e-4 + 1e-5
    q.grad = gradient
    moment = optimizer.state[q]
    beta1, beta2 = optimizer.param_groups[0]["betas"]
    next_step = int(moment["step"]) + 1
    mhat = (beta1 * moment["exp_avg"] + (1 - beta1) * gradient) / (1 - beta1**next_step)
    vhat = (beta2 * moment["exp_avg_sq"] + (1 - beta2) * gradient.square()) / (
        1 - beta2**next_step
    )
    expected = original - 0.3 * mhat / (vhat.sqrt() + 0.01)
    optimizer.step()
    error = float((q.detach() - expected).abs().max())
    assert error < 1e-12
    assert int(optimizer.state[q]["step"]) == 65
    assert int(checkpoint["optimizer"]["state"][0]["step"]) == 64
    assert torch.equal(checkpoint["q"], original)
    project(q, 10.0)
    sources = out / "sources"
    sources.mkdir()
    for name in (
        Path(__file__).name,
        "continuation_helpers.py",
        "73-calibrate-face-step.py",
        "74-face-continuation.py",
        "tensor_controls.py",
        "experiment_profile.py",
    ):
        (sources / name).write_bytes((HERE / "src" / name).read_bytes())
    summary = {
        "status": "passed",
        "scope": "CPU checkpoint identity, Adam counter/moment continuation, physical tensor RMS, and projected-gradient boundary checks; no face solve",
        "checkpoint": {
            "path": str(continuation_cfg.resume),
            "sha256": BASE.sha256(continuation_cfg.resume),
        },
        "control_count": original.numel(),
        "checkpoint_global_step": 64,
        "next_adam_step": 65,
        "full_checkpoint_adam_closed_form_max_abs": error,
        "parent_state_unmodified": True,
        "projected_gradient": {
            "free_descent": free_descent,
            "blocked_lower": blocked_lower,
            "blocked_upper": blocked_upper,
        },
        "physical_rms_identity_mpa": physical_rms,
        "sources": {name.name: BASE.sha256(name) for name in sources.iterdir()},
    }
    BASE.write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
