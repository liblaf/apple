"""CPU-only recovery gate for the interrupted fixed-reference Raw6 fit."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import torch
from experiment import Profile

from liblaf import cherries

BRANCH = "smooth-off-normal"
INTERRUPTED_STEP = 102


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison_dir: Path = Path("130-reference-fit")
    output: Path = Path("131-reference-resume-checks")


def old35() -> Any:
    spec = importlib.util.spec_from_file_location(
        "old35_reference_resume", Path(__file__).with_name("35-check-resume.py")
    )
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OLD = old35()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict), path
    return value


def record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": digest(path)}


def check_provenance(protocol: dict[str, Any]) -> dict[str, int]:
    sources = protocol["sources"]
    assert len(sources) == 99
    for item in sources.values():
        current, snapshot = Path(item["path"]), Path(item["snapshot"])
        assert current.exists()
        assert snapshot.exists()
        assert digest(current) == item["sha256"]
        assert digest(snapshot) == item["sha256"]
    fixture = protocol["fixture"]
    assert len(fixture) == 3
    for item in fixture.values():
        path = Path(item["path"])
        assert path.exists()
        assert digest(path) == item["sha256"]
    return {
        "source_records_checked": len(sources),
        "fixture_records_checked": len(fixture),
    }


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        result = list(csv.DictReader(stream))
    assert result
    assert [int(row["step"]) for row in result] == list(range(len(result)))
    return result


def receipts(path: Path) -> list[dict[str, Any]]:
    result = [json.loads(line) for line in path.read_text().splitlines()]
    assert [int(row["step"]) for row in result] == list(range(len(result)))
    for row in result:
        assert row["forward"]["success"] is True
        assert row["adjoint"]["success"] is True
    return result


def check_adam(latest: dict[str, Any]) -> dict[str, Any]:
    saved_optimizer = latest["optimizer"]
    q = torch.nn.Parameter(latest["q"].detach().clone())
    optimizer = torch.optim.Adam(
        [q], lr=0.3, eps=5.707952862381702e-05, betas=(0.9, 0.999)
    )
    optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
    state, group = optimizer.state[q], optimizer.param_groups[0]
    assert group["lr"] == 0.3
    assert group["eps"] == 5.707952862381702e-05
    assert group["betas"] == (0.9, 0.999)
    assert int(state["step"]) == INTERRUPTED_STEP
    assert state["exp_avg"].shape == q.shape == (288235, 6)
    assert state["exp_avg_sq"].shape == q.shape
    gradient = OLD.synthetic_gradient(tuple(q.shape))
    expected_q, expected_m, expected_v, expected_step = OLD.expected_adam(
        q.detach().clone(),
        state["exp_avg"].detach().clone(),
        state["exp_avg_sq"].detach().clone(),
        INTERRUPTED_STEP,
        gradient,
        group,
    )
    q.grad = gradient.clone()
    optimizer.step()
    torch.testing.assert_close(q.detach(), expected_q, rtol=0.0, atol=2e-15)
    torch.testing.assert_close(state["exp_avg"], expected_m, rtol=0.0, atol=2e-18)
    torch.testing.assert_close(state["exp_avg_sq"], expected_v, rtol=0.0, atol=2e-20)
    assert int(state["step"]) == expected_step
    return {
        "lr": group["lr"],
        "eps": group["eps"],
        "betas": list(group["betas"]),
        "step_before": INTERRUPTED_STEP,
        "step_after_synthetic_update": expected_step,
        "synthetic_adam_formula_passed": True,
    }


def check_branch(source: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    folder = source / BRANCH
    trace_path, receipt_path = folder / "trace.csv", folder / "solver-receipts.jsonl"
    trace, accepted = rows(trace_path), receipts(receipt_path)
    assert len(trace) == len(accepted) == INTERRUPTED_STEP + 1
    summary = read(folder / "summary.json")
    assert summary["status"] == "running"
    assert summary["failure"] is None
    assert int(summary["last_step"]) == INTERRUPTED_STEP
    latest = torch.load(
        folder / "optimizer-latest.pt", map_location="cpu", weights_only=False
    )
    assert latest["branch"] == BRANCH
    assert int(latest["step"]) == INTERRUPTED_STEP
    item = next(iter(latest["optimizer"]["state"].values()))
    assert int(item["step"]) == INTERRUPTED_STEP
    with np.load(folder / "last.npz", allow_pickle=False) as state:
        assert int(state["step"]) == INTERRUPTED_STEP
        assert bool(state["solver_valid"])
        assert bool(state["physical_volume_energy"])
        assert np.array_equal(state["q"], latest["q"].numpy())
        assert np.array_equal(state["u"], latest["u"])
    with np.load(folder / "initial-state.npz", allow_pickle=False) as state:
        assert int(state["step"]) == 0
        assert bool(state["activation_identity"])
        assert bool(state["adjoint_initial_guess_zero"])
        for name in ("q", "u", "m", "v"):
            assert not np.count_nonzero(state[name]), name
    # This is the optimizer state that the continuation must load verbatim.
    checkpoint = check_adam(latest)
    artifacts = [
        record(folder / name)
        for name in (
            "initial-state.npz",
            "initial-gradient.npz",
            "initial-update.json",
            "last.npz",
            "optimizer-latest.pt",
            "trace.csv",
            "solver-receipts.jsonl",
            "summary.json",
        )
    ]
    return {
        "step": INTERRUPTED_STEP,
        "trace_rows": len(trace),
        "accepted_receipts": len(accepted),
        "summary_status": summary["status"],
        "summary_failure": summary["failure"],
        "checkpoint": checkpoint,
    }, artifacts


def main(cfg: Config) -> None:
    assert not torch.cuda.is_initialized(), "CPU preflight must not initialize CUDA"
    source = cherries.input(cfg.comparison_dir).resolve()
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = source / "protocol.json"
    protocol = read(protocol_path)
    config = protocol["config"]
    assert int(config["steps"]) == 200
    assert config["branches"] == "smooth-off-normal,smooth-on-normal"
    provenance = check_provenance(protocol)
    branch, artifacts = check_branch(source)
    checks: dict[str, Any] = {
        "passed": True,
        "parent_dir": str(source),
        "parent_protocol_record": record(protocol_path),
        "parent_provenance": provenance,
        "parent_artifacts": artifacts,
        "branches": {BRANCH: branch},
    }
    (output / "checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    cherries.log_metric("resume/off_preflight_passed", 1.0, step=INTERRUPTED_STEP)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
