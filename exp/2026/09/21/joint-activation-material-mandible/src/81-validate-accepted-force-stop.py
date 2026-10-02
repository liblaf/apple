"""Regression for PNCG stopping on the force of the saved accepted state."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from joint_common import ProfileJoint, archive_sources, write_json

from liblaf import cherries


class Config(cherries.BaseConfig):
    output_dir: Path


class Quadratic:
    def fun(self, state: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.dot(state, state)

    def grad(self, state: torch.Tensor) -> torch.Tensor:
        return state.clone()


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    spec = importlib.util.spec_from_file_location(
        "forward_runner", Path(__file__).with_name("68-run-simple-skin-forward.py")
    )
    assert spec is not None
    assert spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    problem = Quadratic()
    optimizer = runner.MonitoredPncg(
        monitor=SimpleNamespace(last_force_norm=None),
        criteria=runner.StrictPncg.ConvergenceCriteria(
            atol_primary=0.15, rtol_primary=0.0, max_steps=10
        ),
    )
    records = []
    for before, accepted, should_stop in ((0.1, 0.2, False), (0.2, 0.1, True)):
        state = torch.tensor([accepted], dtype=torch.float64)
        opt_state = optimizer.init(problem, state, state)
        convergence = opt_state.convergence_state
        convergence.step = 1
        convergence.grad_norm_first = torch.tensor(before, dtype=torch.float64)
        convergence.grad_norm = torch.tensor(before, dtype=torch.float64)
        legacy_stop, _ = optimizer.criteria.terminate(convergence)
        stop, result = optimizer.terminate(problem, state, opt_state)
        assert bool(legacy_stop) is not should_stop
        assert bool(stop) is should_stop
        assert float(convergence.grad_norm) == accepted
        assert optimizer.monitor.last_force_norm == accepted
        assert convergence.step == 1
        records.append(
            {
                "pre_step_force": before,
                "accepted_state_force": accepted,
                "threshold": 0.15,
                "legacy_stop": bool(legacy_stop),
                "accepted_state_stop": bool(stop),
                "result": str(result),
            }
        )
    write_json(
        cfg.output_dir / "summary.json",
        {
            "schema": "joint-accepted-force-stop-validation-v1",
            "success": True,
            "cases": records,
        },
    )
    cherries.log_output(cfg.output_dir / "summary.json")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
