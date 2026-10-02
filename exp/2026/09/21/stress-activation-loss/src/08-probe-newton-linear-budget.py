# Copyright (c) 2026 liblaf
"""Probe whether the unshifted Newton-CG linear budget triggers shift retries."""

from __future__ import annotations

import time
from pathlib import Path

import pydantic_settings as ps
import torch
from experiment import Profile
from run_support import archive, write_json
from stress_study import StressStudy

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("08-newton-linear-budget-v2")
    validation_accuracy_factor: float = 0.1
    linear_caps: tuple[int, int] = (1_000, 10_000)


def main(cfg: Config) -> None:
    assert 0 < cfg.validation_accuracy_factor < 1
    assert cfg.linear_caps == (1_000, 10_000)
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    study = StressStudy(max_newton_steps=1)
    physics = study.physics
    q = torch.zeros((len(physics.ids), 6), device="cuda", dtype=torch.float64)
    q[:, :3] = 0.04
    seed = torch.zeros_like(torch.as_tensor(physics.points)).cpu().numpy()
    records = []
    with physics.accuracy(cfg.validation_accuracy_factor), physics.approximate_solves():
        for linear_cap in cfg.linear_caps:
            optimizer = physics.forward.optimizer
            optimizer.linear_max_steps = linear_cap
            physics.forward_tolerance["newton_linear_max_steps"] = linear_cap
            started = time.perf_counter()
            result = study.evaluate(
                q.detach(),
                "symmetric6",
                None,
                seed,
                0.0,
                0.0,
                backward=False,
            )
            records.append(
                {
                    "linear_max_steps": linear_cap,
                    "wall_seconds": time.perf_counter() - started,
                    "solver_valid": result["solver_valid"],
                    "tolerance": dict(physics.forward_tolerance),
                    "metrics": {
                        key: result[key]
                        for key in (
                            "objective",
                            "fit_rms_mm",
                            "detF_min",
                            "detF_max",
                        )
                    },
                    "forward": result["forward"],
                }
            )
    write_json(
        out / "probe.json",
        {
            "schema": "smile-stress-newton-linear-budget-probe-v2",
            "purpose": "Compare only the unshifted Newton-CG PCG budget on one fixed q=0.04 I state.",
            "approximate_solves": True,
            "validation_accuracy_factor": cfg.validation_accuracy_factor,
            "force_tolerance": dict(physics.forward_tolerance),
            "stress_control": "symmetric6 q=(0.04, 0.04, 0.04, 0, 0, 0) for every active tetrahedron",
            "normal_weight": 0.0,
            "smooth_weight": 0.0,
            "backward": False,
            "newton_max_steps": 1,
            "records": records,
            "sources": archive(out),
        },
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
