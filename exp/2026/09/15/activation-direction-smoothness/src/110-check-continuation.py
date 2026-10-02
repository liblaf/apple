"""Verify the last checkpoint of the interrupted continuation diagnostic."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import scipy.linalg as la
import study

from liblaf import cherries

resume = importlib.import_module("100-continue")
runner = importlib.import_module("10-run")
GROUP = Path(__file__).resolve().parents[1]


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("110-continuation-checks")


def main(cfg: Config):
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    base = GROUP / "data/tune-w0/h200-unconstrained-w0"
    continued = GROUP / "data/100-backtracked-continuation"
    protocol = json.loads((continued / "protocol.json").read_text())
    assert (
        hashlib.sha256((base / "checkpoint.npz").read_bytes()).hexdigest()
        == protocol["base_checkpoint_sha256"]
    )
    mesh = study.ph.build_mesh(100, 10)
    results = []
    for directory in (base, continued):
        with np.load(directory / "checkpoint.npz") as saved:
            q, full_u, B, step = (
                saved["controls"],
                saved["u"],
                saved["B"],
                int(saved["step"]),
            )
        np.testing.assert_array_equal(B, study.matrices(mesh, q, "unconstrained"))
        u = resume.pack(mesh, full_u)
        _, residual, hessian, J = study.ph.assemble(mesh, u, B)
        eigenvalue = float(
            la.eigh(
                ((hessian + hessian.T) * 0.5).toarray(),
                subset_by_index=[0, 0],
                eigvals_only=True,
                driver="evr",
            )[0]
        )
        _, _, _, values = study.evaluate(mesh, q, "unconstrained", 0.2, 0.0, u)
        fixed_error = float(np.abs(full_u.ravel()[mesh.lookup < 0]).max())
        index = int(np.argmin(J))
        assert np.linalg.norm(residual, np.inf) <= 1e-10
        assert J.min() > 1e-6
        assert eigenvalue > 0
        assert fixed_error == 0
        results.append(
            {
                "step": step,
                "normalized_loss": values["normalized_loss"],
                "raw_loss": values["raw_loss"],
                "min_J": float(J.min()),
                "residual_inf": float(np.linalg.norm(residual, np.inf)),
                "hessian_min_eigenvalue": eigenvalue,
                "fixed_error": fixed_error,
                "minimum_J_cell": index,
                "minimum_J_cell_is_muscle": bool(mesh.muscle[index]),
                "minimum_J_cell_reference_centroid": mesh.p[mesh.tri[index]]
                .mean(axis=0)
                .tolist(),
                "projected_gradient_inf": values["projected_gradient_inf"],
                "nonpositive_det_B_fraction": values["nonpositive_det_B_fraction"],
            }
        )
    log = (GROUP / "logs/continue-terminal.log").read_text()
    runner.write_json(
        output / "checks.json",
        {
            "checkpoints": results,
            "relative_loss_improvement_percent": 100
            * (1 - results[1]["raw_loss"] / results[0]["raw_loss"]),
            "original_checkpoint_unchanged": True,
            "continuation_run_exit_code": 130,
            "termination": "Manually interrupted after steps 268-270 repeated identical displacement loss and min J; retained accepted checkpoint270. The original interrupted run did not flush its in-memory trial/history arrays.",
            "future_runner_change": "The source runner now exits cleanly when a newly accepted update leaves displacement and loss exactly unchanged.",
            "continuation_terminal_has_interrupt": "KeyboardInterrupt" in log,
        },
    )
    (output / "110-check-continuation.py").write_bytes(Path(__file__).read_bytes())
    cherries.log_metrics(
        {
            "normalized_loss": results[-1]["normalized_loss"],
            "min_J": results[-1]["min_J"],
            "verified_step": results[-1]["step"],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=runner.ProfileActivationStudy)
