"""Driver regression for usable approximate parent endpoints."""
# ruff: noqa: ANN001, ANN204, ARG002, ARG005, EM101, E402, PT018, RUF012, TRY003

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "exp/2026/09/21/stress-activation-loss/src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from stress_physics import ForwardConvergenceError


def _module():
    spec = importlib.util.spec_from_file_location(
        "stress_chains_test", SOURCE / "40-run-chains.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_chain_transfers_approximate_parent_and_survives_endpoint_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    output, gate, calibration = tmp_path / "out", tmp_path / "gate", tmp_path / "cal"
    gate.mkdir()
    (gate / "checks.json").write_text('{"passed": true}')
    (gate / "protocol.json").write_text('{"sources": {}, "fixture": {}}')
    calibration.mkdir()
    (calibration / "calibration.json").write_text(
        json.dumps(
            {
                "passed": True,
                "schema": "smile-stress-gradient-calibration-v2",
                "sources": {},
                "validation": {
                    "path": str((gate / "checks.json").resolve()),
                    "sha256": module.receipt(gate / "checks.json")["sha256"],
                },
                "selected_weight": 0.1,
                "learning_rate": 0.01,
                "adam_eps": 1e-8,
            }
        )
    )
    calls: list[tuple[str, torch.Tensor]] = []

    class Physics:
        ids = np.array((0,))
        points = np.zeros((1, 3))
        material_spec = {}
        forward_tolerance = {}

    class Study:
        def __init__(self):
            self.physics = Physics()

        def save_geometry(self, path):
            path.write_bytes(b"fixture")

        def evaluate(self, *args, **kwargs):
            raise ForwardConvergenceError("strict endpoint unavailable")

    def fake_run_stage(study, folder, mode, beta, weight, q0, seed, **kwargs):
        del study, beta, weight, seed, kwargs
        calls.append((mode, q0.clone()))
        folder.mkdir()
        np.savez_compressed(
            folder / "last.npz",
            q=np.zeros((1, 6)),
            fixed_axes=np.empty((0, 3)),
            Qhat=np.ones((1, 3, 3)),
            u=np.zeros((1, 3)),
            solver_valid=False,
        )
        return {
            "status": "completed_budget_not_convergence_certified",
            "last_step": 1,
            "attempted_steps": 2,
            "budget": 2,
            "last_metrics": {},
            "failure": None,
        }

    monkeypatch.setattr(module, "StressStudy", Study)
    monkeypatch.setattr(module, "run_stage", fake_run_stage)
    monkeypatch.setattr(module, "archive", lambda path: {})
    monkeypatch.setattr(module, "verify_sources", lambda sources: None)
    monkeypatch.setattr(module.cherries, "input", lambda path: path)
    monkeypatch.setattr(module.cherries, "output", lambda path: path)
    cfg = module.Config(
        output=output,
        validation=gate,
        calibration=calibration / "calibration.json",
        steps=2,
        _cli_parse_args=[],
    )

    module.main(cfg)

    assert len(calls) == 8
    assert torch.equal(calls[1][1], torch.ones((1, 3, 3)))
    assert torch.equal(calls[4][1], torch.zeros((1, 3, 3)))
    parent = json.loads((output / "l2-psd6-parent.json").read_text())
    assert parent["parent_solver_valid"] is False
    diagnostic = json.loads(
        (output / "l2-symmetric6" / "gradient-balance.json").read_text()
    )
    assert diagnostic["status"] == "unavailable"
    assert (output / "normal-rankone_learned" / "gradient-balance.json").is_file()
