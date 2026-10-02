"""CPU orchestration regression for the portable active-strain chain driver."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[6]
SOURCE = ROOT / "exp/2026/09/21/stress-activation-loss/src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def _module():
    spec = importlib.util.spec_from_file_location(
        "active_strain_chain_test", SOURCE / "46-run-active-strain-chain.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_l2_chain_transfers_approximate_strain_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    output = tmp_path / "chain"
    calls = []

    class Physics:
        def __init__(self) -> None:
            self.ids = np.array((0,))
            self.points = np.zeros((1, 3))
            self.material_spec = {"fake": True}
            self.forward_tolerance = {"fake": True}

    class Study:
        def __init__(self, *, activation_model: str) -> None:
            assert activation_model == "strain"
            self.physics = Physics()

        def save_geometry(self, path: Path) -> None:
            np.savez(path, points=self.physics.points)

    def run_stage(
        _study: object,
        folder: Path,
        mode: str,
        beta: float,
        eta: float,
        q0: torch.Tensor,
        seed: np.ndarray,
        **kwargs: object,
    ) -> dict:
        calls.append((mode, beta, eta, q0.clone(), seed.copy(), kwargs))
        folder.mkdir()
        value = len(calls)
        np.savez_compressed(
            folder / "last.npz",
            q=np.zeros((1, 6)),
            fixed_axes=np.array([[1.0, 0.0, 0.0]])
            if mode == "rankone_fixed"
            else np.empty((0, 3)),
            S=np.full((1, 3, 3), value),
            B=np.eye(3)[None] + value,
            u=np.full((1, 3), value),
            activation_model="strain",
            solver_valid=False,
        )
        return {
            "status": "completed_budget_not_convergence_certified",
            "last_step": 1,
            "budget": 2,
            "last_metrics": {
                "solver_valid": False,
                "l2_gradient_dual_norm": 1.0,
                "smoothness_gradient_dual_norm": 2.0,
            },
        }

    def output_(path: Path) -> Path:
        return Path(path)

    def archive_(_out: Path) -> dict:
        return {}

    monkeypatch.setattr(module.cherries, "output", output_)
    monkeypatch.setattr(module, "StressStudy", Study)
    monkeypatch.setattr(module, "archive", archive_)
    monkeypatch.setattr(module, "run_stage", run_stage)
    module.main(module.Config(_cli_parse_args=[], output=output, loss="l2", steps=2))

    assert [call[0] for call in calls] == list(module.STAGES)
    assert all(call[1] == 0.0 for call in calls)
    assert all(call[2] == 7.2e-7 for call in calls)
    assert torch.equal(calls[1][3], torch.ones((1, 3, 3)))
    assert np.array_equal(calls[1][4], np.ones((1, 3)))
    assert torch.equal(calls[3][3], torch.full((1, 3, 3), 3))
    assert torch.equal(calls[3][5]["parent_axes"], torch.tensor([[1.0, 0.0, 0.0]]))
    assert all(call[5]["activation_model"] == "strain" for call in calls)
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["activation_model"] == "strain"
    assert protocol["loss"]["normal_weight"] == 0.0
    status = json.loads((output / "chain-status.json").read_text())
    assert status["status"] == "completed"
    assert status["current_stage"] is None
    assert [stage["status"] for stage in status["stages"]] == ["completed"] * 4
    assert [stage["budget"] for stage in status["stages"]] == [2] * 4
    assert (output / "l2-symmetric6" / "gradient-balance.json").is_file()


def test_normal_weight_matches_the_declared_physical_balance() -> None:
    module = _module()
    assert module.normal_weight("l2") == 0.0
    assert module.normal_weight("l2-normal") == pytest.approx(1.0)
