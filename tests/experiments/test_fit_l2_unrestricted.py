"""CPU orchestration checks for the single-stage L2 fitting entrypoint."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "exp/2026/09/21/stress-activation-loss/src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from run_support import run_stage  # noqa: E402


def _entrypoint():
    spec = importlib.util.spec_from_file_location(
        "fit_l2_unrestricted", SOURCE / "41-fit-l2-unrestricted.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_explicit_zero_smoothness_starts_fresh_without_pilot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fit = _entrypoint()
    reference = tmp_path / "reference"
    reference.mkdir()
    protocol = {"fixture": "fake", "sources": {}}
    (reference / "protocol.json").write_text(json.dumps(protocol))
    checks = {
        "passed": True,
        "source_protocol": fit.receipt(reference / "protocol.json"),
    }
    (reference / "checks.json").write_text(json.dumps(checks))
    captured = {}

    class Physics:
        def __init__(self) -> None:
            self.ids = [0, 1]
            self.points = np.zeros((3, 3))
            self.material_spec = {"fake": True}
            self.forward_tolerance = 1.0

    constructed = []

    class Study:
        def __init__(self, activation_model: str | None = None) -> None:
            constructed.append(activation_model)
            self.physics = Physics()

        def save_geometry(self, path: Path) -> None:
            np.savez(path, points=self.physics.points)

    def run_stage(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"last_step": None, "last_metrics": None}

    def output(path: Path) -> Path:
        return Path(path)

    def input_(path: Path) -> Path:
        return Path(path)

    def log_metrics(_values: dict) -> None:
        pass

    def archive_(_out: Path) -> dict:
        (_out / "sources.json").write_text("{}")
        return {}

    def pilot_must_not_run(*_args: object) -> None:
        raise AssertionError

    monkeypatch.setattr(fit.cherries, "output", output)
    monkeypatch.setattr(fit.cherries, "input", input_)
    monkeypatch.setattr(fit.cherries, "log_metrics", log_metrics)
    monkeypatch.setattr(fit, "archive", archive_)
    monkeypatch.setattr(fit, "StressStudy", Study)
    monkeypatch.setattr(fit, "run_stage", run_stage)
    monkeypatch.setattr(
        fit,
        "calibrate_current_runtime",
        pilot_must_not_run,
    )

    out = tmp_path / "out"
    fit.main(
        fit.Config(
            _cli_parse_args=[],
            output=out,
            mechanics_reference=reference,
            smooth_weight=0.0,
            activation_model="strain",
            steps=0,
        )
    )

    args, kwargs = captured["args"], captured["kwargs"]
    assert args[2:5] == ("symmetric6", 0.0, 0.0)
    assert np.array_equal(args[5].numpy(), np.zeros((2, 3, 3)))
    assert np.array_equal(args[6], np.zeros((3, 3)))
    assert kwargs["learning_rate"] == 0.05
    assert kwargs["resume_checkpoint"] is None
    assert kwargs["activation_model"] == "strain"
    assert constructed == ["strain"]
    selection = json.loads((out / "calibration.json").read_text())
    assert selection["calibration_status"] == "not_run"
    assert selection["selected_weight"] == 0.0
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["activation_model"] == "strain"
    assert (
        protocol["activation_units"] == "dimensionless active strain S with B = I + S"
    )
    assert "activation_reference_MPa" not in protocol


def test_runner_rejects_cross_model_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stress.pt"
    torch.save(
        {"mode": "symmetric6", "activation_model": "stress", "step": 0}, checkpoint
    )
    with pytest.raises(AssertionError):
        run_stage(
            object(),
            tmp_path / "strain-stage",
            "symmetric6",
            0.0,
            0.0,
            torch.zeros((1, 3, 3)),
            np.zeros((1, 3)),
            steps=1,
            activation_model="strain",
            resume_checkpoint=checkpoint,
        )
