# Copyright (c) 2026 liblaf
"""Continue audited Smile 003 as Smile 006 with isolated baseline config parsing."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
SRC = GROUP / "src"
SOURCE_SHA256 = {
    "3000-inverse-smile-expression-gradient.py": "a86e4608da182dc90b5a0fc253aeb5a1c6f63d4ac304df03ec5a6929ab51475e",
    "3010-run-smile-expression-gradient.py": "956dbe4ef31d38f6d908701855d4411aa190cea08c76d6c4c48cdc883b6edcfc",
    "expression_gradient_check.py": "4cfacce63707334299861d4838f9345f0f3887d8d812defbf473062705ac40a9",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


for source_name, expected in SOURCE_SHA256.items():
    assert sha256(SRC / source_name) == expected, source_name
launcher = load_module(
    "reviewed_smile_003_launcher", SRC / "3010-run-smile-expression-gradient.py"
)
runner = launcher.runner


class Config(launcher.Config):
    """Use the exact saved 003 state and the reviewed 003 physical policy."""

    output_dir: Path = GROUP / "data/inverse-smile-coupled-006"
    initialization_checkpoint: Path | None = None
    continue_optimizer_state: bool = True
    convergence_patience: int = 0
    parent_audit_path: Path | None = None
    expected_parent_audit_sha256: str = ""
    expected_parent_checkpoint_sha256: str = ""
    expected_parent_progress_sha256: str = ""


def preflight(cfg: Config) -> dict:  # noqa: PLR0915
    """Bind the last completed 003 state before any CUDA work or output creation."""
    assert not torch.cuda.is_initialized()
    assert cfg.initialization_checkpoint is not None
    assert cfg.parent_audit_path is not None
    assert len(cfg.expected_parent_audit_sha256) == 64
    assert len(cfg.expected_parent_checkpoint_sha256) == 64
    assert len(cfg.expected_parent_progress_sha256) == 64
    checkpoint = cfg.initialization_checkpoint.resolve()
    parent = checkpoint.parent
    output = cfg.output_dir.resolve()
    assert parent != output
    assert not output.exists(), output
    assert output.name == "inverse-smile-coupled-006"
    assert checkpoint.name == "checkpoint.pt"
    assert parent.name == "inverse-smile-coupled-003"
    assert sha256(checkpoint) == cfg.expected_parent_checkpoint_sha256
    audit_path = cfg.parent_audit_path.resolve()
    assert sha256(audit_path) == cfg.expected_parent_audit_sha256
    audit = json.loads(audit_path.read_text())
    assert audit["schema"] == "expression-coupled-independent-audit-v1"
    assert audit["expression_name"] == "Smile"
    assert audit["valid_forward"] is True
    assert audit["force"]["converged"] is True
    assert audit["collision"]["feasible"] is True
    files = ("endpoint.npz", "protocol.json", "summary.json", "progress.jsonl")
    hashes = {name: sha256(parent / name) for name in files}
    assert hashes["progress.jsonl"] == cfg.expected_parent_progress_sha256
    for name, audit_key in (
        ("endpoint.npz", "endpoint"),
        ("protocol.json", "protocol"),
        ("summary.json", "summary"),
    ):
        assert audit["inputs"][audit_key]["sha256"] == hashes[name], name
    protocol = json.loads((parent / "protocol.json").read_text())
    assert protocol["expression_name"] == "Smile"
    assert protocol["config"]["objective_mode"] == "l2-normal-smooth"
    assert protocol["config"]["continue_optimizer_state"] is False
    for name, expected in SOURCE_SHA256.items():
        assert sha256(parent / "sources/new-neutral" / name) == expected, name
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    with np.load(parent / "endpoint.npz", allow_pickle=False) as archive:
        for name in ("activation_inv", "pose_rad_m", "displacement_m"):
            np.testing.assert_array_equal(archive[name], state[name].numpy())
        active_ids = np.asarray(archive["active_cell_ids"])
    assert state["iteration"] == state["local_iteration"] == 32
    assert state["optimizer_steps"] == {"q": 32, "pose": 32}
    assert state["optimizer_step"] == 32
    assert len(state["moments"]) == 4
    assert (
        state["moments"][0].shape
        == state["moments"][1].shape
        == state["activation_inv"].shape
    )
    assert (
        state["moments"][2].shape
        == state["moments"][3].shape
        == state["pose_normalized"].shape
    )
    for key in ("activation_inv", "pose_normalized", "pose_rad_m", "displacement_m"):
        assert bool(torch.isfinite(state[key]).all()), key
    assert all(bool(torch.isfinite(moment).all()) for moment in state["moments"])
    scale = torch.tensor(
        [math.pi / 18.0] * 3 + [0.01] * 3, dtype=state["pose_normalized"].dtype
    )
    torch.testing.assert_close(
        state["pose_normalized"] * scale, state["pose_rad_m"], rtol=0, atol=0
    )
    progress = [
        json.loads(line)
        for line in (parent / "progress.jsonl").read_text().splitlines()
    ]
    saved_rows = [row for row in progress if row["iteration"] == 32]
    assert len(saved_rows) == 1
    saved = saved_rows[0]
    assert saved["optimizer_steps"] == state["optimizer_steps"]
    assert saved["valid_forward"] is True
    assert progress[-1]["iteration"] >= 32
    summary = json.loads((parent / "summary.json").read_text())
    assert summary["endpoint"]["sha256"] == hashes["endpoint.npz"]
    assert summary["final"] == saved
    uncheckpointed = [row for row in progress if row["iteration"] > 32]
    assert len(uncheckpointed) <= 1
    if uncheckpointed:
        assert uncheckpointed[0]["iteration"] == 33
    return {
        "schema": "smile-003-disk-full-continuation-preflight-v1",
        "parent": str(parent),
        "parent_files_sha256": {
            **hashes,
            "checkpoint.pt": sha256(checkpoint),
            "independent_audit.json": sha256(audit_path),
        },
        "source_sha256": SOURCE_SHA256,
        "last_committed_iteration": 32,
        "optimizer_steps": dict(state["optimizer_steps"]),
        "preserved_arrays": [
            "activation_inv",
            "pose_normalized",
            "pose_rad_m",
            "displacement_m",
            "moments[0:4]",
        ],
        "active_cell_count": len(active_ids),
        "last_progress_iteration": progress[-1]["iteration"],
        "uncheckpointed_progress_rows": uncheckpointed,
        "uncheckpointed_rows_excluded_from_restart": True,
        "uncheckpointed_rows_may_have_passed_numerical_admission": bool(uncheckpointed),
        "parent_history_sha256": hashes["progress.jsonl"],
        "stationarity_monitor_disabled_across_recovery": True,
        "parent_audit_valid_forward": True,
        "cuda_initialized": torch.cuda.is_initialized(),
    }


def main(cfg: Config) -> None:
    # BaseSettings would otherwise parse this process's 3042-only CLI flags.
    original = launcher.Config(_cli_parse_args=False)
    changed_fields = {
        "output_dir",
        "initialization_checkpoint",
        "continue_optimizer_state",
        "convergence_patience",
        "wall_seconds",
    }
    for name in launcher.Config.model_fields:
        if name not in changed_fields:
            assert getattr(cfg, name) == getattr(original, name), name
    assert cfg.expression_name == "Smile"
    assert cfg.objective_mode == "l2-normal-smooth"
    assert cfg.continue_optimizer_state
    assert not cfg.resume
    assert cfg.initialization_refinement is None
    assert cfg.convergence_patience == 0
    assert cfg.maximum_iterations == 10000
    assert cfg.deadline_iso_utc == "2026-09-30T05:50:00+00:00"
    assert cfg.forward_atol == 1e-8
    assert cfg.internal_forward_atol == 1e-9
    assert cfg.adjoint_relative_shift == cfg.predictor_relative_shift == 1e-5
    assert cfg.learning_rate == 0.002
    assert cfg.pose_learning_rate == 0.1
    assert cfg.maximum_inverted_tetrahedra == 100
    assert cfg.maximum_inverted_rest_volume_fraction == 1e-4
    assert cfg.exclude_fully_fixed_tets
    assert cfg.projection_bounded_joint
    assert cfg.projection_descent_fraction == 0.1
    assert cfg.max_rotation_increment_deg is None
    assert cfg.max_translation_increment_m is None
    assert cfg.seed_method == "coupled_tangent"
    assert cfg.q_only_iterations == 0
    assert cfg.adaptive_trial_alpha
    assert cfg.initial_trial_alpha == 0.0625
    assert cfg.minimum_trial_alpha == 1e-6
    assert cfg.gradient_check_epsilons == (1e-4, 1e-5, 1e-6, 3e-7, 1e-7)
    receipt = preflight(cfg)
    control = cfg.output_dir.resolve().with_name(
        cfg.output_dir.name + "-recovery-control"
    )
    control.mkdir(exist_ok=False)
    (control / "preflight.json").write_text(json.dumps(receipt, indent=2) + "\n")
    bounded = cfg.model_copy(
        update={"wall_seconds": launcher.remaining_wall_seconds(cfg)}
    )
    runner.main(bounded)


if __name__ == "__main__":
    cherries.main(main, profile=runner.ProfileJoint)
