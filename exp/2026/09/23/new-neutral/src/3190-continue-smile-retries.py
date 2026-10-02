# Copyright (c) 2026 liblaf
"""Continue audited Smile 008 with bounded numerical trial rejection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from datetime import UTC, datetime
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

PARENT_SHA256 = {
    "checkpoint.pt": "0174fbe53d59eec46578f7822886258f17336ea01bae275642d8eca2043f9e48",
    "progress.jsonl": "a5ca1a4aa5e69b3af2e0f5a4c90e9da510085653d6750e92a1f36899a263cf42",
    "endpoint.npz": "35379a2a063b69e608a1ae9ef140da69562434b7018ae736068e145419190dcc",
    "protocol.json": "fd595b877ba7c705a7ce9b6a169fcc3bf425305e3249c2d6dde9246f6473793d",
    "summary.json": "9e085ac37acba4eff2729f39c265a11e59302c062ae69dbc152be640c303307e",
    "independent-audit.json": "a41b4fb3e29995e7bef138e5b3576400d3169ef038fb5b6f9945f9fbeefd78ad",
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
SUCCESSOR_SHA256 = "35b1915686169668e678857745f2c7a298eab68f2ad7946c16693b9b201b2173"
assert sha256(SRC / "3180-inverse-smile-retries.py") == SUCCESSOR_SHA256
runner = load_module(
    "smile_bounded_retry_runner", SRC / "3180-inverse-smile-retries.py"
)


class Config(launcher.Config):
    """Use the exact saved 008 state and unchanged physical policy."""

    output_dir: Path = GROUP / "data/inverse-smile-coupled-009"
    initialization_checkpoint: Path | None = None
    continue_optimizer_state: bool = True
    convergence_patience: int = 0
    initial_trial_alpha: float = 0.00048828125
    maximum_iterations: int = 25
    wall_seconds: float | None = 3600
    deadline_iso_utc: str | None = None
    maximum_projection_failures: int = 3
    maximum_forward_failures: int = 3
    maximum_adjoint_failures: int = 1
    parent_audit_path: Path | None = None
    expected_parent_audit_sha256: str = ""
    expected_parent_checkpoint_sha256: str = ""
    expected_parent_progress_sha256: str = ""
    alpha_certificate_path: Path = GROUP / "data/smile009-alpha-certificate.json"
    alpha_summary_path: Path = GROUP / "data/remote-smile-alpha-summary-009.json"
    expected_alpha_certificate_sha256: str = (
        "9e58dea1556cfff00ae5c3a57e6d8fb58ae1c5506f31fc5ef4c3b2d889d63db4"
    )
    successor_worker_sha256: str = SUCCESSOR_SHA256


def preflight(cfg: Config) -> dict:  # noqa: PLR0915
    """Bind the last completed 008 state before any CUDA work or output creation."""
    assert not torch.cuda.is_initialized()
    assert cfg.successor_worker_sha256 == SUCCESSOR_SHA256
    assert (
        cfg.expected_alpha_certificate_sha256
        == "9e58dea1556cfff00ae5c3a57e6d8fb58ae1c5506f31fc5ef4c3b2d889d63db4"
    )
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
    assert output.name == "inverse-smile-coupled-009"
    assert checkpoint.name == "checkpoint.pt"
    assert parent.name == "inverse-smile-coupled-008"
    for name, expected in PARENT_SHA256.items():
        assert sha256(parent / name) == expected, name
    assert (
        sha256(cfg.alpha_certificate_path)
        == "9e58dea1556cfff00ae5c3a57e6d8fb58ae1c5506f31fc5ef4c3b2d889d63db4"
    )
    assert (
        sha256(cfg.alpha_summary_path)
        == "5390ac281d12fd981dbc5d93edce30193ec2390caec603e93a5e04a5af8062e4"
    )
    alpha_summary = json.loads(cfg.alpha_summary_path.read_text())
    alpha_certificate = json.loads(cfg.alpha_certificate_path.read_text())
    cache = parent / "joint-projections/00005/model/cache/coefficients.npz"
    assert (
        sha256(cache)
        == alpha_summary["cache"]["sha256"]
        == "6d3acdd2c17d80e707d53ac8a7d298edb5df5c35513a5206934a08a926604bed"
    )
    assert alpha_summary["status"] == "certified_joint_increment"
    assert alpha_summary["alpha"] == cfg.initial_trial_alpha == 0.00048828125
    for name, expected in alpha_certificate.items():
        assert alpha_summary["projection"][name] == expected, name
    assert alpha_certificate["certified"] is True
    assert alpha_summary["actual_joint_increment_slope"] < 0
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
    assert protocol["config"]["continue_optimizer_state"] is True
    for name, expected in SOURCE_SHA256.items():
        assert sha256(parent / "sources/new-neutral" / name) == expected, name
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    with np.load(parent / "endpoint.npz", allow_pickle=False) as archive:
        for name in ("activation_inv", "pose_rad_m", "displacement_m"):
            np.testing.assert_array_equal(archive[name], state[name].numpy())
        active_ids = np.asarray(archive["active_cell_ids"])
    assert state["iteration"] == state["local_iteration"] == 4
    assert state["optimizer_steps"] == {"q": 48, "pose": 48}
    assert state["optimizer_step"] == 48
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
    saved_rows = [row for row in progress if row["iteration"] == 4]
    assert len(saved_rows) == 1
    saved = saved_rows[0]
    assert saved["optimizer_steps"] == state["optimizer_steps"]
    assert saved["valid_forward"] is True
    assert progress[-1]["iteration"] == 4
    summary = json.loads((parent / "summary.json").read_text())
    assert summary["endpoint"]["sha256"] == hashes["endpoint.npz"]
    assert summary["final"] == saved
    uncheckpointed = [row for row in progress if row["iteration"] > 4]
    assert not uncheckpointed
    return {
        "schema": "smile-008-bounded-retry-continuation-preflight-v1",
        "parent": str(parent),
        "parent_files_sha256": {
            **hashes,
            "checkpoint.pt": sha256(checkpoint),
            "independent_audit.json": sha256(audit_path),
        },
        "source_sha256": SOURCE_SHA256,
        "last_committed_iteration": 4,
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
        "maximum_new_accepted_updates": 25,
        "maximum_iterations": cfg.maximum_iterations,
        "alpha_certificate": {
            "path": str(cfg.alpha_certificate_path.resolve()),
            "sha256": sha256(cfg.alpha_certificate_path),
        },
        "alpha_summary": {
            "path": str(cfg.alpha_summary_path.resolve()),
            "sha256": sha256(cfg.alpha_summary_path),
        },
        "alpha_cache_sha256": sha256(cache),
        "fresh_forward_and_nonlinear_armijo_required": True,
        "cuda_initialized": torch.cuda.is_initialized(),
    }


def main(cfg: Config) -> None:
    # BaseSettings would otherwise parse this process's recovery-only CLI flags.
    original = launcher.Config(_cli_parse_args=False)
    changed_fields = {
        "output_dir",
        "initialization_checkpoint",
        "continue_optimizer_state",
        "convergence_patience",
        "initial_trial_alpha",
        "wall_seconds",
        "deadline_iso_utc",
        "maximum_iterations",
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
    assert cfg.maximum_iterations == 25
    assert cfg.deadline_iso_utc is not None
    deadline = datetime.fromisoformat(cfg.deadline_iso_utc)
    assert deadline.tzinfo is not None and deadline > datetime.now(UTC)
    assert cfg.wall_seconds is not None and 0 < cfg.wall_seconds <= 3600
    assert cfg.maximum_projection_failures == 3
    assert cfg.maximum_forward_failures == 3
    assert cfg.maximum_adjoint_failures == 1
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
    assert cfg.initial_trial_alpha == 0.00048828125
    assert cfg.minimum_trial_alpha == 1e-6
    assert cfg.gradient_check_epsilons == (1e-4, 1e-5, 1e-6, 3e-7, 1e-7)
    receipt = preflight(cfg)
    receipt["initial_trial_alpha"] = cfg.initial_trial_alpha
    receipt["physical_policy_unchanged"] = True
    receipt["objective_unchanged"] = True
    receipt["successor_worker_sha256"] = SUCCESSOR_SHA256
    receipt["authorization"] = (
        "User explicitly requested continuing Smile after the prior deadline and allowed a few forward/adjoint failures"
    )
    receipt["retry_policy"] = {
        "projection": 3,
        "forward": 3,
        "adjoint": 1,
        "per": "outer iteration",
        "failed_trials_commit_state": False,
    }
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
