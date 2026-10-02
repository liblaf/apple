# Copyright (c) 2026 liblaf
# ruff: noqa: C901, EM101, EM102, PLR0912, TRY003
"""Measure step-512 repeatability before a fixed learning-rate comparison."""

from __future__ import annotations

import copy
import importlib.util
import itertools
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np
import pydantic_settings as ps
import torch
from continuation_helpers import (
    BASE,
    initial_replay_metrics,
    load_checkpoint,
    verify_parent_sources,
)
from experiment_profile import ProfileCometNoCommit
from face_physics import FacePhysics, configure

from liblaf import cherries

ROOT = Path(__file__).resolve().parent.parent
STEP512 = ROOT / "data/92-fit512/optimizer-latest.pt"
EXPECTED_CHECKPOINT_SHA256 = (
    "3864ed0bef1c7f71ab7e4384653afbd0690910225eb67ee677fe23c4184a548a"
)
EXPECTED_PROJECTED_GRADIENT_ANCHOR_SHA256 = (
    "8133d0f14f68d5eaea4867119858377e097701f6949568cd661a426a831f12c7"
)
DEFAULT_OUTPUT = ROOT / "data/100-learning-rate-repeatability"
DEFAULT_PROTOCOL = ROOT / "docs/101-learning-rate-protocol.md"
SOURCE90 = ROOT / "src/90-audit-repeatability.py"
SAMPLE_COUNT = 3
BASELINE_LR = 0.3
CANDIDATE_LR = 0.6
ADAM_EPS = 0.01
REPLAY_MAX_ABS_TOLERANCE = 1e-10
OWN_UPDATE_NOISE_FRACTION = 0.01
DIRECTION_SIGNAL_NOISE_FRACTION = 0.10
COMPLETED = False


def load_source(name: str, path: Path) -> ModuleType:
    """Load one numbered source without invoking its entry point."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R90 = load_source("repeatability_source90", SOURCE90)


class Config(BASE.Config):
    """One step-512 sampling or fixed-rate aggregation process."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    mode: Literal["sample", "summarize"]
    output_dir: Path = DEFAULT_OUTPUT
    resume: Path = STEP512
    manifest: Path
    source101_protocol: Path = DEFAULT_PROTOCOL
    sample_id: str = "step512-sample0"
    candidate_learning_rate: float = CANDIDATE_LR
    projected_gradient_eta: float = 1.0


def array_receipt(value: np.ndarray | torch.Tensor) -> dict[str, Any]:
    """Record exact type, shape, and contiguous bytes for one array."""
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    array = np.asarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": R90.array_sha256(array),
    }


def resolve_path(manifest_path: Path, raw: Any, label: str) -> Path:
    """Resolve one required manifest-relative path."""
    if not isinstance(raw, str) or not raw:
        raise TypeError(f"{label} path must be a nonempty string")
    return (manifest_path.parent / raw).resolve()


def validate_record(raw: Any, label: str) -> Path:
    """Recompute one complete path/size/hash receipt."""
    if not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256"}:
        raise TypeError(f"{label} must be a complete file receipt")
    path = Path(raw["path"]).resolve()
    if R90.record(path) != raw:
        raise ValueError(f"{label} file receipt differs")
    return path


def optimizer_arrays(state: dict[str, Any]) -> dict[str, np.ndarray]:
    """Expose the exact q/u/m/v/t and active-cell state arrays."""
    groups = state["optimizer"]["param_groups"]
    if len(groups) != 1 or len(groups[0]["params"]) != 1:
        raise ValueError("step-512 checkpoint must contain one Adam parameter")
    moment = state["optimizer"]["state"][groups[0]["params"][0]]
    return {
        "q": state["q"].numpy(),
        "u": np.asarray(state["u"]),
        "m": moment["exp_avg"].numpy(),
        "v": moment["exp_avg_sq"].numpy(),
        "t": moment["step"].numpy(),
        "active_ids": np.asarray(state["active_ids"]),
    }


def validate_anchor(raw: Any, state: dict[str, Any]) -> None:
    """Bind the loaded checkpoint to every frozen optimizer-state array."""
    if not isinstance(raw, dict) or raw.get("step") != 512:
        raise ValueError("manifest must contain the step-512 anchor")
    checkpoint = validate_record(raw.get("checkpoint"), "step-512 checkpoint")
    if checkpoint != STEP512.resolve():
        raise ValueError("manifest checkpoint path differs from step 512")
    if raw["checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("step-512 checkpoint SHA-256 differs")
    if int(state["step"]) != 512:
        raise ValueError("loaded optimizer checkpoint is not step 512")
    if state["config"] != raw.get("config"):
        raise ValueError("step-512 checkpoint configuration differs")
    for name, value in optimizer_arrays(state).items():
        if array_receipt(value) != raw.get(name):
            raise ValueError(f"step-512 {name} differs from frozen manifest")
    evidence = raw.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("step-512 evidence inventory is missing")
    for name, value in evidence.items():
        validate_record(value, f"step-512 evidence {name}")


def validate_frozen_inventory(freeze: dict[str, Any]) -> None:
    """Validate every source and ancestor receipt in the immutable freeze."""
    sources = freeze.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("frozen source inventory is missing")
    if not any(name.endswith("100-learning-rate-repeatability.py") for name in sources):
        raise ValueError("source 100 is absent from the frozen inventory")
    for name, value in sources.items():
        validate_record(value, f"source {name}")
    supporting_sources = freeze.get("supporting_sources")
    if not isinstance(supporting_sources, dict) or not supporting_sources:
        raise ValueError("frozen supporting-source inventory is missing")
    for name, value in supporting_sources.items():
        validate_record(value, f"supporting source {name}")
    lineage = freeze.get("lineage")
    if not isinstance(lineage, list) or len(lineage) != 10:
        raise ValueError("frozen parent lineage must contain ten ancestors")
    lineage_steps = []
    for index, item in enumerate(lineage):
        if not isinstance(item, dict) or set(item) != {
            "checkpoint",
            "evidence",
            "step",
        }:
            raise TypeError(f"lineage ancestor {index} schema differs")
        validate_record(item["checkpoint"], f"lineage ancestor {index} checkpoint")
        evidence = item["evidence"]
        if not isinstance(evidence, dict) or set(evidence) != {
            "config.json",
            "final.npz",
            "final.vtu",
            "provenance.json",
            "summary.json",
            "trace.csv",
        }:
            raise ValueError(f"lineage ancestor {index} evidence differs")
        for name, value in evidence.items():
            validate_record(value, f"lineage ancestor {index} evidence {name}")
        if not isinstance(item["step"], int):
            raise TypeError(f"lineage ancestor {index} step must be an integer")
        lineage_steps.append(item["step"])
    if lineage_steps[0] != 512 or any(
        left <= right for left, right in itertools.pairwise(lineage_steps)
    ):
        raise ValueError("frozen parent lineage step order differs")
    validate_record(freeze.get("freeze_builder"), "pre-run freeze builder")


def validate_manifest(
    cfg: Config,
    manifest_path: Path,
    manifest: dict[str, Any],
    state: dict[str, Any],
    provenance: dict[str, Any] | None = None,
) -> None:
    """Validate the complete pre-sample source/input/anchor freeze."""
    if manifest.get("schema_version") != 1:
        raise ValueError("learning-rate sampling manifest schema differs")
    freeze_path = resolve_path(
        manifest_path, manifest.get("prerun_freeze"), "pre-run freeze"
    )
    freeze = R90.read_json(freeze_path)
    if (
        freeze.get("schema_version") != 1
        or freeze.get("status") != "frozen_before_any_fresh_GPU_sample"
        or freeze.get("same_saved_state_required_for_all_three_samples") is not True
    ):
        raise ValueError("learning-rate repeatability pre-run freeze differs")
    if freeze.get("protocol") != R90.record(cfg.source101_protocol.resolve()):
        raise ValueError("source-101 protocol receipt differs")
    if freeze.get("fixed_optimizer_arms") != {
        "baseline": {"adam_eps": ADAM_EPS, "learning_rate": BASELINE_LR},
        "selected": {"adam_eps": ADAM_EPS, "learning_rate": CANDIDATE_LR},
    }:
        raise ValueError("frozen learning-rate settings differ")
    samples_by_step = manifest.get("samples")
    if not isinstance(samples_by_step, dict) or set(samples_by_step) != {"512"}:
        raise ValueError("manifest must contain only step-512 samples")
    samples = samples_by_step["512"]
    if (
        not isinstance(samples, list)
        or len(samples) != SAMPLE_COUNT
        or len(set(samples)) != SAMPLE_COUNT
    ):
        raise ValueError("manifest must predeclare exactly three sample paths")
    for index, value in enumerate(samples):
        resolve_path(manifest_path, value, f"sample {index}")
    anchors = freeze.get("anchors")
    if not isinstance(anchors, dict) or set(anchors) != {"512"}:
        raise ValueError("pre-run freeze must contain only the step-512 anchor")
    validate_anchor(anchors["512"], state)
    projected_gradient_anchor = freeze.get("projected_gradient_anchor")
    validate_record(projected_gradient_anchor, "projected-gradient anchor")
    if projected_gradient_anchor["sha256"] != EXPECTED_PROJECTED_GRADIENT_ANCHOR_SHA256:
        raise ValueError("projected-gradient anchor SHA-256 differs")
    validate_frozen_inventory(freeze)
    sources = freeze["sources"]
    inputs = freeze.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "skin.vtp",
        "summary.json",
        "volume.vtu",
    }:
        raise ValueError("frozen fixture input inventory differs")
    for name, value in inputs.items():
        validate_record(value, f"fixture input {name}")
    if freeze.get("python") != f"Python {sys.version.split()[0]}":
        raise ValueError("frozen Python version differs")
    if provenance is None:
        return
    frozen_python = {
        name: value["sha256"] for name, value in sources.items() if name.endswith(".py")
    }
    frozen_inputs = {
        name: {key: value[key] for key in ("path", "sha256")}
        for name, value in inputs.items()
    }
    for key in ("git_sha", "torch"):
        if provenance[key] != freeze.get(key):
            raise ValueError(f"runtime {key} differs from frozen manifest")
    if provenance["sources"] != frozen_python:
        raise ValueError("Python source hashes differ from frozen manifest")
    if provenance["inputs"] != frozen_inputs:
        raise ValueError("fixture input hashes differ from frozen manifest")


def parent_provenance(cfg: Config) -> dict[str, Any]:
    """Load the completed step-512 source identity."""
    return R90.read_json(cfg.resume.resolve().parent / "provenance.json")


def assert_fixed_config(cfg: Config) -> None:
    """Fail if a CLI override changes the preregistered study."""
    if cfg.resume.resolve() != STEP512.resolve():
        raise ValueError("source 100 only accepts the frozen step-512 checkpoint")
    if (
        cfg.model != "tensor"
        or cfg.learning_rate != BASELINE_LR
        or cfg.adam_eps != ADAM_EPS
        or cfg.candidate_learning_rate != CANDIDATE_LR
        or cfg.smoothness_weight != 0
        or cfg.magnitude_weight != 0
        or cfg.rank_weight != 0
        or cfg.projected_gradient_eta != 1.0
    ):
        raise ValueError("source 100 scientific settings are fixed")


def sample(cfg: Config) -> None:
    """Run one fresh forward/adjoint sample without an optimizer update."""
    assert_fixed_config(cfg)
    if not cfg.sample_id or any(character.isspace() for character in cfg.sample_id):
        raise ValueError("sample-id must be one nonempty token")
    manifest_path = cfg.manifest.resolve()
    manifest = R90.read_json(manifest_path)
    state = load_checkpoint(cfg.resume.resolve(), cfg)
    validate_manifest(cfg, manifest_path, manifest, state)
    declared_outputs = {
        resolve_path(manifest_path, value, "sample")
        for value in manifest["samples"]["512"]
    }
    if cfg.output_dir.resolve() not in declared_outputs:
        raise ValueError("sample output directory was not frozen in the manifest")
    output = R90.prepare_output(cfg.output_dir)
    R90.write_json(output / "config.json", cfg.model_dump(mode="json"))
    provenance = BASE.archive(output, cfg)
    verify_parent_sources(parent_provenance(cfg), provenance)
    validate_manifest(cfg, manifest_path, manifest, state, provenance)
    configure()
    physics = FacePhysics(cfg.fixture, activation_model="tensor")
    if not np.array_equal(physics.ids, state["active_ids"]):
        raise ValueError("active cell IDs differ from the step-512 checkpoint")
    objective = BASE.Objective(physics, cfg)
    q = torch.nn.Parameter(state["q"].to(device="cuda").clone())
    q.grad = None
    result = objective(q, np.asarray(state["u"]).copy())
    if q.grad is None:
        raise RuntimeError("objective did not produce a gradient")
    q_array = q.detach().cpu().numpy()
    u_array = np.asarray(result["u"])
    gradient = q.grad.detach().cpu().numpy()
    if not np.array_equal(q_array, state["q"].numpy()):
        raise ValueError("no-update sample changed q")
    metrics = {
        **{
            key: result[key]
            for key in (
                "data_objective_mm2",
                "smoothness",
                "magnitude",
                "rank_penalty",
                "objective",
                "fit_gradient_rms",
                "gradient_rms",
                "regularizer_gradient_rms",
            )
        },
        **BASE.metrics(physics, q.detach(), result, cfg),
    }
    for label, value in (
        ("q", q_array),
        ("u", u_array),
        ("gradient", gradient),
        ("metrics", metrics),
        ("forward", result["forward"]),
        ("adjoint", result["adjoint"]),
    ):
        R90.require_finite(value, label)
    replay = initial_replay_metrics(metrics, state["parent_endpoint"])
    if not replay["passed"]:
        raise ValueError("sample does not replay the step-512 physical state")
    R90.write_json(output / "initial-replay.json", replay)
    R90.write_json(output / "result.json", R90.result_json(result))
    np.savez_compressed(
        output / "sample.npz",
        q=q_array,
        u=u_array,
        gradient=gradient,
        active_ids=state["active_ids"],
        step=np.asarray(512),
    )
    summary = {
        "schema_version": 1,
        "status": "completed_no_update_sample",
        "mode": "fresh_process_forward_adjoint",
        "sample_id": cfg.sample_id,
        "process_identity": R90.process_identity(),
        "solver_valid": result["forward"]["success"] and result["adjoint"]["success"],
        "checkpoint": R90.checkpoint_receipt(cfg.resume.resolve(), state),
        "sample": R90.record(output / "sample.npz"),
        "result": R90.record(output / "result.json"),
        "initial_replay": replay,
        "metrics": metrics,
        "array_sha256": {
            "q": R90.array_sha256(q_array),
            "u": R90.array_sha256(u_array),
            "gradient": R90.array_sha256(gradient),
            "active_ids": R90.array_sha256(state["active_ids"]),
        },
        "forward": result["forward"],
        "adjoint": result["adjoint"],
        "config": R90.record(output / "config.json"),
        "provenance": provenance,
        "provenance_file": R90.record(output / "provenance.json"),
        "manifest": R90.record(manifest_path),
        "frozen_manifest": R90.record(
            resolve_path(manifest_path, manifest["prerun_freeze"], "pre-run freeze")
        ),
        "scope": "one fresh forward and adjoint; no optimizer update or projection",
    }
    R90.write_json(output / "summary.json", summary)
    for name in (
        "summary.json",
        "sample.npz",
        "result.json",
        "initial-replay.json",
        "config.json",
        "provenance.json",
    ):
        cherries.log_output(output / name)


def load_samples(
    cfg: Config,
    manifest_path: Path,
    manifest: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load and bind exactly three independently executed step-512 samples."""
    expected_checkpoint = R90.checkpoint_receipt(STEP512.resolve(), state)
    expected_manifest = R90.record(manifest_path)
    samples: list[dict[str, Any]] = []
    frozen_manifest = R90.record(
        resolve_path(manifest_path, manifest["prerun_freeze"], "pre-run freeze")
    )
    for index, raw in enumerate(manifest["samples"]["512"]):
        directory = resolve_path(manifest_path, raw, f"sample {index}")
        summary = R90.read_json(directory / "summary.json")
        if (
            summary.get("schema_version") != 1
            or summary.get("status") != "completed_no_update_sample"
            or summary.get("mode") != "fresh_process_forward_adjoint"
            or summary.get("solver_valid") is not True
            or summary.get("initial_replay", {}).get("passed") is not True
            or summary.get("checkpoint") != expected_checkpoint
            or summary.get("manifest") != expected_manifest
            or summary.get("frozen_manifest") != frozen_manifest
        ):
            raise ValueError(f"sample {index} contract differs: {directory}")
        provenance = R90.read_json(directory / "provenance.json")
        if summary.get("provenance") != provenance:
            raise ValueError(f"sample {index} provenance differs")
        validate_manifest(cfg, manifest_path, manifest, state, provenance)
        records: dict[str, dict[str, Any]] = {}
        for name, key in (
            ("sample.npz", "sample"),
            ("result.json", "result"),
            ("config.json", "config"),
            ("provenance.json", "provenance_file"),
        ):
            records[name] = R90.record(directory / name)
            if records[name] != summary.get(key):
                raise ValueError(f"sample {index} {name} receipt differs")
        records["summary.json"] = R90.record(directory / "summary.json")
        records["initial-replay.json"] = R90.record(directory / "initial-replay.json")
        if (
            R90.read_json(directory / "initial-replay.json")
            != summary["initial_replay"]
        ):
            raise ValueError(f"sample {index} initial replay file differs")
        result = R90.read_json(directory / "result.json")
        if result.get("forward") != summary.get("forward") or result.get(
            "adjoint"
        ) != summary.get("adjoint"):
            raise ValueError(f"sample {index} solver receipt differs")
        with np.load(directory / "sample.npz", allow_pickle=False) as saved:
            arrays = {
                name: saved[name].copy()
                for name in ("q", "u", "gradient", "active_ids")
            }
            step = int(saved["step"])
        if step != 512:
            raise ValueError(f"sample {index} array step differs")
        for name, value in arrays.items():
            R90.require_finite(value, f"sample {index} {name}")
            if R90.array_sha256(value) != summary["array_sha256"][name]:
                raise ValueError(f"sample {index} {name} hash differs")
        if not np.array_equal(arrays["q"], state["q"].numpy()):
            raise ValueError(f"sample {index} q differs from checkpoint")
        if not np.array_equal(arrays["active_ids"], state["active_ids"]):
            raise ValueError(f"sample {index} active IDs differ from checkpoint")
        samples.append(
            {
                "directory": directory,
                "summary": summary,
                "arrays": arrays,
                "records": records,
            }
        )
    identities = {
        tuple(
            sample["summary"]["process_identity"][key]
            for key in (
                "hostname",
                "boot_id",
                "pid",
                "start_ticks_since_boot",
            )
        )
        for sample in samples
    }
    sample_ids = {sample["summary"]["sample_id"] for sample in samples}
    configs = [R90.read_json(sample["directory"] / "config.json") for sample in samples]
    config_keys = (
        "fixture",
        "model",
        "stress_reference_mpa",
        "stress_cap_mpa",
        "smooth_length_m",
        "smoothness_weight",
        "magnitude_weight",
        "rank_weight",
        "learning_rate",
        "adam_eps",
        "candidate_learning_rate",
        "projected_gradient_eta",
        "resume",
        "manifest",
        "source101_protocol",
    )
    if len(identities) != SAMPLE_COUNT or len(sample_ids) != SAMPLE_COUNT:
        raise ValueError("samples must come from three distinct processes and IDs")
    if any(
        any(config[key] != configs[0][key] for key in config_keys)
        for config in configs[1:]
    ):
        raise ValueError("sample scientific configurations differ")
    return samples


@torch.no_grad()
def closed_form_trial(
    initial_q: torch.Tensor,
    direction: torch.Tensor,
    learning_rate: float,
    maximum: float,
    qref: float,
) -> dict[str, Any]:
    """Retain both stages of the closed-form Adam and production projection."""
    trial = R90.closed_form_trial(initial_q, direction, learning_rate, maximum, qref)
    unprojected_delta = learning_rate * direction
    trial["unprojected_delta_q"] = unprojected_delta.cpu().numpy()
    trial["unprojected_delta_Q"] = (
        (qref * R90.matrices(unprojected_delta)).cpu().numpy()
    )
    return trial


@torch.no_grad()
def installed_trial(
    state: dict[str, Any],
    gradient: torch.Tensor,
    learning_rate: float,
    epsilon: float,
    maximum: float,
    qref: float,
) -> dict[str, Any]:
    """Retain installed Adam output before and after production projection."""
    q = torch.nn.Parameter(state["q"].clone())
    optimizer = torch.optim.Adam([q], lr=learning_rate, eps=epsilon)
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    optimizer.param_groups[0]["lr"] = learning_rate
    optimizer.param_groups[0]["eps"] = epsilon
    q.grad = gradient.clone()
    optimizer.step()
    unprojected_delta = q.detach() - state["q"]
    clipping = R90.clipping_signature(q.detach(), maximum)
    projection = R90.project(q, maximum)
    delta = q.detach() - state["q"]
    delta_q = delta.cpu().numpy()
    delta_Q = (qref * R90.matrices(delta)).cpu().numpy()
    return {
        "unprojected_delta_q": unprojected_delta.cpu().numpy(),
        "unprojected_delta_Q": (qref * R90.matrices(unprojected_delta)).cpu().numpy(),
        "delta_q": delta_q,
        "delta_Q": delta_Q,
        "clipping": clipping,
        "metrics": {
            "learning_rate": learning_rate,
            "physical_update_rms_mpa": R90.tensor_rms_mpa(delta, qref),
            "physical_update_frobenius_max_mpa": float(
                np.linalg.norm(delta_Q, axis=(1, 2)).max()
            ),
            "actual_update_rms": R90.rms(delta),
            "unprojected_update_rms": R90.rms(unprojected_delta),
            **projection,
        },
        "adam_step_after": int(optimizer.state[q]["step"]),
    }


def update_stage_receipt(
    initial_q: np.ndarray,
    delta_q: np.ndarray,
    delta_Q: np.ndarray,
) -> dict[str, Any]:
    """Hash one complete normalized and physical full-field update stage."""
    per_cell = np.linalg.norm(delta_Q, axis=(1, 2))
    return {
        "q_sha256": R90.array_sha256(initial_q + delta_q),
        "delta_q_sha256": R90.array_sha256(delta_q),
        "delta_Q_sha256": R90.array_sha256(delta_Q),
        "normalized_update_rms": float(np.sqrt(np.mean(delta_q**2))),
        "normalized_update_max_abs": float(np.abs(delta_q).max()),
        "physical_update_rms_mpa": float(np.sqrt(np.mean(per_cell**2))),
        "physical_update_frobenius_max_mpa": float(per_cell.max()),
    }


def replay_trial_receipt(
    initial_q: np.ndarray, trial: dict[str, Any]
) -> dict[str, Any]:
    """Describe both stages of one same-gradient optimizer replay."""
    return {
        "unprojected": update_stage_receipt(
            initial_q,
            trial["unprojected_delta_q"],
            trial["unprojected_delta_Q"],
        ),
        "projected": update_stage_receipt(
            initial_q, trial["delta_q"], trial["delta_Q"]
        ),
        "adam_step_after": trial.get("adam_step_after"),
    }


def same_gradient_replay(
    state: dict[str, Any],
    gradient: np.ndarray,
    closed: dict[str, Any],
    installed: dict[str, Any],
    repeated: dict[str, Any],
) -> dict[str, Any]:
    """Compare all three serialized-gradient replay paths pairwise."""
    named = [
        ("closed_form_adam_with_installed_projection", closed),
        ("installed_adam_with_installed_projection", installed),
        ("repeated_installed_adam_with_installed_projection", repeated),
    ]
    comparisons = []
    for (left_name, left), (right_name, right) in itertools.combinations(named, 2):
        unprojected_max_abs = float(
            np.abs(left["unprojected_delta_q"] - right["unprojected_delta_q"]).max()
        )
        projected_max_abs = float(np.abs(left["delta_q"] - right["delta_q"]).max())
        comparisons.append(
            {
                "left": left_name,
                "right": right_name,
                "unprojected_max_abs_q": unprojected_max_abs,
                "projected_max_abs_q": projected_max_abs,
                "unprojected_physical_delta_Q": R90.tensor_difference(
                    left["unprojected_delta_Q"], right["unprojected_delta_Q"]
                ),
                "projected_physical_delta_Q": R90.tensor_difference(
                    left["delta_Q"], right["delta_Q"]
                ),
                "passed": unprojected_max_abs < REPLAY_MAX_ABS_TOLERANCE
                and projected_max_abs < REPLAY_MAX_ABS_TOLERANCE,
            }
        )
    return {
        "gradient_sha256": R90.array_sha256(gradient),
        "same_serialized_gradient_and_cloned_q_m_v_t": True,
        "max_abs_q_tolerance": REPLAY_MAX_ABS_TOLERANCE,
        "runs": {
            name: replay_trial_receipt(state["q"].numpy(), trial)
            for name, trial in named
        },
        "pairwise": comparisons,
        "passed": all(row["passed"] for row in comparisons),
    }


def arm_updates(
    state: dict[str, Any],
    samples: list[dict[str, Any]],
    learning_rate: float,
    qref: float,
    maximum: float,
) -> dict[str, Any]:
    """Replay one fixed rate on all sample gradients and quantify variation."""
    closed_trials = []
    installed_trials = []
    agreements = []
    for sample_item in samples:
        gradient = torch.as_tensor(sample_item["arrays"]["gradient"])
        direction = R90.next_adam_direction(state, gradient, ADAM_EPS)
        closed = closed_form_trial(state["q"], direction, learning_rate, maximum, qref)
        installed = installed_trial(
            state, gradient, learning_rate, ADAM_EPS, maximum, qref
        )
        closed_trials.append(closed)
        installed_trials.append(installed)
        agreements.append(R90.installed_agreement(closed, installed, qref))
    repeated = installed_trial(
        state,
        torch.as_tensor(samples[0]["arrays"]["gradient"]),
        learning_rate,
        ADAM_EPS,
        maximum,
        qref,
    )
    installed_repeatability = R90.installed_repeatability(installed_trials[0], repeated)
    replay = same_gradient_replay(
        state,
        samples[0]["arrays"]["gradient"],
        closed_trials[0],
        installed_trials[0],
        repeated,
    )
    pairs = R90.pairwise_arrays([trial["delta_Q"] for trial in closed_trials], "tensor")
    clippings = [trial["clipping"] for trial in closed_trials]
    update_rms = [
        trial["metrics"]["physical_update_rms_mpa"] for trial in closed_trials
    ]
    for row, (left, right) in zip(
        pairs, itertools.combinations(clippings, 2), strict=True
    ):
        row["clipping_disagreement"] = R90.clipping_disagreement(left, right)
    for row, (left_rms, right_rms) in zip(
        pairs, itertools.combinations(update_rms, 2), strict=True
    ):
        row["noise_over_left_update"] = (
            row["rms_mpa"] / left_rms if left_rms > 0 else None
        )
        row["noise_over_right_update"] = (
            row["rms_mpa"] / right_rms if right_rms > 0 else None
        )
    return {
        "closed_trials": closed_trials,
        "installed_agreements": agreements,
        "installed_repeatability": installed_repeatability,
        "same_gradient_replay": replay,
        "pairs": pairs,
        "sample0_update_rms_mpa": update_rms[0],
        "max_pairwise_update_rms_mpa": max(row["rms_mpa"] for row in pairs),
    }


def arm_receipt(value: dict[str, Any], signal_rms_mpa: float) -> dict[str, Any]:
    """Remove full arrays while retaining every replay and noise gate."""
    own = value["sample0_update_rms_mpa"]
    noise = value["max_pairwise_update_rms_mpa"]
    own_fraction = noise / own if own > 0 else None
    signal_fraction = noise / signal_rms_mpa if signal_rms_mpa > 0 else None
    replay_passed = (
        all(receipt["passed"] for receipt in value["installed_agreements"])
        and value["installed_repeatability"]["passed"]
        and value["same_gradient_replay"]["passed"]
    )
    passed = (
        replay_passed
        and own_fraction is not None
        and own_fraction <= OWN_UPDATE_NOISE_FRACTION
        and signal_fraction is not None
        and signal_fraction <= DIRECTION_SIGNAL_NOISE_FRACTION
    )
    return {
        "sample0_physical_update_rms_mpa": own,
        "sample0_update": R90.trial_receipt(value["closed_trials"][0]),
        "same_gradient_installed_agreement": value["installed_agreements"],
        "sample0_installed_repeatability": value["installed_repeatability"],
        "sample0_same_gradient_replay": value["same_gradient_replay"],
        "pairwise_projected_update": value["pairs"],
        "max_pairwise_physical_delta_Q_rms_mpa": noise,
        "noise_over_own_sample0_update": own_fraction,
        "noise_over_baseline_candidate_signal": signal_fraction,
        "same_gradient_replay_passed": replay_passed,
        "passed": passed,
    }


def checkpoint_noise(
    state: dict[str, Any], samples: list[dict[str, Any]], qref: float, maximum: float
) -> dict[str, Any]:
    """Measure fixed-rate update noise and baseline/candidate separation."""
    raw = {
        "baseline": arm_updates(state, samples, BASELINE_LR, qref, maximum),
        "selected": arm_updates(state, samples, CANDIDATE_LR, qref, maximum),
    }
    baseline0 = raw["baseline"]["closed_trials"][0]
    selected0 = raw["selected"]["closed_trials"][0]
    difference = R90.tensor_difference(baseline0["delta_Q"], selected0["delta_Q"])
    signal = difference["rms_mpa"]
    clipping = R90.clipping_disagreement(baseline0["clipping"], selected0["clipping"])
    arms = {name: arm_receipt(value, signal) for name, value in raw.items()}
    passed = (
        signal > 0
        and np.isfinite(signal)
        and all(value["passed"] for value in arms.values())
    )
    return {
        "settings": {
            "baseline": {"adam_eps": ADAM_EPS, "learning_rate": BASELINE_LR},
            "selected": {"adam_eps": ADAM_EPS, "learning_rate": CANDIDATE_LR},
        },
        "gradient_pairwise": R90.pairwise_arrays(
            [sample["arrays"]["gradient"] for sample in samples], "vector"
        ),
        "u_pairwise": R90.pairwise_arrays(
            [sample["arrays"]["u"] for sample in samples], "vector"
        ),
        "baseline_candidate_update_difference": difference,
        "baseline_candidate_clipping_disagreement": clipping,
        "baseline_candidate_direction_difference_rms_mpa": signal,
        "signal_is_nonzero": signal > 0,
        "arms": arms,
        "passed": passed,
    }


def summarize(cfg: Config) -> None:
    """Aggregate three samples and freeze the two fixed learning-rate updates."""
    assert_fixed_config(cfg)
    manifest_path = cfg.manifest.resolve()
    manifest = R90.read_json(manifest_path)
    state = load_checkpoint(cfg.resume.resolve(), cfg)
    validate_manifest(cfg, manifest_path, manifest, state)
    samples = load_samples(cfg, manifest_path, manifest, state)
    output = R90.prepare_output(cfg.output_dir)
    R90.write_json(output / "config.json", cfg.model_dump(mode="json"))
    provenance = BASE.archive(output, cfg)
    verify_parent_sources(parent_provenance(cfg), provenance)
    validate_manifest(cfg, manifest_path, manifest, state, provenance)
    qref = cfg.stress_reference_mpa
    maximum = cfg.stress_cap_mpa / qref
    noise = checkpoint_noise(state, samples, qref, maximum)
    first = samples[0]
    np.savez_compressed(
        output / "initial-gradient.npz",
        q=first["arrays"]["q"],
        u=first["arrays"]["u"],
        gradient=first["arrays"]["gradient"],
        active_ids=first["arrays"]["active_ids"],
        step=np.asarray(512),
    )
    (output / "initial-result.json").write_bytes(
        (first["directory"] / "result.json").read_bytes()
    )
    protocol = R90.record(cfg.source101_protocol.resolve())
    area_fit = [sample["summary"]["metrics"]["area_fit_rms_mm"] for sample in samples]
    unweighted_fit = [sample["summary"]["metrics"]["fit_rms_mm"] for sample in samples]
    baseline_rms = noise["arms"]["baseline"]["sample0_physical_update_rms_mpa"]
    selected_rms = noise["arms"]["selected"]["sample0_physical_update_rms_mpa"]
    passed = noise["passed"]
    summary = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "gate_passed": passed,
        "mode": "learning_rate_repeatability_aggregate",
        "rule": {
            "checkpoint": 512,
            "sample_count": SAMPLE_COUNT,
            "solver_valid_required": True,
            "same_gradient_optimizer_replay_max_abs_q": (REPLAY_MAX_ABS_TOLERANCE),
            "max_pairwise_noise_over_own_sample0_update": (OWN_UPDATE_NOISE_FRACTION),
            "max_pairwise_noise_over_baseline_candidate_direction_difference": (
                DIRECTION_SIGNAL_NOISE_FRACTION
            ),
            "baseline_candidate_signal_must_be_nonzero": True,
            "rates_fixed_before_outcomes": True,
            "calibration_or_bisection_used": False,
            "geometry_used_for_gate": False,
        },
        "baseline": {
            "adam_eps": ADAM_EPS,
            "learning_rate": BASELINE_LR,
            "physical_update_rms_mpa": baseline_rms,
        },
        "selected": {
            "adam_eps": ADAM_EPS,
            "learning_rate": CANDIDATE_LR,
            "physical_update_rms_mpa": selected_rms,
        },
        "parent_checkpoint": R90.checkpoint_receipt(STEP512.resolve(), state),
        "initial_gradient": {
            **R90.record(output / "initial-gradient.npz"),
            "step": 512,
            "q_sha256": R90.array_sha256(first["arrays"]["q"]),
            "u_sha256": R90.array_sha256(first["arrays"]["u"]),
            "gradient_sha256": R90.array_sha256(first["arrays"]["gradient"]),
            "active_ids_sha256": R90.array_sha256(first["arrays"]["active_ids"]),
            "source_sample": first["records"]["summary.json"],
            "source_sample_provenance": first["records"]["provenance.json"],
        },
        "initial_result": {
            **R90.record(output / "initial-result.json"),
            "source_sample": first["records"]["result.json"],
            "copied_byte_for_byte": R90.sha256(output / "initial-result.json")
            == R90.sha256(first["directory"] / "result.json"),
        },
        "fit_rms_range_mm": {
            "definition": "area-weighted fit RMS across three step-512 samples",
            "minimum": min(area_fit),
            "maximum": max(area_fit),
            "range": max(area_fit) - min(area_fit),
        },
        "unweighted_fit_rms_range_mm": {
            "minimum": min(unweighted_fit),
            "maximum": max(unweighted_fit),
            "range": max(unweighted_fit) - min(unweighted_fit),
        },
        "checkpoints": {
            "512": {
                "samples": [sample["records"] for sample in samples],
                "noise": noise,
            }
        },
        "source101_protocol": protocol,
        "manifest": R90.record(manifest_path),
        "frozen_manifest": R90.record(
            resolve_path(manifest_path, manifest["prerun_freeze"], "pre-run freeze")
        ),
        "sources": provenance["sources"],
        "inputs": {
            "fixture": provenance["inputs"],
            "step512_checkpoint": R90.record(STEP512),
            "source101_protocol": protocol,
        },
        "provenance": provenance,
        "provenance_file": R90.record(output / "provenance.json"),
        "scope": "CPU aggregation and projected Adam replay at two fixed learning rates; no forward, adjoint, optimizer trajectory, physics, solver-tolerance, or geometry gate change",
    }
    R90.write_json(output / "summary.json", summary)
    for name in (
        "summary.json",
        "initial-gradient.npz",
        "initial-result.json",
        "config.json",
        "provenance.json",
    ):
        cherries.log_output(output / name)
    if not passed:
        raise RuntimeError("step-512 learning-rate repeatability gate failed")


def run(cfg: Config) -> None:
    """Dispatch exactly one isolated source-100 mode."""
    global COMPLETED  # noqa: PLW0603
    if cfg.mode == "sample":
        sample(cfg)
    else:
        summarize(cfg)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(run, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
