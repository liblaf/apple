# Copyright (c) 2026 liblaf
# ruff: noqa: C901, EM101, EM102, TRY003
"""Separate fixed-gradient replay from fresh forward/adjoint repeatability."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import os
import shutil
import socket
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
    rms,
    tensor_rms_mpa,
    verify_parent_sources,
)
from experiment_profile import ProfileCometNoCommit
from face_physics import FacePhysics, configure
from tensor_controls import matrices, project

from liblaf import cherries

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data/90-repeatability"
STEP64 = ROOT / "data/21-psd/optimizer-latest.pt"
STEP256 = ROOT / "data/74-fit256/optimizer-latest.pt"
OLD_CALIBRATION = ROOT / "data/73-face-step-calibration-v2/summary.json"
OLD_GRADIENT = ROOT / "data/73-face-step-calibration-v2/initial-gradient.npz"
OLD_REJECTED_TRIAL = ROOT / "data/74-calibrated16/failed-trial.npz"
DEFAULT_PROTOCOL = ROOT / "docs/91-next-optimizer-protocol.md"
SAMPLE_COUNT = 3
REPLAY_MAX_ABS_TOLERANCE = 1e-10
PHYSICAL_REPLAY_RELATIVE_TOLERANCE = 1e-5
OWN_UPDATE_NOISE_FRACTION = 0.01
DIRECTION_SIGNAL_NOISE_FRACTION = 0.10
COMPLETED = False
SAMPLE_CONFIG_KEYS = (
    "fixture",
    "model",
    "stress_reference_mpa",
    "stress_cap_mpa",
    "smooth_length_m",
    "smoothness_weight",
    "magnitude_weight",
    "rank_weight",
    "projected_gradient_eta",
)


class Config(BASE.Config):
    """One isolated repeatability mode and its immutable inputs."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    mode: Literal["fixed-replay", "sample", "summarize"]
    output_dir: Path = DEFAULT_OUTPUT
    resume: Path = STEP64
    sample_id: str = "sample-01"
    manifest: Path | None = None
    source91_protocol: Path = DEFAULT_PROTOCOL
    frozen_gradient: Path = OLD_GRADIENT
    frozen_calibration_summary: Path = OLD_CALIBRATION
    rejected_trial: Path = OLD_REJECTED_TRIAL
    reduced_eps: float = 1e-6
    physical_step_multiplier: float = 1.0
    projected_gradient_eta: float = 1.0


def sha256(path: Path) -> str:
    """Hash one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash exact contiguous typed-array bytes."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def record(path: Path) -> dict[str, Any]:
    """Describe one immutable file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    """Read an object-only JSON document."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    """Write strict finite JSON atomically."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_output(path: Path) -> Path:
    """Create one new empty evidence directory."""
    output = path.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    return output


def load_source(name: str, path: Path) -> ModuleType:
    """Load one numbered source without invoking its entry point."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parent_provenance(cfg: Config) -> dict[str, Any]:
    """Read the exact source/input identity beside a parent checkpoint."""
    return read_json(cfg.resume.resolve().parent / "provenance.json")


def checkpoint_receipt(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Record the checkpoint and exact controls, seed, and Adam moments."""
    groups = state["optimizer"]["param_groups"]
    moment = state["optimizer"]["state"][groups[0]["params"][0]]
    q = state["q"].numpy()
    u = np.asarray(state["u"])
    return {
        **record(path),
        "step": int(state["step"]),
        "q_shape": list(q.shape),
        "u_shape": list(u.shape),
        "q_sha256": array_sha256(q),
        "u_sha256": array_sha256(u),
        "active_ids_shape": list(state["active_ids"].shape),
        "active_ids_sha256": array_sha256(state["active_ids"]),
        "optimizer_state_count": len(state["optimizer"]["state"]),
        "optimizer_moment_sha256": {
            name: array_sha256(moment[name].numpy())
            for name in ("exp_avg", "exp_avg_sq")
        },
        "optimizer_moment_shape": {
            name: list(moment[name].shape) for name in ("exp_avg", "exp_avg_sq")
        },
        "adam_step": int(moment["step"]),
    }


def process_identity() -> dict[str, Any]:
    """Identify one Linux process independently of its user-facing sample ID."""
    stat_tail = Path("/proc/self/stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
    return {
        "hostname": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id")
        .read_text(encoding="utf-8")
        .strip(),
        "pid": os.getpid(),
        "start_ticks_since_boot": int(stat_tail.split()[19]),
    }


def require_finite(value: Any, label: str) -> None:
    """Reject nonfinite numerical evidence before it is serialized."""
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite {label}")
    elif isinstance(value, np.ndarray):
        if not bool(np.isfinite(value).all()):
            raise ValueError(f"nonfinite {label}")
    elif isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite(item, f"{label}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"nonfinite {label}")


def result_json(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the exact JSON Objective receipt while storing u separately."""
    return {
        key: value
        for key, value in result.items()
        if key not in {"u", "component_gradients"}
    }


def validate_frozen_anchor(
    manifest_path: Path,
    raw: Any,
    step: int,
    state: dict[str, Any] | None = None,
) -> None:
    """Validate one complete q/u/m/v/t anchor in the pre-run freeze."""
    if not isinstance(raw, dict):
        raise TypeError(f"anchor {step} must be an object")
    expected_path = STEP64 if step == 64 else STEP256
    checkpoint = raw.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"anchor {step} checkpoint receipt must be an object")
    path = resolve_manifest_path(
        manifest_path, checkpoint.get("path"), f"checkpoint {step}"
    )
    if path != expected_path.resolve():
        raise ValueError(f"checkpoint {step} path differs")
    if record(path) != checkpoint:
        raise ValueError(f"checkpoint {step} file receipt differs")
    expected_arrays = {
        "q": ([288235, 6], "float64"),
        "u": ([228660, 3], "float64"),
        "m": ([288235, 6], "float64"),
        "v": ([288235, 6], "float64"),
        "active_ids": ([288235], None),
    }
    for name, (shape, dtype) in expected_arrays.items():
        receipt = raw.get(name)
        if (
            not isinstance(receipt, dict)
            or receipt.get("shape") != shape
            or (dtype is not None and receipt.get("dtype") != dtype)
            or not isinstance(receipt.get("sha256"), str)
            or len(receipt["sha256"]) != 64
        ):
            raise ValueError(f"anchor {step} {name} receipt differs")
    counter = raw.get("t")
    if (
        not isinstance(counter, dict)
        or counter.get("shape") != []
        or not isinstance(counter.get("sha256"), str)
        or raw.get("step") != step
    ):
        raise ValueError(f"anchor {step} Adam counter receipt differs")
    if state is None:
        return
    groups = state["optimizer"]["param_groups"]
    moment = state["optimizer"]["state"][groups[0]["params"][0]]
    actual = {
        "q": state["q"].numpy(),
        "u": np.asarray(state["u"]),
        "m": moment["exp_avg"].numpy(),
        "v": moment["exp_avg_sq"].numpy(),
        "t": moment["step"].numpy(),
        "active_ids": state["active_ids"],
    }
    for name, value in actual.items():
        if (
            list(value.shape) != raw[name]["shape"]
            or str(value.dtype) != raw[name]["dtype"]
            or array_sha256(value) != raw[name]["sha256"]
        ):
            raise ValueError(f"loaded anchor {step} {name} differs from freeze")
    if state["config"] != raw.get("config"):
        raise ValueError(f"loaded anchor {step} config differs from freeze")


def validate_frozen_manifest(
    cfg: Config,
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
    step: int | None = None,
    provenance: dict[str, Any] | None = None,
) -> None:
    """Enforce the manifest frozen before any forward/adjoint sample."""
    if manifest.get("schema_version") != 1:
        raise ValueError("repeatability manifest schema differs")
    freeze_path = resolve_manifest_path(
        manifest_path, manifest.get("prerun_freeze"), "pre-run freeze"
    )
    freeze = read_json(freeze_path)
    if (
        freeze.get("schema_version") != 1
        or freeze.get("status") != "frozen_before_any_fresh_GPU_sample"
        or freeze.get("same_saved_state_required_for_all_three_samples") is not True
        or freeze.get("protocol") != record(cfg.source91_protocol.resolve())
    ):
        raise ValueError("pre-run freeze identity or protocol differs")
    anchors = freeze.get("anchors")
    if not isinstance(anchors, dict) or set(anchors) != {"64", "256"}:
        raise ValueError("pre-run freeze must contain anchors 64 and 256")
    for anchor in (64, 256):
        validate_frozen_anchor(
            freeze_path,
            anchors[str(anchor)],
            anchor,
            state if step == anchor else None,
        )
    samples = manifest.get("samples")
    if not isinstance(samples, dict) or set(samples) != {"64", "256"}:
        raise ValueError("manifest must predeclare samples 64 and 256")
    for anchor in (64, 256):
        raw = samples[str(anchor)]
        if (
            not isinstance(raw, list)
            or len(raw) != SAMPLE_COUNT
            or len(set(raw)) != SAMPLE_COUNT
        ):
            raise ValueError(f"manifest must predeclare three step-{anchor} samples")
        for index, value in enumerate(raw):
            resolve_manifest_path(manifest_path, value, f"step {anchor} sample {index}")
    resolve_manifest_path(manifest_path, manifest.get("fixed_replay"), "fixed replay")
    if provenance is not None:
        frozen_sources = freeze.get("sources")
        if not isinstance(frozen_sources, dict):
            raise ValueError("pre-run source inventory is missing")
        source_hashes = {
            name: value.get("sha256")
            for name, value in frozen_sources.items()
            if isinstance(value, dict) and name.endswith(".py")
        }
        frozen_inputs = freeze.get("inputs")
        input_identity = {
            name: {key: value[key] for key in ("path", "sha256")}
            for name, value in frozen_inputs.items()
        }
        if (
            provenance["sources"] != source_hashes
            or provenance["inputs"] != input_identity
            or provenance["git_sha"] != freeze.get("git_sha")
            or provenance["torch"] != freeze.get("torch")
        ):
            raise ValueError(
                "sample scientific environment differs from pre-run freeze"
            )


def sample(cfg: Config) -> None:  # noqa: PLR0915
    """Run one fresh-process forward/adjoint evaluation without an update."""
    assert cfg.resume.resolve() in {STEP64.resolve(), STEP256.resolve()}
    assert cfg.model == "tensor"
    assert cfg.smoothness_weight == cfg.magnitude_weight == cfg.rank_weight == 0
    assert cfg.projected_gradient_eta == 1.0
    if cfg.manifest is None:
        raise ValueError("sample mode requires the pre-frozen --manifest")
    if not cfg.sample_id or any(char.isspace() for char in cfg.sample_id):
        raise ValueError("sample-id must be one nonempty token")
    manifest_path = cfg.manifest.resolve()
    manifest = read_json(manifest_path)
    state = load_checkpoint(cfg.resume.resolve(), cfg)
    step = int(state["step"])
    if step not in {64, 256}:
        raise ValueError("sample checkpoint step must be 64 or 256")
    validate_frozen_manifest(cfg, manifest_path, manifest, state=state, step=step)
    declared_outputs = {
        resolve_manifest_path(manifest_path, value, f"step {step} sample")
        for value in manifest["samples"][str(step)]
    }
    if cfg.output_dir.resolve() not in declared_outputs:
        raise ValueError("sample output directory was not predeclared")
    output = prepare_output(cfg.output_dir)
    write_json(output / "config.json", cfg.model_dump(mode="json"))
    provenance = BASE.archive(output, cfg)
    verify_parent_sources(parent_provenance(cfg), provenance)
    validate_frozen_manifest(cfg, manifest_path, manifest, provenance=provenance)
    configure()
    physics = FacePhysics(cfg.fixture, activation_model="tensor")
    if not np.array_equal(physics.ids, state["active_ids"]):
        raise ValueError("active cell IDs differ from checkpoint")
    objective = BASE.Objective(physics, cfg)
    q = torch.nn.Parameter(state["q"].to(device="cuda").clone())
    q.grad = None
    result = objective(q, np.asarray(state["u"]).copy())
    if q.grad is None:
        raise RuntimeError("objective did not produce a gradient")
    gradient = q.grad.detach().cpu().numpy()
    q_cpu = q.detach().cpu().numpy()
    u = np.asarray(result["u"])
    if not np.array_equal(q_cpu, state["q"].numpy()):
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
    require_finite(q_cpu, "q")
    require_finite(u, "u")
    require_finite(gradient, "gradient")
    require_finite(metrics, "metrics")
    require_finite(result["forward"], "forward")
    require_finite(result["adjoint"], "adjoint")
    replay = initial_replay_metrics(metrics, state["parent_endpoint"])
    write_json(output / "initial-replay.json", replay)
    if not replay["passed"]:
        raise ValueError("sample does not replay the accepted parent physical state")
    write_json(output / "result.json", result_json(result))
    np.savez_compressed(
        output / "sample.npz",
        q=q_cpu,
        u=u,
        gradient=gradient,
        active_ids=state["active_ids"],
        step=np.asarray(step),
    )
    summary = {
        "schema_version": 1,
        "status": "completed_no_update_sample",
        "mode": "fresh_process_forward_adjoint",
        "sample_id": cfg.sample_id,
        "process_identity": process_identity(),
        "solver_valid": result["forward"]["success"] and result["adjoint"]["success"],
        "checkpoint": checkpoint_receipt(cfg.resume.resolve(), state),
        "sample": record(output / "sample.npz"),
        "result": record(output / "result.json"),
        "initial_replay": replay,
        "metrics": metrics,
        "array_sha256": {
            "q": array_sha256(q_cpu),
            "u": array_sha256(u),
            "gradient": array_sha256(gradient),
            "active_ids": array_sha256(state["active_ids"]),
        },
        "forward": result["forward"],
        "adjoint": result["adjoint"],
        "config": record(output / "config.json"),
        "provenance": provenance,
        "provenance_file": record(output / "provenance.json"),
        "manifest": record(manifest_path),
        "scope": "one fresh forward and adjoint; no optimizer update or projection",
    }
    write_json(output / "summary.json", summary)
    for name in (
        "summary.json",
        "sample.npz",
        "result.json",
        "initial-replay.json",
        "config.json",
        "provenance.json",
    ):
        cherries.log_output(output / name)


def next_adam_direction(
    state: dict[str, Any], gradient: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Return the exact next bias-corrected Adam direction at fixed gradient."""
    groups = state["optimizer"]["param_groups"]
    group = groups[0]
    moment = state["optimizer"]["state"][group["params"][0]]
    if group["weight_decay"] != 0 or group["amsgrad"] or group["maximize"]:
        raise ValueError("unsupported Adam variant")
    step = int(moment["step"]) + 1
    beta1, beta2 = group["betas"]
    mhat = (beta1 * moment["exp_avg"] + (1 - beta1) * gradient) / (1 - beta1**step)
    vhat = (beta2 * moment["exp_avg_sq"] + (1 - beta2) * gradient.square()) / (
        1 - beta2**step
    )
    return -mhat / (vhat.sqrt() + epsilon)


def clipping_signature(unprojected: torch.Tensor, maximum: float) -> dict[str, Any]:
    """Record which cells require lower or upper spectral clipping."""
    eigenvalues = torch.linalg.eigvalsh(matrices(unprojected)).cpu().numpy()
    lower = np.any(eigenvalues < 0, axis=1)
    upper = np.any(eigenvalues > maximum, axis=1)
    return {
        "lower": lower,
        "upper": upper,
        "lower_fraction": float(lower.mean()),
        "upper_fraction": float(upper.mean()),
        "lower_sha256": array_sha256(lower),
        "upper_sha256": array_sha256(upper),
    }


@torch.no_grad()
def closed_form_trial(
    initial_q: torch.Tensor,
    direction: torch.Tensor,
    learning_rate: float,
    maximum: float,
    qref: float,
) -> dict[str, Any]:
    """Apply one cloned closed-form direction and the production projection."""
    unprojected = initial_q + learning_rate * direction
    clipping = clipping_signature(unprojected, maximum)
    candidate = unprojected.clone()
    projection = project(candidate, maximum)
    delta = candidate - initial_q
    delta_q = delta.cpu().numpy()
    delta_Q = (qref * matrices(delta)).cpu().numpy()
    return {
        "delta_q": delta_q,
        "delta_Q": delta_Q,
        "clipping": clipping,
        "metrics": {
            "learning_rate": learning_rate,
            "physical_update_rms_mpa": tensor_rms_mpa(delta, qref),
            "physical_update_frobenius_max_mpa": float(
                np.linalg.norm(delta_Q, axis=(1, 2)).max()
            ),
            "actual_update_rms": rms(delta),
            "unprojected_update_rms": rms(learning_rate * direction),
            **projection,
        },
    }


@torch.no_grad()
def installed_trial(
    state: dict[str, Any],
    gradient: torch.Tensor,
    learning_rate: float,
    epsilon: float,
    maximum: float,
    qref: float,
) -> dict[str, Any]:
    """Run installed Adam once from an immutable copied optimizer state."""
    q = torch.nn.Parameter(state["q"].clone())
    optimizer = torch.optim.Adam([q], lr=learning_rate, eps=epsilon)
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    optimizer.param_groups[0]["lr"] = learning_rate
    optimizer.param_groups[0]["eps"] = epsilon
    q.grad = gradient.clone()
    optimizer.step()
    project(q, maximum)
    delta = q.detach() - state["q"]
    return {
        "delta_q": delta.numpy(),
        "delta_Q": (qref * matrices(delta)).numpy(),
        "adam_step_after": int(optimizer.state[q]["step"]),
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Return one full-field cosine, requiring nonzero finite fields."""
    a, b = left.reshape(-1), right.reshape(-1)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("full-field cosine is undefined")
    return float(np.dot(a, b) / denominator)


def tensor_difference(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Compare two physical tensor fields without subtracting scalar norms."""
    difference = left - right
    per_cell = np.linalg.norm(difference, axis=(1, 2))
    return {
        "rms_mpa": float(np.sqrt(np.mean(per_cell**2))),
        "max_per_tet_frobenius_mpa": float(per_cell.max()),
        "cosine": cosine(left, right),
    }


def installed_agreement(
    closed: dict[str, Any], installed: dict[str, Any], qref: float
) -> dict[str, Any]:
    """Check installed Adam against its same-gradient closed form."""
    difference = installed["delta_q"] - closed["delta_q"]
    installed_rms = float(
        np.sqrt(np.mean(np.sum(installed["delta_Q"] ** 2, axis=(1, 2))))
    )
    declared = closed["metrics"]["physical_update_rms_mpa"]
    return {
        "max_abs_q": float(np.abs(difference).max()),
        "relative_physical_update_error": abs(installed_rms / declared - 1),
        "max_abs_q_tolerance": REPLAY_MAX_ABS_TOLERANCE,
        "relative_physical_update_tolerance": PHYSICAL_REPLAY_RELATIVE_TOLERANCE,
        "passed": float(np.abs(difference).max()) < REPLAY_MAX_ABS_TOLERANCE
        and abs(installed_rms / declared - 1) < PHYSICAL_REPLAY_RELATIVE_TOLERANCE,
        "gradient_reused_without_recomputation": True,
        "qref_mpa": qref,
    }


def installed_repeatability(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    """Compare two independent installed-Adam replays of one frozen gradient."""
    max_abs = float(np.abs(left["delta_q"] - right["delta_q"]).max())
    return {
        "max_abs_q": max_abs,
        "max_abs_q_tolerance": REPLAY_MAX_ABS_TOLERANCE,
        "physical_delta_Q": tensor_difference(left["delta_Q"], right["delta_Q"]),
        "passed": max_abs < REPLAY_MAX_ABS_TOLERANCE,
        "same_serialized_gradient_and_cloned_state": True,
    }


def trial_receipt(value: dict[str, Any]) -> dict[str, Any]:
    """Describe a full-field trial without embedding its large arrays in JSON."""
    clipping = value["clipping"]
    return {
        "metrics": value["metrics"],
        "delta_q_sha256": array_sha256(value["delta_q"]),
        "delta_Q_sha256": array_sha256(value["delta_Q"]),
        "clipping": {
            key: clipping[key]
            for key in (
                "lower_fraction",
                "upper_fraction",
                "lower_sha256",
                "upper_sha256",
            )
        },
    }


def calibration_receipt(value: dict[str, Any]) -> dict[str, Any]:
    """Return the finite JSON receipt for one cached-gradient calibration."""
    return {
        "gradient_sha256": value["gradient_sha256"],
        "physical_step_multiplier": value["physical_step_multiplier"],
        "target_physical_update_rms_mpa": value["target_physical_update_rms_mpa"],
        "baseline": trial_receipt(value["baseline"]),
        "selected": trial_receipt(value["selected"]),
        "baseline_installed_agreement": value["baseline_installed_agreement"],
        "selected_installed_agreement": value["selected_installed_agreement"],
        "baseline_installed_repeatability": value["baseline_installed_repeatability"],
        "selected_installed_repeatability": value["selected_installed_repeatability"],
        "baseline_vs_candidate": value["baseline_vs_candidate"],
        "bisection_trials": value["bisection_trials"],
        "passed": value["passed"],
    }


def calibrate(
    state: dict[str, Any],
    gradient_array: np.ndarray,
    reduced_eps: float,
    multiplier: float,
    qref: float,
    maximum: float,
) -> dict[str, Any]:
    """Calibrate reduced epsilon to one same-gradient projected baseline step."""
    gradient = torch.as_tensor(gradient_array, dtype=torch.float64)
    initial = state["q"]
    baseline_lr = float(state["optimizer"]["param_groups"][0]["lr"])
    baseline_eps = float(state["optimizer"]["param_groups"][0]["eps"])
    baseline = closed_form_trial(
        initial,
        next_adam_direction(state, gradient, baseline_eps),
        baseline_lr,
        maximum,
        qref,
    )
    target = multiplier * baseline["metrics"]["physical_update_rms_mpa"]
    if target <= 0 or not np.isfinite(target):
        raise RuntimeError("baseline physical step is zero or nonfinite")
    candidate_direction = next_adam_direction(state, gradient, reduced_eps)
    lower, upper = 0.0, baseline_lr
    trials: list[dict[str, float]] = []
    upper_trial = closed_form_trial(initial, candidate_direction, upper, maximum, qref)
    if upper_trial["metrics"]["physical_update_rms_mpa"] < target:
        raise RuntimeError("candidate physical step is not bracketed")
    for _ in range(40):
        middle = 0.5 * (lower + upper)
        current = closed_form_trial(initial, candidate_direction, middle, maximum, qref)
        physical = current["metrics"]["physical_update_rms_mpa"]
        trials.append({"learning_rate": middle, "physical_update_rms_mpa": physical})
        if physical < target:
            lower = middle
        else:
            upper = middle
    learning_rate = 0.5 * (lower + upper)
    selected = closed_form_trial(
        initial, candidate_direction, learning_rate, maximum, qref
    )
    ordered = sorted(
        [*trials, upper_trial["metrics"], selected["metrics"]],
        key=lambda value: value["learning_rate"],
    )
    for left, right in itertools.pairwise(ordered):
        if right["physical_update_rms_mpa"] + 1e-12 < left["physical_update_rms_mpa"]:
            raise RuntimeError("projected physical-step calibration is nonmonotone")
    baseline_installed = [
        installed_trial(state, gradient, baseline_lr, baseline_eps, maximum, qref)
        for _ in range(2)
    ]
    selected_installed = [
        installed_trial(state, gradient, learning_rate, reduced_eps, maximum, qref)
        for _ in range(2)
    ]
    baseline_agreement = installed_agreement(baseline, baseline_installed[0], qref)
    selected_agreement = installed_agreement(selected, selected_installed[0], qref)
    baseline_repeatability = installed_repeatability(*baseline_installed)
    selected_repeatability = installed_repeatability(*selected_installed)
    return {
        "gradient_sha256": array_sha256(gradient_array),
        "physical_step_multiplier": multiplier,
        "target_physical_update_rms_mpa": target,
        "baseline": baseline,
        "selected": selected,
        "baseline_installed_agreement": baseline_agreement,
        "selected_installed_agreement": selected_agreement,
        "baseline_installed_repeatability": baseline_repeatability,
        "selected_installed_repeatability": selected_repeatability,
        "baseline_vs_candidate": tensor_difference(
            baseline["delta_Q"], selected["delta_Q"]
        ),
        "bisection_trials": trials,
        "passed": baseline_agreement["passed"]
        and selected_agreement["passed"]
        and baseline_repeatability["passed"]
        and selected_repeatability["passed"]
        and abs(selected["metrics"]["physical_update_rms_mpa"] / target - 1)
        < PHYSICAL_REPLAY_RELATIVE_TOLERANCE,
    }


def fixed_replay(cfg: Config) -> None:
    """Diagnose the old step-64 failure without a forward or adjoint solve."""
    assert cfg.resume.resolve() == STEP64.resolve()
    output = prepare_output(cfg.output_dir)
    state = load_checkpoint(cfg.resume.resolve(), cfg)
    if int(state["step"]) != 64:
        raise ValueError("fixed replay requires step 64")
    write_json(output / "config.json", cfg.model_dump(mode="json"))
    provenance = BASE.archive(output, cfg)
    verify_parent_sources(parent_provenance(cfg), provenance)
    calibration = read_json(cfg.frozen_calibration_summary.resolve())
    with np.load(cfg.frozen_gradient.resolve(), allow_pickle=False) as saved:
        q = saved["q"].copy()
        u = saved["u"].copy()
        gradient_array = saved["gradient"].copy()
        step = int(saved["step"])
    if (
        step != 64
        or not np.array_equal(q, state["q"].numpy())
        or calibration.get("gradient_sha256") != array_sha256(gradient_array)
    ):
        raise ValueError("old frozen gradient does not bind step-64 checkpoint")
    qref = cfg.stress_reference_mpa
    maximum = cfg.stress_cap_mpa / qref
    selected_cfg = calibration["selected"]
    replay = calibrate(
        state,
        gradient_array,
        float(selected_cfg["adam_eps"]),
        float(calibration["physical_step_multiplier"]),
        qref,
        maximum,
    )
    if replay["selected"]["metrics"]["learning_rate"] != selected_cfg[
        "learning_rate"
    ] and not np.isclose(
        replay["selected"]["metrics"]["learning_rate"],
        selected_cfg["learning_rate"],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("old selected learning rate does not replay")
    with np.load(cfg.rejected_trial.resolve(), allow_pickle=False) as saved:
        rejected_q = saved["q"].copy()
    rejected_delta_q = rejected_q - q
    rejected_delta_Q = qref * matrices(torch.from_numpy(rejected_delta_q)).numpy()
    old_difference = tensor_difference(replay["selected"]["delta_Q"], rejected_delta_Q)
    np.savez_compressed(
        output / "fixed-replay.npz",
        q=q,
        u=u,
        gradient=gradient_array,
        baseline_delta_q=replay["baseline"]["delta_q"],
        baseline_delta_Q=replay["baseline"]["delta_Q"],
        selected_delta_q=replay["selected"]["delta_q"],
        selected_delta_Q=replay["selected"]["delta_Q"],
        rejected_delta_q=rejected_delta_q,
        rejected_delta_Q=rejected_delta_Q,
        step=np.asarray(64),
    )
    summary = {
        "schema_version": 1,
        "status": "passed" if replay["passed"] else "failed",
        "mode": "fixed_gradient_replay_step64",
        "checkpoint": checkpoint_receipt(cfg.resume.resolve(), state),
        "cached_gradient": record(cfg.frozen_gradient.resolve()),
        "old_calibration": record(cfg.frozen_calibration_summary.resolve()),
        "rejected_trial": record(cfg.rejected_trial.resolve()),
        "replay": calibration_receipt(replay),
        "closed_form_selected_vs_rejected_trial": old_difference,
        "arrays": record(output / "fixed-replay.npz"),
        "config": record(output / "config.json"),
        "provenance": provenance,
        "provenance_file": record(output / "provenance.json"),
        "scope": "saved state and frozen gradient only; no forward or adjoint solve",
    }
    write_json(output / "summary.json", summary)
    for name in (
        "summary.json",
        "fixed-replay.npz",
        "config.json",
        "provenance.json",
    ):
        cherries.log_output(output / name)
    if summary["status"] != "passed":
        raise RuntimeError("fixed-gradient replay failed")


def resolve_manifest_path(manifest: Path, raw: Any, label: str) -> Path:
    """Resolve one nonempty manifest-relative path."""
    if not isinstance(raw, str) or not raw:
        raise TypeError(f"{label} path must be a nonempty string")
    return (manifest.parent / raw).resolve()


def load_samples(
    manifest_path: Path, raw: Any, expected_step: int
) -> list[dict[str, Any]]:
    """Load exactly three unique completed samples for one checkpoint."""
    if not isinstance(raw, list) or len(raw) != SAMPLE_COUNT or len(set(raw)) != 3:
        raise ValueError(f"step {expected_step} requires three unique sample paths")
    samples: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        directory = resolve_manifest_path(
            manifest_path, value, f"step {expected_step} sample {index}"
        )
        summary = read_json(directory / "summary.json")
        if (
            summary.get("status") != "completed_no_update_sample"
            or summary.get("solver_valid") is not True
            or summary.get("checkpoint", {}).get("step") != expected_step
            or summary.get("initial_replay", {}).get("passed") is not True
        ):
            raise ValueError(f"invalid step-{expected_step} sample: {directory}")
        if summary.get("provenance") != read_json(directory / "provenance.json"):
            raise ValueError(f"sample provenance differs: {directory}")
        for name, key in (
            ("sample.npz", "sample"),
            ("result.json", "result"),
            ("config.json", "config"),
            ("provenance.json", "provenance_file"),
        ):
            if record(directory / name) != summary.get(key):
                raise ValueError(f"sample {name} receipt differs: {directory}")
        with np.load(directory / "sample.npz", allow_pickle=False) as saved:
            arrays = {
                name: saved[name].copy()
                for name in ("q", "u", "gradient", "active_ids")
            }
            step = int(saved["step"])
        if step != expected_step:
            raise ValueError(f"sample array step differs: {directory}")
        for name in ("q", "u", "gradient"):
            if array_sha256(arrays[name]) != summary["array_sha256"][name]:
                raise ValueError(f"sample {name} hash differs: {directory}")
        samples.append(
            {
                "directory": directory,
                "summary": summary,
                "arrays": arrays,
                "records": {
                    name: record(directory / name)
                    for name in (
                        "summary.json",
                        "sample.npz",
                        "result.json",
                        "initial-replay.json",
                        "config.json",
                        "provenance.json",
                    )
                },
            }
        )
    checkpoint_hashes = {
        sample["summary"]["checkpoint"]["sha256"] for sample in samples
    }
    q_hashes = {sample["summary"]["array_sha256"]["q"] for sample in samples}
    active_hashes = {array_sha256(sample["arrays"]["active_ids"]) for sample in samples}
    sample_ids = {sample["summary"]["sample_id"] for sample in samples}
    if (
        len(checkpoint_hashes) != 1
        or len(q_hashes) != 1
        or len(active_hashes) != 1
        or len(sample_ids) != SAMPLE_COUNT
    ):
        raise ValueError(f"step-{expected_step} samples do not share one source state")
    return samples


def pairwise_arrays(
    values: list[np.ndarray], kind: Literal["vector", "tensor"]
) -> list[dict[str, Any]]:
    """Compute all three full-field pairwise differences."""
    rows = []
    for (left_id, left), (right_id, right) in itertools.combinations(
        enumerate(values), 2
    ):
        difference = left - right
        if kind == "tensor":
            metrics = tensor_difference(left, right)
        else:
            metrics = {
                "rms": float(np.sqrt(np.mean(difference**2))),
                "max_abs": float(np.abs(difference).max()),
                "cosine": cosine(left, right),
            }
        rows.append({"left": left_id, "right": right_id, **metrics})
    return rows


def clipping_disagreement(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, float]:
    """Compare per-cell lower and upper clipping membership."""
    return {
        "lower_fraction": float(np.mean(left["lower"] != right["lower"])),
        "upper_fraction": float(np.mean(left["upper"] != right["upper"])),
    }


def checkpoint_noise(
    state: dict[str, Any],
    samples: list[dict[str, Any]],
    reduced_eps: float,
    qref: float,
    maximum: float,
) -> dict[str, Any]:
    """Calibrate from sample zero, then quantify pairwise update-field noise."""
    calibration = calibrate(
        state,
        samples[0]["arrays"]["gradient"],
        reduced_eps,
        1.0,
        qref,
        maximum,
    )
    settings = {
        "baseline": {
            "epsilon": float(state["optimizer"]["param_groups"][0]["eps"]),
            "learning_rate": float(state["optimizer"]["param_groups"][0]["lr"]),
        },
        "candidate": {
            "epsilon": reduced_eps,
            "learning_rate": calibration["selected"]["metrics"]["learning_rate"],
        },
    }
    arms: dict[str, Any] = {}
    signal = calibration["baseline_vs_candidate"]["rms_mpa"]
    for arm, setting in settings.items():
        updates = []
        update_rms = []
        clippings = []
        agreements = []
        for sample_item in samples:
            gradient_array = sample_item["arrays"]["gradient"]
            gradient = torch.as_tensor(gradient_array, dtype=torch.float64)
            closed = closed_form_trial(
                state["q"],
                next_adam_direction(state, gradient, setting["epsilon"]),
                setting["learning_rate"],
                maximum,
                qref,
            )
            installed = installed_trial(
                state,
                gradient,
                setting["learning_rate"],
                setting["epsilon"],
                maximum,
                qref,
            )
            updates.append(closed["delta_Q"])
            update_rms.append(closed["metrics"]["physical_update_rms_mpa"])
            clippings.append(closed["clipping"])
            agreements.append(installed_agreement(closed, installed, qref))
        pairs = pairwise_arrays(updates, "tensor")
        for row, (left, right) in zip(
            pairs, itertools.combinations(clippings, 2), strict=True
        ):
            row["clipping_disagreement"] = clipping_disagreement(left, right)
        for row, (left_rms, right_rms) in zip(
            pairs, itertools.combinations(update_rms, 2), strict=True
        ):
            row["noise_over_left_update"] = (
                row["rms_mpa"] / left_rms if left_rms > 0 else None
            )
            row["noise_over_right_update"] = (
                row["rms_mpa"] / right_rms if right_rms > 0 else None
            )
        max_noise = max(row["rms_mpa"] for row in pairs)
        own = calibration["baseline" if arm == "baseline" else "selected"]["metrics"][
            "physical_update_rms_mpa"
        ]
        own_fraction = max_noise / own if own > 0 else None
        signal_fraction = max_noise / signal if signal > 0 else None
        passed = (
            all(value["passed"] for value in agreements)
            and own_fraction is not None
            and own_fraction <= OWN_UPDATE_NOISE_FRACTION
            and signal_fraction is not None
            and signal_fraction <= DIRECTION_SIGNAL_NOISE_FRACTION
        )
        arms[arm] = {
            "setting": setting,
            "sample0_physical_update_rms_mpa": own,
            "same_gradient_installed_agreement": agreements,
            "pairwise_projected_update": pairs,
            "max_pairwise_physical_delta_Q_rms_mpa": max_noise,
            "noise_over_own_sample0_update": own_fraction,
            "noise_over_baseline_candidate_signal": signal_fraction,
            "passed": passed,
        }
    return {
        "calibration": calibration_receipt(calibration),
        "gradient_pairwise": pairwise_arrays(
            [sample_item["arrays"]["gradient"] for sample_item in samples],
            "vector",
        ),
        "u_pairwise": pairwise_arrays(
            [sample_item["arrays"]["u"] for sample_item in samples], "vector"
        ),
        "baseline_candidate_direction_difference_rms_mpa": signal,
        "signal_is_nonzero": signal > 0,
        "arms": arms,
        "passed": signal > 0
        and calibration["passed"]
        and all(value["passed"] for value in arms.values()),
    }


def summarize(cfg: Config) -> None:  # noqa: PLR0915
    """Aggregate six samples and freeze the step-256 one-step calibration."""
    if cfg.manifest is None:
        raise ValueError("summarize mode requires --manifest")
    if cfg.physical_step_multiplier != 1.0 or cfg.reduced_eps != 1e-6:
        raise ValueError("aggregate requires eps=1e-6 calibrated to 1x baseline")
    output = prepare_output(cfg.output_dir)
    manifest_path = cfg.manifest.resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("repeatability manifest schema differs")
    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, dict) or set(raw_samples) != {"64", "256"}:
        raise ValueError("manifest must declare samples 64 and 256")
    fixed_path = resolve_manifest_path(
        manifest_path, manifest.get("fixed_replay"), "fixed replay"
    )
    fixed_summary = read_json(fixed_path / "summary.json")
    if fixed_summary.get("status") != "passed":
        raise ValueError("fixed-gradient replay did not pass")
    samples = {
        step: load_samples(manifest_path, raw_samples[str(step)], step)
        for step in (64, 256)
    }
    states: dict[int, dict[str, Any]] = {}
    for step in (64, 256):
        checkpoint = Path(samples[step][0]["summary"]["checkpoint"]["path"])
        expected_checkpoint = STEP64 if step == 64 else STEP256
        if checkpoint.resolve() != expected_checkpoint.resolve():
            raise ValueError(f"step-{step} sample uses the wrong checkpoint")
        sample_config = read_json(samples[step][0]["directory"] / "config.json")
        for key in (
            "fixture",
            "model",
            "stress_reference_mpa",
            "stress_cap_mpa",
            "smooth_length_m",
        ):
            value = Path(sample_config[key]) if key == "fixture" else sample_config[key]
            setattr(cfg, key, value)
        cfg.resume = checkpoint
        states[step] = load_checkpoint(checkpoint, cfg)
        expected_receipt = checkpoint_receipt(checkpoint, states[step])
        identities = set()
        for sample_item in samples[step]:
            assert sample_item["summary"]["checkpoint"] == expected_receipt
            assert all(
                sample_item["summary"]["provenance"][key]
                == samples[step][0]["summary"]["provenance"][key]
                for key in ("sources", "inputs", "git_sha", "python", "torch", "cuda")
            )
            identities.add(
                json.dumps(sample_item["summary"]["process_identity"], sort_keys=True)
            )
        assert len(identities) == SAMPLE_COUNT
    output_config = cfg.model_dump(mode="json")
    output_config["resume"] = str(STEP256.resolve())
    write_json(output / "config.json", output_config)
    cfg.resume = STEP256
    provenance = BASE.archive(output, cfg)
    verify_parent_sources(parent_provenance(cfg), provenance)
    noises = {
        step: checkpoint_noise(
            states[step],
            samples[step],
            cfg.reduced_eps,
            cfg.stress_reference_mpa,
            cfg.stress_cap_mpa / cfg.stress_reference_mpa,
        )
        for step in (64, 256)
    }
    first = samples[256][0]
    np.savez_compressed(
        output / "initial-gradient.npz",
        q=first["arrays"]["q"],
        u=first["arrays"]["u"],
        gradient=first["arrays"]["gradient"],
        active_ids=first["arrays"]["active_ids"],
        step=np.asarray(256),
    )
    shutil.copyfile(first["directory"] / "result.json", output / "initial-result.json")
    protocol = cfg.source91_protocol.resolve()
    calibration = noises[256]["calibration"]
    baseline_metrics = calibration["baseline"]["metrics"]
    selected_metrics = calibration["selected"]["metrics"]
    area_fit_values = [
        item["summary"]["metrics"]["area_fit_rms_mm"] for item in samples[256]
    ]
    unweighted_fit_values = [
        item["summary"]["metrics"]["fit_rms_mm"] for item in samples[256]
    ]
    passed = fixed_summary["status"] == "passed" and all(
        noises[step]["passed"] for step in (64, 256)
    )
    summary = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "gate_passed": passed,
        "mode": "repeatability_aggregate",
        "rule": {
            "sample_count_per_checkpoint": SAMPLE_COUNT,
            "checkpoints": [64, 256],
            "solver_valid_required": True,
            "same_gradient_optimizer_replay_max_abs_q": REPLAY_MAX_ABS_TOLERANCE,
            "max_pairwise_noise_over_own_sample0_update": OWN_UPDATE_NOISE_FRACTION,
            "max_pairwise_noise_over_baseline_candidate_direction_difference": DIRECTION_SIGNAL_NOISE_FRACTION,
            "baseline_candidate_signal_must_be_nonzero": True,
            "physical_step_multiplier": 1.0,
            "geometry_used_for_gate": False,
        },
        "baseline": {
            "adam_eps": float(states[256]["optimizer"]["param_groups"][0]["eps"]),
            "learning_rate": baseline_metrics["learning_rate"],
            "physical_update_rms_mpa": baseline_metrics["physical_update_rms_mpa"],
        },
        "selected": {
            "adam_eps": cfg.reduced_eps,
            "learning_rate": selected_metrics["learning_rate"],
            "physical_update_rms_mpa": selected_metrics["physical_update_rms_mpa"],
        },
        "parent_checkpoint": checkpoint_receipt(STEP256.resolve(), states[256]),
        "initial_gradient": {
            **record(output / "initial-gradient.npz"),
            "step": 256,
            "q_sha256": array_sha256(first["arrays"]["q"]),
            "u_sha256": array_sha256(first["arrays"]["u"]),
            "gradient_sha256": array_sha256(first["arrays"]["gradient"]),
            "source_sample": first["records"]["summary.json"],
            "source_sample_provenance": first["records"]["provenance.json"],
        },
        "initial_result": {
            **record(output / "initial-result.json"),
            "source_sample": first["records"]["result.json"],
            "copied_byte_for_byte": sha256(output / "initial-result.json")
            == sha256(first["directory"] / "result.json"),
        },
        "fit_rms_range_mm": {
            "definition": "area-weighted fit RMS across the three step-256 samples",
            "minimum": min(area_fit_values),
            "maximum": max(area_fit_values),
            "range": max(area_fit_values) - min(area_fit_values),
        },
        "unweighted_fit_rms_range_mm": {
            "minimum": min(unweighted_fit_values),
            "maximum": max(unweighted_fit_values),
            "range": max(unweighted_fit_values) - min(unweighted_fit_values),
        },
        "fixed_gradient_replay_step64": record(fixed_path / "summary.json"),
        "checkpoints": {
            str(step): {
                "samples": [item["records"] for item in samples[step]],
                "noise": noises[step],
            }
            for step in (64, 256)
        },
        "calibration": {
            "checkpoint_step": 256,
            "cached_gradient_sha256": calibration["gradient_sha256"],
            "physical_step_multiplier": 1.0,
            "baseline_installed_agreement": calibration["baseline_installed_agreement"],
            "selected_installed_agreement": calibration["selected_installed_agreement"],
            "baseline_vs_candidate": calibration["baseline_vs_candidate"],
            "bisection_trials": calibration["bisection_trials"],
        },
        "source91_protocol": record(protocol),
        "manifest": record(manifest_path),
        "sources": provenance["sources"],
        "inputs": {
            "fixture": provenance["inputs"],
            "step64_checkpoint": record(STEP64),
            "step256_checkpoint": record(STEP256),
            "fixed_replay": record(fixed_path / "summary.json"),
            "source91_protocol": record(protocol),
        },
        "provenance": provenance,
        "provenance_file": record(output / "provenance.json"),
        "scope": "CPU aggregation and projected Adam replay; no forward, adjoint, optimizer trajectory, physics, solver-tolerance, or geometry gate change",
    }
    write_json(output / "summary.json", summary)
    for name in (
        "summary.json",
        "initial-gradient.npz",
        "initial-result.json",
        "config.json",
        "provenance.json",
    ):
        cherries.log_output(output / name)
    if not passed:
        raise RuntimeError("repeatability gate failed")


def run(cfg: Config) -> None:
    """Dispatch exactly one isolated repeatability mode."""
    global COMPLETED  # noqa: PLW0603
    if cfg.mode == "fixed-replay":
        fixed_replay(cfg)
    elif cfg.mode == "sample":
        sample(cfg)
    else:
        summarize(cfg)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(run, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
