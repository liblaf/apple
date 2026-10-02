"""Continue matched historical Adam runs from a shared immutable step-50 state."""

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pydantic_settings as ps
import torch
from experiment_profile import ProfileCometNoCommit
from historical_adam_physics import FacePhysics, ForwardConvergenceError, configure

from liblaf import cherries

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[4]
BASE_RUNNER_PATH = Path(__file__).with_name("30-run-historical-adam.py")
SOURCE_GLOBAL_STEP = 50
STOP_REQUESTED: int | None = None


def load_base_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "historical_adam_base_for_continuation", BASE_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(BASE_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_runner()
AdjointConvergenceError = BASE.AdjointConvergenceError
Objective = BASE.Objective
activation_matrix = BASE.activation_matrix
endpoint_metrics = BASE.endpoint_metrics


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = HERE / "data/12-historical-fixture"
    source_run: Path = HERE / "data/30-historical-adam-raw6"
    source_checkpoint: Path = source_run / "step-0050.npz"
    output_dir: Path = HERE / "data/38-historical-adam-raw6-continuation"
    steps: int = 150
    checkpoint_interval: int = 10
    inverse_lr: float = 0.3
    adam_eps: float = 0.01
    smoothness_weight: float = 0.0
    smooth_length: float = 0.005
    forward_rtol: float = 5e-4
    forward_atol: float = 1e-10
    adjoint_rtol: float = 5e-4
    resume: bool = True
    preflight: bool = False


def signal_handler(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED  # noqa: PLW0603
    STOP_REQUESTED = signum


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def json_default(item: Any) -> Any:
    if isinstance(item, np.generic):
        return item.item()
    raise TypeError(f"object of type {type(item).__name__} is not JSON serializable")


def atomic_write_json(path: Path, value: Any) -> None:
    data = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=json_default,
        )
        + "\n"
    ).encode()
    atomic_write_bytes(path, data)


def atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_snapshot(
    path: Path,
    q: torch.Tensor,
    result: dict[str, Any],
    local_step: int,
) -> None:
    q_numpy = q.detach().cpu().numpy()
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            q=q_numpy,
            u=result["u"],
            Ainv=activation_matrix(q_numpy),
            step=np.asarray(local_step, dtype=np.int64),
            source_global_step=np.asarray(SOURCE_GLOBAL_STEP, dtype=np.int64),
            nominal_global_step=np.asarray(
                SOURCE_GLOBAL_STEP + local_step, dtype=np.int64
            ),
            forward_success=np.asarray(result["forward"]["success"], dtype=np.bool_),
            adjoint_success=np.asarray(result["adjoint"]["success"], dtype=np.bool_),
            solver_valid=np.asarray(
                result["forward"]["success"] and result["adjoint"]["success"],
                dtype=np.bool_,
            ),
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def canonical_config(cfg: Config) -> dict[str, Any]:
    payload = cfg.model_dump(mode="json")
    for name in ("fixture", "source_run", "source_checkpoint", "output_dir"):
        payload[name] = str(Path(payload[name]).resolve())
    return payload


def validate_config(cfg: Config) -> None:
    expected = {
        "steps": 150,
        "checkpoint_interval": 10,
        "inverse_lr": 0.3,
        "adam_eps": 0.01,
        "smooth_length": 0.005,
        "forward_rtol": 5e-4,
        "forward_atol": 1e-10,
        "adjoint_rtol": 5e-4,
        "resume": True,
    }
    actual = {key: getattr(cfg, key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"matched continuation constants changed: {actual} != {expected}"
        )
    if cfg.smoothness_weight not in {0.0, 5e-4}:
        raise ValueError("use Raw6 weight 0 or Raw6-S weight 5e-4")
    if cfg.source_checkpoint.resolve().parent != cfg.source_run.resolve():
        raise ValueError("source checkpoint must be an explicit file in source-run")


def source_evidence(cfg: Config) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    validate_config(cfg)
    paths = {
        "checkpoint": cfg.source_checkpoint.resolve(),
        "source_config": (cfg.source_run / "config.json").resolve(),
        "source_provenance": (cfg.source_run / "provenance.json").resolve(),
        "source_trace": (cfg.source_run / "trace.csv").resolve(),
        "source_receipts": (cfg.source_run / "solver-receipts.jsonl").resolve(),
        "fixture_volume": (cfg.fixture / "volume.vtu").resolve(),
        "fixture_skin": (cfg.fixture / "skin.vtp").resolve(),
        "fixture_summary": (cfg.fixture / "summary.json").resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    source_config = json.loads(paths["source_config"].read_text())
    source_provenance = json.loads(paths["source_provenance"].read_text())
    matched = {
        "fixture": str(cfg.fixture.resolve()),
        "steps": 200,
        "checkpoint_interval": 10,
        "inverse_lr": cfg.inverse_lr,
        "adam_eps": cfg.adam_eps,
        "smoothness_weight": cfg.smoothness_weight,
        "smooth_length": cfg.smooth_length,
        "forward_rtol": cfg.forward_rtol,
        "forward_atol": cfg.forward_atol,
        "adjoint_rtol": cfg.adjoint_rtol,
        "preflight": False,
    }
    actual = {key: source_config[key] for key in matched}
    if actual != matched:
        raise ValueError(f"source-run contract differs: {actual} != {matched}")
    expected_fixture_hashes = source_provenance["inputs"]
    actual_fixture_hashes = {
        "volume.vtu": sha256(paths["fixture_volume"]),
        "skin.vtp": sha256(paths["fixture_skin"]),
        "summary.json": sha256(paths["fixture_summary"]),
    }
    if actual_fixture_hashes != expected_fixture_hashes:
        raise ValueError(
            "live fixture hashes differ from the source-run provenance: "
            f"{actual_fixture_hashes} != {expected_fixture_hashes}"
        )
    with np.load(paths["checkpoint"]) as saved:
        required = {
            "q",
            "u",
            "Ainv",
            "step",
            "forward_success",
            "adjoint_success",
            "solver_valid",
        }
        if missing := required.difference(saved.files):
            raise KeyError(f"source checkpoint lacks {sorted(missing)}")
        arrays = {name: np.asarray(saved[name]).copy() for name in required}
    if int(arrays["step"]) != SOURCE_GLOBAL_STEP:
        raise ValueError("continuation must start from source step 50")
    if arrays["q"].shape != (288_235, 6) or arrays["u"].shape != (228_660, 3):
        raise ValueError("source checkpoint control or displacement shape changed")
    matrix = np.asarray(activation_matrix(arrays["q"]))
    matrix_error = float(np.max(np.abs(matrix - arrays["Ainv"])))
    if matrix_error != 0.0:
        raise ValueError("source checkpoint q and Ainv differ")
    arrays_finite = bool(
        np.isfinite(arrays["q"]).all()
        and np.isfinite(arrays["u"]).all()
        and np.isfinite(arrays["Ainv"]).all()
    )
    if not arrays_finite:
        raise ValueError("source checkpoint q, u, or Ainv contains nonfinite values")
    if not (
        bool(arrays["forward_success"])
        and bool(arrays["adjoint_success"])
        and bool(arrays["solver_valid"])
    ):
        raise ValueError("shared source checkpoint step 50 is not solver-valid")
    trace_rows = list(csv.DictReader(paths["source_trace"].read_text().splitlines()))
    trace = {int(row["step"]): row for row in trace_rows}
    receipts = {
        int(row["step"]): row
        for row in (
            json.loads(line)
            for line in paths["source_receipts"].read_text().splitlines()
            if line
        )
    }
    if SOURCE_GLOBAL_STEP not in trace or SOURCE_GLOBAL_STEP not in receipts:
        raise KeyError("source trace and receipts must contain step 50")
    if trace[SOURCE_GLOBAL_STEP]["solver_valid"] != "True" or not (
        receipts[SOURCE_GLOBAL_STEP]["forward"]["success"]
        and receipts[SOURCE_GLOBAL_STEP]["adjoint"]["success"]
    ):
        raise ValueError("source step-50 trace/receipt status is not valid")
    evidence = {
        "source_global_step": SOURCE_GLOBAL_STEP,
        "checkpoint": {
            "path": str(paths["checkpoint"]),
            "bytes": paths["checkpoint"].stat().st_size,
            "sha256": sha256(paths["checkpoint"]),
        },
        "inputs": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in paths.items()
            if name != "checkpoint"
        },
        "source_step_trace": trace[SOURCE_GLOBAL_STEP],
        "source_step_solver_receipt": receipts[SOURCE_GLOBAL_STEP],
        "control_contract": {
            "shape": list(arrays["q"].shape),
            "scalar_controls": int(arrays["q"].size),
            "q_vs_Ainv_max_abs_error": matrix_error,
            "all_finite": arrays_finite,
        },
    }
    return evidence, arrays


def preflight(cfg: Config) -> dict[str, Any]:
    evidence, _ = source_evidence(cfg)
    report = {
        "status": "cpu_preflight_passed",
        "gpu_used": False,
        "config": canonical_config(cfg),
        "continuation": {
            "source_global_step": SOURCE_GLOBAL_STEP,
            "local_evaluations": [0, cfg.steps],
            "new_adam_updates": cfg.steps,
            "optimizer_moments": "explicit reset at shared source step 50 for both methods",
            "uninterrupted_trajectory_claimed": False,
            "resume_bundle": "atomic resume.pt after every completed local evaluation",
        },
        "source": evidence,
    }
    path = cfg.output_dir.with_name(cfg.output_dir.name + "-preflight.json")
    atomic_write_json(path, report)
    return report


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def capture_transition_state(
    q: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    current: dict[str, Any],
    optimizer_events: list[dict[str, Any]],
    consecutive_solver_failures: int,
    optimizer_updates: int,
) -> dict[str, Any]:
    return {
        "q": q.detach().clone(),
        "q_grad": q.grad.detach().clone() if q.grad is not None else None,
        "optimizer_state": cpu_tree(optimizer.state_dict()),
        "current": current,
        "optimizer_events": copy.deepcopy(optimizer_events),
        "consecutive_solver_failures": consecutive_solver_failures,
        "optimizer_updates": optimizer_updates,
    }


def restore_transition_state(
    q: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    optimizer_events: list[dict[str, Any]],
    state: dict[str, Any],
) -> tuple[dict[str, Any], int, int]:
    with torch.no_grad():
        q.copy_(state["q"])
    saved_grad = state["q_grad"]
    q.grad = None if saved_grad is None else saved_grad.to(q.device)
    optimizer.load_state_dict(state["optimizer_state"])
    optimizer_events[:] = state["optimizer_events"]
    return (
        state["current"],
        int(state["consecutive_solver_failures"]),
        int(state["optimizer_updates"]),
    )


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def archive_sources(output: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    source_dir = output / "sources"
    source_dir.mkdir()
    paths = [
        Path(__file__),
        BASE_RUNNER_PATH,
        Path(__file__).with_name("historical_adam_physics.py"),
        Path(__file__).with_name("experiment_profile.py"),
    ]
    sources = {}
    for path in paths:
        shutil.copy2(path, source_dir / path.name)
        sources[path.name] = sha256(path)
    provenance = {
        "sources": sources,
        "source_checkpoint": evidence["checkpoint"],
        "frozen_inputs": evidence["inputs"],
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    atomic_write_json(output / "provenance.json", provenance)
    return provenance


def verify_resume_provenance(output: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    provenance = json.loads((output / "provenance.json").read_text())
    for name, expected_hash in provenance["sources"].items():
        if sha256(Path(__file__).with_name(name)) != expected_hash:
            raise ValueError(
                f"runtime source changed since continuation started: {name}"
            )
    if evidence["checkpoint"] != provenance["source_checkpoint"]:
        raise ValueError("source checkpoint changed since continuation started")
    if evidence["inputs"] != provenance["frozen_inputs"]:
        raise ValueError("frozen source-run or fixture inputs changed since start")
    return provenance


def bootstrap_record(
    physics: FacePhysics,
    source_u: np.ndarray,
    current: dict[str, Any],
) -> dict[str, Any]:
    delta = current["u"] - source_u
    top = physics.top
    return {
        "source_seed_reused": True,
        "forward": current["forward"],
        "adjoint": current["adjoint"],
        "all_vertex_displacement_delta_rms_mm": float(
            1000 * np.linalg.norm(delta) / math.sqrt(len(delta))
        ),
        "all_vertex_displacement_delta_max_mm": float(
            1000 * np.linalg.norm(delta, axis=1).max()
        ),
        "target_vertex_displacement_delta_rms_mm": float(
            1000 * np.linalg.norm(delta[top]) / math.sqrt(len(top))
        ),
        "bootstrap_data_objective_mm2": current["data_objective_mm2"],
        "bootstrap_objective_mm2": current["objective"],
        "meaning": (
            "new runtime re-equilibration at the saved q and u seed after both Adam "
            "moment histories were explicitly reset"
        ),
    }


def write_trace(output: Path, trace: list[dict[str, Any]]) -> None:
    if not trace:
        return
    temporary = output / ".trace.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(trace)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output / "trace.csv")


def write_receipts(output: Path, receipts: list[dict[str, Any]]) -> None:
    data = "".join(json.dumps(item, sort_keys=True) + "\n" for item in receipts)
    atomic_write_bytes(output / "solver-receipts.jsonl", data.encode())


def persist_resume(
    output: Path,
    cfg: Config,
    *,
    local_step: int,
    q: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    current: dict[str, Any],
    best: dict[str, Any],
    last_solver_valid: dict[str, Any],
    trace: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    optimizer_events: list[dict[str, Any]],
    consecutive_solver_failures: int,
    optimizer_updates: int,
    bootstrap: dict[str, Any],
    terminal_status: str | None,
) -> None:
    if q.grad is None:
        raise RuntimeError("cannot persist exact resume state without q.grad")
    state = {
        "schema_version": 1,
        "config": canonical_config(cfg),
        "source_global_step": SOURCE_GLOBAL_STEP,
        "evaluated_local_step": local_step,
        "next_local_step": local_step + 1,
        "q": q.detach().cpu(),
        "q_grad": q.grad.detach().cpu(),
        "optimizer_state": cpu_tree(optimizer.state_dict()),
        "current": cpu_tree(current),
        "best": cpu_tree(best),
        "last_solver_valid": cpu_tree(last_solver_valid),
        "trace": copy.deepcopy(trace),
        "receipts": copy.deepcopy(receipts),
        "numerical_failures": copy.deepcopy(failures),
        "optimizer_events": copy.deepcopy(optimizer_events),
        "consecutive_solver_failures": consecutive_solver_failures,
        "optimizer_updates": optimizer_updates,
        "bootstrap": copy.deepcopy(bootstrap),
        "rng_state": rng_state(),
        "terminal_status": terminal_status,
        "resume_semantics": (
            "state is after a completed evaluation and before its next Adam/recovery "
            "transition; retrying from this file repeats no accepted update"
        ),
    }
    atomic_torch_save(output / "resume.pt", state)


def load_resume(
    output: Path,
    cfg: Config,
    q: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    state = torch.load(output / "resume.pt", map_location="cpu", weights_only=False)
    if state["schema_version"] != 1 or state["config"] != canonical_config(cfg):
        raise ValueError("resume bundle schema or resolved config differs")
    with torch.no_grad():
        q.copy_(state["q"].to(q.device))
    q.grad = state["q_grad"].to(q.device)
    optimizer.load_state_dict(state["optimizer_state"])
    restore_rng(state["rng_state"])
    write_trace(output, state["trace"])
    write_receipts(output, state["receipts"])
    return state


def result_is_valid(result: dict[str, Any]) -> bool:
    return bool(result["forward"]["success"] and result["adjoint"]["success"])


def atomic_final_mesh(
    physics: FacePhysics, path: Path, u: np.ndarray, q: torch.Tensor
) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    physics.save_mesh(
        temporary,
        u,
        np.asarray(activation_matrix(q.detach().cpu().numpy())),
    )
    temporary.replace(path)


def archive_terminal_outputs(output: Path, next_local_step: int) -> None:
    names = ("summary.json", "final.npz", "final.vtu")
    sources = [output / name for name in names if (output / name).is_file()]
    if not sources:
        return
    archive = output / "interruptions" / f"before-local-{next_local_step:04d}"
    archive.mkdir(parents=True, exist_ok=True)
    for source in sources:
        destination = archive / source.name
        source_hash = sha256(source)
        if destination.exists():
            if sha256(destination) != source_hash:
                raise ValueError(f"archived terminal output differs: {destination}")
            continue
        temporary = destination.with_name(f".{destination.name}.tmp")
        shutil.copy2(source, temporary)
        if sha256(temporary) != source_hash:
            temporary.unlink(missing_ok=True)
            raise OSError(f"terminal output archive verification failed: {source}")
        temporary.replace(destination)
    for source in sources:
        source.unlink()


def run(cfg: Config) -> None:
    validate_config(cfg)
    if cfg.preflight:
        print(json.dumps(preflight(cfg), indent=2))
        return
    evidence, source_arrays = source_evidence(cfg)
    output = cfg.output_dir.resolve()
    resume_path = output / "resume.pt"
    continuing = output.exists()
    if continuing:
        if not cfg.resume or not resume_path.is_file():
            raise FileExistsError(
                "nonempty continuation output requires its atomic resume.pt"
            )
    else:
        output.mkdir(parents=True)
        atomic_write_json(output / "config.json", canonical_config(cfg))
        atomic_write_json(output / "source.json", evidence)
        provenance = archive_sources(output, evidence)
    if continuing:
        stored_config = json.loads((output / "config.json").read_text())
        if stored_config != canonical_config(cfg):
            raise ValueError("continuation config differs from existing output")
        provenance = verify_resume_provenance(output, evidence)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    configure()
    physics = FacePhysics(
        cfg.fixture,
        skin_factor=0.0,
        fat_factor=1.0,
        muscle_factor=1.0,
        rtol=cfg.forward_rtol,
        atol=cfg.forward_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        soft_nu=0.49,
        fat_nu=0.49,
        fat_model="stable",
        target_name="Smile",
        target_scale=1.0,
    )
    objective = Objective(physics, cfg)
    q = torch.nn.Parameter(torch.zeros((len(physics.ids), 6)))
    optimizer = torch.optim.Adam([q], lr=cfg.inverse_lr, eps=cfg.adam_eps)
    start = time.perf_counter()

    if continuing:
        state = load_resume(output, cfg, q, optimizer)
        current = state["current"]
        best = state["best"]
        best["q"] = best["q"].to(q.device)
        last_solver_valid = state["last_solver_valid"]
        trace = state["trace"]
        receipts = state["receipts"]
        failures = state["numerical_failures"]
        optimizer_events = state["optimizer_events"]
        consecutive_solver_failures = int(state["consecutive_solver_failures"])
        optimizer_updates = int(state["optimizer_updates"])
        bootstrap = state["bootstrap"]
        next_local_step = int(state["next_local_step"])
        if next_local_step <= cfg.steps:
            archive_terminal_outputs(output, next_local_step)
    else:
        with torch.no_grad():
            q.copy_(torch.as_tensor(source_arrays["q"]))
        current = objective(q, source_arrays["u"])
        if not result_is_valid(current):
            raise RuntimeError(
                "continuation bootstrap forward or adjoint was unsuccessful"
            )
        bootstrap = bootstrap_record(physics, source_arrays["u"], current)
        best = {"step": -1, "q": q.detach().clone(), "result": current, "row": None}
        last_solver_valid: dict[str, Any] = {}
        trace: list[dict[str, Any]] = []
        receipts: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        optimizer_events: list[dict[str, Any]] = [
            {
                "local_step": 0,
                "source_global_step": SOURCE_GLOBAL_STEP,
                "event": "explicit_shared_source_adam_moment_reset",
                "learning_rate": cfg.inverse_lr,
                "epsilon": cfg.adam_eps,
            }
        ]
        consecutive_solver_failures = 0
        optimizer_updates = 0
        next_local_step = 0

    status = "fixed_budget_completed_not_stationarity_certified"
    while next_local_step <= cfg.steps:
        if STOP_REQUESTED is not None and trace:
            status = "signal_checkpointed_best_valid_retained"
            break
        local_step = next_local_step
        if local_step > 0:
            accepted = capture_transition_state(
                q,
                optimizer,
                current,
                optimizer_events,
                consecutive_solver_failures,
                optimizer_updates,
            )
            attempted_transition = {
                "local_step": local_step,
                "event": "adam_step",
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            try:
                if consecutive_solver_failures >= 3:
                    with torch.no_grad():
                        q.copy_(best["q"])
                    optimizer.param_groups[0]["lr"] *= 0.5
                    optimizer.state.clear()
                    attempted_transition = {
                        "local_step": local_step,
                        "event": "three_solver_failures_restore_best_halve_lr_reset_adam_moments",
                        "restored_local_step": best["step"],
                        "new_learning_rate": optimizer.param_groups[0]["lr"],
                    }
                    optimizer_events.append(attempted_transition)
                    consecutive_solver_failures = 0
                    current = objective(q, best["result"]["u"])
                else:
                    optimizer.step()
                    optimizer_updates += 1
                    current = objective(q, accepted["current"]["u"])
            except (ForwardConvergenceError, AdjointConvergenceError) as error:
                current, consecutive_solver_failures, optimizer_updates = (
                    restore_transition_state(q, optimizer, optimizer_events, accepted)
                )
                failures.append(
                    {
                        "attempted_local_step": local_step,
                        "nominal_global_step": SOURCE_GLOBAL_STEP + local_step,
                        "type": type(error).__name__,
                        "message": str(error),
                        "receipt": error.receipt,
                        "rolled_back_transition": attempted_transition,
                    }
                )
                status = "numerical_failure_best_valid_retained"
                persist_resume(
                    output,
                    cfg,
                    local_step=local_step - 1,
                    q=q,
                    optimizer=optimizer,
                    current=current,
                    best=best,
                    last_solver_valid=last_solver_valid,
                    trace=trace,
                    receipts=receipts,
                    failures=failures,
                    optimizer_events=optimizer_events,
                    consecutive_solver_failures=consecutive_solver_failures,
                    optimizer_updates=optimizer_updates,
                    bootstrap=bootstrap,
                    terminal_status=status,
                )
                break

        metrics = endpoint_metrics(physics, q, current)
        if q.grad is None:
            raise RuntimeError("continuation evaluation has no activation gradient")
        row = {
            "step": local_step,
            "local_step": local_step,
            "source_global_step": SOURCE_GLOBAL_STEP,
            "nominal_global_step": SOURCE_GLOBAL_STEP + local_step,
            "objective": current["objective"],
            "data_objective_mm2": current["data_objective_mm2"],
            "smoothness": current["smoothness"],
            "smoothness_penalty_mm2": current["smoothness_penalty_mm2"],
            "gradient_rms": float(
                torch.linalg.vector_norm(q.grad) / math.sqrt(q.numel())
            ),
            "elapsed_s_this_process": time.perf_counter() - start,
            "forward_steps": current["forward"]["steps"],
            "forward_grad_norm": current["forward"]["grad_norm"],
            **metrics,
        }
        valid = result_is_valid(current)
        row.update(
            forward_success=bool(current["forward"]["success"]),
            adjoint_success=bool(current["adjoint"]["success"]),
            solver_valid=valid,
        )
        trace.append(row)
        receipts.append(
            {
                "step": local_step,
                "local_step": local_step,
                "source_global_step": SOURCE_GLOBAL_STEP,
                "nominal_global_step": SOURCE_GLOBAL_STEP + local_step,
                "forward": current["forward"],
                "adjoint": current["adjoint"],
            }
        )
        consecutive_solver_failures = 0 if valid else consecutive_solver_failures + 1
        if valid:
            last_solver_valid = {"step": local_step, "row": dict(row)}
        if valid and (
            best["step"] < 0 or current["objective"] < best["result"]["objective"]
        ):
            best = {
                "step": local_step,
                "q": q.detach().clone(),
                "result": current,
                "row": dict(row),
            }
        write_trace(output, trace)
        write_receipts(output, receipts)
        atomic_write_json(output / "numerical-failures.json", failures)
        atomic_write_json(output / "optimizer-events.json", optimizer_events)
        if local_step % cfg.checkpoint_interval == 0 or local_step == cfg.steps:
            atomic_snapshot(
                output / f"step-{local_step:04d}.npz", q, current, local_step
            )
        atomic_snapshot(output / "latest.npz", q, current, local_step)
        persist_resume(
            output,
            cfg,
            local_step=local_step,
            q=q,
            optimizer=optimizer,
            current=current,
            best=best,
            last_solver_valid=last_solver_valid,
            trace=trace,
            receipts=receipts,
            failures=failures,
            optimizer_events=optimizer_events,
            consecutive_solver_failures=consecutive_solver_failures,
            optimizer_updates=optimizer_updates,
            bootstrap=bootstrap,
            terminal_status=None,
        )
        cherries.set_step(local_step)
        cherries.log_metrics(
            {
                "historical_continuation/objective": row["objective"],
                "historical_continuation/fit_rms_mm": row["fit_rms_mm"],
                "historical_continuation/smoothness": row["smoothness"],
                "historical_continuation/gradient_rms": row["gradient_rms"],
            },
            step=local_step,
        )
        if STOP_REQUESTED is not None:
            status = "signal_checkpointed_best_valid_retained"
            break
        if local_step == cfg.steps:
            if consecutive_solver_failures:
                status = (
                    "fixed_budget_completed_with_unresolved_solver_failures_"
                    "best_valid_retained"
                )
            break
        next_local_step += 1

    if not trace or best["step"] < 0 or not last_solver_valid:
        raise RuntimeError("continuation has no solver-valid evaluated state")
    best_q = best["q"]
    best_result = best["result"]
    atomic_snapshot(output / "final.npz", best_q, best_result, best["step"])
    atomic_final_mesh(physics, output / "final.vtu", best_result["u"], best_q)
    summary = {
        "schema_version": 1,
        "status": status,
        "convergence": {
            "claimed": False,
            "label": status,
            "declared_steps": cfg.steps,
            "last_evaluated_step": trace[-1]["step"],
            "last_solver_valid_step": last_solver_valid["step"],
            "best_valid_step": best["step"],
            "best_endpoint_policy": (
                "lowest continuation objective with successful forward and adjoint solves"
            ),
        },
        "config": canonical_config(cfg),
        "continuation": {
            "source_global_step": SOURCE_GLOBAL_STEP,
            "local_step_range": [0, trace[-1]["step"]],
            "nominal_global_step_range": [
                SOURCE_GLOBAL_STEP,
                SOURCE_GLOBAL_STEP + trace[-1]["step"],
            ],
            "optimizer_moments_at_source": "explicitly reset for both methods",
            "uninterrupted_original_trajectory_claimed": False,
            "best_selection_scope": "continuation evaluations only",
            "source": evidence,
            "bootstrap_re_equilibration": bootstrap,
            "resume_bundle": {
                "path": "resume.pt",
                "atomic_after_every_completed_evaluation": True,
                "contains_optimizer_moments_and_q_gradient": True,
            },
        },
        "objective": {
            "data": "uniform Cartesian MSE times 1e6, exactly matching June",
            "smoothness_weight_mm2": cfg.smoothness_weight,
            "smoothness_graph": "within identical MuscleId shared tetrahedral faces only",
            "smoothness_length_m": cfg.smooth_length,
            "smoothness_normalization": (
                "finite-volume conductance divided by active muscle-fraction volume and "
                "(-log(0.8))^2"
            ),
        },
        "optimizer": {
            "implementation": "torch.optim.Adam",
            "learning_rate": cfg.inverse_lr,
            "epsilon": cfg.adam_eps,
            "state_updates": optimizer_updates,
            "source_moment_reset": True,
            "events": optimizer_events,
        },
        "materials": physics.material_spec,
        "solver": {
            "forward": physics.forward_tolerance,
            "adjoint": {
                "implementations": ["CupyCG", "CupyMinRes"],
                "max_steps": 10_000,
                "rtol": cfg.adjoint_rtol,
                "atol": 0.0,
            },
        },
        "mesh": {
            "points": len(physics.points),
            "tetrahedra": len(physics.tets),
            "active_tetrahedra": len(physics.ids),
            "scalar_controls": int(q.numel()),
            "muscle_regions": physics.n_regions,
            "within_same_MuscleId_shared_face_edges": len(physics.graph[0]),
            "fixed_vertices": int(
                np.count_nonzero(np.asarray(physics.mesh.point_data["IsFixed"], bool))
            ),
            "target_vertices": len(physics.top),
        },
        "best": {**best["row"], "output": "final.npz and final.vtu"},
        "last_solver_valid": last_solver_valid["row"],
        "last_evaluated": trace[-1],
        "numerical_failures": failures,
        "geometry_rejection_enabled": False,
        "provenance": provenance,
        "wall_s_this_process": time.perf_counter() - start,
        "signal": STOP_REQUESTED,
    }
    atomic_write_json(output / "summary.json", summary)
    persist_resume(
        output,
        cfg,
        local_step=trace[-1]["step"],
        q=q,
        optimizer=optimizer,
        current=current,
        best=best,
        last_solver_valid=last_solver_valid,
        trace=trace,
        receipts=receipts,
        failures=failures,
        optimizer_events=optimizer_events,
        consecutive_solver_failures=consecutive_solver_failures,
        optimizer_updates=optimizer_updates,
        bootstrap=bootstrap,
        terminal_status=status,
    )
    for name in (
        "config.json",
        "source.json",
        "provenance.json",
        "trace.csv",
        "solver-receipts.jsonl",
        "optimizer-events.json",
        "numerical-failures.json",
        "resume.pt",
        "final.npz",
        "final.vtu",
        "summary.json",
    ):
        cherries.log_output(output / name)


if __name__ == "__main__":
    cherries.main(
        run, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
