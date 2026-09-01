"""Isolated fixed-horizon shared-control refinement audit."""

from __future__ import annotations

# ruff: noqa: BLE001, EM101, EM102, FBT003, SLF001, TRY003
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import scipy.sparse.linalg as spla

ROOT = Path("/home/liblaf/Projects/liblaf/apple")
SOURCE = (
    ROOT / "exp/2026/08/31/unreachable-pork-factor-study/src/60-run-pork-folding-2d.py"
)
DATA = ROOT / "exp/2026/08/31/unreachable-pork-factor-study/data/60-pork-folding-2d"
OUT = ROOT / "tmp/audit-fem-refinement/results"
NAMES = (
    "l010-band-shared-nu49",
    "l010-full-shared-nu35",
    "l100-band-shared-nu35",
    "l100-full-shared-nu49",
)
CYCLE_LRS = (0.003, 0.001, 0.0003, 0.0001)
UPDATES = 300
DECAY = 0.995
TIGHT_FORWARD_TOLERANCE = 1.0e-10


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("folding2d_refinement", SOURCE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate(
    module: Any,
    mesh: Any,
    a: np.ndarray,
    u: np.ndarray,
    case: Any,
    forward_tolerance: float = TIGHT_FORWARD_TOLERANCE,
) -> dict[str, Any]:
    cfg = module.Config(
        max_steps=0,
        validate_derivatives=False,
        forward_max_iterations=1000,
        forward_tolerance=forward_tolerance,
    )
    state = module.core.solve(mesh, a, "stable", u, cfg)
    if not state.converged:
        raise RuntimeError(
            f"forward failed: {state.failure!r}, iterations={state.iterations}"
        )
    objective, du, _ = module.loss(mesh, state.u, module.HEIGHT, "l2", case.length)
    _, _, _, B, *_ = module.assembly(mesh, state.u, a, "stable", False, True)
    gradient = np.asarray(B.T @ spla.spsolve(state.h, -du)).ravel()
    if not (np.isfinite(objective) and np.isfinite(gradient).all()):
        raise FloatingPointError("non-finite objective or exact implicit gradient")
    metrics = module.metrics(mesh, state.u, a, module.HEIGHT, case.length)
    return {
        "objective": float(objective),
        "gradient": gradient,
        "gradient_rms": float(np.linalg.norm(gradient) / math.sqrt(len(gradient))),
        "gradient_inf": float(np.max(np.abs(gradient))),
        "target_rms": float(metrics["top_target_rms"]),
        "min_det_f": float(state.det_f.min()),
        "min_det_g": float(state.det_g.min()),
        "min_det_ainv": float(state.det_a.min()),
        "forward_iterations": int(state.iterations),
        "forward_tolerance": forward_tolerance,
        "state": state,
    }


def scalar(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in record.items() if key not in {"gradient", "state"}
    }


def production_seed(
    module: Any, case: Any
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    module._SHARED = True
    mesh = module.build_mesh(case)
    frame = pv.read(DATA / case.name / "final.vtu")
    activation = np.column_stack(
        [
            np.asarray(frame.cell_data[name])
            for name in ("ActivationXX", "ActivationYY", "ActivationXY")
        ]
    )
    active = activation[mesh.muscle]
    a = active[0].copy()
    if not np.allclose(active, a, rtol=0.0, atol=1.0e-12):
        raise AssertionError(f"{case.name} production final is not one shared 3-vector")
    full_u = np.asarray(frame.point_data["Displacement"], float)[:, :2].ravel()
    u = full_u[mesh.free]
    loose = evaluate(module, mesh, a, u, case, forward_tolerance=1.0e-8)
    tight = evaluate(
        module, mesh, a, u, case, forward_tolerance=TIGHT_FORWARD_TOLERANCE
    )
    return mesh, a, tight["state"].u, loose, tight


def cycle(
    module: Any, mesh: Any, a: np.ndarray, u: np.ndarray, case: Any, base_lr: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    first, second = np.zeros_like(a), np.zeros_like(a)
    history, updates = [], []
    for step in range(UPDATES + 1):
        current = evaluate(module, mesh, a, u, case)
        u = current["state"].u
        history.append({"step": step, "controls": a.tolist(), **scalar(current)})
        if step == UPDATES:
            break
        first = 0.9 * first + 0.1 * current["gradient"]
        second = 0.999 * second + 0.001 * current["gradient"] ** 2
        lr = base_lr * DECAY**step
        delta = (
            lr
            * (first / (1.0 - 0.9 ** (step + 1)))
            / (np.sqrt(second / (1.0 - 0.999 ** (step + 1))) + 1.0e-8)
        )
        a -= delta
        updates.append(float(np.linalg.norm(delta) / math.sqrt(len(delta))))
    tail = history[-100:]
    tail_losses = np.asarray([row["objective"] for row in tail])
    return (
        a,
        u,
        {
            "base_learning_rate": base_lr,
            "decay": DECAY,
            "updates": UPDATES,
            "evaluations": len(history),
            "initial": history[0],
            "final": history[-1],
            "minimum_det_f": min(row["min_det_f"] for row in history),
            "minimum_det_g": min(row["min_det_g"] for row in history),
            "minimum_det_ainv": min(row["min_det_ainv"] for row in history),
            "final_100_relative_loss_span": float(
                np.ptp(tail_losses) / max(abs(float(tail_losses.min())), 1.0e-30)
            ),
            "final_100_mean_update_rms": float(np.mean(updates[-100:])),
            "final_100_max_update_rms": float(np.max(updates[-100:])),
        },
    )


def schedule(module: Any, case: Any, lrs: tuple[float, ...]) -> dict[str, Any]:
    mesh, a, u, loose_initial, tight_initial = production_seed(module, case)
    cycles = []
    for lr in lrs:
        a, u, report = cycle(module, mesh, a, u, case, lr)
        cycles.append(report)
    return {
        "case": case.name,
        "schedule": list(lrs),
        "moments": "reset at each cycle",
        "constraints": "none; no activation bounds, determinant guards, or repairs",
        "production_initial_forward_tolerance_1e-8": scalar(loose_initial),
        "production_initial_forward_tolerance_1e-10": scalar(tight_initial),
        "final_controls": a.tolist(),
        "cycles": cycles,
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    module = load_module()
    available = {case.name: case for case in module.cases()}
    reports, failures = [], []
    for name in NAMES:
        try:
            reports.append(schedule(module, available[name], CYCLE_LRS))
        except Exception as error:
            failures.append({"case": name, "error": repr(error)})
    repeated = None
    hard = "l100-full-shared-nu49"
    try:
        repeated = schedule(module, available[hard], (0.003, 0.003, 0.003, 0.003))
    except Exception as error:
        failures.append(
            {"case": hard, "schedule": "repeated-0.003", "error": repr(error)}
        )
    report = {"reports": reports, "repeated_003": repeated, "failures": failures}
    (OUT / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n")
    if failures:
        raise RuntimeError(f"refinement audit failures: {failures}")


if __name__ == "__main__":
    main()
