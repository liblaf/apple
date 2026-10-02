"""Audit and refine selected endpoints with tighter inner numerical tolerances.

This starts from persisted controls, uses the same objective and constraints,
and keeps the initial fixed-budget results intact in their original folders.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path

import activation_models as am
import numpy as np
import pydantic_settings as ps
import torch
from block_physics import Physics, configure
from experiment_profile import ProfileCometNoCommit
from liblaf.peach.linalg.cupy import CupyCG

from liblaf import cherries

spec = importlib.util.spec_from_file_location(
    "polish_matrix", Path(__file__).with_name("20-inverse-constraint-matrix.py")
)
assert spec and spec.loader
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("35-polish", mkdir=True)
    source_dir: Path = Path("data/30-followups")
    endpoints: str = (
        "strength-0.1/G-MS,strength-0.1/F-MS,strength-1/G-MS,strength-1/F-MS,"
        "noise005-seed20260917/F-MS,fiber-25deg/F-MS,"
        "transverse-natural-stretch-fixed/F-MS,fiber-10deg/F-MS,cap-0.1/F-MS"
    )
    steps: int = 100


def resolve_original(path: Path, method: str) -> Path:
    candidates = [
        path / target / method
        for target in ("clean", "noisy")
        if (path / target / method / "summary.json").exists()
    ]
    assert len(candidates) == 1, (
        f"Expected exactly one clean/noisy source for {path.name}/{method}, "
        f"found {candidates}"
    )
    return candidates[0]


def polish(p, original, fixture, source_cfg, cfg, dest):
    start = time.perf_counter()
    parent = base.json.loads((original / "summary.json").read_text())
    vals = source_cfg.copy()
    vals.pop("output_dir", None)
    bc = base.Config(_cli_parse_args=False, **vals)
    case = base.Case(**parent["case"])
    graph = am.face_graph(p.points, p.tets, p.ids)
    angle = math.radians(bc.fiber_angle)
    fibers = torch.zeros((len(p.ids), 3))
    fibers[:, 0] = math.cos(angle)
    fibers[:, 2] = math.sin(angle)
    clean, target, scale = (
        fixture["clean"],
        fixture[parent["target"]],
        float(fixture["D"]),
    )
    obj = base.Objective(p, target, scale, case, bc, fibers, graph)
    with np.load(original / "final.npz") as old:
        q = torch.as_tensor(old["q"].copy())
        seed = old["u"].copy()
    initial = obj(q, seed)
    current = initial
    history = []
    trace = []
    n = len(p.ids) if case.mode != "Shared" else 1
    status = "step_budget"
    for step in range(cfg.steps + 1):
        grad = current["grad"]
        pg = q - am.project(q - n * grad, case.mode, bc.amax)
        kkt = float(torch.linalg.vector_norm(pg) / math.sqrt(q.numel()) / base.AREF)
        trace.append(
            {
                "step": step,
                "objective": current["value"],
                "kkt": kkt,
                "forward_grad_norm": current["forward"]["grad_norm"],
            }
        )
        if step % 20 == 0:
            LOG.info(
                "%s polish=%d KKT=%.5g objective=%.9g",
                case.name,
                step,
                kkt,
                current["value"],
            )
        cherries.log_metrics(
            {
                case.name + "/polished_kkt": kkt,
                case.name + "/polished_objective": current["value"],
            },
            step=step,
        )
        if kkt < bc.kkt_tol:
            status = "projected_stationary"
            break
        if step == cfg.steps:
            break
        d = base.direction_lbfgs(grad, history, n * 0.01)
        if (
            float(torch.sum(grad * (am.project(q + d, case.mode, bc.amax) - q)))
            >= -1e-18
        ):
            history.clear()
            d = -n * 0.01 * grad
        accepted = None
        for trial in range(18):
            candidate = am.project(q + (0.5**trial) * d, case.mode, bc.amax)
            slope = float(torch.sum(grad * (candidate - q)))
            if slope >= 0:
                continue
            attempt = obj(candidate, current["u"])
            if attempt["value"] <= current["value"] + 1e-4 * slope:
                accepted = candidate, attempt
                break
        if accepted is None:
            status = "line_search_stalled"
            break
        new_q, new = accepted
        s = new_q - q
        y = new["grad"] - grad
        sy = torch.sum(s * y)
        if float(sy) > 1e-10 * float(
            torch.linalg.vector_norm(s) * torch.linalg.vector_norm(y)
        ):
            history.append((s, y, 1 / sy))
            history = history[-12:]
        q, current = new_q, new
    met = base.measurements(
        p,
        current["u"],
        target,
        clean,
        current["Ainv"],
        current["H"],
        q.cpu().numpy(),
        (case, parent["target"]),
        bc,
        graph,
        scale,
    )
    endpoint_change = (
        base.rms(current["u"][p.top] - initial["u"][p.top], p.weights) / scale
    )
    grad = current["grad"]
    interior_margin = 2e-3
    interior_mask = (q > interior_margin) & (q < bc.amax - interior_margin)
    bound_active_scalar = case.mode == "F" and not bool(torch.all(interior_mask))
    if bound_active_scalar:
        direction_kind = "interior_masked_gradient"
        direction_mask = interior_mask
    else:
        direction_kind = "full_gradient"
        direction_mask = torch.ones_like(q, dtype=torch.bool)
    direction_source = torch.where(direction_mask, grad, torch.zeros_like(grad))
    direction_norm = torch.linalg.vector_norm(direction_source)
    direction_available = float(direction_norm) > 0
    unavailable_reason = None
    if direction_available:
        direction = direction_source / direction_norm
        analytic = float(torch.sum(grad * direction))
    else:
        direction = torch.zeros_like(grad)
        analytic = None
        unavailable_reason = (
            "No nonzero gradient remains on coordinates more than 0.002 from "
            "both scalar-F bounds."
            if bound_active_scalar
            else "The full gradient is zero, so no normalized direction is available."
        )
    checks = []
    if direction_available:
        for eps in (1e-3, 3e-4):
            for sign in (-1, 1):
                trial = q + sign * eps * direction
                assert torch.allclose(
                    am.project(trial, case.mode, bc.amax),
                    trial,
                    atol=1e-12,
                    rtol=0,
                )
            plus = obj(q + eps * direction, current["u"])
            minus = obj(q - eps * direction, current["u"])
            fd = (plus["value"] - minus["value"]) / (2 * eps)
            checks.append(
                {
                    "epsilon": eps,
                    "analytic": analytic,
                    "finite_difference": fd,
                    "relative_error": abs(fd - analytic)
                    / max(abs(fd), abs(analytic), 1e-12),
                    "plus_value": plus["value"],
                    "minus_value": minus["value"],
                    "plus_forward": plus["forward"],
                    "minus_forward": minus["forward"],
                    "plus_adjoint": plus["adjoint"],
                    "minus_adjoint": minus["adjoint"],
                }
            )
    fd_agreement = (
        abs(checks[0]["finite_difference"] - checks[1]["finite_difference"])
        / max(
            abs(checks[0]["finite_difference"]),
            abs(checks[1]["finite_difference"]),
            1e-12,
        )
        if direction_available
        else None
    )
    reset_u = (
        p.solve(am.packed(torch.as_tensor(current["Ainv"])), np.zeros_like(p.points))
        .detach()
        .cpu()
        .numpy()
    )
    reset_forward = dict(p.last_forward)
    branch_delta = base.rms(reset_u[p.top] - current["u"][p.top], p.weights) / scale
    summary = {
        "source": str(original),
        "source_summary_sha256": hashlib.sha256(
            (original / "summary.json").read_bytes()
        ).hexdigest(),
        "source_final_sha256": hashlib.sha256(
            (original / "final.npz").read_bytes()
        ).hexdigest(),
        "case": dataclasses.asdict(case),
        "target": parent["target"],
        "status": status,
        "steps": step,
        "original_status": parent["status"],
        "original_kkt": parent["final"]["projected_kkt"],
        "strict_start_kkt": trace[0]["kkt"],
        "projected_kkt": trace[-1]["kkt"],
        "endpoint_change_rms_over_D": endpoint_change,
        "objective_start": initial["value"],
        "persisted_objective": parent["final"]["objective"],
        "strict_start_minus_persisted_objective": initial["value"]
        - parent["final"]["objective"],
        "objective_final": current["value"],
        "forward_atol": p.atol,
        "forward_rtol": p.rtol,
        "wall_s": time.perf_counter() - start,
        "final": met,
        "directional_gradient_audit": {
            "available": direction_available,
            "unavailable_reason": unavailable_reason,
            "direction_kind": direction_kind,
            "interior_margin": interior_margin,
            "direction_mask_count": int(torch.count_nonzero(direction_mask)),
            "direction_mask_fraction": float(torch.mean(direction_mask.double())),
            "direction_analytic": analytic,
            "finite_difference_relative_errors": [
                check["relative_error"] for check in checks
            ],
            "array_file": "final.npz",
            "mask_array": "directional_gradient_mask",
            "direction_array": "directional_gradient_direction",
        },
        "directional_gradient": checks,
        "finite_difference_scale_disagreement": fd_agreement,
        "branch_reset_difference_over_D": branch_delta,
        "reset_forward": reset_forward,
    }
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest / "final.npz",
        q=q.cpu().numpy(),
        u=current["u"],
        Ainv=current["Ainv"],
        H=current["H"],
        reset_u=reset_u,
        directional_gradient_mask=direction_mask.cpu().numpy(),
        directional_gradient_direction=direction.cpu().numpy(),
    )
    base.write_json(dest / "trace.json", trace)
    base.write_json(dest / "summary.json", summary)
    return summary


def main(cfg: Config):
    configure()
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    base.write_json(out / "config.json", cfg.model_dump(mode="json"))
    sources = out / "sources"
    sources.mkdir(exist_ok=True)
    for f in Path(__file__).parent.glob("*.py"):
        shutil.copy2(f, sources / f.name)
    p = Physics(24, 10, rtol=1e-8, atol=1e-13)
    p.diff.adjoint_solver = CupyCG(maxiter=40000, rtol=1e-9, atol=0.0)
    results = []
    for endpoint in cfg.endpoints.split(","):
        group, method = endpoint.split("/")
        path = cfg.source_dir / group
        source_cfg = base.json.loads((path / "run-config.json").read_text())
        fixture = dict(np.load(path / "fixture.npz"))
        original = resolve_original(path, method)
        assert (
            original.parent.name
            == base.json.loads((original / "summary.json").read_text())["target"]
        )
        results.append(
            polish(p, original, fixture, source_cfg, cfg, out / group / method)
        )
        base.write_json(out / "summary.json", results)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
