"""Matched nonlinear tetrahedral inverse fits with activation-space priors.

The surface high-pass diagnostics are evaluation only. They never enter the
objective. The optimizer is projected L-BFGS with Armijo backtracking; every
trial is solved from the last accepted equilibrium (a common branch policy).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import activation_models as am
import numpy as np
import pydantic_settings as ps
import scipy
import scipy.ndimage as ndi
import torch
from block_physics import ROOT, SOURCE, Physics, array_hash, configure
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

LOG = logging.getLogger(__name__)
AREF = -math.log(0.8)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("20-matrix", mkdir=True)
    nx: int = 24
    ny: int = 10
    steps: int = 160
    rows: str = "all"
    targets: str = "clean,noisy,mismatch"
    seed: int = 20260906
    noise_rms: float = 0.02
    mismatch_rms: float = 0.5
    weight: float = 0.01
    amax: float = -math.log(0.65)
    fiber_angle: float = 0.0
    gamma: float = 0.5
    init: float = 0.0
    forward_rtol: float = 1e-6
    forward_atol: float = 1e-11
    kkt_tol: float = 1e-3
    validate_gradients: bool = True
    resume: bool = False


@dataclass(frozen=True)
class Case:
    name: str
    mode: str
    lm: float = 0.0
    ls: float = 0.0


def cases(cfg):
    rows = [Case("Raw6", "Raw6"), Case("G6", "G6")]
    for label, mode in (("G", "G5"), ("F", "F")):
        rows.extend(
            [
                Case(label, mode),
                Case(label + "-M", mode, cfg.weight),
                Case(label + "-S", mode, 0.0, cfg.weight),
                Case(label + "-MS", mode, cfg.weight, cfg.weight),
            ]
        )
    rows.extend([Case("Shared", "Shared"), Case("G6-MS", "G6", cfg.weight, cfg.weight)])
    selected = (
        [x for x in rows if x.name != "G6-MS"]
        if cfg.rows == "all"
        else [x for x in rows if x.name in cfg.rows.split(",")]
    )
    assert selected
    return selected


def write_json(path, data):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x

    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")


def rms(a, w):
    return (
        float(np.sqrt(np.sum(w * np.sum(a * a, axis=-1))))
        if a.ndim > 1
        else float(np.sqrt(np.sum(w * a * a)))
    )


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / np.sum(weights)
    return float(
        values[order[min(np.searchsorted(cumulative, quantile), len(order) - 1)]]
    )


def lowpass(physics, a, length=0.06):
    size = physics.nx + 1
    shape = (size, size) + a.shape[1:]
    sigma = (length * physics.nx, length * physics.nx) + ((0,) if a.ndim == 2 else ())
    return ndi.gaussian_filter(a.reshape(shape), sigma=sigma, mode="reflect").reshape(
        a.shape
    )


def measurements(p, u, target, clean, Ainv, H, q, case, cfg, graph, scale):
    top, w = p.top, p.weights
    residual = u[top] - target[top]
    ref = target[top] if "mismatch" in case[1] else clean[top]
    # Both amplitude measures include all vector components; remove the
    # spatial mean for the second so uniform movement cannot hide pattern loss.
    pred_lp, ref_lp = lowpass(p, u[top]), lowpass(p, ref)
    out = {
        "fit_rms_over_D": rms(residual, w) / scale,
        "clean_rms_over_D": rms(u[top] - clean[top], w) / scale,
        "fit_p95_over_D": weighted_quantile(np.linalg.norm(residual, axis=1), w, 0.95)
        / scale,
        "fit_max_over_D": float(np.linalg.norm(residual, axis=1).max()) / scale,
        "motion_rms_over_D": rms(u[top], w) / scale,
        "interior_error_rms_over_D": float(
            np.sqrt(np.mean(np.sum((u - clean) ** 2, axis=1)))
        )
        / scale,
    }
    for suffix, pred, truth in (
        ("", pred_lp, ref_lp),
        (
            "_demeaned",
            pred_lp - np.sum(w[:, None] * pred_lp, axis=0),
            ref_lp - np.sum(w[:, None] * ref_lp, axis=0),
        ),
    ):
        denom = np.sum(w[:, None] * truth**2)
        amp = (
            float(np.sum(w[:, None] * pred * truth) / denom)
            if denom > 1e-30
            else math.nan
        )
        out["low_frequency_amplitude" + suffix] = amp
        out["low_frequency_projection_residual" + suffix] = (
            rms(pred - amp * truth, w) / scale
        )
    for length in (0.03, 0.06, 0.12):
        excess = (u[top] - clean[top])[:, 1]
        hp = excess - lowpass(p, excess, length)
        raw = u[top, 1] - lowpass(p, u[top, 1], length)
        name = f"{length:.2f}"
        out[f"excess_hp_{name}_over_D"] = rms(hp, w) / scale
        out[f"raw_hp_{name}_over_D"] = rms(raw, w) / scale
        xy = p.points[top][:, (0, 2)]
        crop = np.all((xy >= 0.15) & (xy <= 0.85), axis=1)
        out[f"excess_hp_crop_{name}_over_D"] = (
            rms(hp[crop], w[crop] / w[crop].sum()) / scale
        )
    detf, deta, detg = p.determinants(u, Ainv)
    for name, arr in (("detF", detf), ("detAinv", deta), ("detG", detg)):
        out[name + "_min"] = float(arr.min())
        out[name + "_max"] = float(arr.max())
        out[name + "_nonpositive_volume_fraction"] = float(
            np.sum(p.volumes_all[arr <= 0]) / p.volumes_all.sum()
        )
    eig, eigvec = np.linalg.eigh(Ainv)
    singular = np.linalg.svd(Ainv, compute_uv=False)
    out["activation_condition_max"] = float((singular[:, 0] / singular[:, -1]).max())
    out["activation_eigenvalue_min"] = float(eig.min())
    out["activation_offset_rms"] = float(
        np.sqrt(
            np.average(np.sum((Ainv - np.eye(3)) ** 2, axis=(1, 2)), weights=p.volumes)
        )
    )
    out["activation_log_rms"] = (
        float(
            np.sqrt(
                np.average(np.sum(np.log(eig) ** 2, axis=1) / 1.5, weights=p.volumes)
            )
        )
        if np.all(eig > 0)
        else None
    )
    ei, ej, ew = graph
    out["activation_offset_jump_rms"] = float(
        np.sqrt(
            np.average(
                np.sum((Ainv[ei] - Ainv[ej]) ** 2, axis=(1, 2)) / 1.5, weights=ew
            )
        )
    )
    if np.all(eig > 0):
        log_h = (eigvec * np.log(eig)[:, None, :]) @ eigvec.transpose(0, 2, 1)
        out["activation_tensor_jump_rms"] = float(
            np.sqrt(
                np.average(
                    np.sum((log_h[ei] - log_h[ej]) ** 2, axis=(1, 2)) / 1.5, weights=ew
                )
            )
        )
    else:
        out["activation_tensor_jump_rms"] = None
    if case[0].mode in ("F", "Shared"):
        a = np.broadcast_to(q.reshape(-1), (len(p.ids),))
        out["scalar_rms"] = float(np.sqrt(np.average(a * a, weights=p.volumes)))
        out["scalar_neighbor_jump_rms"] = float(
            np.sqrt(np.average((a[ei] - a[ej]) ** 2, weights=ew))
        )
        out["max_natural_shortening"] = float((1 - np.exp(-a)).max())
        out["bound_fraction"] = float(
            np.average((a >= cfg.amax - 1e-7), weights=p.volumes)
        )
    elif case[0].mode != "Raw6":
        out["bound_fraction"] = float(
            np.average(
                np.linalg.norm(q, axis=1) >= math.sqrt(1.5) * cfg.amax - 1e-7,
                weights=p.volumes,
            )
        )
    return out


class Objective:
    def __init__(self, p, target, scale, case, cfg, fibers, graph):
        self.p, self.target = p, torch.as_tensor(target)
        self.scale, self.case, self.cfg, self.fibers = scale, case, cfg, fibers
        self.ei, self.ej, self.ew = (torch.as_tensor(x) for x in graph)
        self.volumes = torch.as_tensor(p.volumes)
        self.calls = 0

    def __call__(self, value, seed):
        q = value.detach().clone().requires_grad_(True)
        Ainv, H = am.matrices(q, self.case.mode, self.fibers, self.cfg.gamma)
        u = self.p.solve(am.packed(Ainv), seed)
        fit = (
            torch.sum(
                self.p.weights_t[:, None]
                * (u[self.p.top_t] - self.target[self.p.top_t]) ** 2
            )
            / self.scale**2
        )
        if self.case.mode in ("F", "Shared"):
            a = (
                q.reshape(-1).expand(len(self.p.ids))
                if self.case.mode == "Shared"
                else q[:, 0]
            )
            magnitude = torch.sum(self.volumes * a * a) / self.volumes.sum() / AREF**2
            smooth = (
                0.1**2
                * torch.sum(self.ew * (a[self.ei] - a[self.ej]) ** 2)
                / self.volumes.sum()
                / AREF**2
            )
        else:
            magnitude = torch.sum(self.volumes * H.square().sum(dim=(1, 2))) / (
                1.5 * self.volumes.sum() * AREF**2
            )
            smooth = (
                0.1**2
                * torch.sum(
                    self.ew * (H[self.ei] - H[self.ej]).square().sum(dim=(1, 2))
                )
                / (1.5 * self.volumes.sum() * AREF**2)
            )
        objective = fit + self.case.lm * magnitude + self.case.ls * smooth
        objective.backward()
        adjoint = self.p.check_adjoint()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        self.calls += 1
        return {
            "value": float(objective.detach()),
            "fit": float(fit.detach()),
            "magnitude": float(magnitude.detach()),
            "smooth": float(smooth.detach()),
            "grad": q.grad.detach().clone(),
            "u": u.detach().cpu().numpy().copy(),
            "Ainv": Ainv.detach().cpu().numpy(),
            "H": H.detach().cpu().numpy(),
            "forward": dict(self.p.last_forward),
            "adjoint": adjoint,
        }


def direction_lbfgs(grad, history, initial_scale):
    v = grad.clone()
    alphas = []
    for s, y, rho in reversed(history):
        alpha = rho * torch.sum(s * v)
        alphas.append(alpha)
        v -= alpha * y
    if history:
        s, y, _ = history[-1]
        v *= torch.sum(s * y) / torch.sum(y * y)
    else:
        v *= initial_scale
    for (s, y, rho), alpha in zip(history, reversed(alphas), strict=True):
        v += s * (alpha - rho * torch.sum(y * v))
    return -v


def run_case(p, target, clean, scale, case, target_name, cfg, graph, fibers, out):
    out.mkdir(parents=True, exist_ok=True)
    if cfg.resume and (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text())
    obj = Objective(p, target, scale, case, cfg, fibers, graph)
    q = torch.zeros(am.shape(case.mode, len(p.ids)))
    if cfg.init:
        if case.mode in ("F", "Shared"):
            q += cfg.init
        else:
            # Same physical constant-fiber state in each coordinate system.
            A0, H0 = am.matrices(
                torch.full((len(p.ids), 1), cfg.init), "F", fibers, cfg.gamma
            )
            if case.mode == "Raw6":
                q = am.packed(A0)
            else:
                basis = (
                    am._g5_basis(q.dtype, q.device)
                    if case.mode == "G5"
                    else am._g6_basis(q.dtype, q.device)
                )
                q = torch.einsum("nij,kij->nk", H0, basis)
    q = am.project(q, case.mode, cfg.amax)
    start = time.perf_counter()
    current = obj(q, np.zeros_like(p.points))
    history, trace, trials = [], [], []
    accepted_min_det = {"detF": math.inf, "detAinv": math.inf, "detG": math.inf}
    nscale = len(p.ids) if case.mode != "Shared" else 1
    status = "step_budget"
    snapshots = []
    for step in range(cfg.steps + 1):
        grad = current["grad"]
        pg = q - am.project(q - nscale * grad, case.mode, cfg.amax)
        kkt = float(torch.linalg.vector_norm(pg) / math.sqrt(q.numel()) / AREF)
        met = measurements(
            p,
            current["u"],
            target,
            clean,
            current["Ainv"],
            current["H"],
            q.cpu().numpy(),
            (case, target_name),
            cfg,
            graph,
            scale,
        )
        for key in accepted_min_det:
            accepted_min_det[key] = min(accepted_min_det[key], met[key + "_min"])
        row = {
            "step": step,
            "objective": current["value"],
            "fit": current["fit"],
            "R_m": current["magnitude"],
            "R_s": current["smooth"],
            "projected_kkt": kkt,
            "elapsed_s": time.perf_counter() - start,
            "forward_steps": current["forward"]["steps"],
            "forward_grad_norm": current["forward"]["grad_norm"],
            **met,
        }
        trace.append(row)
        with (out / "trace.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerows(trace)
        if step % 20 == 0 or step == cfg.steps or kkt < cfg.kkt_tol:
            np.savez_compressed(
                out / f"step-{step:04d}.npz",
                q=q.cpu().numpy(),
                u=current["u"],
                Ainv=current["Ainv"],
                step=step,
            )
            snapshots.append(step)
            LOG.info(
                "%s/%s step=%d fit=%.5g HP=%.5g KKT=%.4g calls=%d elapsed=%.1fs",
                target_name,
                case.name,
                step,
                met["fit_rms_over_D"],
                met["excess_hp_0.06_over_D"],
                kkt,
                obj.calls,
                time.perf_counter() - start,
            )
        cherries.log_metrics(
            {
                f"{target_name}/{case.name}/objective": current["value"],
                f"{target_name}/{case.name}/kkt": kkt,
            },
            step=step,
        )
        if kkt < cfg.kkt_tol:
            status = "projected_stationary"
            break
        if step == cfg.steps:
            break
        d = direction_lbfgs(grad, history, nscale * 0.01)
        projected_d = am.project(q + d, case.mode, cfg.amax) - q
        if float(torch.sum(grad * projected_d)) >= -1e-16:
            history.clear()
            d = -nscale * 0.01 * grad
        accepted = None
        for trial in range(15):
            alpha = 0.5**trial
            candidate = am.project(q + alpha * d, case.mode, cfg.amax)
            move = candidate - q
            slope = float(torch.sum(grad * move))
            if slope >= 0:
                continue
            attempt = obj(candidate, current["u"])
            ok = attempt["value"] <= current["value"] + 1e-4 * slope
            trials.append(
                {
                    "step": step,
                    "trial": trial,
                    "alpha": alpha,
                    "objective": attempt["value"],
                    "armijo": ok,
                    "forward": attempt["forward"],
                }
            )
            if ok:
                accepted = candidate, attempt
                break
        if accepted is None:
            status = "line_search_stalled"
            LOG.warning(
                "%s/%s line search stalled at KKT %.4g", target_name, case.name, kkt
            )
            break
        new_q, new_current = accepted
        s, y = new_q - q, new_current["grad"] - grad
        sy = torch.sum(s * y)
        if float(sy) > 1e-10 * float(
            torch.linalg.vector_norm(s) * torch.linalg.vector_norm(y)
        ):
            history.append((s, y, 1 / sy))
            history = history[-12:]
        q, current = new_q, new_current
    # Audit the chosen endpoint against a fresh rest-start solve, without
    # replacing the continuation endpoint or calling disagreement a failure.
    A = torch.as_tensor(current["Ainv"])
    reset_u = p.solve(am.packed(A), np.zeros_like(p.points)).detach().cpu().numpy()
    branch_delta = rms(reset_u[p.top] - current["u"][p.top], p.weights) / scale
    summary = {
        "case": asdict(case),
        "target": target_name,
        "status": status,
        "steps": step,
        "forward_calls": obj.calls + 1,
        "wall_s": time.perf_counter() - start,
        "D": scale,
        "amax": cfg.amax,
        "fiber_angle_degrees": cfg.fiber_angle,
        "gamma": cfg.gamma,
        "seed": cfg.seed,
        "noise_rms": cfg.noise_rms,
        "nx": cfg.nx,
        "ny": cfg.ny,
        "dofs": q.numel(),
        "active_tets": len(p.ids),
        "total_tets": len(p.tets),
        "branch_reset_difference_over_D": branch_delta,
        "accepted_trajectory_min_determinants": accepted_min_det,
        "snapshots": snapshots,
        "final": trace[-1],
    }
    np.savez_compressed(
        out / "final.npz",
        q=q.cpu().numpy(),
        u=current["u"],
        Ainv=current["Ainv"],
        H=current["H"],
        reset_u=reset_u,
    )
    write_json(out / "trials.json", trials)
    p.save_mesh(out / "final.vtu", current["u"], current["Ainv"])
    write_json(out / "summary.json", summary)
    return summary


def prepare_targets(p, cfg, out):
    centers = p.centers
    truth_a = 0.10 + 0.10 * np.exp(
        -((centers[:, 0] - 0.5) ** 2 + (centers[:, 2] - 0.5) ** 2) / (2 * 0.2**2)
    )
    true_fibers = torch.zeros((len(p.ids), 3))
    true_fibers[:, 0] = 1
    Atrue, _ = am.matrices(torch.as_tensor(truth_a[:, None]), "F", true_fibers)
    u0 = (
        p.solve(
            am.packed(torch.eye(3).expand(len(p.ids), 3, 3)), np.zeros_like(p.points)
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert rms(u0[p.top], p.weights) < 1e-8
    clean = (
        p.solve(am.packed(Atrue), np.zeros_like(p.points)).detach().cpu().numpy().copy()
    )
    scale = rms(clean[p.top], p.weights)
    assert scale > 1e-8
    # Physical frequencies 2--3: at least 8 cells per wavelength at nx=24.
    # This intentionally corrects the design draft's undersampled k=4--6.
    xy = p.points[p.top][:, (0, 2)]
    rng = np.random.default_rng(cfg.seed)
    noise = np.zeros(len(p.top))
    coefficients = []
    for kx in (2, 3):
        for kz in (2, 3):
            coef, phase_x, phase_z = (
                rng.normal(),
                rng.uniform(0, 2 * np.pi),
                rng.uniform(0, 2 * np.pi),
            )
            coefficients.append([kx, kz, coef, phase_x, phase_z])
            noise += (
                coef
                * np.cos(2 * np.pi * kx * xy[:, 0] + phase_x)
                * np.cos(2 * np.pi * kz * xy[:, 1] + phase_z)
            )
    noise -= np.sum(p.weights * noise)
    noise *= cfg.noise_rms * scale / rms(noise, p.weights)
    noisy = clean.copy()
    noisy[p.top, 1] += noise
    bump = 16 * xy[:, 0] * (1 - xy[:, 0]) * xy[:, 1] * (1 - xy[:, 1])
    bump *= cfg.mismatch_rms * scale / rms(bump, p.weights)
    mismatch = clean.copy()
    mismatch[p.top, 1] += bump
    np.savez_compressed(
        out / "fixture.npz",
        points=p.points,
        tets=p.tets,
        active_ids=p.ids,
        top=p.top,
        weights=p.weights,
        volumes=p.volumes,
        clean=clean,
        noisy=noisy,
        mismatch=mismatch,
        truth_a=truth_a,
        truth_Ainv=Atrue.cpu().numpy(),
        zero=u0,
        D=scale,
    )
    write_json(
        out / "fixture.json",
        {
            "D": scale,
            "noise_fourier_coefficients": coefficients,
            "noise_rms_over_D": rms(noise, p.weights) / scale,
            "mismatch_rms_over_D": rms(bump, p.weights) / scale,
            "clean_forward": p.last_forward,
            "points_hash": array_hash(p.points),
            "tets_hash": array_hash(p.tets),
            "active_ids_hash": array_hash(p.ids),
            "target_hashes": {
                k: array_hash(v)
                for k, v in {
                    "clean": clean,
                    "noisy": noisy,
                    "mismatch": mismatch,
                }.items()
            },
        },
    )
    p.save_mesh(out / "rest.vtu")
    p.save_mesh(out / "clean.vtu", clean, Atrue.cpu().numpy())
    return (
        clean,
        scale,
        {"clean": clean, "noisy": noisy, "mismatch": mismatch},
        true_fibers,
    )


def gradient_audit(cfg, out):
    p = Physics(6, 10, rtol=1e-8, atol=1e-13)
    graph = am.face_graph(p.points, p.tets, p.ids)
    fibers = torch.zeros((len(p.ids), 3))
    fibers[:, 0] = 1
    rows = []
    for mode in ("Raw6", "G6", "G5", "F", "Shared"):
        case = Case(
            mode, mode, 0.01 if mode != "Raw6" else 0, 0.01 if mode != "Raw6" else 0
        )
        q = torch.zeros(am.shape(mode, len(p.ids)))
        if mode in ("F", "Shared"):
            q += 0.10
        elif mode == "G5":
            q[:, 0] = 0.10
            q[:, 1] = 0.05
        else:
            q[:, 0] = 0.10
        target = np.zeros_like(p.points)
        target[p.top, 1] = 0.005
        obj = Objective(p, target, 0.005, case, cfg, fibers, graph)
        center = obj(q, np.zeros_like(p.points))
        generator = torch.Generator(device="cuda")
        generator.manual_seed(21)
        direction = torch.randn(q.shape, generator=generator)
        direction /= torch.linalg.vector_norm(direction)
        analytic = float(torch.sum(center["grad"] * direction))
        errors = []
        for eps in (1e-3, 3e-4):
            plus = obj(q + eps * direction, center["u"])
            minus = obj(q - eps * direction, center["u"])
            fd = (plus["value"] - minus["value"]) / (2 * eps)
            error = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-9)
            errors.append(error)
            rows.append(
                {
                    "mode": mode,
                    "epsilon": eps,
                    "analytic": analytic,
                    "finite_difference": fd,
                    "relative_error": error,
                }
            )
        assert min(errors) < 0.02, (
            f"implicit gradient check failed for {mode}: {errors}"
        )
    write_json(out / "gradient-audit.json", rows)
    del p
    return rows


def main(cfg: Config):
    configure()
    am.validate()
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run-config.json", cfg.model_dump(mode="json"))
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("activation_models.py"),
        Path(__file__).with_name("block_physics.py"),
        Path(__file__).with_name("experiment_profile.py"),
        SOURCE,
    ]
    write_json(
        out / "provenance.json",
        {
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "device": torch.cuda.get_device_name(),
            "sources": {
                str(x.relative_to(ROOT)): hashlib.sha256(x.read_bytes()).hexdigest()
                for x in source_paths
            },
            "branch_policy": "Each line-search trial starts at last accepted equilibrium; each case starts at rest; final reset audit does not replace endpoint.",
            "optimizer": "Projected L-BFGS, 12 secant pairs, Armijo c1=1e-4, at most 15 halvings. KKT=RMS(q-P(q-N grad))/a_ref; N=active count except Shared=1.",
        },
    )
    if cfg.validate_gradients:
        gradient_audit(cfg, out)
    p = Physics(cfg.nx, cfg.ny, cfg.forward_rtol, cfg.forward_atol)
    graph = am.face_graph(p.points, p.tets, p.ids)
    np.savez_compressed(
        out / "graph.npz", edge_i=graph[0], edge_j=graph[1], weight=graph[2]
    )
    clean, scale, targets, fibers = prepare_targets(p, cfg, out)
    angle = math.radians(cfg.fiber_angle)
    fibers[:, 0] = math.cos(angle)
    fibers[:, 2] = math.sin(angle)
    selected = cases(cfg)
    results = []
    for target_name in cfg.targets.split(","):
        for case in selected:
            LOG.info(
                "Starting %s / %s (%d controls)",
                target_name,
                case.name,
                np.prod(am.shape(case.mode, len(p.ids))),
            )
            path = out / target_name / case.name
            try:
                summary = run_case(
                    p,
                    targets[target_name],
                    clean,
                    scale,
                    case,
                    target_name,
                    cfg,
                    graph,
                    fibers,
                    path,
                )
            except Exception as error:
                path.mkdir(parents=True, exist_ok=True)
                failure = {
                    "case": asdict(case),
                    "target": target_name,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                write_json(path / "failure.json", failure)
                # Independent cases continue; failed cases are never replaced.
                LOG.exception("Case failed: %s/%s", target_name, case.name)
                summary = failure
            results.append(summary)
            write_json(out / "summary.json", results)
    LOG.info(
        "Finished %d cases; %d failed", len(results), sum("error" in x for x in results)
    )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
