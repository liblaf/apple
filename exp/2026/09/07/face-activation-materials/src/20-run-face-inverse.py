"""Matched nonlinear face inversions; all output geometries are equilibria."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import activation_models as am
import numpy as np
import pydantic_settings as ps
import torch
from experiment_profile import ProfileCometNoCommit
from face_physics import ROOT, FacePhysics, ForwardConvergenceError, configure

from liblaf import cherries

LOG = logging.getLogger(__name__)
AREF = -math.log(0.8)


class TrialAdmissibilityError(RuntimeError):
    def __init__(self, detf_min, activation_eigen_min):
        self.diagnostics = dict(
            detF_min=float(detf_min), activation_eigen_min=float(activation_eigen_min)
        )
        super().__init__(str(self.diagnostics))


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = cherries.input("10-fixture")
    controls: Path = cherries.input("12-controls.npz")
    output_dir: Path = cherries.output("20-pilot", mkdir=True)
    method: str = "Region5"
    steps: int = 40
    snapshot_interval: int = 5
    magnitude: float = 0.001
    smoothness: float = 0.01
    smooth_length: float = 0.005
    amax: float = -math.log(0.65)
    skin_factor: float = 0.12
    fat_factor: float = 1.0
    muscle_factor: float = 0.8
    soft_nu: float = 0.46
    skin_nu: float = 0.46
    fat_model: str = "stable"
    fat_nu: float | None = None
    det_floor: float = 0.2
    forward_rtol: float = 1e-5
    forward_atol: float = 1e-12
    adjoint_rtol: float = 1e-6
    kkt_tol: float = 1e-3
    target_scale: float = 1.0
    target_name: str = "Smile"
    gradient_audit: bool = False
    initial: Path | None = None
    initial_repeat_regions: bool = False


def write_json(path, data):
    path.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
        + "\n"
    )


class Objective:
    def __init__(self, p, cfg):
        self.p, self.cfg = p, cfg
        self.shared = cfg.method in {"Region5", "FiberRegion"}
        self.spatial_modes = cfg.method in {"FiberModes", "Region5Modes"}
        self.mode = {
            "Raw6": "Raw6",
            "G5": "G5",
            "G5-S": "G5",
            "Region5": "G5",
            "FiberRegion": "F",
            "FiberSmooth": "F",
            "FiberModes": "F",
            "Region5Modes": "G5",
        }[cfg.method]
        self.n = p.n_regions if self.shared else len(p.ids)
        if self.spatial_modes:
            basis = np.load(cfg.controls)
            assert np.array_equal(basis["active_ids"], p.ids)
            assert np.array_equal(basis["active_region_ids"], p.region)
            assert np.allclose(basis["active_mass"], p.volumes, rtol=1e-12, atol=0)
            self.control_indices = torch.as_tensor(basis["control_indices"])
            self.control_weights = torch.as_tensor(basis["weights"])
            self.n = len(basis["parameter_mass_fraction"])
            mass = basis["parameter_mass_fraction"]
        else:
            mass = p.region_mass if self.shared else p.volumes / p.volumes.sum()
        self.shape = am.shape(self.mode, self.n)
        self.volume = torch.as_tensor(p.volumes)
        self.mass = torch.as_tensor(mass)[:, None]
        self.ei, self.ej, self.ew = (torch.as_tensor(x) for x in p.graph)
        self.target = torch.as_tensor(p.target)
        self.scale = p.D
        self.calls = 0

    def project(self, q):
        return am.project(q, self.mode, self.cfg.amax)

    def __call__(self, value, seed, *, backward=True):
        q = value.detach().clone().requires_grad_(backward)
        if self.spatial_modes:
            local = (self.control_weights[..., None] * q[self.control_indices]).sum(1)
        else:
            local = q[self.p.region_t] if self.shared else q
        A, H = am.matrices(local, self.mode, self.p.fibers)
        u = self.p.solve(am.packed(A), seed)
        detf_min = self.p.detf(u.detach().cpu().numpy()).min()
        # CUDA's batched eigensolver requests excessive workspace at this mesh
        # size. These detached 3x3 diagnostics need no device-side derivative.
        ainv_np = A.detach().cpu().numpy()
        activation_eigen_min = float(np.linalg.eigvalsh(ainv_np).min())
        if detf_min < self.cfg.det_floor or activation_eigen_min <= 0:
            raise TrialAdmissibilityError(detf_min, activation_eigen_min)
        fit = (
            self.p.weights_t[:, None]
            * (u[self.p.top_t] - self.target[self.p.top_t]).square()
        ).sum() / self.scale**2
        # Regularization is always on the physical activation field; fiber
        # models smooth scalar amplitudes within the same named muscle.
        field = (
            local
            if self.mode == "F"
            else H.reshape(len(self.p.ids), 9) / math.sqrt(1.5)
        )
        mag = (
            (self.volume * field.square().sum(dim=1)).sum()
            / self.volume.sum()
            / AREF**2
        )
        smooth = (
            self.cfg.smooth_length**2
            * (self.ew * (field[self.ei] - field[self.ej]).square().sum(dim=1)).sum()
            / self.volume.sum()
            / AREF**2
        )
        lm = 0.0 if self.cfg.method in {"Raw6", "G5"} else self.cfg.magnitude
        ls = self.cfg.smoothness if self.cfg.method in {"G5-S", "FiberSmooth"} else 0.0
        obj = fit + lm * mag + ls * smooth
        adjoint = None
        if backward:
            obj.backward()
            adjoint = self.p.check_adjoint()
            assert q.grad is not None and torch.isfinite(q.grad).all()
        self.calls += 1
        return dict(
            value=float(obj.detach()),
            fit=float(fit.detach()),
            magnitude=float(mag.detach()),
            smooth=float(smooth.detach()),
            grad=q.grad.detach().clone() if backward else None,
            u=u.detach().cpu().numpy().copy(),
            A=ainv_np,
            forward=dict(self.p.last_forward),
            adjoint=adjoint,
        )


def lbfgs(grad, history, scale):
    v = grad.clone()
    alpha = []
    for s, y, rho in reversed(history):
        a = rho * (s * v).sum()
        alpha.append(a)
        v -= a * y
    if history:
        s, y, _ = history[-1]
        # Scale the volume preconditioner by the last secant curvature.
        scale = scale * ((s * y).sum() / (y * scale * y).sum())
    v *= scale
    for (s, y, rho), a in zip(history, reversed(alpha), strict=True):
        v += s * (a - rho * (y * v).sum())
    return -v


def metrics(p, cfg, result):
    u, A = result["u"], result["A"]
    D = p.D
    target = p.target[p.top]
    pred = u[p.top]
    detf = p.detf(u)
    eig = np.linalg.eigvalsh(A)
    detg = detf.copy()
    detg[p.ids] *= np.linalg.det(A)
    return dict(
        fit_rms_over_D=math.sqrt(result["fit"]),
        fit_rms_mm=1000 * D * math.sqrt(result["fit"]),
        motion_rms_mm=1000 * math.sqrt(np.sum(p.weights[:, None] * pred**2)),
        target_projection_amplitude=float(
            np.sum(p.weights[:, None] * pred * target) / D**2
        ),
        target_projection_residual_over_D=float(
            np.sqrt(
                np.sum(
                    p.weights[:, None]
                    * (
                        pred
                        - np.sum(p.weights[:, None] * pred * target) / D**2 * target
                    )
                    ** 2
                )
            )
            / D
        ),
        detF_min=float(detf.min()),
        detF_max=float(detf.max()),
        inverted_tets=int((detf <= 0).sum()),
        detG_min=float(detg.min()),
        activation_eigen_min=float(eig.min()),
        activation_eigen_max=float(eig.max()),
        activation_det_min=float(np.linalg.det(A).min()),
        activation_det_max=float(np.linalg.det(A).max()),
    )


def main(cfg: Config):
    configure()
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), "Choose an empty output directory"
    assert cfg.det_floor > 0 and cfg.det_floor < 1
    assert 0 < cfg.soft_nu < 0.5 and 0 < cfg.skin_nu < 0.5
    assert cfg.target_scale > 0 and cfg.snapshot_interval > 0
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    source_dir = out / "sources"
    source_dir.mkdir(exist_ok=True)
    manifest = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copy2(path, source_dir / path.name)
        manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(
        out / "provenance.json",
        dict(
            sources=manifest,
            inputs={
                name: hashlib.sha256((cfg.fixture / name).read_bytes()).hexdigest()
                for name in ("volume.vtu", "skin.vtp", "summary.json")
            },
            control_basis_sha256=(
                hashlib.sha256(cfg.controls.read_bytes()).hexdigest()
                if cfg.method in {"FiberModes", "Region5Modes"}
                else None
            ),
            initial_sha256=(
                hashlib.sha256(cfg.initial.read_bytes()).hexdigest()
                if cfg.initial is not None
                else None
            ),
            git_sha=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        ),
    )
    start = time.perf_counter()
    p = FacePhysics(
        cfg.fixture,
        skin_factor=cfg.skin_factor,
        fat_factor=cfg.fat_factor,
        muscle_factor=cfg.muscle_factor,
        rtol=cfg.forward_rtol,
        atol=cfg.forward_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        soft_nu=cfg.soft_nu,
        skin_nu=cfg.skin_nu,
        target_name=cfg.target_name,
        target_scale=cfg.target_scale,
        fat_model=cfg.fat_model,
        fat_nu=cfg.fat_nu,
    )
    obj = Objective(p, cfg)
    q = torch.zeros(obj.shape)
    seed = np.zeros_like(p.points)
    if cfg.initial is not None:
        initial = np.load(cfg.initial)
        q = torch.as_tensor(initial["q"])
        seed = initial["u"]
        if cfg.initial_repeat_regions:
            assert obj.spatial_modes and q.shape == (p.n_regions, obj.shape[1])
            q = torch.repeat_interleave(q, 4, dim=0)
        assert q.shape == obj.shape
        assert torch.allclose(obj.project(q), q)
    current = obj(q, seed)
    LOG.info(
        "Loaded %d tets, %d selected active tets, %d controls; "
        "target RMS %.4f mm; first forward %s",
        len(p.tets),
        len(p.ids),
        q.numel(),
        obj.scale * 1000,
        current["forward"],
    )
    if cfg.gradient_audit:
        direction = -current["grad"] / obj.mass
        direction /= torch.linalg.vector_norm(direction, dim=-1).max()
        # At an active box or ball boundary, audit an inward direction with a
        # second-order one-sided difference. This also supports mixed-bound
        # warm starts without perturbing controls outside their feasible set.
        one_sided = not all(
            torch.allclose(
                obj.project(q + sign * 0.001 * direction), q + sign * 0.001 * direction
            )
            for sign in (-1, 1)
        )
        if one_sided:
            center = (
                torch.full_like(q, cfg.amax / 2)
                if obj.mode == "F"
                else torch.zeros_like(q)
            )
            direction = center - q
            direction /= torch.linalg.vector_norm(direction, dim=-1).max()
        rows = []
        for epsilon in (0.001, 0.0003):
            plus_q = q + epsilon * direction
            other_q = q + (2 if one_sided else -1) * epsilon * direction
            assert torch.allclose(obj.project(plus_q), plus_q) and torch.allclose(
                obj.project(other_q), other_q
            ), "Gradient audit direction must be interior"
            plus = obj(plus_q, current["u"], backward=False)
            other = obj(other_q, current["u"], backward=False)
            fd = (
                (-3 * current["value"] + 4 * plus["value"] - other["value"])
                if one_sided
                else (plus["value"] - other["value"])
            ) / (2 * epsilon)
            analytic = float((current["grad"] * direction).sum())
            rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-12)
            rows.append(
                dict(
                    epsilon=epsilon,
                    method=("one_sided_second_order" if one_sided else "central"),
                    analytic=analytic,
                    finite_difference=fd,
                    relative_error=rel,
                    plus_forward=plus["forward"],
                    other_forward=other["forward"],
                )
            )
        write_json(out / "gradient-audit.json", rows)
        assert min(r["relative_error"] for r in rows) < 0.02, rows
    trace, history, trials, snapshots = [], [], [], []
    status = "step_budget"
    min_det = math.inf
    for step in range(cfg.steps + 1):
        grad = current["grad"]
        pg = q - obj.project(q - grad / obj.mass)
        kkt = float(torch.sqrt((obj.mass * pg.square()).sum() / q.shape[1]) / AREF)
        met = metrics(p, cfg, current)
        assert met["detF_min"] >= cfg.det_floor and met["activation_eigen_min"] > 0, met
        min_det = min(min_det, met["detF_min"])
        row = dict(
            step=step,
            objective=current["value"],
            magnitude=current["magnitude"],
            smoothness=current["smooth"],
            kkt=kkt,
            elapsed_s=time.perf_counter() - start,
            forward_steps=current["forward"]["steps"],
            forward_grad_norm=current["forward"]["grad_norm"],
            **met,
        )
        trace.append(row)
        with (out / "trace.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerows(trace)
        LOG.info(
            "%s step=%d fit=%.4f mm (%.4f D) detF=%.4f KKT=%.4g calls=%d elapsed=%.1fs",
            cfg.method,
            step,
            met["fit_rms_mm"],
            met["fit_rms_over_D"],
            met["detF_min"],
            kkt,
            obj.calls,
            time.perf_counter() - start,
        )
        cherries.log_metrics(
            {
                cfg.method + "/" + k: row[k]
                for k in ("objective", "fit_rms_mm", "detF_min", "kkt")
            },
            step=step,
        )
        if step % cfg.snapshot_interval == 0 or step == cfg.steps or kkt < cfg.kkt_tol:
            p.save_mesh(out / f"step-{step:04d}.vtu", current["u"], current["A"])
            snapshots.append(step)
        np.savez_compressed(
            out / "latest.npz",
            q=q.cpu().numpy(),
            u=current["u"],
            Ainv=current["A"],
            step=step,
        )
        if kkt < cfg.kkt_tol:
            status = "projected_stationary"
            break
        if step == cfg.steps:
            break
        direction = lbfgs(grad, history, 0.01 / obj.mass)
        if float((grad * (obj.project(q + direction) - q)).sum()) >= 0:
            history.clear()
            direction = -0.01 * grad / obj.mass
        # A fixed physical trust radius only limits each proposed optimizer step;
        # it does not clip raw activation or accept a nonequilibrium geometry.
        row_norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
        direction *= (0.1 / row_norm.clamp_min(0.1)).clamp_max(1.0)
        if float((grad * (obj.project(q + direction) - q)).sum()) >= 0:
            history.clear()
            direction = -0.01 * grad / obj.mass
            row_norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
            direction *= (0.1 / row_norm.clamp_min(0.1)).clamp_max(1.0)
        accepted = None
        for trial in range(12):
            candidate = obj.project(q + 0.5**trial * direction)
            slope = float((grad * (candidate - q)).sum())
            if slope >= 0:
                continue
            try:
                attempt = obj(candidate, current["u"])
            except ForwardConvergenceError:
                trials.append(
                    dict(
                        step=step,
                        trial=trial,
                        forward=p.last_forward,
                        armijo=False,
                        rejection="inner_nonconvergence",
                    )
                )
                LOG.warning(
                    "Rejected nonconverged trial %d at outer step %d", trial, step
                )
                continue
            except TrialAdmissibilityError as error:
                trials.append(
                    dict(
                        step=step,
                        trial=trial,
                        forward=p.last_forward,
                        armijo=False,
                        rejection="geometric_admissibility",
                        diagnostics=error.diagnostics,
                    )
                )
                LOG.warning(
                    "Rejected geometry at step %d trial %d: %s", step, trial, error
                )
                continue
            admissible = (
                np.linalg.eigvalsh(attempt["A"]).min() > 0
                and p.detf(attempt["u"]).min() >= cfg.det_floor
            )
            ok = admissible and attempt["value"] <= current["value"] + 1e-4 * slope
            trials.append(
                dict(
                    step=step,
                    trial=trial,
                    objective=attempt["value"],
                    admissible=bool(admissible),
                    armijo=bool(ok),
                    forward=attempt["forward"],
                )
            )
            if ok:
                accepted = candidate, attempt
                break
        write_json(out / "trials.json", trials)
        if accepted is None:
            status = "line_search_stalled"
            break
        newq, new = accepted
        s, y = newq - q, new["grad"] - grad
        sy = (s * y).sum()
        if float(sy) > 1e-10 * float(
            torch.linalg.vector_norm(s) * torch.linalg.vector_norm(y)
        ):
            history.append((s, y, 1 / sy))
            history = history[-10:]
        q, current = newq, new
    p.save_mesh(out / "final.vtu", current["u"], current["A"])
    np.savez_compressed(
        out / "final.npz", q=q.cpu().numpy(), u=current["u"], Ainv=current["A"]
    )
    reset = (
        p.solve(am.packed(torch.as_tensor(current["A"])), np.zeros_like(p.points))
        .detach()
        .cpu()
        .numpy()
    )
    np.savez_compressed(out / "reset.npz", u=reset)
    branch = float(
        np.sqrt(np.sum(p.weights[:, None] * (reset[p.top] - current["u"][p.top]) ** 2))
        / obj.scale
    )
    summary = dict(
        status=status,
        config=cfg.model_dump(mode="json"),
        materials=p.material_spec,
        target=cfg.target_name + ", supplied surface displacement",
        target_rms_mm=obj.scale * 1000,
        n_tets=len(p.tets),
        n_active=len(p.ids),
        controls=q.numel(),
        regions=p.n_regions,
        final=trace[-1],
        snapshots=snapshots,
        accepted_trajectory_min_detF=min_det,
        projected_gradient_scope="activation box or ball only; excludes determinant admissibility constraint",
        rest_reset_difference_over_D=branch,
        reset_forward=p.last_forward,
        forward_calls=p.solve_count,
        wall_s=time.perf_counter() - start,
    )
    write_json(out / "summary.json", summary)
    LOG.info("Completed %s: %s", cfg.method, summary)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
