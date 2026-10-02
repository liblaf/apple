"""Remove secondary effective activation modes and re-equilibrate a saved face."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from experiment_profile import ProfileCometNoCommit
from study_metrics import StudyMetrics
from study_physics import FacePhysics, configure

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
ORIGINAL = ROOT
FIXTURE = (
    ORIGINAL / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
)
CHECKPOINT = (
    Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
    / "exp/2026/09/08/physical-volume-baseline/data/20-baseline/step-0200.npz"
)
CHECKPOINT_SHA = "21b7e546f04c5566629a1d3634659ec0c5738df215699ba23e24eb7ac856abd7"
LOGGER = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    output_dir: Path = cherries.output("10-forward", mkdir=True)


def record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def unpack(q: np.ndarray) -> np.ndarray:
    x, y, z, xy, yz, xz = q.T
    return np.stack((x, xy, xz, xy, y, yz, xz, yz, z), axis=-1).reshape(
        -1, 3, 3
    ) + np.eye(3)


def pack(b: np.ndarray) -> np.ndarray:
    c = b - np.eye(3)
    return np.stack(
        (c[:, 0, 0], c[:, 1, 1], c[:, 2, 2], c[:, 0, 1], c[:, 1, 2], c[:, 0, 2]),
        axis=-1,
    )


def metrics(
    physics: FacePhysics,
    diagnostic: StudyMetrics,
    u: np.ndarray,
    z: np.ndarray,
    baseline: np.ndarray,
    baseline_z: np.ndarray,
) -> dict:
    result = diagnostic.evaluate(u, z, previous_u=baseline, previous_z=baseline_z)
    pred = u[physics.top]
    result["area_weighted_fit_rms_mm"] = float(
        1000
        * np.sqrt(
            np.sum(
                physics.weights
                * np.sum((pred - physics.target[physics.top]) ** 2, axis=1)
            )
        )
    )
    result["area_weighted_motion_rms_mm"] = float(
        1000 * np.sqrt(np.sum(physics.weights * np.sum(pred**2, axis=1)))
    )
    result["area_weighted_change_from_saved_rms_mm"] = float(
        1000
        * np.sqrt(
            np.sum(
                physics.weights * np.sum((pred - baseline[physics.top]) ** 2, axis=1)
            )
        )
    )
    det = physics.detf(u)
    fraction = np.asarray(physics.mesh.cell_data["MuscleFraction"])
    pure_muscle = (
        (fraction >= 1 - 1e-12)
        & (np.asarray(physics.mesh.cell_data["FatFraction"]) == 0)
        & (np.asarray(physics.mesh.cell_data["AponeurosisFraction"]) == 0)
    )
    result.update(
        detF_min=float(det.min()),
        detF_max=float(det.max()),
        inverted_all_cells=int(np.sum(det <= 0)),
        inverted_active_cells=int(np.sum(det[physics.ids] <= 0)),
        inverted_pure_muscle_cells=int(np.sum((det <= 0) & pure_muscle)),
        active_volume_weighted_rms_detF_minus_one=float(
            np.sqrt(np.average((det[physics.ids] - 1) ** 2, weights=physics.volumes))
        ),
    )
    fixed = np.asarray(physics.mesh.point_data["FixedMask"], dtype=bool)
    fixed_value = np.asarray(physics.mesh.point_data["FixedValue"])
    assert fixed.shape == u.shape
    result["fixed_max_error_m"] = float(np.max(np.abs(u[fixed] - fixed_value[fixed])))
    assert result["fixed_max_error_m"] < 1e-14
    return result


def archive_sources(out: Path) -> dict:
    sources = {}
    for name, module in tuple(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if not file or not file.endswith(".py"):
            continue
        path = Path(file).resolve()
        if path.parent == GROUP / "src":
            relative = Path("experiment") / path.name
        elif name.startswith(("liblaf.apple", "liblaf.peach")):
            relative = Path("runtime") / Path(*name.split(".")).with_suffix(".py")
        else:
            continue
        target = out / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sources[name] = {**record(path), "snapshot": str(target)}
    return sources


def main(cfg: Config) -> None:  # noqa: PLR0915
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"Output must be empty: {out}"
    assert record(CHECKPOINT)["sha256"] == CHECKPOINT_SHA
    configure()
    LOGGER.info("Building frozen no-skin, physical-volume face model")
    physics = FacePhysics(FIXTURE, activation_model="raw6")
    diagnostic = StudyMetrics(FIXTURE)
    with np.load(CHECKPOINT, allow_pickle=False) as saved:
        assert bool(saved["solver_valid"])
        assert bool(saved["physical_volume_energy"])
        assert int(saved["step"]) == 200
        assert np.array_equal(saved["active_ids"], physics.ids)
        assert np.array_equal(saved["rest_points"], physics.points)
        q0, b0, u0 = (saved[name].copy() for name in ("q", "Ainv", "u"))
    assert np.max(np.abs(unpack(q0) - b0)) < 1e-14
    z0 = b0 @ b0.swapaxes(-1, -2) - np.eye(3)
    eigenvalues, eigenvectors = np.linalg.eigh(z0)
    strength = np.maximum(eigenvalues[:, -1], 0)
    axis = eigenvectors[:, :, -1]
    projector = axis[:, :, None] * axis[:, None, :]
    z1 = strength[:, None, None] * projector
    b1 = np.eye(3) + (np.sqrt(1 + strength) - 1)[:, None, None] * projector
    z_error = float(np.max(np.abs(b1 @ b1.swapaxes(-1, -2) - np.eye(3) - z1)))
    assert z_error < 1e-10
    assert np.max(np.abs(unpack(pack(b1)) - b1)) < 1e-14
    assert (
        np.max(np.abs(np.einsum("nij,nj->ni", z1, axis) - strength[:, None] * axis))
        < 1e-10
    )
    b1_values = np.linalg.eigvalsh(b1)
    assert np.min(b1_values) > 1 - 1e-12
    assert np.max(np.abs(b1_values[:, :2] - 1)) < 1e-12
    assert np.max(np.abs(b1_values[:, -1] - np.sqrt(1 + strength))) < 1e-12
    z_norm2 = np.sum(z0**2, axis=(1, 2))
    omitted = np.divide(
        np.sum((z0 - z1) ** 2, axis=(1, 2)),
        z_norm2,
        out=np.zeros_like(z_norm2),
        where=z_norm2 > 0,
    )
    gap = eigenvalues[:, -1] - eigenvalues[:, -2]
    protocol = {
        "question": "Shape after deleting all activation except the strongest positive effective mode",
        "baseline": record(CHECKPOINT),
        "fixture": {
            name: record(FIXTURE / name)
            for name in ("volume.vtu", "skin.vtp", "summary.json")
        },
        "control": "Z0=B0 B0^T-I; Z1=max(lambda_max(Z0),0) n n^T; B1=I+(sqrt(1+lambda_max+)-1)n n^T",
        "continuation": "Z(t)=(1-t) Z0+t Z1, t=0,0.25,0.5,0.75,1; positive square root B(t) for t>0",
        "initialization": "Replay original signed B0 at saved equilibrium, then initialize each forward solve from preceding stage",
        "inverse_optimization": False,
        "materials": physics.material_spec,
        "solver": physics.forward_tolerance,
        "projection_checks": {
            "B1_Z1_max_abs_error": z_error,
            "baseline_B_nonpositive_cells": int(
                np.sum(np.linalg.eigvalsh(b0)[:, 0] <= 0)
            ),
            "no_positive_mode_cells": int(np.sum(strength == 0)),
            "near_repeated_top_modes_cells": int(
                np.sum(gap <= 1e-6 * np.maximum(1, np.abs(eigenvalues[:, -1])))
            ),
            "volume_mean_per_cell_omitted_squared_magnitude_fraction": float(
                np.average(omitted, weights=physics.volumes)
            ),
            "global_volume_weighted_omitted_squared_magnitude_fraction": float(
                np.sum(physics.volumes * np.sum((z0 - z1) ** 2, axis=(1, 2)))
                / np.sum(physics.volumes * z_norm2)
            ),
        },
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
        },
    }
    write_json(out / "protocol.json", protocol)
    saved_metrics = metrics(physics, diagnostic, u0, z0, u0, z0)
    assert abs(saved_metrics["area_weighted_fit_rms_mm"] - 1.835697) < 1e-5
    stages = []
    seed = u0
    started = time.perf_counter()
    for index, t in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        cherries.set_step(index)
        if t == 0:
            b, z, q = b0, z0, q0
        elif t == 1:
            b, z, q = b1, z1, pack(b1)
        else:
            z = (1 - t) * z0 + t * z1
            vals, vecs = np.linalg.eigh(np.eye(3) + z)
            assert np.min(vals) > 0
            b = (vecs * np.sqrt(vals)[:, None, :]) @ vecs.swapaxes(-1, -2)
            q = pack(b)
            assert np.max(np.abs(b @ b.swapaxes(-1, -2) - np.eye(3) - z)) < 1e-10
        name = (
            "baseline-replay"
            if t == 0
            else "dominant-only"
            if t == 1
            else f"removal-{round(t * 100):03d}"
        )
        LOGGER.info(
            "Forward solve %s: removed fraction %.2f; no inverse update", name, t
        )
        tick = time.perf_counter()
        u = physics.solve(torch.as_tensor(q), seed).detach().cpu().numpy().copy()
        forward = physics.last_forward
        write_json(out / f"{name}-solver.json", forward)
        assert forward["success"], f"Forward solve failed: {name}: {forward}"
        values = metrics(physics, diagnostic, u, z, u0, z0)
        np.savez_compressed(
            out / f"{name}.npz",
            u=u,
            rest_points=physics.points,
            active_ids=physics.ids,
            B=b,
            Z=z,
            q=q,
            removal_fraction=t,
            solver_valid=True,
            physical_volume_energy=True,
        )
        stage = {
            "name": name,
            "removal_fraction": t,
            "seconds": time.perf_counter() - tick,
            "forward": forward,
            "metrics": values,
            "checkpoint": record(out / f"{name}.npz"),
        }
        stages.append(stage)
        write_json(
            out / "summary.json",
            {
                "status": "complete" if t == 1 else "running",
                "saved_baseline_metrics": saved_metrics,
                "stages": stages,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        cherries.log_metrics(
            {
                key: values[key]
                for key in (
                    "area_weighted_fit_rms_mm",
                    "area_weighted_motion_rms_mm",
                    "area_weighted_change_from_saved_rms_mm",
                    "inverted_all_cells",
                )
            }
        )
        LOGGER.info(
            "Solved %s: fit %.6f mm; motion %.6f mm; change %.6f mm; steps %d; |g| %.3e",
            name,
            values["area_weighted_fit_rms_mm"],
            values["area_weighted_motion_rms_mm"],
            values["area_weighted_change_from_saved_rms_mm"],
            forward["steps"],
            forward["grad_norm"],
        )
        if t == 0:
            assert values["area_weighted_change_from_saved_rms_mm"] < 0.01, (
                "Baseline replay materially changed the saved shape"
            )
        seed = u
    protocol["sources"] = archive_sources(out)
    protocol["metric_definitions"] = (
        "See frozen study_metrics.py; vector RMS on finite IsFace target vertices, rest-skin-area weights where specified"
    )
    write_json(out / "protocol.json", protocol)
    LOGGER.info(
        "Completed dominant-mode removal in %.1f seconds", time.perf_counter() - started
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
