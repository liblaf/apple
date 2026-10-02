"""Check inverse conclusions against an independently refined forward mesh."""

# ruff: noqa: PLR0915

from __future__ import annotations

import gc
import hashlib
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import activation_models as am
import numpy as np
import numpy.typing as npt
import pydantic_settings as ps
import torch
from block_physics import ROOT, Physics, configure
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
FROZEN_SOURCE_DIR = EXPERIMENT / "data" / "20-matrix" / "sources"
FROZEN_RUNNER = FROZEN_SOURCE_DIR / "20-inverse-constraint-matrix.py"
LOG = logging.getLogger(__name__)

spec = importlib.util.spec_from_file_location("activation_matrix_frozen", FROZEN_RUNNER)
assert spec is not None
assert spec.loader is not None
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("40-refinement", mkdir=True)
    coarse_nx: int = 24
    coarse_ny: int = 10
    fine_nx: int = 48
    fine_ny: int = 20
    steps: int = 240
    weight: float = 0.01
    highpass_length: float = 0.06
    seed: int = 20260917
    resume: bool = False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_frozen_sources() -> None:
    """Fail if the live helpers no longer match the frozen matrix sources."""
    for name in ("activation_models.py", "block_physics.py"):
        assert sha256(HERE / name) == sha256(FROZEN_SOURCE_DIR / name), (
            f"Live {name} differs from the frozen matrix implementation"
        )


def exact_nested_point_map(
    coarse_points: npt.NDArray[np.float64], fine_points: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    """Map every coarse point to the exactly coincident refined point."""
    lookup = {tuple(point): index for index, point in enumerate(fine_points)}
    assert len(lookup) == len(fine_points)
    indices = np.asarray([lookup[tuple(point)] for point in coarse_points])
    assert len(np.unique(indices)) == len(coarse_points)
    assert np.array_equal(fine_points[indices], coarse_points)
    return indices


def fine_to_coarse_cell_map(
    coarse: Physics,
    fine_points: npt.NDArray[np.float64],
    fine_tets: npt.NDArray[np.int64],
    fine_active_ids: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], dict[str, Any]]:
    """Locate each fine active centroid in a coarse active tetrahedron."""
    fine_active_tets = fine_tets[fine_active_ids]
    centroids = fine_points[fine_active_tets].mean(axis=1)
    parent_cells = np.asarray(
        coarse.mesh.find_containing_cell(centroids), dtype=np.int64
    )
    assert parent_cells.shape == (len(fine_active_ids),)
    assert np.all(parent_cells >= 0), "Some fine active centroids left the coarse mesh"

    active_lookup = np.full(len(coarse.tets), -1, dtype=np.int64)
    active_lookup[coarse.ids] = np.arange(len(coarse.ids))
    parent_active_rows = active_lookup[parent_cells]
    assert np.all(parent_active_rows >= 0), (
        "Some fine muscle centroids map to non-muscle coarse cells"
    )

    parent_tets = coarse.tets[parent_cells]
    parent_x0 = coarse.points[parent_tets[:, 0]]
    parent_dm = np.transpose(
        coarse.points[parent_tets[:, 1:]] - parent_x0[:, None, :], (0, 2, 1)
    )
    parent_dm_inv = np.linalg.inv(parent_dm)
    deltas = fine_points[fine_active_tets] - parent_x0[:, None, :]
    bary123 = np.einsum("nij,nvj->nvi", parent_dm_inv, deltas)
    barycentric = np.concatenate(
        ((1.0 - bary123.sum(axis=2))[..., None], bary123), axis=2
    )
    tolerance = 1e-10
    contained = np.all(barycentric >= -tolerance, axis=(1, 2))
    contained &= np.all(barycentric <= 1.0 + tolerance, axis=(1, 2))
    transfer_kind = (
        "nested_parent_tet_exact" if np.all(contained) else "centroid_resampling"
    )
    details = {
        "method": "VTK containing-cell query at each fine active tet centroid",
        "kind": transfer_kind,
        "num_fine_active_tets": len(fine_active_ids),
        "num_parent_coarse_active_tets_used": len(np.unique(parent_cells)),
        "num_fine_tets_wholly_in_centroid_parent": int(np.count_nonzero(contained)),
        "num_fine_tets_crossing_centroid_parent": int(np.count_nonzero(~contained)),
        "whole_cell_containment_fraction": float(np.mean(contained)),
        "barycentric_tolerance": tolerance,
    }
    return parent_active_rows, details


def surface_metrics(
    physics: Physics,
    displacement: npt.NDArray[np.float64],
    truth: npt.NDArray[np.float64],
    scale: float,
    highpass_length: float,
) -> dict[str, float]:
    error = displacement[physics.top] - truth[physics.top]
    error_norm = np.linalg.norm(error, axis=1)
    error_y = error[:, 1]
    error_y_hp = error_y - base.lowpass(physics, error_y, highpass_length)
    return {
        "surface_error_rms": base.rms(error, physics.weights),
        "surface_error_rms_over_D": base.rms(error, physics.weights) / scale,
        "surface_error_p95_over_D": base.weighted_quantile(
            error_norm, physics.weights, 0.95
        )
        / scale,
        "surface_error_max_over_D": float(error_norm.max()) / scale,
        f"surface_error_y_hp_{highpass_length:.2f}_rms_over_D": base.rms(
            error_y_hp, physics.weights
        )
        / scale,
        "whole_volume_error_rms_over_D": float(
            np.sqrt(np.mean(np.sum((displacement - truth) ** 2, axis=1))) / scale
        ),
    }


def replay_on_fine_mesh(
    fine: Physics,
    fine_truth: npt.NDArray[np.float64],
    fine_scale: float,
    coarse_point_to_fine: npt.NDArray[np.int64],
    coarse_top: npt.NDArray[np.int64],
    coarse_weights: npt.NDArray[np.float64],
    coarse_final: dict[str, npt.NDArray[np.float64]],
    fine_ainv: npt.NDArray[np.float64],
    highpass_length: float,
) -> tuple[dict[str, Any], npt.NDArray[np.float64]]:
    packed = am.packed(torch.as_tensor(fine_ainv))
    continuation = fine.solve(packed, fine_truth).detach().cpu().numpy().copy()
    continuation_forward = dict(fine.last_forward)
    reset = fine.solve(packed, np.zeros_like(fine.points)).detach().cpu().numpy().copy()
    reset_forward = dict(fine.last_forward)
    detf, deta, detg = fine.determinants(reset, fine_ainv)
    coarse_u = coarse_final["u"]
    restricted_fine = reset[coarse_point_to_fine]
    branch_difference = base.rms(reset[fine.top] - continuation[fine.top], fine.weights)
    metrics = {
        **surface_metrics(fine, reset, fine_truth, fine_scale, highpass_length),
        "coarse_node_replay_difference_rms_over_D": float(
            np.sqrt(np.mean(np.sum((restricted_fine - coarse_u) ** 2, axis=1)))
            / fine_scale
        ),
        "coarse_top_replay_difference_rms_over_D": base.rms(
            restricted_fine[coarse_top] - coarse_u[coarse_top],
            coarse_weights,
        )
        / fine_scale,
        "branch_reset_difference_over_D": branch_difference / fine_scale,
        "continuation_surface_error_rms_over_D": base.rms(
            continuation[fine.top] - fine_truth[fine.top], fine.weights
        )
        / fine_scale,
        "detF_min": float(detf.min()),
        "detF_nonpositive_volume_fraction": float(
            np.sum(fine.volumes_all[detf <= 0]) / np.sum(fine.volumes_all)
        ),
        "detAinv_min": float(deta.min()),
        "detG_min": float(detg.min()),
        "continuation_forward": continuation_forward,
        "reset_forward": reset_forward,
    }
    return metrics, reset


def write_provenance(output_dir: Path) -> None:
    source_dir = output_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__),
        FROZEN_RUNNER,
        FROZEN_SOURCE_DIR / "activation_models.py",
        FROZEN_SOURCE_DIR / "block_physics.py",
        FROZEN_SOURCE_DIR / "experiment_profile.py",
    ]
    hashes = {}
    for source in sources:
        label = source.relative_to(ROOT).as_posix()
        hashes[label] = sha256(source)
        shutil.copy2(source, source_dir / source.name)
    base.write_json(
        output_dir / "provenance.json",
        {
            "sources": hashes,
            "frozen_runner": FROZEN_RUNNER.relative_to(ROOT).as_posix(),
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "target_design": "Fine-mesh analytic clean target restricted exactly to nested coarse nodes",
            "control_replay": "Coarse piecewise-constant Ainv sampled by fine active-tet centroid; exactness classified by whole-tet containment",
        },
    )


def write_findings(
    path: Path,
    cfg: Config,
    transfer: dict[str, Any],
    coarse_summaries: list[dict[str, Any]],
    fine_summaries: list[dict[str, Any]],
) -> None:
    coarse_by_case = {item["case"]["name"]: item for item in coarse_summaries}
    lines = [
        "# Refined-forward validation",
        "",
        (
            f"The clean target was generated on a {cfg.fine_nx}x{cfg.fine_ny}x"
            f"{cfg.fine_nx} mesh and restricted at exactly coincident nodes to the "
            f"{cfg.coarse_nx}x{cfg.coarse_ny}x{cfg.coarse_nx} inverse mesh. The "
            f"inverse used {cfg.steps} projected L-BFGS steps and fixed magnitude "
            f"and smoothness weights of {cfg.weight:g} for F-MS."
        ),
        "",
        (
            f"The coarse activation field was transferred to the refined mesh as "
            f"`{transfer['kind']}`. The containment audit found "
            f"{transfer['num_fine_tets_crossing_centroid_parent']} of "
            f"{transfer['num_fine_active_tets']} fine active tetrahedra crossing "
            "their centroid-selected coarse parent."
        ),
        "",
        "| Method | Coarse fit RMS / D | Fine replay error RMS / D | Fine error HP / D | Fine min det(F) | Fine branch difference / D |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    hp_key = f"surface_error_y_hp_{cfg.highpass_length:.2f}_rms_over_D"
    for fine in fine_summaries:
        name = fine["case"]
        coarse = coarse_by_case[name]
        lines.append(
            f"| {name} | {coarse['final']['fit_rms_over_D']:.6g} | "
            f"{fine['surface_error_rms_over_D']:.6g} | {fine[hp_key]:.6g} | "
            f"{fine['detF_min']:.6g} | "
            f"{fine['branch_reset_difference_over_D']:.6g} |"
        )
    lines.extend(
        [
            "",
            "The fine replay uses a fresh rest-start equilibrium for the reported errors. The branch column compares that solution with a solve initialized from the refined clean target.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main(cfg: Config) -> None:
    assert_frozen_sources()
    configure()
    am.validate()
    output_dir = cfg.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base.write_json(output_dir / "run-config.json", cfg.model_dump(mode="json"))
    write_provenance(output_dir)

    target_dir = output_dir / "fine-target"
    target_dir.mkdir(exist_ok=True)
    fine_target_cfg = base.Config(
        _cli_parse_args=False,
        output_dir=target_dir,
        nx=cfg.fine_nx,
        ny=cfg.fine_ny,
        seed=cfg.seed,
        validate_gradients=False,
    )
    LOG.info(
        "Generating independent clean target on %dx%dx%d mesh",
        cfg.fine_nx,
        cfg.fine_ny,
        cfg.fine_nx,
    )
    fine_target_physics = Physics(cfg.fine_nx, cfg.fine_ny)
    fine_truth, fine_scale, unused_targets, unused_fibers = base.prepare_targets(
        fine_target_physics, fine_target_cfg, target_dir
    )
    fine_points = fine_target_physics.points.copy()
    fine_tets = fine_target_physics.tets.copy()
    fine_active_ids = fine_target_physics.ids.copy()
    del fine_target_physics, unused_targets, unused_fibers
    gc.collect()
    torch.cuda.empty_cache()

    coarse = Physics(cfg.coarse_nx, cfg.coarse_ny)
    coarse_point_to_fine = exact_nested_point_map(coarse.points, fine_points)
    coarse_truth = fine_truth[coarse_point_to_fine].copy()
    coarse_scale = base.rms(coarse_truth[coarse.top], coarse.weights)
    assert coarse_scale > 0
    base.write_json(
        output_dir / "target-summary.json",
        {
            "target": "clean",
            "fine_D": fine_scale,
            "coarse_restricted_D": coarse_scale,
            "coarse_points_all_exactly_nested": True,
            "num_coarse_points": len(coarse.points),
            "num_fine_points": len(fine_points),
        },
    )
    np.savez_compressed(
        output_dir / "targets.npz",
        fine_points=fine_points,
        fine_tets=fine_tets,
        fine_active_ids=fine_active_ids,
        fine_clean=fine_truth,
        coarse_point_to_fine=coarse_point_to_fine,
        coarse_clean=coarse_truth,
        fine_D=fine_scale,
        coarse_D=coarse_scale,
    )
    coarse.save_mesh(output_dir / "coarse-target.vtu", coarse_truth)

    graph = am.face_graph(coarse.points, coarse.tets, coarse.ids)
    fibers = torch.zeros((len(coarse.ids), 3))
    fibers[:, 0] = 1.0
    inverse_cfg = base.Config(
        _cli_parse_args=False,
        output_dir=output_dir / "coarse-inverse",
        nx=cfg.coarse_nx,
        ny=cfg.coarse_ny,
        steps=cfg.steps,
        rows="Raw6,F-MS",
        targets="refined-clean",
        seed=cfg.seed,
        weight=cfg.weight,
        validate_gradients=False,
        resume=cfg.resume,
    )
    inverse_summaries = []
    final_arrays = {}
    for case in base.cases(inverse_cfg):
        case_dir = output_dir / "coarse-inverse" / case.name
        LOG.info("Coarse inverse of refined target: %s", case.name)
        summary = base.run_case(
            coarse,
            coarse_truth,
            coarse_truth,
            coarse_scale,
            case,
            "refined-clean",
            inverse_cfg,
            graph,
            fibers,
            case_dir,
        )
        inverse_summaries.append(summary)
        with np.load(case_dir / "final.npz") as final:
            final_arrays[case.name] = {
                "u": np.asarray(final["u"]).copy(),
                "Ainv": np.asarray(final["Ainv"]).copy(),
            }

    parent_active_rows, transfer = fine_to_coarse_cell_map(
        coarse, fine_points, fine_tets, fine_active_ids
    )
    base.write_json(output_dir / "control-transfer.json", transfer)
    coarse_top = coarse.top.copy()
    coarse_weights = coarse.weights.copy()
    del coarse
    gc.collect()
    torch.cuda.empty_cache()

    fine = Physics(cfg.fine_nx, cfg.fine_ny)
    assert np.array_equal(fine.points, fine_points)
    assert np.array_equal(fine.tets, fine_tets)
    replay_summaries = []
    for case_name, coarse_final in final_arrays.items():
        fine_ainv = coarse_final["Ainv"][parent_active_rows]
        metrics, replay = replay_on_fine_mesh(
            fine,
            fine_truth,
            fine_scale,
            coarse_point_to_fine,
            coarse_top,
            coarse_weights,
            coarse_final,
            fine_ainv,
            cfg.highpass_length,
        )
        replay_dir = output_dir / "fine-replay" / case_name
        replay_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(replay_dir / "final.npz", u=replay, Ainv=fine_ainv)
        fine.save_mesh(replay_dir / "final.vtu", replay, fine_ainv)
        replay_summary = {
            "case": case_name,
            "control_transfer_kind": transfer["kind"],
            **metrics,
        }
        base.write_json(replay_dir / "summary.json", replay_summary)
        replay_summaries.append(replay_summary)

    summary = {
        "target": "independent refined clean forward solution",
        "coarse_mesh": [cfg.coarse_nx, cfg.coarse_ny, cfg.coarse_nx],
        "fine_mesh": [cfg.fine_nx, cfg.fine_ny, cfg.fine_nx],
        "fine_D": fine_scale,
        "coarse_restricted_D": coarse_scale,
        "highpass_length": cfg.highpass_length,
        "transfer": transfer,
        "coarse_inverse": inverse_summaries,
        "fine_replay": replay_summaries,
    }
    base.write_json(output_dir / "summary.json", summary)
    write_findings(
        EXPERIMENT / "docs" / "40-refinement-findings.md",
        cfg,
        transfer,
        inverse_summaries,
        replay_summaries,
    )
    LOG.info("Refinement check finished")


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
