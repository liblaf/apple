"""Summarize completed, independently audited Stage 3 transition branches."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402


class Config(cherries.BaseConfig):
    output: Path = Path("60-comparison")
    prepared: Path = GROUP / "data/10-stage3"
    collision_on: Path = GROUP / "data/20-collision-on"
    collision_off: Path = GROUP / "data/20-collision-off"
    collision_on_audit: Path = GROUP / "data/30-collision-on-audit"
    collision_off_audit: Path = GROUP / "data/30-collision-off-audit"
    collision_on_render: Path = GROUP / "data/40-collision-on-render"
    collision_off_render: Path = GROUP / "data/40-collision-off-render"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path: Path) -> dict[str, Any]:
    assert path.is_file(), path
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def check(row: dict[str, Any]) -> Path:
    path = Path(row["path"])
    assert record(path)["sha256"] == row["sha256"], path
    return path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def verify_branch(
    run_dir: Path,
    audit_dir: Path,
    render_dir: Path,
    *,
    collision_enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[np.ndarray]]:
    run = load_json(run_dir / "summary.json")
    audit = load_json(audit_dir / "summary.json")
    manifest = load_json(render_dir / "manifest.json")
    assert run["schema"] == "fixed-reference-activation-transition-v1"
    assert run["status"] == "completed"
    assert run["provenance_verified"] is True
    assert run["config"]["collision_enabled"] is collision_enabled
    assert len(run["frames"]) == run["config"]["frames"] == 121
    assert audit["schema"] == "stage3-matched-transition-independent-audit-v1"
    assert audit["status"] == "verified_completed"
    assert audit["collision_enabled"] is collision_enabled
    assert audit["counts"]["frames"] == 121
    assert manifest["schema"] == "stage3-fixed-reference-transition-render-v1"
    assert manifest["status"] == "complete"
    assert manifest["collision_enabled"] is collision_enabled
    assert manifest["frame_count"] == 121
    assert manifest["activation_stage"] == "rankone_fixed"
    assert (
        record(run_dir / "summary.json")["sha256"] == audit["numerical_summary_sha256"]
    )
    assert (
        record(run_dir / "summary.json")["sha256"]
        == manifest["source_summary"]["sha256"]
    )
    assert len(audit["frames"]) == 121
    assert len(manifest["state_receipts"]) == 121
    physical: list[np.ndarray] = []
    for index, (frame, checked, rendered) in enumerate(
        zip(run["frames"], audit["frames"], manifest["state_receipts"], strict=True)
    ):
        assert frame["index"] == index
        assert checked["beta"] == frame["beta"]
        assert checked["checkpoint_sha256"] == frame["checkpoint"]["sha256"]
        assert rendered["state_number"] == index
        assert rendered["beta_smile"] == frame["beta"]
        assert rendered["alpha_mouthopen"] == frame["alpha"]
        assert rendered["checkpoint"]["sha256"] == frame["checkpoint"]["sha256"]
        with np.load(check(frame["checkpoint"]), allow_pickle=False) as state:
            u = state["u"].copy()
            assert u.ndim == 2
            assert u.shape[1] == 3
            assert np.isfinite(u).all()
        physical.append(u)
    return run, audit, manifest, physical


def branch_metrics(
    run: dict[str, Any], audit: dict[str, Any], cell_count: int
) -> dict[str, Any]:
    diagnostics = [frame["diagnostics"] for frame in run["frames"]]
    audited = audit["frames"]
    assert len(audited) == len(diagnostics)
    forces_n = np.asarray([row["force_norm_fresh"] for row in audited]) * 1e6
    assert np.isfinite(forces_n).all()
    inverted = np.asarray(
        [row["inverted_cells"] for row in diagnostics], dtype=np.int64
    )
    minimum_j = np.asarray([row["minimum_J"] for row in diagnostics])
    contact = [row["contact"] for row in diagnostics]
    result: dict[str, Any] = {
        "fresh_force_max_N": float(forces_n.max()),
        "fresh_force_min_N": float(forces_n.min()),
        "inverted_cells_min": int(inverted.min()),
        "inverted_cells_max": int(inverted.max()),
        "inverted_fraction_min": float(inverted.min() / cell_count),
        "inverted_fraction_max": float(inverted.max() / cell_count),
        "minimum_J": float(minimum_j.min()),
    }
    if run["config"]["collision_enabled"]:
        active = np.asarray(
            [item["active_contact_count"] for item in contact], dtype=np.int64
        )
        distance = np.asarray([item["minimum_active_distance_m"] for item in contact])
        no_intersections = np.asarray(
            [row["scoped_contact_intersections_independent"] for row in audited],
            dtype=bool,
        )
        result["contact"] = {
            "active_contact_count_min": int(active.min()),
            "active_contact_count_max": int(active.max()),
            "minimum_active_distance_m": float(distance.min()),
            "maximum_active_distance_m": float(distance.max()),
            "no_intersections_all_frames": bool((~no_intersections).all()),
            "intersection_frames": np.flatnonzero(no_intersections).tolist(),
        }
    else:
        intersections = np.asarray(
            [row["scoped_contact_intersections_independent"] for row in audited],
            dtype=bool,
        )
        result["off_diagnostic_intersection_frames"] = np.flatnonzero(
            intersections
        ).tolist()
    return result


def write_report(path: Path, result: dict[str, Any]) -> None:
    on, off, difference = (
        result["collision_on"],
        result["collision_off"],
        result["physical_displacement_difference"],
    )
    path.write_text(
        "# Stage 3 matched contact comparison\n\n"
        "Both branches contain the same 121 beta values and use the prepared Stage 3 fixed-axis endpoint tensors. "
        "The reported force values are independently audited fresh free-force residuals.\n\n"
        "| Metric | Contact on | Contact off |\n| --- | ---: | ---: |\n"
        f"| Maximum force residual | {on['fresh_force_max_N']:.6g} N | {off['fresh_force_max_N']:.6g} N |\n"
        f"| Inverted-cell range | {on['inverted_cells_min']}-{on['inverted_cells_max']} | {off['inverted_cells_min']}-{off['inverted_cells_max']} |\n"
        f"| Minimum J | {on['minimum_J']:.6g} | {off['minimum_J']:.6g} |\n"
        f"| Paired vertex-distance RMS, all frames | {difference['all_frames_vertex_distance_rms_m'] * 1e3:.6g} mm | n/a |\n"
        f"| Paired maximum vertex distance, all frames | {difference['all_frames_max_vertex_norm_m'] * 1e3:.6g} mm | n/a |\n"
        f"| Paired component RMS, all frames | {difference['all_frames_component_rms_m'] * 1e3:.6g} mm | n/a |\n"
        f"| Paired maximum component, all frames | {difference['all_frames_component_abs_max_m'] * 1e3:.6g} mm | n/a |\n\n"
        f"Contact-on active contacts range from {on['contact']['active_contact_count_min']} to {on['contact']['active_contact_count_max']}; "
        f"its minimum active distance is {on['contact']['minimum_active_distance_m']:.6g} m and it reports no scoped intersections on every frame: {on['contact']['no_intersections_all_frames']}.\n\n"
        f"Contact-off diagnostic intersection frames: {off['off_diagnostic_intersection_frames']}.\n\n"
        "Both branches permit a small inversion cap. Their passing force and contact gates therefore do not establish mechanical validity. "
        "See `data/60-comparison/summary.json` for input receipts, exact tensor digests, per-branch details, endpoint differences, and video references.\n"
    )


def main(cfg: Config) -> None:
    output = cherries.output(cfg.output / "summary.json", mkdir=True)
    assert not output.exists(), output
    source_paths = {
        "prepared_summary": cfg.prepared / "summary.json",
        "collision_on_summary": cfg.collision_on / "summary.json",
        "collision_off_summary": cfg.collision_off / "summary.json",
        "collision_on_audit": cfg.collision_on_audit / "summary.json",
        "collision_off_audit": cfg.collision_off_audit / "summary.json",
        "collision_on_render": cfg.collision_on_render / "manifest.json",
        "collision_off_render": cfg.collision_off_render / "manifest.json",
    }
    for path in source_paths.values():
        cherries.input(path)
    prepared = load_json(source_paths["prepared_summary"])
    assert prepared["status"] == "prepared"
    on_run, on_audit, _on_render, on_u = verify_branch(
        cfg.collision_on,
        cfg.collision_on_audit,
        cfg.collision_on_render,
        collision_enabled=True,
    )
    off_run, off_audit, _off_render, off_u = verify_branch(
        cfg.collision_off,
        cfg.collision_off_audit,
        cfg.collision_off_render,
        collision_enabled=False,
    )
    assert (
        on_run["inputs"]["mesh"]["sha256"]
        == off_run["inputs"]["mesh"]["sha256"]
        == prepared["outputs"]["mesh.npz"]["sha256"]
    )
    assert (
        on_run["inputs"]["endpoints"]["sha256"]
        == off_run["inputs"]["endpoints"]["sha256"]
        == prepared["outputs"]["endpoints.npz"]["sha256"]
    )
    assert (
        on_audit["activation_endpoints_sha256"]
        == off_audit["activation_endpoints_sha256"]
        == {
            "Smile": prepared["field_sha256"]["S_smile"],
            "MouthOpen": prepared["field_sha256"]["S_mouthopen"],
        }
    )
    assert (
        on_audit["stage3_checkpoints_sha256"] == off_audit["stage3_checkpoints_sha256"]
    )
    assert len(on_u) == len(off_u) == 121
    point_count = on_u[0].shape[0]
    assert all(u.shape == (point_count, 3) for u in [*on_u, *off_u])
    all_delta = np.stack(
        [left - right for left, right in zip(on_u, off_u, strict=True)]
    )
    norms = np.linalg.norm(all_delta, axis=-1)
    paired = {
        "definitions": {
            "vertex_distance_rms": "sqrt(mean_over_frames_and_vertices(||u_on-u_off||_2^2))",
            "component_rms": "sqrt(mean_over_frames_vertices_and_xyz((u_on-u_off)^2))",
            "maximum_vertex_distance": "max_over_frames_and_vertices(||u_on-u_off||_2)",
            "maximum_component": "max_over_frames_vertices_and_xyz(abs(u_on-u_off))",
        },
        "all_frames_vertex_distance_rms_m": float(np.sqrt(np.mean(norms**2))),
        "all_frames_component_rms_m": float(np.sqrt(np.mean(all_delta**2))),
        "all_frames_component_abs_max_m": float(np.abs(all_delta).max()),
        "all_frames_max_vertex_norm_m": float(norms.max()),
        "mouthopen_beta_0_vertex_distance_rms_m": float(
            np.sqrt(np.mean(norms[0] ** 2))
        ),
        "mouthopen_beta_0_component_rms_m": float(np.sqrt(np.mean(all_delta[0] ** 2))),
        "mouthopen_beta_0_max_vertex_distance_m": float(norms[0].max()),
        "mouthopen_beta_0_max_component_m": float(np.abs(all_delta[0]).max()),
        "smile_beta_1_vertex_distance_rms_m": float(np.sqrt(np.mean(norms[-1] ** 2))),
        "smile_beta_1_component_rms_m": float(np.sqrt(np.mean(all_delta[-1] ** 2))),
        "smile_beta_1_max_vertex_distance_m": float(norms[-1].max()),
        "smile_beta_1_max_component_m": float(np.abs(all_delta[-1]).max()),
    }
    # The run has the canonical physical mesh; retain its cell count for fractions.
    with np.load(cfg.collision_on / "mesh.npz", allow_pickle=False) as mesh:
        cell_count = len(mesh["tets"])
    videos: dict[str, dict[str, Any]] = {}
    for label, directory in (
        ("collision_on", cfg.collision_on_render),
        ("collision_off", cfg.collision_off_render),
    ):
        found = sorted(directory.glob("*.mp4"))
        assert len(found) == 1, found
        videos[label] = record(found[0])
    result = {
        "schema": "stage3-matched-contact-comparison-v1",
        "status": "completed_audited_rendered",
        "inputs": {name: record(path) for name, path in source_paths.items()},
        "activation": {
            "stage": "rankone_fixed",
            "source_checkpoint_sha256": on_audit["stage3_checkpoints_sha256"],
            "endpoint_tensor_sha256": on_audit["activation_endpoints_sha256"],
            "historical_solver_valid": on_audit["historical_solver_valid"],
        },
        "frame_count": 121,
        "collision_on": branch_metrics(on_run, on_audit, cell_count),
        "collision_off": branch_metrics(off_run, off_audit, cell_count),
        "physical_displacement_difference": paired,
        "videos": videos,
        "mechanical_validity_claim": False,
        "limitation": "The declared inversion cap permits inverted cells; no mechanically-valid claim is made.",
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report = cherries.output("../docs/60-results.md", mkdir=True)
    write_report(report, result)
    cherries.log_metrics(
        {
            "on/force_max_N": result["collision_on"]["fresh_force_max_N"],
            "off/force_max_N": result["collision_off"]["fresh_force_max_N"],
            "paired_u/vertex_distance_rms_m": paired[
                "all_frames_vertex_distance_rms_m"
            ],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
