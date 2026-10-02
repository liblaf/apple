"""Render a saved corrected-neutral expression endpoint without solving."""

# ruff: noqa: C901, PLR0912, PLR0915, PLW0603

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from coupled_review_curves import (  # noqa: E402
    break_at_zero_update,
    plot_objective_components,
)
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402
from smile_recovery_lineage import (  # noqa: E402
    resolve_data_binding,
    verified_recovery_lineage,
    verified_terminal_interruption,
)

SPEC = importlib.util.spec_from_file_location(
    "rigid_review", GROUP / "src/131-review-rigid-inverse.py"
)
assert SPEC is not None
assert SPEC.loader is not None
review = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = review
SPEC.loader.exec_module(review)

RIGID_COLORS = {"cranium": "#e5d8bb", "mandible": "#cdb787", "eyes": "#91c9da"}


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/inverse-expression-coupled-001"
    neutral_dir: Path = GROUP / "data/forward-isfixed-001"
    output_dir: Path = GROUP / "data/review-expression-coupled-001"
    parent_run_dir: Path | None = None
    binding_mirror_root: Path | None = None
    local_data_root: Path | None = None
    recovery_lineage_receipt: Path | None = None
    recovery_preflight: Path | None = None
    recovery_parent_audit: Path | None = None
    terminal_receipt: Path | None = None
    terminal_bundle_manifest: Path | None = None


BINDING_MIRROR_ROOT: Path | None = None
LOCAL_DATA_ROOT: Path | None = None


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verified(item: dict[str, Any]) -> Path:
    """Resolve source bindings from the original path or a mirror of remote `data/`."""
    return resolve_data_binding(item, BINDING_MIRROR_ROOT, LOCAL_DATA_ROOT)


def objective_terms(protocol: dict) -> dict[str, Any]:
    """Return the recorded objective contract, preserving legacy run semantics."""
    terms = protocol.get("objective_terms")
    if terms is None:
        return {
            "mode": "legacy_position_only",
            "requires_explicit_positional_metric": False,
        }
    assert isinstance(terms, dict)
    assert "mode" in terms
    terms = dict(terms)
    terms["requires_explicit_positional_metric"] = True
    return terms


def verify_objective_continuity(protocol: dict) -> dict[str, Any] | None:
    """Bind an unchanged objective continuation to its exact source protocol."""
    initialization = protocol["initialization"]
    continuity = initialization.get("objective_continuity")
    if continuity is None:
        return None
    assert isinstance(continuity, dict)
    assert continuity["same_state"] is True
    assert continuity["objective_terms_exact"] is True
    source_protocol = verified(continuity["source_protocol"])
    checkpoint = verified(initialization["checkpoint"])
    assert source_protocol == checkpoint.parent / "protocol.json"
    parent_protocol = json.loads(source_protocol.read_text())
    assert parent_protocol["objective_terms"] == protocol["objective_terms"]
    return {
        **continuity,
        "source_protocol": record(source_protocol),
        "checkpoint": record(checkpoint),
    }


def positional_fit_rms_mm(row: dict, terms: dict[str, Any]) -> float:
    """Use the explicit data-fit metric; never infer it from total objective loss."""
    value = row.get("positional_fit_rms_mm")
    if value is None:
        assert not terms["requires_explicit_positional_metric"], (
            "regularized objective requires positional_fit_rms_mm in every progress row"
        )
        value = row["fit_rms_mm"]
    value = float(value)
    assert np.isfinite(value)
    return value


def attach_objective_metrics(rows: list[dict], protocol: dict) -> list[dict]:
    terms = objective_terms(protocol)
    return [
        {**row, "_positional_fit_rms_mm": positional_fit_rms_mm(row, terms)}
        for row in rows
    ]


def verify_initialization_refinement(
    refinement: dict, parent: Path, parent_final: dict, child_initial: dict
) -> dict:
    """Verify a fixed-control refinement without treating it as an Adam update."""
    import torch

    result_path = verified(refinement["result"])
    protocol_path = verified(refinement["protocol"])
    endpoint_path = verified(refinement["endpoint"])
    result = json.loads(result_path.read_text())
    polish_protocol = json.loads(protocol_path.read_text())
    assert result["refinement_succeeded"]
    assert result["failure"] is None
    assert result["candidate_kind"] == "completed_physical_corrector"
    assert result["final"]["original_acceptance_gates_met"]
    assert result["final"]["internal_force_target_met"]
    assert verified(result["endpoint"]) == endpoint_path
    assert refinement["optimizer_state_unchanged"]
    assert verified(result["source_checkpoint"]) == parent / "checkpoint.pt"
    for name, binding in polish_protocol["source_refs"].items():
        assert verified(binding) == parent / name
    assert (
        polish_protocol["internal_force_target"] == refinement["internal_force_target"]
    )
    assert (
        result["final"]["raw_free_force_mpa_m2"] <= refinement["internal_force_target"]
    )
    parent_checkpoint = torch.load(
        parent / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    with (
        np.load(parent / "endpoint.npz", allow_pickle=False) as original,
        np.load(endpoint_path, allow_pickle=False) as refined,
    ):
        for key in ("activation_inv", "pose_rad_m", "active_cell_ids"):
            np.testing.assert_array_equal(refined[key], original[key])
        for key in ("activation_inv", "pose_rad_m", "pose_normalized"):
            np.testing.assert_array_equal(refined[key], parent_checkpoint[key].numpy())
    assert parent_checkpoint["optimizer_steps"] == result["source_optimizer_steps"]
    assert child_initial["optimizer_steps"] == parent_checkpoint["optimizer_steps"]
    for actual, expected in (
        (refinement["original_fit_rms_mm"], parent_final["fit_rms_mm"]),
        (result["initial"]["fit_rms_mm"], parent_final["fit_rms_mm"]),
        (refinement["refined_fit_rms_mm"], result["final"]["fit_rms_mm"]),
        (child_initial["fit_rms_mm"], result["final"]["fit_rms_mm"]),
        (child_initial["force_norm_n"], result["final"]["force_norm_n"]),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)
    return {
        **refinement,
        "optimizer_steps": parent_checkpoint["optimizer_steps"],
        "original_force_norm_n": result["initial"]["force_norm_n"],
        "refined_force_norm_n": result["final"]["force_norm_n"],
        "counts_as_accepted_optimizer_update": False,
    }


def faces(triangles: np.ndarray) -> np.ndarray:
    assert triangles.ndim == 2
    assert triangles.shape[1] == 3
    assert triangles.min() >= 0
    return np.column_stack((np.full(len(triangles), 3), triangles)).ravel()


def surface(points: np.ndarray, ids: np.ndarray, triangles: np.ndarray) -> pv.PolyData:
    assert points.shape == (len(ids), 3)
    assert len(ids) == len(np.unique(ids))
    assert triangles.max() < len(ids)
    mesh = pv.PolyData(points, faces(triangles))
    mesh.point_data["GlobalPointId"] = ids
    return mesh


def camera(meshes: list[pv.DataSet], view: str) -> tuple[list, float]:
    points = np.concatenate([np.asarray(mesh.points) for mesh in meshes])
    low, high = points.min(axis=0), points.max(axis=0)
    center, span = (low + high) / 2, float(np.max(high - low))
    direction = np.asarray((0, 0, 2.7) if view == "front" else (2.7, 0, 0))
    return [list(center + span * direction), list(center), [0, 1, 0]], 0.60 * span


def add_anatomy(plot: pv.Plotter, anatomy: dict[str, pv.PolyData]) -> None:
    for name, mesh in anatomy.items():
        plot.add_mesh(
            mesh,
            color=RIGID_COLORS[name],
            smooth_shading=True,
            opacity=0.82,
        )


def render_target_fit(
    output: Path,
    target: pv.PolyData,
    fit_boundary: pv.PolyData,
    anatomy: dict[str, pv.PolyData],
    *,
    view: str,
) -> str:
    position, scale = camera([target, fit_boundary, *anatomy.values()], view)
    plot = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2000, 1000))
    plot.set_background("#f7f7f5")
    for column, (title, mesh, color) in enumerate(
        (
            ("Transferred expression target skin", target, "#737b86"),
            ("Saved inverse FEM fit full tetmesh boundary", fit_boundary, "#c75f42"),
        )
    ):
        plot.subplot(0, column)
        if column == 1:
            add_anatomy(plot, anatomy)
        plot.add_mesh(mesh, color=color, smooth_shading=True)
        plot.add_text(
            f"{title}\n{view} · identical true scale",
            position="upper_left",
            font_size=15,
            color="#202124",
        )
        plot.camera_position = position
        plot.camera.parallel_projection = True
        plot.camera.parallel_scale = scale
        plot.reset_camera_clipping_range()
    name = f"target-vs-fit-{view}.png"
    plot.show(screenshot=output / name, auto_close=True)
    return name


def render_tetmesh(
    output: Path,
    neutral: pv.PolyData,
    fit: pv.PolyData,
    neutral_anatomy: dict[str, pv.PolyData],
    fit_anatomy: dict[str, pv.PolyData],
    *,
    view: str,
    edges: bool = False,
) -> str:
    position, scale = camera(
        [neutral, fit, *neutral_anatomy.values(), *fit_anatomy.values()], view
    )
    plot = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2000, 1000))
    plot.set_background("#f7f7f5")
    for column, (title, mesh, anatomy, color) in enumerate(
        (
            ("Corrected neutral tetmesh boundary", neutral, neutral_anatomy, "#737b86"),
            ("Saved inverse fit tetmesh boundary", fit, fit_anatomy, "#c75f42"),
        )
    ):
        plot.subplot(0, column)
        add_anatomy(plot, anatomy)
        plot.add_mesh(
            mesh,
            color=color,
            smooth_shading=not edges,
            show_edges=edges,
            edge_color="#3c3734",
            line_width=0.45,
            ambient=0.25,
            diffuse=0.7,
        )
        plot.add_text(
            f"{title}\ncomplete boundary{' with edges' if edges else ''} · {view} · true scale",
            position="upper_left",
            font_size=15,
            color="#202124",
        )
        plot.camera_position = position
        plot.camera.parallel_projection = True
        plot.camera.parallel_scale = scale
        plot.reset_camera_clipping_range()
    name = f"neutral-vs-fit-tetmesh-{'edges-' if edges else ''}{view}.png"
    plot.show(screenshot=output / name, auto_close=True)
    return name


def render_error(
    output: Path, fit: pv.PolyData, target: pv.PolyData, *, view: str
) -> str:
    error_mm = np.linalg.norm(
        np.asarray(fit.points) - np.asarray(target.points), axis=1
    )
    error_mm *= 1000
    plot = pv.Plotter(off_screen=True, window_size=(1200, 1000))
    plot.set_background("#f7f7f5")
    plot.add_mesh(
        fit,
        scalars=error_mm,
        cmap="magma",
        clim=(0, float(np.quantile(error_mm, 0.99))),
        scalar_bar_args={"title": "target error (mm)"},
        smooth_shading=True,
    )
    position, scale = camera([fit], view)
    plot.add_text(
        f"Transferred expression target error on saved FEM skin\n{view} · 99th percentile color scale",
        position="upper_left",
        font_size=15,
        color="#202124",
    )
    plot.camera_position = position
    plot.camera.parallel_projection = True
    plot.camera.parallel_scale = scale
    plot.reset_camera_clipping_range()
    name = f"fit-error-{view}.png"
    plot.show(screenshot=output / name, auto_close=True)
    return name


def curves(
    output: Path, rows: list[dict], error_mm: np.ndarray, parent_rows: list[dict]
) -> str:
    combined_rows = list(parent_rows)
    if parent_rows:
        parent_last = parent_rows[-1]["iteration"]
        continuation = [
            {**row, "iteration": parent_last + row["iteration"]} for row in rows
        ]
        if (
            continuation
            and continuation[0]["iteration"] == parent_last
            and "equilibrium_refinement" not in continuation[0]
            and "objective_change" not in continuation[0]
            and "recovery_zero_update" not in continuation[0]
        ):
            continuation = continuation[1:]
        combined_rows.extend(continuation)
    else:
        combined_rows = rows
    assert combined_rows
    iterations = np.asarray([row["iteration"] for row in combined_rows])
    fit_rms = np.asarray([row["_positional_fit_rms_mm"] for row in combined_rows])
    force = np.asarray([row["force_norm_n"] for row in combined_rows])
    threshold = np.asarray([row["force_threshold_n"] for row in combined_rows])
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    axes[0].plot(
        iterations,
        break_at_zero_update(combined_rows, fit_rms),
        color="#c75f42",
        marker="o",
    )
    axes[0].set(
        xlabel="Accepted optimizer iteration",
        ylabel="Positional fit RMS (mm)",
        title="L2 data fit",
    )
    axes[1].semilogy(
        iterations,
        break_at_zero_update(combined_rows, force),
        color="#087d81",
        marker="o",
        label="force",
    )
    axes[1].semilogy(iterations, threshold, color="#555", ls=":", label="threshold")
    axes[1].set(
        xlabel="Accepted optimizer iteration",
        ylabel="Residual (N)",
        title="Forward force",
    )
    axes[1].legend()
    for row in combined_rows:
        if "equilibrium_refinement" not in row:
            continue
        for axis, field in zip(
            axes[:2], ("_positional_fit_rms_mm", "force_norm_n"), strict=True
        ):
            axis.scatter(
                row["iteration"], row[field], color="#6a4c93", marker="D", zorder=4
            )
            axis.annotate(
                "Fixed-control polish\n(no optimizer update)",
                (row["iteration"], row[field]),
                xytext=(8, 22),
                textcoords="offset points",
                fontsize=8,
                arrowprops={"arrowstyle": "-", "color": "#6a4c93"},
            )
    for row in combined_rows:
        if "recovery_zero_update" not in row:
            continue
        for axis, field in zip(
            axes[:2], ("_positional_fit_rms_mm", "force_norm_n"), strict=True
        ):
            axis.scatter(
                row["iteration"], row[field], color="#555", marker="s", zorder=5
            )
    for row in combined_rows:
        if "objective_change" not in row:
            continue
        axes[0].scatter(
            row["iteration"],
            row["_positional_fit_rms_mm"],
            color="#bc6c25",
            marker="X",
            zorder=5,
        )
        axes[0].annotate(
            "Objective changed\noptimizer moments retained",
            (row["iteration"], row["_positional_fit_rms_mm"]),
            xytext=(10, 22),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#bc6c25"},
        )
    ordered = np.sort(error_mm)
    axes[2].plot(ordered, np.linspace(0, 1, len(ordered)), color="#6a4c93")
    axes[2].set(
        xlabel="Target error (mm)",
        ylabel="Cumulative skin fraction",
        title="Saved fit error",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    name = "fit-error-force-curves.png"
    figure.savefig(output / name, dpi=180)
    plt.close(figure)
    return name


def main(cfg: Config) -> None:
    global BINDING_MIRROR_ROOT, LOCAL_DATA_ROOT
    BINDING_MIRROR_ROOT = (
        None if cfg.binding_mirror_root is None else cfg.binding_mirror_root.resolve()
    )
    LOCAL_DATA_ROOT = (
        None if cfg.local_data_root is None else cfg.local_data_root.resolve()
    )
    run, neutral_dir, output = (
        cfg.run_dir.resolve(),
        cfg.neutral_dir.resolve(),
        cfg.output_dir.resolve(),
    )
    assert not output.exists(), output
    paths = {
        "protocol": run / "protocol.json",
        "summary": run / "summary.json",
        "progress": run / "progress.jsonl",
        "endpoint": run / "endpoint.npz",
    }
    # A saved endpoint is required: this renderer never replaces an unfinished fit
    # with its neutral initialization.
    assert all(path.is_file() for path in paths.values()), paths
    inputs = {name: record(path) for name, path in paths.items()}
    protocol, summary = (
        json.loads(paths[name].read_text()) for name in ("protocol", "summary")
    )
    assert protocol["schema"] == "new-neutral-expression-rigid6-inverse-v1"
    expression_name = str(protocol["expression_name"])
    expression_index = int(protocol["expression_index"])
    assert expression_name
    assert verified(summary["endpoint"]) == paths["endpoint"]
    audit_path = run / "independent-audit.json"
    audit = json.loads(audit_path.read_text())
    assert verified(audit["inputs"]["endpoint"]) == paths["endpoint"]
    assert verified(audit["inputs"]["summary"]) == paths["summary"]
    assert audit["valid_forward"]
    inputs["independent_audit"] = record(audit_path)
    interruption_path = run / "objective-change-interruption.json"
    if summary["status"] == "running":
        if interruption_path.is_file():
            inputs["objective_change_interruption"] = record(interruption_path)
            saved_status = "interrupted_for_requested_objective_change"
        else:
            assert cfg.terminal_receipt is not None, summary["status"]
            assert cfg.terminal_bundle_manifest is not None, summary["status"]
            assert BINDING_MIRROR_ROOT is not None, summary["status"]
            terminal_evidence = verified_terminal_interruption(
                cfg.terminal_receipt,
                cfg.terminal_bundle_manifest,
                BINDING_MIRROR_ROOT,
                run,
                summary,
                [
                    json.loads(line)
                    for line in paths["progress"].read_text().splitlines()
                ],
                audit_path,
            )
            inputs["terminal_receipt"] = terminal_evidence["terminal_receipt"]
            inputs["terminal_bundle_manifest"] = terminal_evidence["bundle_manifest"]
            inputs["terminal_evidence"] = terminal_evidence
            saved_status = (
                "interrupted_after_saved_audited_partial_endpoint"
                if terminal_evidence["terminal_status"] == "audited_partial_endpoint"
                else "saved_audited_endpoint"
            )
    else:
        saved_status = summary["status"]
    rendering_path = verified(protocol["rendering"]["archive"])
    blendshape_path = verified(protocol["sources"]["blendshapes"])
    neutral_endpoint_path = verified(protocol["sources"]["neutral_endpoint"])
    neutral_summary_path = verified(protocol["sources"]["neutral_summary"])
    inputs.update(
        rendering=record(rendering_path),
        blendshapes=record(blendshape_path),
        neutral_endpoint=record(neutral_endpoint_path),
        neutral_summary=record(neutral_summary_path),
    )
    neutral_protocol = json.loads((neutral_dir / "protocol.json").read_text())
    assert sha256(neutral_dir / "summary.json") == sha256(neutral_summary_path)
    reference_path = verified(
        neutral_protocol["reference_configuration"]["constitutive_volume"]
    )
    coverage = neutral_protocol["coverage"]
    assert coverage["coverage"]["soft_cranium"]
    assert coverage["coverage"]["soft_mandible"]
    assert coverage["coverage"]["soft_eyes"]
    assert coverage["coverage"]["soft_soft"] is False
    assert coverage["coverage"]["rigid_rigid"] is False
    inputs["corrected_reference_volume"] = record(reference_path)
    volume = pv.read(reference_path)
    assert isinstance(volume, pv.UnstructuredGrid)
    assert np.all(volume.celltypes == pv.CellType.TETRA)
    with np.load(rendering_path, allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    required = {
        "full_reference_points_m",
        "skin_global_ids",
        "skin_triangles",
        "cranium_global_ids",
        "cranium_triangles",
        "mandible_global_ids",
        "mandible_triangles",
        "eye_global_ids",
        "eye_triangles",
    }
    assert set(fields) == required
    full_reference = fields["full_reference_points_m"]
    assert np.array_equal(full_reference[: volume.n_points], volume.points)
    with np.load(paths["endpoint"], allow_pickle=False) as archive:
        displacement = archive["displacement_m"]
        pose = archive["pose_rad_m"]
    with np.load(neutral_endpoint_path, allow_pickle=False) as archive:
        neutral_displacement = archive["displacement_m"]
    assert displacement.shape == full_reference.shape
    assert np.isfinite(displacement).all()
    assert neutral_displacement.shape == volume.points.shape
    assert pose.shape == (6,)
    assert np.isfinite(pose).all()
    final = summary["final"]
    objective_continuity = verify_objective_continuity(protocol)
    final_positional_fit = positional_fit_rms_mm(final, objective_terms(protocol))
    assert np.array_equal(pose, np.asarray(final["pose_rad_m"]))
    rows = attach_objective_metrics(
        [json.loads(line) for line in paths["progress"].read_text().splitlines()],
        protocol,
    )
    assert rows
    assert rows[-1]["iteration"] == final["iteration"]
    parent_rows = []
    parent_binding = []
    refinement_binding = []
    recovery_lineage = None
    if cfg.recovery_lineage_receipt is not None:
        assert cfg.parent_run_dir is not None
        assert cfg.recovery_preflight is not None
        assert cfg.recovery_parent_audit is not None
        parent = cfg.parent_run_dir.resolve()
        certified, recovery_lineage = verified_recovery_lineage(
            cfg.recovery_lineage_receipt,
            cfg.recovery_preflight,
            parent,
            cfg.recovery_parent_audit,
            run,
        )
        parent_protocol = json.loads((parent / "protocol.json").read_text())
        parent_rows = attach_objective_metrics(certified, parent_protocol)
        rows[0]["recovery_zero_update"] = recovery_lineage["lineage_clip"]
        parent_binding.append(recovery_lineage)
        inputs["recovery_lineage"] = recovery_lineage["metadata"]
    if cfg.parent_run_dir is None and protocol["initialization"].get("refinement"):
        parent = verified(protocol["initialization"]["checkpoint"]).parent
        assert (
            verified(protocol["initialization"]["endpoint"]) == parent / "endpoint.npz"
        )
        parent_summary = json.loads((parent / "summary.json").read_text())
        verified_refinement = verify_initialization_refinement(
            protocol["initialization"]["refinement"],
            parent,
            parent_summary["final"],
            rows[0],
        )
        rows[0]["equilibrium_refinement"] = verified_refinement
        refinement_binding.append(verified_refinement)
    if cfg.parent_run_dir is not None and recovery_lineage is None:
        parent = cfg.parent_run_dir.resolve()
        child_protocol, child_rows = protocol, rows
        visited = {run}
        while True:
            assert parent not in visited, parent
            visited.add(parent)
            parent_summary = json.loads((parent / "summary.json").read_text())
            assert verified(child_protocol["initialization"]["checkpoint"]) == (
                parent / "checkpoint.pt"
            )
            assert verified(child_protocol["initialization"]["endpoint"]) == (
                parent / "endpoint.npz"
            )
            parent_protocol = json.loads((parent / "protocol.json").read_text())
            prefix = attach_objective_metrics(
                [
                    json.loads(line)
                    for line in (parent / "progress.jsonl").read_text().splitlines()
                ],
                parent_protocol,
            )
            assert prefix[-1]["iteration"] == parent_summary["final"]["iteration"]
            refinement = child_protocol["initialization"].get("refinement")
            if refinement is None:
                np.testing.assert_allclose(
                    prefix[-1]["_positional_fit_rms_mm"],
                    child_rows[0]["_positional_fit_rms_mm"],
                    rtol=1e-10,
                )
            else:
                verified_refinement = verify_initialization_refinement(
                    refinement, parent, prefix[-1], child_rows[0]
                )
                child_rows[0]["equilibrium_refinement"] = verified_refinement
                refinement_binding.append(verified_refinement)
            if objective_change := child_protocol["initialization"].get(
                "objective_change"
            ):
                assert isinstance(objective_change, dict)
                child_rows[0]["objective_change"] = objective_change
            retained_children = (
                parent_rows
                if parent_rows
                and {
                    "equilibrium_refinement",
                    "objective_change",
                    "recovery_zero_update",
                }.intersection(parent_rows[0])
                else parent_rows[1:]
            )
            parent_rows = prefix + [
                {**row, "iteration": prefix[-1]["iteration"] + row["iteration"]}
                for row in retained_children
            ]
            parent_binding.append(
                {
                    "summary": record(parent / "summary.json"),
                    "progress": record(parent / "progress.jsonl"),
                    "endpoint": record(parent / "endpoint.npz"),
                    "checkpoint": record(parent / "checkpoint.pt"),
                    "protocol": record(parent / "protocol.json"),
                }
            )
            child_protocol = json.loads((parent / "protocol.json").read_text())
            child_rows = prefix
            parent = verified(child_protocol["initialization"]["checkpoint"]).parent
            if not (parent / "protocol.json").exists():
                break
    initial_fit = float((parent_rows or rows)[0]["_positional_fit_rms_mm"])
    total_iterations = final["iteration"] + (
        parent_rows[-1]["iteration"] if parent_rows else 0
    )
    with np.load(blendshape_path, allow_pickle=False) as archive:
        index = [str(name) for name in archive["expression_names"]].index(
            expression_name
        )
        assert index == expression_index
        skin_ids, skin_triangles = archive["skin_global_ids"], archive["skin_triangles"]
        target = surface(archive["target_points_m"][index], skin_ids, skin_triangles)
    assert np.array_equal(skin_ids, fields["skin_global_ids"])
    fit_skin = surface(
        full_reference[skin_ids] + displacement[skin_ids], skin_ids, skin_triangles
    )
    neutral_volume = volume.copy(deep=True)
    neutral_volume.points = volume.points + neutral_displacement
    fit_volume = volume.copy(deep=True)
    fit_volume.points = volume.points + displacement[: volume.n_points]
    neutral_boundary, fit_boundary = (
        mesh.extract_surface(algorithm=None) for mesh in (neutral_volume, fit_volume)
    )
    assert np.array_equal(neutral_boundary.faces, fit_boundary.faces)
    neutral_full_displacement = np.zeros_like(full_reference)
    neutral_full_displacement[: volume.n_points] = neutral_displacement
    anatomy = {}
    for key, label in (
        ("cranium", "cranium"),
        ("mandible", "mandible"),
        ("eye", "eyes"),
    ):
        ids, triangles = fields[f"{key}_global_ids"], fields[f"{key}_triangles"]
        anatomy[label] = (
            surface(
                full_reference[ids] + neutral_full_displacement[ids], ids, triangles
            ),
            surface(full_reference[ids] + displacement[ids], ids, triangles),
        )
    neutral_anatomy = {name: pair[0] for name, pair in anatomy.items()}
    fit_anatomy = {name: pair[1] for name, pair in anatomy.items()}
    error_mm = (
        np.linalg.norm(np.asarray(fit_skin.points) - np.asarray(target.points), axis=1)
        * 1000
    )
    output.mkdir(parents=True)
    shutil.copy2(__file__, output / Path(__file__).name)
    shutil.copy2(audit_path, output / "independent-audit.json")
    fit_skin.save(output / "fit-skin.vtp")
    target.save(output / "expression-transferred-skin-target.vtp")
    neutral_boundary.save(output / "neutral-full-tetmesh-boundary.vtp")
    fit_boundary.save(output / "fit-full-tetmesh-boundary.vtp")
    images = [
        *(
            render_target_fit(output, target, fit_boundary, fit_anatomy, view=view)
            for view in ("front", "side")
        ),
        render_tetmesh(
            output,
            neutral_boundary,
            fit_boundary,
            neutral_anatomy,
            fit_anatomy,
            view="front",
            edges=True,
        ),
        *(
            render_tetmesh(
                output,
                neutral_boundary,
                fit_boundary,
                neutral_anatomy,
                fit_anatomy,
                view=view,
            )
            for view in ("front", "side")
        ),
        render_error(output, fit_skin, target, view="front"),
        curves(output, rows, error_mm, parent_rows),
    ]
    component_curve, component_coverage = plot_objective_components(
        output, rows, parent_rows
    )
    images.append(component_curve)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    motion_rows = list(parent_rows)
    if parent_rows:
        motion_rows.extend(
            {**row, "iteration": parent_rows[-1]["iteration"] + row["iteration"]}
            for row in (rows if "recovery_zero_update" in rows[0] else rows[1:])
        )
    else:
        motion_rows = rows
    iteration = [row["iteration"] for row in motion_rows]
    for axis, field, label in zip(
        axes[:2],
        ("pose_rotation_degrees", "pose_translation_mm"),
        ("Jaw rotation (degrees)", "Jaw translation magnitude (mm)"),
        strict=True,
    ):
        axis.plot(
            iteration, [row[field] for row in motion_rows], marker="o", color="#087d81"
        )
        axis.set(xlabel="Accepted optimizer iteration", ylabel=label)
    axes[2].plot(
        iteration,
        [row["geometry"]["inverted_tetrahedra"] for row in motion_rows],
        marker="o",
        color="#c75f42",
    )
    axes[2].set(
        xlabel="Accepted optimizer iteration", ylabel="Inverted retained tetrahedra"
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    name = "jaw-and-inversion-curves.png"
    figure.savefig(output / name, dpi=180)
    plt.close(figure)
    images.append(name)
    status = {
        "saved_status": saved_status,
        "forward_converged": bool(final["forward_converged"]),
        "contact_valid": bool(final["contact_valid"]),
        "forward_valid": bool(final["valid_forward"]),
        "inverse_converged": bool(summary["inverse_converged"]),
        "inverted_tetrahedra": int(final["geometry"]["inverted_tetrahedra"]),
        "detF_min": float(final["geometry"]["detF_min"]),
    }
    receipt = {
        "schema": "coupled-expression-saved-review-v1",
        "inputs": inputs,
        "status": status,
        "geometry": {
            "fem_vertices": volume.n_points,
            "tetrahedra": volume.n_cells,
            "boundary_triangles": fit_boundary.n_cells,
        },
        "target": {
            "name": expression_name,
            "expression_index": index,
            "scope": "Transferred skin-only kinematic target; no volumetric target was invented.",
            "vertices": target.n_points,
            "triangles": target.n_cells,
        },
        "fit_error_mm": {
            "unweighted_rms": float(
                np.sqrt(np.mean(error_mm**2)),
            ),
            "mean": float(error_mm.mean()),
            "max": float(error_mm.max()),
            "p95": float(np.quantile(error_mm, 0.95)),
        },
        "material_and_contact": {
            "active_strain": protocol["materials"]["formulation"],
            "skin_prestretch": protocol["materials"]["skin"],
            "ipc_policy": protocol["ipc_policy"],
            "ipc_stiffness_mpa": protocol["ipc_stiffness_mpa"],
            "collision_scope": "Soft tissue against complete cranium, mandible, and fixed eyes. Soft-soft and rigid-rigid contact are disabled. Rendering performs no collision query.",
        },
        "positional_fit_rms_mm": {
            "area_weighted_initial": initial_fit,
            "area_weighted_final": final_positional_fit,
        },
        "objective_terms": objective_terms(protocol),
        "saved_pose_rad_m": pose.tolist(),
        "curve_lineage": {
            "total_accepted_optimizer_iterations": total_iterations,
            "parent": parent_binding,
            "fixed_control_refinements": refinement_binding,
            "objective_component_coverage": component_coverage,
            "objective_continuity": objective_continuity,
            "storage_recovery": recovery_lineage,
        },
        "tetrahedron_policy": protocol["tetrahedron_policy"],
        "inversion_policy": protocol["inversion_policy"],
        "retained_geometry": final["geometry"],
        "solver_rerun": False,
        "assets": {
            path.name: record(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    image_html = "".join(
        f'<figure><img src="{name}" alt="Saved expression fit review"></figure>'
        for name in images
    )
    refinement_html = (
        "<p>Diamond markers show fixed-control equilibrium refinement at the same "
        "optimizer counter. These changes in fit and force use a stricter internal "
        "force target and add no accepted optimizer updates.</p>"
        if refinement_binding
        else ""
    )
    (output / "index.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>Coupled {expression_name} fit</title>
<style>body{{font:16px/1.5 system-ui;max-width:1500px;margin:2rem auto;padding:0 1rem}}img{{max-width:100%}}.status{{padding:1rem;background:#fff1c7}}details{{margin:1rem 0;padding:0.8rem;background:#f0f0ee}}</style>
<p><a href="../">Back to corrected neutral</a></p><h1>{expression_name} · coupled jaw and tissue</h1>
<p class="status"><strong>{total_iterations} accepted updates · {saved_status}. Inverse convergence: {summary["inverse_converged"]}.</strong>
The saved forward endpoint meets the force tolerance, configured contact checks, and declared inversion allowance. Its independent audit is linked below.</p>
<p>Area-weighted positional fit RMS: {initial_fit:.4f} → {final_positional_fit:.4f} mm.
Jaw rotation: {final["pose_rotation_degrees"]:.4f}°; translation magnitude: {final["pose_translation_mm"]:.4f} mm.
Residual force: {final["force_norm_n"]:.6g} N; tolerance: {final["force_threshold_n"]:.6g} N.</p>
{refinement_html}
<p>{"Storage recovery preserves the locally recorded zero-update boundary at optimizer step 32; curves break there because the initial displacement identity was not archived." if recovery_lineage else ""}</p>
<p><strong>Objective-components plot:</strong> full recorded objective, normalized positional L2, weighted normal and smoothness contributions, and raw activation roughness on its own scale. Historic L2-only rows leave unavailable components as gaps.</p>
<p>{protocol["tetrahedron_policy"]["excluded_tetrahedra"]:,} fully fixed tetrahedra are excluded from bulk mechanics.
The full original tetmesh boundary is retained for display. Among {final["geometry"]["retained_tetrahedra"]:,} retained mechanical tetrahedra,
{status["inverted_tetrahedra"]:,} are inverted; their rest-volume fraction is {100 * final["geometry"]["inverted_rest_volume_fraction"]:.6g}%; minimum det(F) is {status["detF_min"]:.6g}.
The declared allowance is {protocol["inversion_policy"]["maximum_inverted_tetrahedra"]} cells and {100 * protocol["inversion_policy"]["maximum_inverted_rest_volume_fraction"]:.4g}% rest volume.</p>
<p>The run uses a damped equilibrium-tangent predictor, CCD on combined jaw and tissue motion, and collision-on forward correction.
Skin pre-strain is preserved. Contact covers the selected soft surface against the complete cranium, mandible and eyes; soft-soft and rigid-rigid contact are disabled. The soft collision surface retains only faces whose three vertices are labelled neither Cranium nor Mandible; mixed faces remain excluded.</p>
<p>The target is transferred skin geometry. The error map uses unweighted errors: RMS {receipt["fit_error_mm"]["unweighted_rms"]:.4f} mm, p95 {receipt["fit_error_mm"]["p95"]:.4f} mm, maximum {receipt["fit_error_mm"]["max"]:.4f} mm.</p>
{image_html}
<details><summary>Verification and provenance</summary><p><a href="independent-audit.json">Independent endpoint audit</a> · <a href="receipt.json">Input and asset hashes</a></p></details>
<p><a href="expression-transferred-skin-target.vtp">Target skin</a> · <a href="neutral-full-tetmesh-boundary.vtp">Neutral full boundary</a> · <a href="fit-full-tetmesh-boundary.vtp">Fit full boundary</a></p>"""
    )
    receipt["assets"]["index.html"] = record(output / "index.html")
    write_json(output / "receipt.json", receipt)
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "review/fit_rms_mm": receipt["fit_error_mm"]["unweighted_rms"],
            "review/forward_valid": float(status["forward_valid"]),
            "review/inverse_converged": float(status["inverse_converged"]),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
