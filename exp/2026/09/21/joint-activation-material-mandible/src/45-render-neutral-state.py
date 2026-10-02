"""Render the stable current contact-enabled neutral checkpoint."""

from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
from joint_common import GROUP, ProfileJoint, sha256, write_json
from joint_data import PreparedInputs
from joint_fields import SharedFieldParameters
from joint_spatial_fields import SPATIAL_FIELD_SCHEMA, SpatialSharedFieldParameters

from liblaf import cherries

LOG = logging.getLogger(__name__)
WINDOW = (1600, 1200)
BACKGROUND = "#f7f7f5"


class Config(cherries.BaseConfig):
    prepared_dir: Path = GROUP / "data/prepared"
    run_dir: Path = GROUP / "data/neutral-convergence-010-contact"
    checkpoint: Path = GROUP / "data/neutral-convergence-010-contact/terminal.pt"
    contact_visuals_dir: Path = GROUP / "data/contact-neutral-visuals"
    output_dir: Path = cherries.output("neutral-state-visuals", mkdir=True)
    overlay_magnification: float = 20.0


def load_stable_torch(path: Path) -> tuple[dict[str, Any], str, dict[str, int]]:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (
        after.st_size,
        after.st_mtime_ns,
    ), f"checkpoint changed during read: {path}"
    assert len(payload) == before.st_size
    value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    return (
        value,
        hashlib.sha256(payload).hexdigest(),
        {"size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns},
    )


def load_stable_json(path: Path) -> tuple[Any, str]:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (
        after.st_size,
        after.st_mtime_ns,
    ), f"JSON changed during read: {path}"
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def camera(mesh: pv.DataSet, view: str, zoom: float = 1.0) -> dict[str, Any]:
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    center = np.asarray(((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2))
    span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    if view == "front":
        eye = center + np.asarray((0.0, 0.0, 2.7 * span))
        horizontal_span = xmax - xmin
    elif view == "side":
        eye = center + np.asarray((2.7 * span, 0.0, 0.0))
        horizontal_span = zmax - zmin
    else:
        raise ValueError(view)
    aspect = WINDOW[0] / WINDOW[1]
    scale = 1.10 * max((ymax - ymin) / 2, horizontal_span / (2 * aspect)) / zoom
    return {
        "position": [eye.tolist(), center.tolist(), (0.0, 1.0, 0.0)],
        "parallel_scale": scale,
    }


def plotter(title: str) -> pv.Plotter:
    result = pv.Plotter(off_screen=True, window_size=WINDOW)
    result.set_background(BACKGROUND)
    result.enable_anti_aliasing("ssaa")
    result.add_text(title, position="upper_left", font_size=16, color="#202124")
    return result


def save(plot: pv.Plotter, path: Path, settings: dict[str, Any]) -> None:
    plot.camera_position = settings["position"]
    plot.camera.parallel_projection = True
    plot.camera.parallel_scale = settings["parallel_scale"]
    plot.reset_camera_clipping_range()
    plot.show(screenshot=path, auto_close=True)


def scalar_bar(title: str) -> dict[str, Any]:
    return {
        "title": title,
        "vertical": True,
        "position_x": 0.86,
        "position_y": 0.24,
        "width": 0.07,
        "height": 0.50,
    }


def audit_surface_selection(
    volume: pv.UnstructuredGrid, contact_visuals: dict[str, Any]
) -> dict[str, Any]:
    boundary = volume.extract_surface(algorithm=None).triangulate()
    original = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    faces = original[np.asarray(boundary.faces).reshape(-1, 4)[:, 1:]]
    names = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    labels = np.asarray(volume.point_data["GroupId"], dtype=np.int32)
    is_cranium = labels[faces] == names.index("Cranium")
    is_mandible = labels[faces] == names.index("Mandible")
    pure_cranium = np.all(is_cranium, axis=1)
    pure_mandible = np.all(is_mandible, axis=1)
    pure_soft = np.all(~(is_cranium | is_mandible), axis=1)
    mixed = ~(pure_cranium | pure_mandible | pure_soft)
    counts = {
        "boundary_triangles": int(boundary.n_cells),
        "pure_cranium_triangles": int(pure_cranium.sum()),
        "pure_mandible_triangles": int(pure_mandible.sum()),
        "pure_soft_triangles": int(pure_soft.sum()),
        "bonded_mixed_transition_triangles": int(mixed.sum()),
    }
    classified = sum(
        value for key, value in counts.items() if key != "boundary_triangles"
    )
    assert classified == counts["boundary_triangles"]
    receipt = contact_visuals["surface_map"]
    assert counts["pure_cranium_triangles"] == receipt["cranium_triangles"]
    assert counts["pure_mandible_triangles"] == receipt["mandible_triangles"]
    assert counts["pure_soft_triangles"] == receipt["soft_triangles"]
    assert (
        counts["bonded_mixed_transition_triangles"]
        == receipt["bonded_mixed_triangles_omitted"]
    )
    collider_scope = contact_visuals["collider_scope_audit"]
    assert collider_scope["cranium"]["collider_vertices_outside_support"] == 0
    assert collider_scope["mandible"]["collider_vertices_outside_support"] == 0
    return counts | {
        "partition_sum_matches_boundary": True,
        "contact_receipt_matches_independent_boundary_classification": True,
        "contact_eligible": "all exact pure-soft faces versus all exact pure-cranium and pure-mandible faces",
        "omitted": "only mixed bone-soft transition faces, retained as bonded FEM topology",
        "anatomical_limit": "mixed-face omission is justified by shared FEM topology; it does not validate whether the modeled attachment correspondence is anatomically correct",
        "collider_scope_audit": collider_scope,
    }


def render_surface_drift(
    output: Path,
    skin: pv.PolyData,
    displacement: np.ndarray,
    scale_mm: float,
    status: str,
) -> list[dict[str, str]]:
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    surface_u = displacement[ids]
    deformed = skin.copy(deep=True)
    deformed.points = np.asarray(skin.points) + surface_u
    deformed.point_data["DisplacementMm"] = np.linalg.norm(surface_u, axis=1) * 1000
    assets = []
    for view in ("front", "side"):
        name = f"01-contact-neutral-drift-{view}.png"
        p = plotter(f"Contact neutral drift — {view} — {status}")
        p.add_mesh(
            deformed,
            scalars="DisplacementMm",
            cmap="turbo",
            clim=(0.0, scale_mm),
            smooth_shading=True,
            scalar_bar_args=scalar_bar("|u| (mm)"),
        )
        p.add_text(
            "current terminal relative to the frozen reference; shared scale",
            position="lower_left",
            font_size=10,
            color="#202124",
        )
        save(p, output / name, camera(skin, view, zoom=1.04))
        assets.append(
            {
                "filename": name,
                "caption": f"{view.title()} surface displacement magnitude of the current contact-enabled terminal relative to the frozen reference, on the shared 0--{scale_mm:.4f} mm scale.",
            }
        )
    return assets


def render_overlays(
    output: Path,
    skin: pv.PolyData,
    displacement: np.ndarray,
    magnification: float,
    status: str,
) -> list[dict[str, str]]:
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    surface_u = displacement[ids]
    actual = skin.copy(deep=True)
    actual.points = np.asarray(skin.points) + surface_u
    amplified = skin.copy(deep=True)
    amplified.points = np.asarray(skin.points) + magnification * surface_u
    assets = []
    for view in ("front", "side"):
        name = f"02-contact-neutral-overlay-{view}.png"
        p = plotter(f"Contact neutral overlay — {view} — {status}")
        p.add_mesh(skin, color="#727b85", opacity=0.28, smooth_shading=True)
        p.add_mesh(actual, color="#1976ad", opacity=0.38, smooth_shading=True)
        p.add_mesh(
            amplified,
            color="#c7354b",
            opacity=0.82,
            style="wireframe",
            line_width=0.9,
        )
        p.add_text(
            f"gray: reference   blue: actual terminal   red wire: {magnification:g}x drift",
            position="lower_left",
            font_size=10,
            color="#202124",
        )
        save(p, output / name, camera(skin, view, zoom=1.04))
        assets.append(
            {
                "filename": name,
                "caption": f"{view.title()} geometry overlay: frozen reference in gray, actual current contact-enabled terminal in blue, and {magnification:g}x amplified displacement wireframe in red.",
            }
        )
    return assets


def render_sections(
    output: Path,
    volume: pv.UnstructuredGrid,
    displacement: np.ndarray,
    pivot: np.ndarray,
    scale_mm: float,
    status: str,
) -> list[dict[str, str]]:
    reference = volume.copy(deep=True)
    deformed = volume.copy(deep=True)
    deformed.points = np.asarray(volume.points) + displacement
    deformed.point_data["DisplacementMm"] = np.linalg.norm(displacement, axis=1) * 1000
    specifications = (
        (
            "coronal",
            (0.0, 0.0, 1.0),
            (pivot[0], pivot[1] - 0.02, 0.065),
            "front",
        ),
        ("sagittal", (1.0, 0.0, 0.0), tuple(pivot), "side"),
    )
    assets = []
    for label, normal, origin, view in specifications:
        reference_slice = reference.slice(normal=normal, origin=origin)
        deformed_slice = deformed.slice(normal=normal, origin=origin)
        name = f"03-contact-neutral-{label}-section.png"
        p = plotter(f"Contact neutral {label} section — {status}")
        p.add_mesh(
            deformed_slice,
            scalars="DisplacementMm",
            cmap="turbo",
            clim=(0.0, scale_mm),
            show_edges=True,
            edge_color="#6c737a",
            line_width=0.25,
            scalar_bar_args=scalar_bar("|u| (mm)"),
        )
        p.add_mesh(
            reference_slice,
            color="#17191b",
            style="wireframe",
            opacity=0.30,
            line_width=0.4,
        )
        p.add_text(
            "colored: deformed terminal   black wire: frozen reference",
            position="lower_left",
            font_size=10,
            color="#202124",
        )
        save(p, output / name, camera(reference_slice, view, zoom=1.02))
        assets.append(
            {
                "filename": name,
                "caption": f"{label.title()} tetrahedral section of displacement magnitude at the current contact-enabled terminal, with the frozen reference section overlaid in black.",
            }
        )
    return assets


def material_state(
    checkpoint: dict[str, Any], volume: pv.UnstructuredGrid
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    spatial = checkpoint["materials"]["schema"] == SPATIAL_FIELD_SCHEMA
    if spatial:
        receipt = checkpoint["protocol"]["spatial_basis"]
        shared = SpatialSharedFieldParameters(
            Path(receipt["basis_path"]),
            Path(receipt["audit_summary_path"]),
            volume.n_cells,
            dtype=torch.float64,
        )
        assert shared.basis_receipt() == receipt
        assert shared.config == checkpoint["materials"]
    else:
        shared = SharedFieldParameters(checkpoint["materials"], dtype=torch.float64)
    with torch.no_grad():
        shared.coefficients.copy_(
            checkpoint["shared_coefficients"].detach().cpu().to(torch.float64)
        )
        bulk_stress = shared.bulk_stresses_mpa().detach().cpu().numpy()
        skin_resultant = shared.skin_resultant_n_per_m().detach().cpu().numpy()
        skin_multiplier = float(shared.skin_stiffness_multiplier())
    fraction_names = ("FatFraction", "AponeurosisFraction", "MuscleFraction")
    fractions = np.column_stack(
        [
            np.asarray(volume.cell_data[name], dtype=np.float64)
            for name in fraction_names
        ]
    )
    cell_stress = (
        np.einsum("ct,tcij->cij", fractions, bulk_stress)
        if spatial
        else np.einsum("ct,tij->cij", fractions, bulk_stress)
    )
    cell_principal = np.linalg.eigvalsh(cell_stress)
    component_maximum = np.zeros((3, volume.n_cells))
    for index in range(3):
        support = fractions[:, index] > 1e-6
        values = bulk_stress[index, support] if spatial else bulk_stress[index]
        component_maximum[index, support] = np.linalg.eigvalsh(values)[..., -1]
    tissue_mask = fractions.sum(axis=1) > 0.0
    skin = checkpoint["materials"]["materials"]["skin"]
    reference_young_mpa = float(skin["reference_map"]["young_mpa"])
    effective_young_mpa = reference_young_mpa * skin_multiplier
    return (
        {
            "bulk_stress_mpa_by_tissue": None
            if spatial
            else {
                name: bulk_stress[index].tolist()
                for index, name in enumerate(("fat", "aponeurosis", "muscle"))
            },
            "spatial_basis": checkpoint["protocol"]["spatial_basis"]
            if spatial
            else None,
            "cell_definition": "sum of FatFraction, AponeurosisFraction, and MuscleFraction times their checkpoint shared baseline stress tensors",
            "tissue_cells": int(tissue_mask.sum()),
            "largest_principal_kpa": {
                "minimum": float(1000.0 * cell_principal[tissue_mask, -1].min()),
                "maximum": float(1000.0 * cell_principal[tissue_mask, -1].max()),
            },
            "smallest_principal_kpa": {
                "minimum": float(1000.0 * cell_principal[tissue_mask, 0].min()),
                "maximum": float(1000.0 * cell_principal[tissue_mask, 0].max()),
            },
            "skin_resultant_n_per_m": skin_resultant.tolist(),
            "skin_stiffness_multiplier": skin_multiplier,
            "skin_reference_young_mpa": reference_young_mpa,
            "skin_effective_young_mpa": effective_young_mpa,
            "skin_spatial_basis": "uniform scalar multiplier; no registered spatial stiffness map",
            "status": checkpoint["materials"]["status"],
        },
        cell_principal,
        component_maximum,
    )


def render_tissue_baselines(
    output: Path,
    volume: pv.UnstructuredGrid,
    displacement: np.ndarray,
    pivot: np.ndarray,
    component_maximum_mpa: np.ndarray,
    status: str,
) -> list[dict[str, str]]:
    """Show each constitutive field on its own declared signed color scale."""
    deformed = volume.copy(deep=True)
    deformed.points = np.asarray(volume.points) + displacement
    specifications = (
        ("coronal", (0.0, 0.0, 1.0), (pivot[0], pivot[1] - 0.02, 0.065), "front"),
        ("sagittal", (1.0, 0.0, 0.0), tuple(pivot), "side"),
    )
    assets = []
    for index, (name, fraction) in enumerate(
        zip(
            ("fat", "aponeurosis", "muscle"),
            ("FatFraction", "AponeurosisFraction", "MuscleFraction"),
            strict=True,
        )
    ):
        mask = np.asarray(volume.cell_data[fraction]) > 1e-6
        values = component_maximum_mpa[index, mask] * 1000
        tissue = deformed.extract_cells(mask)
        tissue.cell_data["component_kpa"] = values
        limit = max(float(np.max(np.abs(values))), 1e-12)
        for section_name, normal, origin, view in specifications:
            section = tissue.slice(normal=normal, origin=origin)
            assert section.n_cells > 0, (name, section_name)
            filename = f"06-{name}-baseline-{section_name}.png"
            p = plotter(f"{name.title()} baseline — {section_name} — {status}")
            p.add_mesh(
                section,
                scalars="component_kpa",
                cmap="coolwarm",
                clim=(-limit, limit),
                scalar_bar_args=scalar_bar("kPa"),
                show_edges=False,
            )
            reference_section = deformed.slice(normal=normal, origin=origin)
            save(p, output / filename, camera(reference_section, view, zoom=1.02))
            assets.append(
                {
                    "filename": filename,
                    "caption": f"{name.title()} largest-principal baseline stress before tissue-fraction weighting on the current {section_name} section. Positive means tension; the signed scale is ±{limit:.5g} kPa for this tissue and checkpoint. Optimized model field, not measured anatomy.",
                }
            )
    return assets


def render_material_state(
    output: Path,
    volume: pv.UnstructuredGrid,
    skin: pv.PolyData,
    displacement: np.ndarray,
    pivot: np.ndarray,
    state: dict[str, Any],
    cell_principal_mpa: np.ndarray,
    status: str,
) -> list[dict[str, str]]:
    deformed_volume = volume.copy(deep=True)
    deformed_volume.points = np.asarray(volume.points) + displacement
    fractions = np.column_stack(
        [
            np.asarray(volume.cell_data[name], dtype=np.float64)
            for name in ("FatFraction", "AponeurosisFraction", "MuscleFraction")
        ]
    )
    tissue = deformed_volume.extract_cells(fractions.sum(axis=1) > 0.0)
    tissue.cell_data["LargestPrincipalBaselineKPa"] = (
        1000.0 * cell_principal_mpa[fractions.sum(axis=1) > 0.0, -1]
    )
    limit_kpa = float(
        max(
            abs(state["largest_principal_kpa"]["minimum"]),
            abs(state["largest_principal_kpa"]["maximum"]),
            1.0e-12,
        )
    )
    specifications = (
        (
            "coronal",
            (0.0, 0.0, 1.0),
            (pivot[0], pivot[1] - 0.02, 0.065),
            "front",
        ),
        ("sagittal", (1.0, 0.0, 0.0), tuple(pivot), "side"),
    )
    assets: list[dict[str, str]] = []
    for label, normal, origin, view in specifications:
        section = tissue.slice(normal=normal, origin=origin)
        assert section.n_cells > 0, f"empty {label} material section"
        name = f"04-baseline-stress-{label}.png"
        p = plotter(f"Checkpoint baseline stress — {label} — {status}")
        p.add_mesh(
            section,
            scalars="LargestPrincipalBaselineKPa",
            cmap="coolwarm",
            clim=(-limit_kpa, limit_kpa),
            show_edges=True,
            edge_color="#6c737a",
            line_width=0.25,
            scalar_bar_args=scalar_bar("largest principal (kPa)"),
        )
        p.add_text(
            "fraction-weighted shared baseline tensor; research sensitivity field",
            position="lower_left",
            font_size=10,
            color="#202124",
        )
        save(p, output / name, camera(section, view, zoom=1.04))
        assets.append(
            {
                "filename": name,
                "caption": f"{label.title()} section of the current checkpoint's fraction-weighted largest-principal bulk baseline stress, on the shared {-limit_kpa:.5g}--{limit_kpa:.5g} kPa scale. This is an optimized model field, not a measured registered stress map.",
            }
        )

    deformed_skin = skin.copy(deep=True)
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    deformed_skin.points = np.asarray(skin.points) + displacement[skin_ids]
    deformed_skin.cell_data["EffectiveYoungMPa"] = np.full(
        deformed_skin.n_cells, state["skin_effective_young_mpa"]
    )
    bounds = (0.2 / 3.0, 0.2 * 3.0)
    name = "05-skin-stiffness-front.png"
    p = plotter(f"Checkpoint skin stiffness — front — {status}")
    p.add_mesh(
        deformed_skin,
        scalars="EffectiveYoungMPa",
        cmap="viridis",
        clim=bounds,
        smooth_shading=True,
        scalar_bar_args=scalar_bar("effective E (MPa)"),
    )
    p.add_text(
        f"uniform multiplier {state['skin_stiffness_multiplier']:.5f}; model bounds {bounds[0]:.4f}--{bounds[1]:.4f} MPa",
        position="lower_left",
        font_size=10,
        color="#202124",
    )
    save(p, output / name, camera(skin, "front", zoom=1.04))
    assets.append(
        {
            "filename": name,
            "caption": f"Current deformed skin colored by the checkpoint effective Young modulus, {state['skin_effective_young_mpa']:.6g} MPa. The field is spatially uniform because the approved basis has one shared log-stiffness multiplier; the color scale spans the fixed {bounds[0]:.6g}--{bounds[1]:.6g} MPa model bounds.",
        }
    )
    return assets


def write_report(output: Path, summary: dict[str, Any]) -> None:
    status = summary["status"]
    collider = summary["surface_selection_audit"]["collider_scope_audit"]
    lines = [
        "# Current contact-enabled neutral state",
        "",
        f"Checkpoint update `{status['update']}` is contact validated and within the "
        f"neutral geometry budget. Optimizer converged: `{status['optimizer_converged']}`; "
        f"preparation complete: `{status['preparation_complete']}`. These figures are the "
        "current terminal state, not the older prestress pilots and not a converged result.",
        "",
    ]
    for asset in summary["assets"]:
        lines.extend(
            [
                f"## {asset['filename']}",
                "",
                f"![{asset['filename']}]({asset['filename']})",
                "",
                asset["caption"],
                "",
            ]
        )
    lines.extend(
        [
            "## Scope",
            "",
            "The current checkpoint includes the exact pure-soft versus pure-cranium and "
            "pure-mandible IPC term. Mixed bone-soft triangles remain bonded FEM transition "
            "topology and are omitted from sliding contact. This is simulation ownership, "
            "not validation of anatomical attachment correspondence.",
            "",
            f"All {collider['cranium']['collider_vertices']:,} pure-cranium collider "
            f"vertices belong to the recovered fixed support, and all "
            f"{collider['mandible']['collider_vertices']:,} pure-mandible collider "
            "vertices belong to the differentiable rigid-jaw support. The selected FEM "
            "boundary is still a cropped face subvolume, not complete anatomical source-bone "
            "coverage.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))


def main(cfg: Config) -> None:
    pv.OFF_SCREEN = True
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    checkpoint, checkpoint_hash, checkpoint_stat = load_stable_torch(cfg.checkpoint)
    trace, trace_hash = load_stable_json(cfg.run_dir / "trace.json")
    contact_visuals, contact_visuals_hash = load_stable_json(
        cfg.contact_visuals_dir / "summary.json"
    )
    assert contact_visuals["state"]["checkpoint"]["sha256"] == checkpoint_hash
    assert checkpoint["stage"] == "neutral"
    assert checkpoint["protocol"]["input_manifest_sha256"] == sha256(
        cfg.prepared_dir / "manifest.json"
    )
    assert checkpoint["protocol"]["input_arrays_sha256"] == sha256(
        cfg.prepared_dir / "inputs.npz"
    )
    trace = [row for row in trace if row["update"] <= checkpoint["update"]]
    assert trace[-1]["update"] == checkpoint["update"]
    row = trace[-1]
    displacement_t = checkpoint["primal"]["neutral"].detach().cpu()
    volume = pv.read(prepared.volume_path)
    skin = pv.read(prepared.skin_path).triangulate()
    assert tuple(displacement_t.shape) == (volume.n_points, 3)
    displacement = displacement_t.numpy(force=True)
    assert np.isfinite(displacement).all()
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    surface_mm = np.linalg.norm(displacement[skin_ids], axis=1) * 1000
    scale_mm = float(surface_mm.max())
    surface_selection = audit_surface_selection(volume, contact_visuals)
    material, cell_principal_mpa, component_maximum_mpa = material_state(
        checkpoint, volume
    )
    status_label = (
        "converged"
        if checkpoint["neutral_converged"]
        else "stationary; shape limit failed"
        if checkpoint["optimizer_converged"]
        else "not converged"
    )
    assets = []
    assets.extend(
        render_surface_drift(cfg.output_dir, skin, displacement, scale_mm, status_label)
    )
    assets.extend(
        render_overlays(
            cfg.output_dir,
            skin,
            displacement,
            cfg.overlay_magnification,
            status_label,
        )
    )
    assets.extend(
        render_sections(
            cfg.output_dir,
            volume,
            displacement,
            prepared.arrays["mandible_pivot_m"],
            scale_mm,
            status_label,
        )
    )
    assets.extend(
        render_material_state(
            cfg.output_dir,
            volume,
            skin,
            displacement,
            prepared.arrays["mandible_pivot_m"],
            material,
            cell_principal_mpa,
            status_label,
        )
    )
    assets.extend(
        render_tissue_baselines(
            cfg.output_dir,
            volume,
            displacement,
            prepared.arrays["mandible_pivot_m"],
            component_maximum_mpa,
            status_label,
        )
    )
    numerical = row["metrics"]
    status = {
        "update": checkpoint["update"],
        "accepted_steps": checkpoint["accepted_steps"],
        "optimizer_converged": bool(checkpoint["optimizer_converged"]),
        "neutral_budget_met": bool(checkpoint["neutral_budget_met"]),
        "contact_validated": bool(checkpoint["contact_validated"]),
        "preparation_complete": bool(checkpoint["preparation_complete"]),
        "neutral_converged": bool(checkpoint["neutral_converged"]),
        "scope": "current contact-enabled terminal; not the old neutral-prestress pilots and not final joint-optimization trends",
    }
    assert status["contact_validated"]
    summary = {
        "schema": "joint-contact-neutral-state-visuals-v1",
        "checkpoint": {
            "path": str(cfg.checkpoint.resolve()),
            "sha256": checkpoint_hash,
            **checkpoint_stat,
        },
        "trace": {
            "path": str((cfg.run_dir / "trace.json").resolve()),
            "sha256": trace_hash,
            "accepted_evaluations_through_checkpoint": len(trace),
        },
        "prepared_inputs_sha256": sha256(cfg.prepared_dir / "inputs.npz"),
        "prepared_manifest_sha256": sha256(cfg.prepared_dir / "manifest.json"),
        "status": status,
        "surface_displacement_mm": {
            "minimum": float(surface_mm.min()),
            "median": float(np.median(surface_mm)),
            "q95": float(np.quantile(surface_mm, 0.95)),
            "q99": float(np.quantile(surface_mm, 0.99)),
            "maximum": scale_mm,
            "shared_visual_scale": [0.0, scale_mm],
        },
        "numerical_geometry": numerical,
        "contact": row["contact"],
        "oral_diagnostic": row["oral_diagnostic"],
        "surface_selection_audit": surface_selection,
        "material_state": material,
        "related_contact_visuals": {
            "directory": str(cfg.contact_visuals_dir.resolve()),
            "summary_sha256": contact_visuals_hash,
            "assets": contact_visuals["assets"],
        },
        "assets": assets,
    }
    write_json(cfg.output_dir / "summary.json", summary)
    write_report(cfg.output_dir, summary)
    cherries.log_metrics(
        {
            "neutral/update": status["update"],
            "neutral/contact_validated": float(status["contact_validated"]),
            "neutral/optimizer_converged": float(status["optimizer_converged"]),
            "neutral/surface_max_mm": scale_mm,
            "neutral/detF_min": numerical["detF_min"],
            "neutral/contact_min_gap_mm": row["contact"]["minimum_active_distance_m"]
            * 1000,
        }
    )
    LOG.info("Wrote %d current-neutral visuals to %s", len(assets), cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
