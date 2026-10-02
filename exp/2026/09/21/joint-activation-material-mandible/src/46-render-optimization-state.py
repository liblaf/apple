"""Render real accepted control/final spatial checkpoints without an FEM solve."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from joint_common import GROUP, ProfileJoint, sha256, write_json
from joint_contact import build_owned_contact
from joint_data import PreparedInputs

from liblaf import cherries

LOG = logging.getLogger(__name__)
WINDOW = (1600, 1200)
BACKGROUND = "#f7f7f5"
SNAPSHOT_SCHEMA = "joint-optimization-visualization-snapshot-v2"
LEGACY_SNAPSHOT_SCHEMA = "joint-optimization-visualization-snapshot-v1"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    run_dir: Path
    spatial_dir: Path
    prepared_dir: Path = GROUP / "data/prepared"
    contact_spec: Path = GROUP / "data/contact/config.json"
    output_dir: Path = cherries.output("optimization-state-visuals", mkdir=True)
    checkpoint_labels: str = "0000,best,terminal"
    contact_expression: str = "MouthOpenSlightly"
    residual_cap_mm: float | None = None
    maximum_principal_axes: int = 260


def load_stable_json(path: Path) -> tuple[Any, str]:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (
        after.st_size,
        after.st_mtime_ns,
    ), f"JSON changed during read: {path}"
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def load_stable_snapshot(path: Path) -> tuple[dict[str, np.ndarray], str]:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (
        after.st_size,
        after.st_mtime_ns,
    ), f"snapshot changed during read: {path}"
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    return data, hashlib.sha256(payload).hexdigest()


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
    result.add_text(title, position="upper_left", font_size=15, color="#202124")
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


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def boundary_partition(
    volume: pv.UnstructuredGrid, points: np.ndarray
) -> dict[str, pv.PolyData]:
    boundary = volume.extract_surface(algorithm=None).triangulate()
    original = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    boundary.points = points[original]
    names = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    labels = np.asarray(volume.point_data["GroupId"], dtype=np.int32)[original]
    faces = np.asarray(boundary.faces).reshape(-1, 4)[:, 1:]
    cranium_id, mandible_id = names.index("Cranium"), names.index("Mandible")
    is_cranium = labels[faces] == cranium_id
    is_mandible = labels[faces] == mandible_id
    masks = {
        "cranium": np.all(is_cranium, axis=1),
        "mandible": np.all(is_mandible, axis=1),
        "soft": np.all(~(is_cranium | is_mandible), axis=1),
    }
    masks["bonded"] = ~(masks["cranium"] | masks["mandible"] | masks["soft"])
    return {
        name: boundary.extract_cells(np.flatnonzero(mask))
        .extract_surface(algorithm=None)
        .triangulate()
        for name, mask in masks.items()
    }


def contact_records(
    contact: Any,
    state: Any,
    displacement: torch.Tensor,
    volume: pv.UnstructuredGrid,
) -> list[dict[str, Any]]:
    positions = (contact.vertices + displacement[contact.indices]).numpy(force=True)
    edges = contact.collision_mesh.edges
    faces = contact.collision_mesh.faces
    global_ids = contact.indices.numpy(force=True)
    names = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    labels = np.asarray(volume.point_data["GroupId"], dtype=np.int32)
    bone_ids = {"cranium": names.index("Cranium"), "mandible": names.index("Mandible")}
    records: list[dict[str, Any]] = []
    for collection in (
        "vv_collisions",
        "ev_collisions",
        "ee_collisions",
        "fv_collisions",
    ):
        for collision in getattr(state.collisions, collection):
            local_ids = np.asarray(collision.vertex_ids(edges, faces), dtype=np.int64)
            local_ids = local_ids[local_ids >= 0]
            stencil = np.asarray(collision.dof(positions, edges, faces))
            xyz = stencil.reshape(-1, 3)
            coefficients = np.asarray(
                collision.compute_coefficients(stencil), dtype=np.float64
            ).reshape(-1)
            positive, negative = coefficients > 0, coefficients < 0
            assert positive.any()
            assert negative.any()
            point_a = (xyz[positive] * coefficients[positive, None]).sum(axis=0) / (
                coefficients[positive].sum()
            )
            point_b = (xyz[negative] * (-coefficients[negative, None])).sum(axis=0) / (
                -coefficients[negative].sum()
            )
            stencil_labels = labels[global_ids[local_ids]]
            targets = [
                name for name, value in bone_ids.items() if value in stencil_labels
            ]
            assert len(targets) == 1, (collection, targets, stencil_labels)
            records.append(
                {
                    "kind": collection.removesuffix("_collisions"),
                    "bone": targets[0],
                    "gap_m": float(np.linalg.norm(point_a - point_b)),
                    "location_m": ((point_a + point_b) / 2).tolist(),
                }
            )
    return records


def clip(mesh: pv.PolyData, bounds: tuple[float, ...]) -> pv.PolyData:
    return (
        mesh.clip_box(bounds, invert=False)
        .extract_surface(algorithm=None)
        .triangulate()
    )


def oral_bounds(volume: pv.UnstructuredGrid) -> tuple[float, ...]:
    mask = np.zeros(volume.n_points, dtype=bool)
    for name in ("IsLip", "IsGingiva", "IsTeeth"):
        mask |= np.asarray(volume.point_data[name], dtype=bool)
    assert mask.any()
    points = np.asarray(volume.points)[mask]
    padding = np.asarray((0.004, 0.004, 0.004))
    lower, upper = points.min(axis=0) - padding, points.max(axis=0) + padding
    return (
        lower[0],
        upper[0],
        lower[1],
        upper[1],
        lower[2],
        upper[2],
    )


def observation_on_skin(
    values: np.ndarray, observation_ids: np.ndarray, skin: pv.PolyData
) -> np.ndarray:
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    assert len(np.unique(observation_ids)) == len(observation_ids)
    lookup = {int(value): index for index, value in enumerate(observation_ids)}
    indices = np.asarray([lookup[int(value)] for value in skin_ids], dtype=np.int64)
    return values[indices]


def validate_snapshot(  # noqa: PLR0915 - explicit versioned scientific data contract.
    path: Path,
    data: dict[str, np.ndarray],
    prepared: PreparedInputs,
    volume: pv.UnstructuredGrid,
    skin: pv.PolyData,
) -> dict[str, Any]:
    required = {
        "schema",
        "stage",
        "checkpoint_label",
        "status",
        "update",
        "activation",
        "activation_reference_mpa",
        "activation_cap_mpa",
        "symmetric_coordinate_order",
        "active_cell_ids",
        "jaw_pose_rad_m",
        "shared",
        "observation_node_ids",
        "predicted_observation_displacement_m",
        "target_displacement_m",
        "target_indices",
        "full_displacement_m",
    }
    assert required <= data.keys(), sorted(required - data.keys())
    snapshot_schema = str(data["schema"].item())
    assert snapshot_schema in {SNAPSHOT_SCHEMA, LEGACY_SNAPSHOT_SCHEMA}
    if snapshot_schema == SNAPSHOT_SCHEMA:
        shared_field = json.loads(str(data["shared_field_json"].item()))
        materials = json.loads(str(data["material_config_json"].item()))
        spatial_basis = json.loads(str(data["spatial_basis_json"].item()))
        shared_count = int(data["shared_coefficient_count"].item())
        shared_basis = str(data["shared_basis"].item())
        assert shared_field["coefficient_count"] == shared_count
        assert materials["parameterization"]["shared_coefficient_count"] == shared_count
        assert shared_field["basis"] == shared_basis
        if shared_basis == "spatial80":
            assert shared_count == 80
            assert materials["schema"] == "joint-additive-spatial-stress-fields-v1"
            assert spatial_basis["input_arrays_sha256"] == sha256(prepared.npz_path)
            assert spatial_basis["input_manifest_sha256"] == sha256(
                prepared.manifest_path
            )
            assert (
                sha256(Path(spatial_basis["basis_path"]))
                == spatial_basis["basis_sha256"]
            )
            assert (
                sha256(Path(spatial_basis["audit_summary_path"]))
                == spatial_basis["audit_summary_sha256"]
            )
            assert shared_field["spatial_smoothness"]["weight"] == 100.0
            assert shared_field["spatial_smoothness"]["factor"] == 0.5
        else:
            assert shared_basis == "constant20"
            assert shared_count == 20
            assert spatial_basis is None
            assert materials["schema"] == "joint-additive-stress-fields-v1"
    else:
        # Version 1 was exclusively the original constant20 model.
        shared_count = 20
        shared_basis = "constant20"
    label = path.stem.removeprefix("visualization-")
    assert str(data["checkpoint_label"].item()) == label
    stage = str(data["stage"].item())
    assert stage in {"control_converge", "joint_trend"}
    assert str(data["status"].item()) == "accepted_numerically_valid_state"
    assert data["symmetric_coordinate_order"].tolist() == [
        "xx",
        "yy",
        "zz",
        "sqrt2_xy",
        "sqrt2_yz",
        "sqrt2_xz",
    ]
    expression_count = len(data["target_indices"])
    assert data["activation"].shape == (
        expression_count,
        len(prepared.arrays["active_cell_ids"]),
        6,
    )
    assert data["jaw_pose_rad_m"].shape == (expression_count, 6)
    assert data["shared"].shape == (shared_count,)
    assert np.isfinite(data["shared"]).all()
    assert data["full_displacement_m"].shape == (
        expression_count,
        volume.n_points,
        3,
    )
    assert data["full_displacement_m"].dtype == np.float64
    assert data["predicted_observation_displacement_m"].shape == (
        expression_count,
        len(data["observation_node_ids"]),
        3,
    )
    assert (
        data["target_displacement_m"].shape
        == data["predicted_observation_displacement_m"].shape
    )
    assert data["predicted_observation_displacement_m"].dtype == np.float64
    assert np.array_equal(
        data["predicted_observation_displacement_m"],
        data["full_displacement_m"][:, data["observation_node_ids"]],
    )
    assert np.array_equal(data["active_cell_ids"], prepared.arrays["active_cell_ids"])
    assert np.array_equal(
        data["observation_node_ids"], prepared.arrays["observation_node_ids"]
    )
    assert np.isfinite(data["full_displacement_m"]).all()
    assert np.isfinite(data["predicted_observation_displacement_m"]).all()
    assert np.isfinite(data["target_displacement_m"]).all()
    observation_on_skin(
        data["predicted_observation_displacement_m"][0],
        data["observation_node_ids"],
        skin,
    )
    reference_mpa = float(data["activation_reference_mpa"])
    cap_mpa = float(data["activation_cap_mpa"])
    assert reference_mpa > 0
    assert cap_mpa >= reference_mpa
    return {
        "label": label,
        "stage": stage,
        "update": int(data["update"]),
        "expressions": expression_count,
        "activation_reference_mpa": reference_mpa,
        "activation_cap_mpa": cap_mpa,
        "snapshot_schema": snapshot_schema,
        "shared_basis": shared_basis,
        "shared_coefficient_count": shared_count,
    }


def render_shared_baseline_snapshots(  # noqa: C901, PLR0915 - shared fixed-view figure protocol.
    output: Path,
    spatial_dir: Path,
    loaded: list[dict[str, Any]],
    volume: pv.UnstructuredGrid,
    skin: pv.PolyData,
    pivot: np.ndarray,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Render accepted shared fields on identical reference sections and scales."""
    assert all(item["snapshot_schema"] == SNAPSHOT_SCHEMA for item in loaded)
    meshes = []
    receipts = []
    limit_kpa = 0.0
    fraction_names = {
        "fat": "FatFraction",
        "aponeurosis": "AponeurosisFraction",
        "muscle": "MuscleFraction",
    }
    tissue_masks = {
        name: np.asarray(volume.cell_data[fraction]) > 1e-6
        for name, fraction in fraction_names.items()
    }
    tissue_limits_kpa = dict.fromkeys(fraction_names, 0.0)
    tissue_mask = (
        sum(
            np.asarray(volume.cell_data[name])
            for name in ("FatFraction", "AponeurosisFraction", "MuscleFraction")
        )
        > 0.0
    )
    for item in loaded:
        label = item["label"]
        bulk_path = spatial_dir / f"shared-baseline-{label}.vtu"
        skin_path = spatial_dir / f"shared-skin-{label}.vtp"
        bulk = pv.read(bulk_path)
        skin_field = pv.read(skin_path)
        np.testing.assert_array_equal(bulk.points, volume.points)
        np.testing.assert_array_equal(bulk.cells, volume.cells)
        np.testing.assert_array_equal(skin_field.points, skin.points)
        np.testing.assert_array_equal(
            skin_field.point_data["GlobalPointId"], skin.point_data["GlobalPointId"]
        )
        for name in ("baseline_min_principal_mpa", "baseline_max_principal_mpa"):
            values = np.asarray(bulk.cell_data[name])
            assert values.shape == (volume.n_cells,)
            assert np.isfinite(values).all()
            limit_kpa = max(
                limit_kpa, float(np.max(np.abs(values[tissue_mask]))) * 1000
            )
        for name, mask in tissue_masks.items():
            values = np.asarray(bulk.cell_data[f"{name}_baseline_max_principal_mpa"])
            assert values.shape == (volume.n_cells,)
            assert np.isfinite(values).all()
            assert np.all(values[~mask] == 0)
            tissue_limits_kpa[name] = max(
                tissue_limits_kpa[name], float(np.max(np.abs(values[mask]))) * 1000
            )
        materials = json.loads(str(item["data"]["material_config_json"].item()))
        layout = materials["parameterization"]
        reference_young = materials["materials"]["skin"]["reference_map"]["young_mpa"]
        expected_young = reference_young * np.exp(
            item["data"]["shared"][layout["skin_log_stiffness_multiplier_coordinate"]]
        )
        np.testing.assert_allclose(
            skin_field.point_data["effective_young_mpa"],
            expected_young,
            rtol=1e-12,
            atol=1e-12,
        )
        meshes.append((bulk, skin_field, materials))
        receipts.append(
            {
                "label": label,
                "snapshot_sha256": item["sha256"],
                "bulk": {"path": str(bulk_path.resolve()), "sha256": sha256(bulk_path)},
                "skin": {"path": str(skin_path.resolve()), "sha256": sha256(skin_path)},
            }
        )
    assert limit_kpa > 0
    for name, limit in tissue_limits_kpa.items():
        if limit == 0:
            # A truly zero field still needs a defined display scale.
            tissue_limits_kpa[name] = meshes[0][2]["materials"][name]["mu_mpa"] * 1000
    assets = []
    sections = (
        ("coronal", (0.0, 0.0, 1.0), (pivot[0], pivot[1] - 0.02, 0.065), "front"),
        ("sagittal", (1.0, 0.0, 0.0), tuple(pivot), "side"),
    )
    for item, (bulk, skin_field, materials) in zip(loaded, meshes, strict=True):
        label = item["label"]
        tissue = bulk.extract_cells(tissue_mask)
        for section_name, normal, origin, view in sections:
            section = tissue.slice(normal=normal, origin=origin)
            assert section.n_cells > 0
            for principal in ("min", "max"):
                section.cell_data["baseline_kpa"] = (
                    section.cell_data[f"baseline_{principal}_principal_mpa"] * 1000
                )
                filename = f"shared-{label}-{section_name}-{principal}-principal.png"
                p = plotter(
                    f"Shared baseline: {principal} principal — {section_name} — {label}"
                )
                p.add_mesh(
                    section,
                    scalars="baseline_kpa",
                    cmap="coolwarm",
                    clim=(-limit_kpa, limit_kpa),
                    scalar_bar_args=scalar_bar("kPa"),
                    show_edges=False,
                )
                save(p, output / filename, camera(section, view, zoom=1.02))
                assets.append(
                    {
                        "filename": filename,
                        "caption": f"{label}: {section_name} reference section of fraction-weighted {principal} principal baseline stress. Fixed signed scale across all checkpoints; positive means tension. This is an optimized model field.",
                    }
                )
        for tissue_name, mask in tissue_masks.items():
            component = bulk.extract_cells(mask)
            limit = tissue_limits_kpa[tissue_name]
            for section_name, normal, origin, view in sections:
                section = component.slice(normal=normal, origin=origin)
                assert section.n_cells > 0, (tissue_name, section_name)
                section.cell_data["component_kpa"] = (
                    section.cell_data[f"{tissue_name}_baseline_max_principal_mpa"]
                    * 1000
                )
                filename = f"shared-{label}-{tissue_name}-{section_name}.png"
                p = plotter(
                    f"{tissue_name.title()} baseline: max principal — {section_name} — {label}"
                )
                p.add_mesh(
                    section,
                    scalars="component_kpa",
                    cmap="coolwarm",
                    clim=(-limit, limit),
                    scalar_bar_args=scalar_bar("kPa"),
                    show_edges=False,
                )
                # Use the full tissue section camera to retain anatomical context.
                reference_section = tissue.slice(normal=normal, origin=origin)
                save(p, output / filename, camera(reference_section, view, zoom=1.02))
                assets.append(
                    {
                        "filename": filename,
                        "caption": f"{label}: {tissue_name} largest-principal constitutive baseline stress before tissue-fraction weighting, shown on its positive-fraction {section_name} section. This tissue's signed scale is fixed across checkpoints.",
                    }
                )
        filename = f"shared-{label}-skin-stiffness.png"
        reference_young = materials["materials"]["skin"]["reference_map"]["young_mpa"]
        bounds = (
            np.asarray(materials["constraints"]["skin_multiplier_bounds"])
            * reference_young
        )
        p = plotter(f"Shared skin stiffness on reference surface — {label}")
        p.add_mesh(
            skin_field,
            scalars="effective_young_mpa",
            cmap="viridis",
            clim=tuple(bounds),
            scalar_bar_args=scalar_bar("E (MPa)"),
            smooth_shading=True,
        )
        save(p, output / filename, camera(skin, "front"))
        assets.append(
            {
                "filename": filename,
                "caption": f"{label}: effective skin Young modulus on the reference surface, using the fixed model-bound color scale. The current skin basis remains spatially uniform.",
            }
        )
    return assets, {
        "scope": "shared fields on fixed reference geometry",
        "stress_scale_kpa": [-limit_kpa, limit_kpa],
        "per_tissue_max_principal_scale_kpa": {
            name: [-limit, limit] for name, limit in tissue_limits_kpa.items()
        },
        "exports": receipts,
    }


def render_overlay(
    output: Path,
    skin: pv.PolyData,
    predicted: np.ndarray,
    target: np.ndarray,
    label: str,
    expression: str,
    stage_update: str,
    cameras: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    reference = skin.copy(deep=True)
    predicted_skin = skin.copy(deep=True)
    target_skin = skin.copy(deep=True)
    predicted_skin.points = np.asarray(skin.points) + predicted
    target_skin.points = np.asarray(skin.points) + target
    assets: list[dict[str, str]] = []
    for view in ("front", "side"):
        name = f"{label}-{slug(expression)}-01-target-prediction-{view}.png"
        p = plotter(f"Target / prediction — {stage_update} — {expression} — {view}")
        p.add_mesh(reference, color="#7a7f85", opacity=0.18, smooth_shading=True)
        p.add_mesh(
            target_skin,
            color="#158cba",
            opacity=0.85,
            style="wireframe",
            line_width=0.7,
        )
        p.add_mesh(
            predicted_skin,
            color="#d95f02",
            opacity=0.85,
            style="wireframe",
            line_width=0.7,
        )
        p.add_text(
            "gray: frozen reference   cyan: transferred target   orange: FEM prediction",
            position="lower_left",
            font_size=10,
            color="#202124",
        )
        save(p, output / name, cameras[view])
        assets.append(
            {
                "filename": name,
                "kind": "target_prediction_overlay",
                "checkpoint": label,
                "expression": expression,
                "caption": f"{view.title()} overlay at {stage_update} for {expression}: frozen skin reference in gray, transferred target in cyan, and solved FEM prediction in orange, using the fixed {view} camera.",
            }
        )
    return assets


def render_residual(
    output: Path,
    skin: pv.PolyData,
    predicted: np.ndarray,
    residual_vtp: Path,
    expected_residual_mm: np.ndarray,
    cap_mm: float,
    label: str,
    expression: str,
    stage_update: str,
    front_camera: dict[str, Any],
) -> tuple[dict[str, str], dict[str, float]]:
    rendered = pv.read(residual_vtp).triangulate()
    assert rendered.n_points == skin.n_points
    residual = np.asarray(rendered.point_data["surface_residual_mm"], dtype=np.float64)
    finite = np.isfinite(residual)
    expected = expected_residual_mm
    assert np.allclose(residual[finite], expected[finite], rtol=1.0e-5, atol=1.0e-7)
    rendered.points = np.asarray(skin.points) + predicted
    name = f"{label}-{slug(expression)}-02-skin-residual-front.png"
    p = plotter(f"Skin residual — {stage_update} — {expression} — front")
    p.add_mesh(
        rendered,
        scalars="surface_residual_mm",
        cmap="turbo",
        clim=(0.0, cap_mm),
        smooth_shading=True,
        scalar_bar_args=scalar_bar("|prediction-target| (mm)"),
    )
    clipped_fraction = float(np.mean(residual[finite] > cap_mm))
    p.add_text(
        f"fixed 0--{cap_mm:.4g} mm scale; {100 * clipped_fraction:.3g}% above cap",
        position="lower_left",
        font_size=10,
        color="#202124",
    )
    save(p, output / name, front_camera)
    return (
        {
            "filename": name,
            "kind": "skin_residual",
            "checkpoint": label,
            "expression": expression,
            "caption": f"Predicted skin colored by Euclidean target residual for {expression} at {stage_update}, on the fixed 0--{cap_mm:.6g} mm scale; {100 * clipped_fraction:.4g}% of valid observations exceed the display cap.",
        },
        {
            "minimum_mm": float(residual[finite].min()),
            "median_mm": float(np.median(residual[finite])),
            "q95_mm": float(np.quantile(residual[finite], 0.95)),
            "q99_mm": float(np.quantile(residual[finite], 0.99)),
            "maximum_mm": float(residual[finite].max()),
            "fraction_above_display_cap": clipped_fraction,
        },
    )


def render_activation_sections(
    output: Path,
    activation_vtu: Path,
    pivot: np.ndarray,
    cap_mpa: float,
    axis_limit: int,
    label: str,
    expression: str,
    stage_update: str,
    section_cameras: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    field = pv.read(activation_vtu)
    magnitude = np.asarray(field.cell_data["activation_max_principal_mpa"])
    direction = np.asarray(field.cell_data["activation_max_principal_direction_xyz"])
    active = np.isfinite(magnitude)
    assert active.any()
    norms = np.linalg.norm(direction[active], axis=1)
    assert np.allclose(norms, 1.0, rtol=1.0e-5, atol=1.0e-6)
    active_field = field.extract_cells(np.flatnonzero(active))
    specifications = (
        (
            "coronal",
            (0.0, 0.0, 1.0),
            (pivot[0], pivot[1] - 0.02, 0.065),
        ),
        ("sagittal", (1.0, 0.0, 0.0), tuple(pivot)),
    )
    assets: list[dict[str, str]] = []
    for section_name, normal, origin in specifications:
        section = active_field.slice(normal=normal, origin=origin)
        assert section.n_cells > 0, f"empty {section_name} activation section"
        name = f"{label}-{slug(expression)}-03-activation-{section_name}.png"
        p = plotter(f"Activation — {stage_update} — {expression} — {section_name}")
        p.add_mesh(
            section,
            scalars="activation_max_principal_mpa",
            cmap="magma",
            clim=(0.0, cap_mpa),
            show_edges=True,
            edge_color="#71767b",
            line_width=0.25,
            scalar_bar_args=scalar_bar("max principal activation (MPa)"),
        )
        values = np.asarray(section.cell_data["activation_max_principal_mpa"])
        vectors = np.asarray(
            section.cell_data["activation_max_principal_direction_xyz"]
        )
        candidates = np.flatnonzero(values > max(cap_mpa * 1.0e-6, 1.0e-12))
        if len(candidates):
            stride = max(math.ceil(len(candidates) / axis_limit), 1)
            selected = candidates[::stride][:axis_limit]
            centers = section.cell_centers().points[selected]
            cloud = pv.PolyData(centers)
            cloud.point_data["PrincipalAxis"] = vectors[selected]
            axes = cloud.glyph(
                orient="PrincipalAxis",
                scale=False,
                geom=pv.Line((-0.00075, 0.0, 0.0), (0.00075, 0.0, 0.0)),
            )
            p.add_mesh(axes, color="#141414", line_width=1.2)
        p.add_text(
            f"fixed 0--{cap_mpa:.5g} MPa cap; black 1.5 mm lines are unsigned principal axes",
            position="lower_left",
            font_size=10,
            color="#202124",
        )
        save(p, output / name, section_cameras[section_name])
        assets.append(
            {
                "filename": name,
                "kind": "activation_section",
                "checkpoint": label,
                "expression": expression,
                "caption": f"{section_name.title()} active-muscle section for {expression} at {stage_update}. Color is largest-principal activation stress on the fixed 0--{cap_mpa:.6g} MPa model-cap scale; black 1.5 mm line glyphs show the canonicalized but physically unsigned principal axis.",
            }
        )
    return assets


def evaluate_contact(
    contact: Any,
    volume: pv.UnstructuredGrid,
    displacement_m: np.ndarray,
) -> dict[str, Any]:
    displacement = torch.as_tensor(displacement_m, dtype=torch.float64)
    state = contact.state_at(displacement)
    diagnostics = contact.diagnostics(state, displacement)
    assert diagnostics["contact_numerically_valid"] is True
    records = contact_records(contact, state, displacement, volume)
    assert len(records) == diagnostics["active_contact_count"]
    gradient = torch.zeros_like(displacement)
    contact.grad(state, displacement, gradient)
    force_n = -gradient.numpy(force=True) * 1.0e6
    names = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    labels = np.asarray(volume.point_data["GroupId"], dtype=np.int32)
    magnitude = np.linalg.norm(force_n, axis=1)
    cutoff = max(float(magnitude.max()) * 1.0e-12, 1.0e-18)
    bone_nodes: dict[str, np.ndarray] = {}
    force_summary: dict[str, Any] = {}
    for bone in ("cranium", "mandible"):
        ids = np.flatnonzero(
            (labels == names.index(bone.title())) & (magnitude > cutoff)
        )
        bone_nodes[bone] = ids
        resultant = force_n[ids].sum(axis=0)
        values = magnitude[ids]
        force_summary[bone] = {
            "active_nodes": len(ids),
            "resultant_vector_N": resultant.tolist(),
            "resultant_magnitude_N": float(np.linalg.norm(resultant)),
            "sum_nodal_magnitudes_N": float(values.sum()) if len(values) else 0.0,
            "maximum_nodal_magnitude_N": float(values.max()) if len(values) else 0.0,
        }
    return {
        "diagnostics": diagnostics,
        "records": records,
        "force_n": force_n,
        "bone_nodes": bone_nodes,
        "force_summary": force_summary,
        "points": np.asarray(volume.points) + displacement_m,
    }


def render_contact(
    output: Path,
    volume: pv.UnstructuredGrid,
    evaluated: dict[str, Any],
    bounds: tuple[float, ...],
    force_clim_n: tuple[float, float] | None,
    label: str,
    expression: str,
    stage_update: str,
    settings: dict[str, Any],
) -> dict[str, str]:
    surfaces = boundary_partition(volume, evaluated["points"])
    name = f"{label}-{slug(expression)}-04-exact-ipc-contact-side.png"
    p = plotter(f"Exact IPC contact — {stage_update} — {expression} — side")
    for key, color in (("cranium", "#4c83b6"), ("mandible", "#df8c2f")):
        p.add_mesh(
            clip(surfaces[key], bounds),
            color=color,
            opacity=0.42,
            show_edges=True,
            edge_color="#5e6670",
            line_width=0.35,
        )
    p.add_mesh(clip(surfaces["soft"], bounds), color="#ead0c7", opacity=0.14)
    p.add_mesh(
        clip(surfaces["bonded"], bounds),
        color="#d52bd0",
        opacity=0.28,
        style="wireframe",
        line_width=0.8,
    )
    records = evaluated["records"]
    if records:
        locations = np.asarray([record["location_m"] for record in records])
        targets = np.asarray(
            [0 if record["bone"] == "cranium" else 1 for record in records]
        )
        for value, color in ((0, "#1155cc"), (1, "#e65100")):
            selected = locations[targets == value]
            if len(selected):
                spheres = pv.PolyData(selected).glyph(
                    geom=pv.Sphere(
                        radius=0.00040, theta_resolution=16, phi_resolution=16
                    ),
                    scale=False,
                    orient=False,
                )
                p.add_mesh(spheres, color=color)
    all_ids = np.concatenate(
        [evaluated["bone_nodes"][name] for name in ("cranium", "mandible")]
    )
    if len(all_ids):
        force_n = evaluated["force_n"][all_ids]
        force_magnitude = np.linalg.norm(force_n, axis=1)
        cloud = pv.PolyData(evaluated["points"][all_ids])
        cloud.point_data["ForceVectorN"] = force_n
        cloud.point_data["ForceMagnitudeN"] = force_magnitude
        spheres = cloud.glyph(
            geom=pv.Sphere(radius=0.00050, theta_resolution=16, phi_resolution=16),
            scale=False,
            orient=False,
        )
        p.add_mesh(
            spheres,
            scalars="ForceMagnitudeN",
            cmap="viridis",
            clim=force_clim_n,
            log_scale=True,
            scalar_bar_args=scalar_bar("bone nodal |f| (N, log)"),
        )
        arrows = cloud.glyph(orient="ForceVectorN", scale="ForceMagnitudeN", factor=4.0)
        p.add_mesh(arrows, color="#202124", opacity=0.75)
    p.add_text(
        "blue/orange: active cranium/mandible stencils; magenta: bonded; force is -IPC gradient",
        position="lower_left",
        font_size=9,
        color="#202124",
    )
    save(p, output / name, settings)
    return {
        "filename": name,
        "kind": "exact_ipc_contact",
        "checkpoint": label,
        "expression": expression,
        "caption": f"Exact owned-IPC state rebuilt from the full solved displacement for {expression} at {stage_update}. Blue/orange markers are active cranium/mandible collision representatives; spheres/arrows show bone nodal negative-energy-gradient force in N on the run-wide scale; magenta is bonded mixed transition topology. Counts are collision-set representatives, not anatomical area.",
    }


def write_report(output: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Accepted optimization spatial states",
        "",
        f"Run stage: `{summary['run']['stage']}`. Selected accepted checkpoints: "
        + ", ".join(f"`{item['label']}`" for item in summary["snapshots"])
        + ".",
        "",
        "All cameras and activation/residual/contact-force scales are fixed across the "
        "selected checkpoints. These figures report accepted numerical states. They do "
        "not establish anatomical validation or, by themselves, convergence.",
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
    (output / "report.md").write_text("\n".join(lines))


def main(cfg: Config) -> None:  # noqa: PLR0915
    pv.OFF_SCREEN = True
    labels = [
        value.strip() for value in cfg.checkpoint_labels.split(",") if value.strip()
    ]
    assert labels
    assert len(labels) == len(set(labels))
    snapshots = [cfg.run_dir / f"visualization-{label}.npz" for label in labels]
    for path in snapshots:
        assert path.is_file(), f"missing real accepted snapshot: {path}"
    spatial_summary, spatial_summary_hash = load_stable_json(
        cfg.spatial_dir / "summary.json"
    )
    assert spatial_summary["success"] is True
    assert spatial_summary["schema"] == "joint-final-visualization-summary-v1"
    assert Path(spatial_summary["run_dir"]).resolve() == cfg.run_dir.resolve()

    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    volume = pv.read(prepared.volume_path)
    skin = pv.read(prepared.skin_path).triangulate()
    loaded: list[dict[str, Any]] = []
    for path in snapshots:
        data, digest = load_stable_snapshot(path)
        metadata = validate_snapshot(path, data, prepared, volume, skin)
        loaded.append({"path": path, "sha256": digest, "data": data, **metadata})
    stages = {item["stage"] for item in loaded}
    assert len(stages) == 1
    reference_values = {item["activation_reference_mpa"] for item in loaded}
    cap_values = {item["activation_cap_mpa"] for item in loaded}
    assert len(reference_values) == len(cap_values) == 1
    cap_mpa = cap_values.pop()

    cohort = prepared.manifest["cohort"]
    cohort_names = list(cohort["names"])
    expression_names_by_snapshot: list[list[str]] = []
    all_residual_mm: list[np.ndarray] = []
    for item in loaded:
        data = item["data"]
        expression_names = [
            cohort_names[int(index)] for index in data["target_indices"]
        ]
        expression_names_by_snapshot.append(expression_names)
        all_residual_mm.extend(
            1000.0
            * np.linalg.norm(
                data["predicted_observation_displacement_m"]
                - data["target_displacement_m"],
                axis=-1,
            )
        )
    assert all(
        names == expression_names_by_snapshot[0]
        for names in expression_names_by_snapshot
    )
    expression_names = expression_names_by_snapshot[0]
    assert cfg.contact_expression in expression_names
    residual_values = np.concatenate(all_residual_mm)
    computed_cap_mm = float(np.quantile(residual_values, 0.99))
    residual_cap_mm = (
        float(cfg.residual_cap_mm)
        if cfg.residual_cap_mm is not None
        else max(computed_cap_mm, 1.0e-12)
    )
    assert residual_cap_mm > 0

    reference_skin_cameras = {
        view: camera(skin, view, zoom=1.04) for view in ("front", "side")
    }
    pivot = prepared.arrays["mandible_pivot_m"]
    active_reference = volume.extract_cells(prepared.arrays["active_cell_ids"])
    reference_sections = {
        "coronal": active_reference.slice(
            normal=(0.0, 0.0, 1.0), origin=(pivot[0], pivot[1] - 0.02, 0.065)
        ),
        "sagittal": active_reference.slice(normal=(1.0, 0.0, 0.0), origin=pivot),
    }
    section_cameras = {
        "coronal": camera(reference_sections["coronal"], "front", zoom=1.04),
        "sagittal": camera(reference_sections["sagittal"], "side", zoom=1.04),
    }
    mouth_bounds = oral_bounds(volume)
    contact_camera = camera(pv.Box(mouth_bounds), "side", zoom=1.0)

    contact_config = json.loads(cfg.contact_spec.read_text())
    fixed = np.unique(
        np.concatenate(
            [
                prepared.arrays[name]
                for name in (
                    "cranium_node_ids",
                    "mandible_node_ids",
                    "historical_fixed_node_ids",
                )
            ]
        )
    )
    contact, contact_surface_map = build_owned_contact(volume, fixed, contact_config)
    contact_index = expression_names.index(cfg.contact_expression)
    evaluated_contacts = []
    force_magnitudes = []
    for item in loaded:
        evaluated = evaluate_contact(
            contact,
            volume,
            item["data"]["full_displacement_m"][contact_index],
        )
        evaluated_contacts.append(evaluated)
        magnitude = np.linalg.norm(evaluated["force_n"], axis=1)
        force_magnitudes.append(magnitude[magnitude > 0.0])
    nonempty_force = [value for value in force_magnitudes if len(value)]
    force_values = np.concatenate(nonempty_force) if nonempty_force else np.asarray([])
    force_clim_n = (
        (float(force_values.min()), float(force_values.max()))
        if len(force_values)
        else None
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    assets: list[dict[str, str]] = []
    schemas = {item["snapshot_schema"] for item in loaded}
    assert len(schemas) == 1, "Do not mix legacy and version 2 snapshots"
    if schemas == {SNAPSHOT_SCHEMA}:
        shared_assets, shared_visuals = render_shared_baseline_snapshots(
            cfg.output_dir,
            cfg.spatial_dir,
            loaded,
            volume,
            skin,
            prepared.arrays["mandible_pivot_m"],
        )
        assets.extend(shared_assets)
    else:
        shared_visuals = {
            "scope": "Legacy version 1 snapshots did not export shared material maps"
        }
    residual_summary: dict[str, Any] = {}
    contact_summary: dict[str, Any] = {}
    for snapshot_index, item in enumerate(loaded):
        data = item["data"]
        label = item["label"]
        stage_update = f"{item['stage']} update {item['update']} / {label}"
        residual_summary[label] = {}
        for expression_index, expression in enumerate(expression_names):
            observation_ids = data["observation_node_ids"]
            predicted = observation_on_skin(
                data["predicted_observation_displacement_m"][expression_index],
                observation_ids,
                skin,
            )
            target = observation_on_skin(
                data["target_displacement_m"][expression_index],
                observation_ids,
                skin,
            )
            assets.extend(
                render_overlay(
                    cfg.output_dir,
                    skin,
                    predicted,
                    target,
                    label,
                    expression,
                    stage_update,
                    reference_skin_cameras,
                )
            )
            residual_vtp = (
                cfg.spatial_dir
                / f"surface-residual-{label}-expression-{expression_index}.vtp"
            )
            assert residual_vtp.is_file(), residual_vtp
            expected_residual = np.linalg.norm(predicted - target, axis=1) * 1000.0
            residual_asset, residual_metrics = render_residual(
                cfg.output_dir,
                skin,
                predicted,
                residual_vtp,
                expected_residual,
                residual_cap_mm,
                label,
                expression,
                stage_update,
                reference_skin_cameras["front"],
            )
            assets.append(residual_asset)
            residual_summary[label][expression] = residual_metrics
            activation_vtu = (
                cfg.spatial_dir
                / f"activation-{label}-expression-{expression_index}.vtu"
            )
            assert activation_vtu.is_file(), activation_vtu
            assets.extend(
                render_activation_sections(
                    cfg.output_dir,
                    activation_vtu,
                    pivot,
                    cap_mpa,
                    cfg.maximum_principal_axes,
                    label,
                    expression,
                    stage_update,
                    section_cameras,
                )
            )
        evaluated = evaluated_contacts[snapshot_index]
        assets.append(
            render_contact(
                cfg.output_dir,
                volume,
                evaluated,
                mouth_bounds,
                force_clim_n,
                label,
                cfg.contact_expression,
                stage_update,
                contact_camera,
            )
        )
        contact_summary[label] = {
            "expression": cfg.contact_expression,
            "jaw_pose_rad_m": data["jaw_pose_rad_m"][contact_index].tolist(),
            "diagnostics": evaluated["diagnostics"],
            "bone_force_N": evaluated["force_summary"],
        }

    summary = {
        "schema": "joint-optimization-spatial-visuals-v1",
        "run": {
            "directory": str(cfg.run_dir.resolve()),
            "stage": stages.pop(),
            "claim": "accepted numerical checkpoint visualization; convergence and promotion require the run summary gates",
        },
        "prepared_inputs_sha256": sha256(cfg.prepared_dir / "inputs.npz"),
        "prepared_manifest_sha256": sha256(cfg.prepared_dir / "manifest.json"),
        "spatial_summary": {
            "path": str((cfg.spatial_dir / "summary.json").resolve()),
            "sha256": spatial_summary_hash,
        },
        "snapshots": [
            {
                key: item[key]
                for key in (
                    "label",
                    "stage",
                    "update",
                    "sha256",
                    "activation_reference_mpa",
                    "activation_cap_mpa",
                    "snapshot_schema",
                    "shared_basis",
                    "shared_coefficient_count",
                )
            }
            | {"path": str(item["path"].resolve())}
            for item in loaded
        ],
        "expressions": expression_names,
        "fixed_visual_contract": {
            "cameras": "reference front/side, reference active-muscle sections, and frozen oral box; identical across checkpoints",
            "activation_scale_mpa": [0.0, cap_mpa],
            "residual_scale_mm": [0.0, residual_cap_mm],
            "residual_cap_selection": (
                "explicit configuration"
                if cfg.residual_cap_mm is not None
                else "99th percentile over every selected checkpoint and expression"
            ),
            "residual_uncapped_maximum_mm": float(residual_values.max()),
            "contact_force_scale_N": list(force_clim_n) if force_clim_n else None,
            "principal_axis_glyph": "1.5 mm centered line; eigenvector sign is physically unsigned",
        },
        "residuals": residual_summary,
        "shared_material_visuals": shared_visuals,
        "contact": {
            "expression": cfg.contact_expression,
            "contact_spec_sha256": sha256(cfg.contact_spec),
            "surface_map": contact_surface_map,
            "force_definition": "negative IPC energy gradient; MPa*m^2 multiplied by 1e6 gives N; nodal force is not pressure",
            "checkpoints": contact_summary,
        },
        "assets": assets,
        "anatomical_validation": False,
        "promotion_ready": False,
    }
    write_json(cfg.output_dir / "summary.json", summary)
    write_report(cfg.output_dir, summary)
    cherries.log_metrics(
        {
            "visuals/snapshots": len(loaded),
            "visuals/expressions": len(expression_names),
            "visuals/assets": len(assets),
            "visuals/residual_cap_mm": residual_cap_mm,
        }
    )
    LOG.info("Wrote %d optimization-state assets to %s", len(assets), cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
