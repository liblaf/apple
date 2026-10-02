# Copyright 2026 liblaf
"""Controlled forehead forward solve for baseline and transferred fiber fields."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from anatomy_common import (
    BASELINE,
    BASELINE_SOURCE,
    ROOT,
    ProfileCometNoCommit,
    camera,
    sha256,
    write_json,
)
from scipy.spatial import cKDTree

from liblaf import cherries

FOREHEAD_MUSCLE_ID = 28
EXPECTED_FOREHEAD_CELLS = 35_171
SOURCE_NAMES = ("face_physics.py", "activation_models.py")
ALLOWED_CANDIDATE_CELL_DATA = {"PublicReferenceDistance", "PublicReferenceId"}
ALLOWED_CANDIDATE_FIELD_DATA = {"PublicReferenceMethod", "PublicReferenceName"}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    baseline: Path = BASELINE
    candidate: Path = (
        ROOT / "exp/2026/09/12/public-head-anatomy/data/20-reference-transfer/fixture"
    )
    output: Path = cherries.output("30-forward", mkdir=True)
    contraction: float = 0.1
    gamma: float = 0.5
    roi_distance_m: float = 0.006
    rtol: float = 1e-5
    atol: float = 1e-12
    analysis_only: bool = False


def freeze_and_import(output: Path) -> tuple[ModuleType, ModuleType, dict[str, object]]:
    source = output / "source"
    source.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, object] = {}
    modules = []
    for name in SOURCE_NAMES:
        live = BASELINE_SOURCE / name
        frozen = source / name
        shutil.copy2(live, frozen)
        assert frozen.read_bytes() == live.read_bytes()
        digest = sha256(frozen)
        hashes[name] = {
            "live_path": str(live),
            "frozen_path": str(frozen),
            "live_sha256": sha256(live),
            "frozen_sha256": digest,
        }
        module_name = f"forward_snapshot_{name.removesuffix('.py')}"
        spec = importlib.util.spec_from_file_location(module_name, frozen)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        assert Path(module.__file__).resolve() == frozen.resolve()
        modules.append(module)
    return modules[0], modules[1], hashes


def fixture_hashes(fixture: Path) -> dict[str, str]:
    return {name: sha256(fixture / name) for name in ("volume.vtu", "skin.vtp")}


def arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if np.issubdtype(np.asarray(left).dtype, np.inexact):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def raw_bitarray_metadata(path: Path) -> dict[str, list[str]]:
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    field_data = reader.GetOutput().GetFieldData()
    result = {}
    for name in ("_PYVISTA_BITARRAY_POINT_", "_PYVISTA_BITARRAY_CELL_"):
        array = field_data.GetAbstractArray(name)
        assert array is not None
        result[name] = [
            array.GetValue(index) for index in range(array.GetNumberOfValues())
        ]
    return result


def delivery_equivalence_receipt(
    consumed_path: Path, delivery_path: Path
) -> dict[str, object]:
    consumed, delivery = pv.read(consumed_path), pv.read(delivery_path)
    checks = {
        "points": arrays_equal(consumed.points, delivery.points),
        "cells": arrays_equal(consumed.cells, delivery.cells),
        "celltypes": arrays_equal(consumed.celltypes, delivery.celltypes),
        "point_array_names": set(consumed.point_data) == set(delivery.point_data),
        "cell_array_names": set(consumed.cell_data) == set(delivery.cell_data),
        "field_array_names": set(consumed.field_data) == set(delivery.field_data),
    }
    array_counts = {}
    for association in ("point_data", "cell_data", "field_data"):
        left, right = getattr(consumed, association), getattr(delivery, association)
        names = sorted(left)
        assert set(names) == set(right)
        for name in names:
            assert left[name].dtype == right[name].dtype, (
                f"changed dtype: {association}/{name}"
            )
            assert left[name].shape == right[name].shape, (
                f"changed shape: {association}/{name}"
            )
            assert arrays_equal(left[name], right[name]), (
                f"changed values: {association}/{name}"
            )
        array_counts[association] = len(names)
    assert all(checks.values())

    metadata = {
        "consumed": raw_bitarray_metadata(consumed_path),
        "current_delivery": raw_bitarray_metadata(delivery_path),
    }
    point_key = "_PYVISTA_BITARRAY_POINT_"
    cell_key = "_PYVISTA_BITARRAY_CELL_"
    assert set(metadata["consumed"][point_key]) == set(
        metadata["current_delivery"][point_key]
    )
    assert metadata["consumed"][point_key] != metadata["current_delivery"][point_key]
    assert metadata["consumed"][cell_key] == metadata["current_delivery"][cell_key]

    pattern = re.compile(
        rb'(<Array type="String" Name="_PYVISTA_BITARRAY_POINT_"[^>]*>).*?(</Array>)',
        re.DOTALL,
    )
    consumed_bytes, delivery_bytes = (
        consumed_path.read_bytes(),
        delivery_path.read_bytes(),
    )
    assert len(pattern.findall(consumed_bytes)) == 1
    assert len(pattern.findall(delivery_bytes)) == 1
    normalized_consumed = pattern.sub(rb"\1NORMALIZED\2", consumed_bytes)
    normalized_delivery = pattern.sub(rb"\1NORMALIZED\2", delivery_bytes)
    assert normalized_consumed == normalized_delivery
    return {
        "status": "verified_semantically_identical",
        "consumed_snapshot": {
            "path": str(consumed_path),
            "sha256": sha256(consumed_path),
            "bytes": consumed_path.stat().st_size,
        },
        "current_delivery": {
            "path": str(delivery_path),
            "sha256": sha256(delivery_path),
            "bytes": delivery_path.stat().st_size,
        },
        "dataset_checks": checks,
        "array_counts": array_counts,
        "all_array_dtypes_shapes_and_values_equal": True,
        "serialized_byte_difference": (
            "only the payload of _PYVISTA_BITARRAY_POINT_ differs; its set of "
            "boolean-array names is identical and only iteration order changed"
        ),
        "raw_pyvista_bitarray_metadata": metadata,
        "physics_reused": True,
        "physics_rerun": False,
    }


def assert_controlled_fixtures(
    baseline: Path, candidate: Path
) -> tuple[pv.UnstructuredGrid, pv.UnstructuredGrid]:
    for fixture in (baseline, candidate):
        for name in ("volume.vtu", "skin.vtp"):
            if not (fixture / name).is_file():
                raise FileNotFoundError(fixture / name)
    a, b = pv.read(baseline / "volume.vtu"), pv.read(candidate / "volume.vtu")
    assert a.n_points == b.n_points
    assert a.n_cells == b.n_cells
    assert np.array_equal(a.cells, b.cells)
    assert np.array_equal(a.celltypes, b.celltypes)
    assert arrays_equal(a.points, b.points)
    assert set(a.point_data) == set(b.point_data)
    assert set(b.cell_data) - set(a.cell_data) == ALLOWED_CANDIDATE_CELL_DATA
    assert set(a.cell_data) - set(b.cell_data) == set()
    assert set(b.field_data) - set(a.field_data) == ALLOWED_CANDIDATE_FIELD_DATA
    assert set(a.field_data) - set(b.field_data) == set()
    for association in ("point_data", "field_data"):
        baseline_data, candidate_data = getattr(a, association), getattr(b, association)
        for name in baseline_data:
            assert arrays_equal(baseline_data[name], candidate_data[name]), (
                f"changed {association}/{name}"
            )
    for name in a.cell_data:
        if name != "ActivationFiber":
            assert arrays_equal(a.cell_data[name], b.cell_data[name]), (
                f"changed cell_data/{name}"
            )
    assert sha256(baseline / "skin.vtp") == sha256(candidate / "skin.vtp"), (
        "candidate skin differs from baseline"
    )
    active = np.asarray(a.cell_data["ActivationMask"], dtype=bool)
    forehead = active & (
        np.asarray(a.cell_data["MuscleId"], dtype=int) == FOREHEAD_MUSCLE_ID
    )
    assert int(forehead.sum()) == EXPECTED_FOREHEAD_CELLS
    changed = np.any(
        np.asarray(a.cell_data["ActivationFiber"])
        != np.asarray(b.cell_data["ActivationFiber"]),
        axis=1,
    )
    assert arrays_equal(changed, forehead), (
        "fiber changes are not exactly the active forehead cells"
    )
    for fibers in (a.cell_data["ActivationFiber"], b.cell_data["ActivationFiber"]):
        norms = np.linalg.norm(np.asarray(fibers)[forehead], axis=1)
        assert np.allclose(norms, 1.0, rtol=1e-12, atol=1e-12)
    return a, b


def activation(
    physics: Any, activation_models: ModuleType, contraction: float, gamma: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = np.asarray(physics.mesh.cell_data["MuscleId"], dtype=int)[physics.ids]
    forehead = labels == FOREHEAD_MUSCLE_ID
    assert int(forehead.sum()) == EXPECTED_FOREHEAD_CELLS
    q = torch.zeros(activation_models.shape("F", len(physics.ids)))
    q[torch.as_tensor(forehead), 0] = contraction
    ainv, _ = activation_models.matrices(q, "F", physics.fibers, gamma=gamma)
    return q, ainv, activation_models.packed(ainv)


def raw6_fiber_independence(
    activation_models: ModuleType,
    fibers_a: torch.Tensor,
    fibers_b: torch.Tensor,
    forehead: np.ndarray,
    contraction: float,
) -> dict[str, object]:
    q = torch.zeros(activation_models.shape("Raw6", len(forehead)))
    q[torch.as_tensor(forehead), 0] = contraction
    a, _ = activation_models.matrices(q, "Raw6", fibers_a)
    b, _ = activation_models.matrices(q, "Raw6", fibers_b)
    difference = float(torch.max(torch.abs(a - b)).detach().cpu())
    assert difference == 0.0
    return {
        "mode": "Raw6",
        "construction": f"q_xx={contraction} on active MuscleId 28 cells and zero elsewhere",
        "max_matrix_difference": difference,
        "interpretation": "Raw6 does not consume ActivationFiber; no second physics solve is needed.",
    }


def skin_area_weights(skin: pv.PolyData) -> np.ndarray:
    triangles = np.asarray(skin.faces).reshape(-1, 4)
    assert np.all(triangles[:, 0] == 3)
    tri = triangles[:, 1:]
    xyz = np.asarray(skin.points)[tri]
    area = (
        np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        / 2
    )
    weights = np.zeros(skin.n_points)
    np.add.at(weights, tri.ravel(), np.repeat(area / 3, 3))
    assert np.all(weights >= 0)
    assert weights.sum() > 0
    return weights


def brow_roi(
    mesh: pv.UnstructuredGrid, skin: pv.PolyData, distance_m: float
) -> tuple[np.ndarray, dict[str, object]]:
    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    forehead = active & (
        np.asarray(mesh.cell_data["MuscleId"], dtype=int) == FOREHEAD_MUSCLE_ID
    )
    centers = np.asarray(mesh.cell_centers().points)[forehead]
    y_min, y_max = float(centers[:, 1].min()), float(centers[:, 1].max())
    y_cut = y_min + (y_max - y_min) / 3
    distance, _ = cKDTree(centers).query(np.asarray(skin.points), k=1)
    global_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=int)
    fixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool)[global_ids]
    roi = (distance <= distance_m) & (np.asarray(skin.points)[:, 1] <= y_cut) & ~fixed
    assert roi.any()
    return roi, {
        "definition": "skin vertices within threshold of an active MuscleId 28 tetrahedron center, in the lower third of the active forehead center y-range, excluding IsFixed",
        "distance_threshold_m": distance_m,
        "forehead_center_y_range_m": [y_min, y_max],
        "lower_third_y_max_m": y_cut,
        "point_count": int(roi.sum()),
        "nearest_center_distance_range_m": [
            float(distance[roi].min()),
            float(distance[roi].max()),
        ],
    }


def bilateral_lower_forehead_roi(
    mesh: pv.UnstructuredGrid, skin: pv.PolyData
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    forehead = active & (
        np.asarray(mesh.cell_data["MuscleId"], dtype=int) == FOREHEAD_MUSCLE_ID
    )
    centers = np.asarray(mesh.cell_centers().points)[forehead]
    points = np.asarray(skin.points)
    x_min, y_min, _ = centers.min(axis=0)
    x_max, y_max, _ = centers.max(axis=0)
    x_mid = (x_min + x_max) / 2
    y_cut = y_min + (y_max - y_min) / 3
    distance, nearest = cKDTree(centers).query(points, k=1)
    global_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=int)
    fixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool)[global_ids]
    anterior = points[:, 2] >= centers[nearest, 2]
    bilateral = (
        (distance <= 0.012)
        & (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_cut)
        & anterior
        & ~fixed
    )
    masks = {
        "left_lower_x": bilateral & (points[:, 0] < x_mid),
        "right_higher_x": bilateral & (points[:, 0] >= x_mid),
        "bilateral": bilateral,
    }
    assert masks["left_lower_x"].any()
    assert masks["right_higher_x"].any()
    assert np.array_equal(masks["left_lower_x"] | masks["right_higher_x"], bilateral)
    return masks, {
        "definition": "baseline skin band within 12 mm of active MuscleId 28 tetrahedron centers; x inside the active forehead center extent; y inside its lower third; skin z at or anterior to the nearest center; IsFixed excluded",
        "side_convention": "front-view left is x < active-forehead x midpoint; front-view right is x >= midpoint",
        "distance_threshold_m": 0.012,
        "x_range_m": [float(x_min), float(x_max)],
        "x_midpoint_m": float(x_mid),
        "y_range_m": [float(y_min), float(y_cut)],
        "anterior_filter": "skin_z >= nearest_active_forehead_cell_center_z",
        "counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "bounds_m": {
            name: {
                "min": points[mask].min(axis=0).tolist(),
                "max": points[mask].max(axis=0).tolist(),
            }
            for name, mask in masks.items()
        },
        "nearest_center_distance_range_m": [
            float(distance[bilateral].min()),
            float(distance[bilateral].max()),
        ],
    }


def displacement_stats(
    displacement: np.ndarray, roi: np.ndarray, weights: np.ndarray
) -> dict[str, object]:
    selected = displacement[roi]
    area_weights = weights[roi] / weights[roi].sum()

    def summarize(w: np.ndarray) -> dict[str, object]:
        mean = np.sum(w[:, None] * selected, axis=0)
        return {
            "mean_components_m": mean.tolist(),
            "mean_components_mm": (mean * 1000).tolist(),
            "mean_abs_lateral_m": float(np.sum(w * np.abs(selected[:, 0]))),
            "mean_abs_lateral_mm": float(np.sum(w * np.abs(selected[:, 0])) * 1000),
            "cranial_mean_m": float(mean[1]),
            "cranial_mean_mm": float(mean[1] * 1000),
        }

    return {
        "axes": {"x": "lateral", "y": "superior/cranial", "z": "anterior"},
        "point_weighted": summarize(np.full(len(selected), 1 / len(selected))),
        "area_weighted": summarize(area_weights),
    }


def save_plot(
    path: Path,
    skin: pv.PolyData,
    roi_masks: dict[str, np.ndarray],
    cases: dict[str, np.ndarray],
) -> None:
    global_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=int)
    magnitudes = [np.linalg.norm(u[global_ids], axis=1) * 1000 for u in cases.values()]
    clim = (0.0, max(float(np.max(value)) for value in magnitudes))
    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1800, 900))
    plotter.set_background("white")
    for column, ((name, u), magnitude) in enumerate(
        zip(cases.items(), magnitudes, strict=True)
    ):
        plotter.subplot(0, column)
        deformed = skin.copy()
        deformed.points = np.asarray(skin.points) + u[global_ids]
        deformed.point_data["Displacement magnitude (mm)"] = magnitude
        plotter.add_mesh(
            deformed,
            scalars="Displacement magnitude (mm)",
            clim=clim,
            cmap="viridis",
            smooth_shading=True,
        )
        plotter.add_points(
            deformed.points[roi_masks["left_lower_x"]],
            color="#e63946",
            point_size=4,
            render_points_as_spheres=True,
        )
        plotter.add_points(
            deformed.points[roi_masks["right_higher_x"]],
            color="#277da1",
            point_size=4,
            render_points_as_spheres=True,
        )
        plotter.add_text(
            f"{name.title()} fibers · q=0.1", color="#202020", font_size=15
        )
        plotter.add_text(
            "red: lower-x side · blue: higher-x side",
            position="upper_right",
            color="#303030",
            font_size=10,
        )
        camera(plotter, skin.center)
        plotter.camera.zoom(1.0)
    plotter.show(screenshot=path, auto_close=True)


def add_roi_analysis(
    cfg: Config,
    evidence: dict[str, object],
    mesh: pv.UnstructuredGrid,
    solutions: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
) -> None:
    skin = pv.read(cfg.baseline / "skin.vtp")
    old_roi, old_definition = brow_roi(mesh, skin, cfg.roi_distance_m)
    masks, bilateral_definition = bilateral_lower_forehead_roi(mesh, skin)
    area_weights = skin_area_weights(skin)
    global_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=int)
    shape_delta = solutions["candidate"] - solutions["baseline"]
    cases = {
        **solutions,
        "candidate_minus_baseline": shape_delta,
    }
    old_stats = {
        name: displacement_stats(u[global_ids], old_roi, area_weights)
        for name, u in cases.items()
    }
    bilateral_stats = {
        subset: {
            name: displacement_stats(u[global_ids], mask, area_weights)
            for name, u in cases.items()
        }
        for subset, mask in masks.items()
    }
    evidence.pop("brow_roi", None)
    evidence.pop("brow_roi_displacement", None)
    evidence["roi_diagnostics"] = {
        "original_proximity_6mm": {
            "status": "preserved diagnostic; too small and asymmetric for a brow-wide claim",
            "definition": old_definition,
            "displacement": old_stats,
        },
        "bilateral_lower_forehead": {
            "status": "primary bilateral geometry diagnostic",
            "definition": bilateral_definition,
            "displacement_by_subset": bilateral_stats,
        },
    }
    arrays.update(
        {
            "candidate_minus_baseline_m": shape_delta,
            "brow_skin_mask": old_roi,
            "original_proximity_6mm_mask": old_roi,
            "bilateral_lower_forehead_mask": masks["bilateral"],
            "bilateral_lower_forehead_left_mask": masks["left_lower_x"],
            "bilateral_lower_forehead_right_mask": masks["right_higher_x"],
            "brow_skin_area_weights_m2": area_weights,
        }
    )
    np.savez_compressed(cfg.output / "forward-arrays.npz", **arrays)
    save_plot(cfg.output / "paired-skin-displacement-front.png", skin, masks, solutions)
    write_json(cfg.output / "evidence.json", evidence)
    primary = bilateral_stats["bilateral"]
    cherries.log_metrics(
        {
            "forward/baseline_min_det_f": evidence["min_det_f"]["baseline"],
            "forward/candidate_min_det_f": evidence["min_det_f"]["candidate"],
            "forward/f_packed_max_abs_difference": evidence["configuration"][
                "f_packed_max_abs_difference"
            ],
            "forward/lower_forehead_points": bilateral_definition["counts"][
                "bilateral"
            ],
            "forward/lower_forehead_area_cranial_baseline_mm": primary["baseline"][
                "area_weighted"
            ]["cranial_mean_mm"],
            "forward/lower_forehead_area_cranial_candidate_mm": primary["candidate"][
                "area_weighted"
            ]["cranial_mean_mm"],
            "forward/lower_forehead_area_cranial_delta_mm": primary[
                "candidate_minus_baseline"
            ]["area_weighted"]["cranial_mean_mm"],
        }
    )


def reuse_saved_analysis(cfg: Config, baseline_mesh: pv.UnstructuredGrid) -> None:
    output = cfg.output
    evidence_path = output / "evidence.json"
    arrays_path = output / "forward-arrays.npz"
    consumed_fixture = output / "consumed-fixture"
    consumed_volume = consumed_fixture / "volume.vtu"
    consumed_skin = consumed_fixture / "skin.vtp"
    if not evidence_path.is_file() or not arrays_path.is_file():
        message = "analysis-only mode requires prior solve evidence"
        raise FileNotFoundError(message)
    if not consumed_volume.is_file() or not consumed_skin.is_file():
        message = "analysis-only mode requires the frozen consumed fixture"
        raise FileNotFoundError(message)
    evidence = json.loads(evidence_path.read_text())
    assert evidence["fixtures"]["baseline"]["hashes"] == fixture_hashes(cfg.baseline)
    assert evidence["fixtures"]["candidate"]["hashes"] == fixture_hashes(
        consumed_fixture
    )
    assert sha256(consumed_skin) == sha256(cfg.baseline / "skin.vtp")
    receipt = delivery_equivalence_receipt(
        consumed_volume, cfg.candidate / "volume.vtu"
    )
    receipt["consumed_snapshot_origin"] = (
        ".cherries/runs/2026/09/12/public-head-anatomy/20-transfer-reference/"
        "2026-09-12T150835-Public-atlas-forehead-transfer-smoke/data/"
        "20-reference-transfer/fixture/volume.vtu"
    )
    receipt["baseline_solver_fixture"] = {
        "path": str(cfg.baseline),
        "hashes": fixture_hashes(cfg.baseline),
        "frozen_skin_path": str(consumed_skin),
    }
    evidence["fixtures"]["candidate"]["role"] = (
        "original path and hashes consumed by the recorded physics solve"
    )
    evidence["fixtures"]["candidate"]["frozen_consumed_snapshot"] = {
        "path": str(consumed_fixture),
        "hashes": fixture_hashes(consumed_fixture),
    }
    evidence["fixtures"]["candidate"]["current_delivery"] = {
        "path": str(cfg.candidate),
        "hashes": fixture_hashes(cfg.candidate),
        "relationship_to_consumed_snapshot": "verified semantic equality; XML metadata order differs",
    }
    evidence["fixture_delivery_reconciliation"] = receipt
    write_json(output / "fixture-receipt.json", receipt)
    with np.load(arrays_path) as saved:
        arrays = {name: saved[name] for name in saved.files}
    solutions = {
        "baseline": arrays["baseline_displacement_m"],
        "candidate": arrays["candidate_displacement_m"],
    }
    add_roi_analysis(cfg, evidence, baseline_mesh, solutions, arrays)


def main(cfg: Config) -> None:
    if cfg.contraction <= 0:
        message = "contraction must be positive"
        raise ValueError(message)
    if cfg.rtol != 1e-5 or cfg.atol != 1e-12:
        message = "this controlled comparison requires rtol=1e-5 and atol=1e-12"
        raise ValueError(message)
    output = cfg.output
    output.mkdir(parents=True, exist_ok=True)
    baseline_mesh, _candidate_mesh = assert_controlled_fixtures(
        cfg.baseline, cfg.candidate
    )
    if cfg.analysis_only:
        reuse_saved_analysis(cfg, baseline_mesh)
        return
    face_physics, activation_models, source_hashes = freeze_and_import(output)
    activation_models.validate()
    face_physics.configure()

    physics = {
        "baseline": face_physics.FacePhysics(
            cfg.baseline, rtol=cfg.rtol, atol=cfg.atol
        ),
        "candidate": face_physics.FacePhysics(
            cfg.candidate, rtol=cfg.rtol, atol=cfg.atol
        ),
    }
    q: dict[str, torch.Tensor] = {}
    ainv: dict[str, torch.Tensor] = {}
    packed: dict[str, torch.Tensor] = {}
    for name, model in physics.items():
        q[name], ainv[name], packed[name] = activation(
            model, activation_models, cfg.contraction, cfg.gamma
        )
    assert torch.equal(q["baseline"], q["candidate"])
    f_packed_difference = float(
        torch.max(torch.abs(packed["candidate"] - packed["baseline"])).detach().cpu()
    )
    assert f_packed_difference > 0
    active_labels = np.asarray(baseline_mesh.cell_data["MuscleId"], dtype=int)[
        physics["baseline"].ids
    ]
    forehead = active_labels == FOREHEAD_MUSCLE_ID
    raw6_check = raw6_fiber_independence(
        activation_models,
        physics["baseline"].fibers,
        physics["candidate"].fibers,
        forehead,
        cfg.contraction,
    )

    solutions: dict[str, np.ndarray] = {}
    receipts: dict[str, object] = {}
    determinants: dict[str, np.ndarray] = {}
    zero_seed = np.zeros_like(physics["baseline"].points)
    for name, model in physics.items():
        u = model.solve(packed[name], zero_seed)
        solutions[name] = u.detach().cpu().numpy()
        receipts[name] = model.last_forward
        determinants[name] = model.detf(solutions[name])
        model.save_mesh(
            output / f"{name}-deformed.vtu",
            solutions[name],
            ainv[name].detach().cpu().numpy(),
        )

    shape_delta = solutions["candidate"] - solutions["baseline"]
    evidence = {
        "scope": "Controlled modeling diagnostic; not anatomical validation or expression-fit validation.",
        "axes": {"x": "lateral", "y": "superior/cranial", "z": "anterior"},
        "fixtures": {
            "baseline": {
                "path": str(cfg.baseline),
                "hashes": fixture_hashes(cfg.baseline),
            },
            "candidate": {
                "path": str(cfg.candidate),
                "hashes": fixture_hashes(cfg.candidate),
            },
            "controlled_solver_input_difference": "ActivationFiber on active MuscleId 28 cells only",
            "candidate_diagnostic_extras_ignored_by_face_physics": {
                "cell_data": sorted(ALLOWED_CANDIDATE_CELL_DATA),
                "field_data": sorted(ALLOWED_CANDIDATE_FIELD_DATA),
            },
        },
        "source_snapshots": source_hashes,
        "configuration": {
            "activation_mode": "F",
            "forehead_muscle_id": FOREHEAD_MUSCLE_ID,
            "forehead_active_cells": EXPECTED_FOREHEAD_CELLS,
            "q": cfg.contraction,
            "gamma": cfg.gamma,
            "initial_seed": "all-zero displacement",
            "rtol": cfg.rtol,
            "atol": cfg.atol,
            "f_packed_max_abs_difference": f_packed_difference,
            "passive_materials_fixed_mesh_and_constraints_identical": True,
        },
        "raw6_independent_check": raw6_check,
        "material_spec": physics["baseline"].material_spec,
        "material_spec_identical": physics["baseline"].material_spec
        == physics["candidate"].material_spec,
        "solve_receipts": receipts,
        "min_det_f": {name: float(value.min()) for name, value in determinants.items()},
        "shape_delta": {
            "candidate_minus_baseline_max_norm_m": float(
                np.linalg.norm(shape_delta, axis=1).max()
            ),
            "candidate_minus_baseline_rms_norm_m": float(
                np.sqrt(np.mean(np.sum(shape_delta**2, axis=1)))
            ),
        },
        "excluded_metric": "Smile target loss is intentionally absent because it is the wrong target for forehead contraction.",
    }
    assert evidence["material_spec_identical"]
    arrays = {
        "q": q["baseline"].detach().cpu().numpy(),
        "baseline_displacement_m": solutions["baseline"],
        "candidate_displacement_m": solutions["candidate"],
        "candidate_minus_baseline_m": shape_delta,
        "baseline_det_f": determinants["baseline"],
        "candidate_det_f": determinants["candidate"],
    }
    add_roi_analysis(cfg, evidence, baseline_mesh, solutions, arrays)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
