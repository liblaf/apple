# Copyright 2026 liblaf
"""Register a public atlas and derive an explicitly geometric forehead field.

The two frontalis axes use surface regression along atlas superior, not measured
fascicles. The superior coordinate is an explicit brow-to-galea modeling prior.  Only
the existing active forehead fibers change; no atlas labels become material or
attachment definitions. Registration excludes the forehead being evaluated.
"""

# The numbered experiment keeps acquisition, validation, and export in order.
# ruff: noqa: C901, PLR0912, PLR0915

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from anatomy_common import BASELINE, ProfileCometNoCommit, camera, sha256, write_json
from scipy.spatial import cKDTree

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = BASELINE
    atlas: Path = cherries.input("12-public-models/zanatomy/extracted")
    output: Path = cherries.output("20-reference-transfer", mkdir=True)


# Current _0 components occupy negative lateral coordinates relative to midline.
# Source .r/.l labels are retained as atlas names, not specimen-side validation.
PAIRS = {
    "Levator labii superioris": (57, 58),
    "Zygomaticus major muscle": (63, 64),
    "Risorius muscle": (73, 74),
    "Mentalis muscle": (93, 94),
    "Orbital part of orbicularis oculi": (97, 98),
    "Depressor anguli oris": (99, 100),
    "Corrugator supercilii": (110, 111),
    "Levator anguli oris": (142, 143),
    "Depressor labii inferioris": (162, 163),
    "Zygomaticus minor muscle": (218, 219),
    "Levator nasolabialis": (283, 284),
}


def surface_moments(mesh: pv.PolyData):
    triangles = np.asarray(mesh.points, dtype=np.float64)[
        np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    ]
    areas = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        / 2
    )
    # Exact first and second moments of uniformly weighted triangular surfaces.
    total = triangles.sum(axis=1)
    center = np.einsum("n,ni->i", areas, total / 3) / areas.sum()
    second = (
        np.einsum("ni,nj->nij", total, total)
        + np.einsum("nki,nkj->nij", triangles, triangles)
    ) / 12
    covariance = np.einsum("n,nij->ij", areas, second) / areas.sum() - np.outer(
        center, center
    )
    values, vectors = np.linalg.eigh(covariance)
    return center, vectors[:, -1], values, covariance


def similarity(source: np.ndarray, target: np.ndarray):
    """Least-squares proper similarity, without a reflection or a local warp."""
    x0, y0 = source.mean(axis=0), target.mean(axis=0)
    x, y = source - x0, target - y0
    u, singular, vt = np.linalg.svd(x.T @ y)
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt  # row-vector convention
    scale = np.dot(singular, sign) / np.sum(x * x)
    assert scale > 0
    assert np.isclose(np.linalg.det(rotation), 1)
    translation = y0 - scale * x0 @ rotation
    return scale, rotation, translation


def main(cfg: Config) -> None:
    out = cfg.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((cfg.atlas / "zanatomy-manifest.json").read_text())
    references = {}
    objects = {row["source_object_name"]: row for row in manifest["objects"]}
    for name, row in objects.items():
        path = cfg.atlas / row["output_ply"]
        assert sha256(path) == row["output_ply_sha256"], name
        references[name] = pv.read(path).triangulate()
    mesh = pv.read(cfg.fixture / "volume.vtu")
    skin = pv.read(cfg.fixture / "skin.vtp")
    centers = mesh.cell_centers().points
    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    ids = np.asarray(mesh.cell_data["MuscleId"], dtype=int)
    weights = np.asarray(mesh.cell_data["Volume"] * mesh.cell_data["MuscleFraction"])
    rows, source, target = [], [], []
    for name, pair in PAIRS.items():
        for side, muscle_id in zip(("r", "l"), pair, strict=True):
            key = f"{name}.{side}"
            selected = active & (ids == muscle_id)
            assert selected.any(), muscle_id
            source.append(surface_moments(references[key])[0])
            target.append(
                np.average(centers[selected], weights=weights[selected], axis=0)
            )
            rows.append({"atlas_name": key, "current_muscle_id": muscle_id})
    source, target = np.asarray(source), np.asarray(target)
    scale, rotation, translation = similarity(source, target)
    aligned = scale * source @ rotation + translation
    errors = np.linalg.norm(aligned - target, axis=1) * 1000
    # Leave an entire bilateral muscle pair out to avoid mirror-pair leakage.
    heldout = np.empty(len(rows))
    for pair in range(len(PAIRS)):
        select = np.arange(len(rows)) // 2 == pair
        s, r, t = similarity(source[~select], target[~select])
        heldout[select] = (
            np.linalg.norm(s * source[select] @ r + t - target[select], axis=1) * 1000
        )
    for row, x, y, error, heldout_error in zip(
        rows, source, target, errors, heldout, strict=True
    ):
        row.update(
            source_centroid_m=x.tolist(),
            target_centroid_m=y.tolist(),
            residual_mm=float(error),
            heldout_pair_residual_mm=float(heldout_error),
        )
    registered = pv.MultiBlock()
    for name, reference in references.items():
        copy = reference.copy()
        copy.points = scale * reference.points @ rotation + translation
        registered[name] = copy
    registered.save(out / "registered-atlas.vtm")
    selected = active & (ids == 28)
    old_fibers = np.asarray(mesh.cell_data["ActivationFiber"]).copy()
    new_fibers = old_fibers.copy()
    # Use the atlas median plane, after the same proper similarity as every mesh.
    in_atlas = ((centers[selected] - translation) @ rotation.T) / scale
    sides = in_atlas[:, 0] >= 0
    reference_id = np.full(mesh.n_cells, -1, dtype=np.int32)
    reference_distance = np.full(mesh.n_cells, -1.0)
    axes = []
    changed_ids = np.flatnonzero(selected)
    for index, side in enumerate(("r", "l")):
        name = f"Frontalis muscle.{side}"
        _, pca_axis, eigenvalues, covariance = surface_moments(references[name])
        # Regress lateral and posterior position against atlas superior (Z).
        # This defines a broad longitudinal axis without assuming that the
        # widest extent of the belly is its anatomical contraction direction.
        axis = covariance[:, 2] / covariance[2, 2]
        axis /= np.linalg.norm(axis)
        axis = axis @ rotation
        axis *= 1 if axis[1] > 0 else -1
        cells = changed_ids[sides == bool(index)]
        new_fibers[cells] = axis
        reference_id[cells] = index
        distance, _ = cKDTree(registered[name].points).query(centers[cells])
        reference_distance[cells] = distance
        axes.append(
            {
                "atlas_name": name,
                "geometric_longitudinal_axis": axis.tolist(),
                "surface_pca_axis_for_diagnostic_only": (pca_axis @ rotation).tolist(),
                "surface_covariance_eigenvalues_m2": eigenvalues.tolist(),
                "cells": len(cells),
                "nearest_source_vertex_distance_mm_p50_p95_max": (
                    np.quantile(distance, [0.5, 0.95, 1]) * 1000
                ).tolist(),
            }
        )
    assert np.array_equal(old_fibers[~selected], new_fibers[~selected])
    assert np.allclose(np.linalg.norm(new_fibers[active], axis=1), 1)
    mesh.cell_data["ActivationFiber"] = new_fibers
    mesh.cell_data["PublicReferenceId"] = reference_id
    mesh.cell_data["PublicReferenceDistance"] = reference_distance
    mesh.field_data["PublicReferenceName"] = np.array(
        ["Frontalis muscle.r", "Frontalis muscle.l"]
    )
    mesh.field_data["PublicReferenceMethod"] = np.array(
        [
            "Z-Anatomy frontalis superior-coordinate surface regression; modeled longitudinal prior, not measured fascicles"
        ]
    )
    fixture = out / "fixture"
    fixture.mkdir(exist_ok=True)
    mesh.save(fixture / "volume.vtu")
    shutil.copy2(cfg.fixture / "skin.vtp", fixture / "skin.vtp")
    # Reload the actual consumer artifact and compare all pre-existing fields.
    original = pv.read(cfg.fixture / "volume.vtu")
    derivative = pv.read(fixture / "volume.vtu")
    assert np.array_equal(original.points, derivative.points)
    assert np.array_equal(original.cells, derivative.cells)
    for attribute in ("point_data", "cell_data", "field_data"):
        for key in getattr(original, attribute):
            if attribute == "cell_data" and key == "ActivationFiber":
                continue
            a, b = (
                np.asarray(getattr(original, attribute)[key]),
                np.asarray(getattr(derivative, attribute)[key]),
            )
            assert (
                np.array_equal(a, b, equal_nan=True)
                if a.dtype.kind in "fc"
                else np.array_equal(a, b)
            ), key
    assert sha256(cfg.fixture / "skin.vtp") == sha256(fixture / "skin.vtp")
    statistics = {
        "units": "m",
        "method": "22 paired muscle centroids, proper similarity; frontalis excluded; separate longitudinal surface-regression axes",
        "source_manifest": str(cfg.atlas / "zanatomy-manifest.json"),
        "source_manifest_sha256": sha256(cfg.atlas / "zanatomy-manifest.json"),
        "baseline_volume_sha256": sha256(cfg.fixture / "volume.vtu"),
        "candidate_volume_sha256": sha256(fixture / "volume.vtu"),
        "scale": float(scale),
        "row_vector_rotation": rotation.tolist(),
        "translation_m": translation.tolist(),
        "fit_rms_mm": float(np.sqrt(np.mean(errors**2))),
        "heldout_bilateral_pair_rms_mm": float(np.sqrt(np.mean(heldout**2))),
        "landmarks": rows,
        "frontalis": axes,
        "changed_cells": int(selected.sum()),
        "mean_squared_superior_alignment_before": float(
            np.average(old_fibers[selected, 1] ** 2, weights=weights[selected])
        ),
        "mean_squared_superior_alignment_after": float(
            np.average(new_fibers[selected, 1] ** 2, weights=weights[selected])
        ),
        "unchanged": [
            "points",
            "tetrahedra",
            "skin",
            "material fractions",
            "fixed constraints",
            "activation mask",
            "control IDs",
            "all other fibers",
        ],
        "limitations": [
            "Atlas geometry and centroid correspondence are modeled references, not subject measurements.",
            "Surface regression on the atlas superior coordinate gives one modeled axis per belly; it does not recover fascicle curvature or attachments.",
            "The entire existing active forehead region is retained, including any original segmentation errors.",
            "Registered fascia and galea are inspection assets; no SMAS, material, or attachment transfer is applied.",
            "Nearest source vertex distances measure atlas mismatch and are not anatomical accuracy errors.",
        ],
    }
    write_json(out / "transfer.json", statistics)
    rng = np.random.default_rng(0)
    samples = rng.choice(changed_ids, size=220, replace=False)
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2100, 900))
    for column, title in enumerate(
        (
            "Existing merged PCA",
            "Registered public atlas",
            "Candidate: separate belly axes",
        )
    ):
        plotter.subplot(0, column)
        plotter.set_background("white")
        plotter.add_mesh(skin, color="#adb0b5", opacity=0.2)
        if column == 1:
            for side in ("r", "l"):
                plotter.add_mesh(
                    registered[f"Frontalis muscle.{side}"], color="#bc6757", opacity=0.9
                )
                plotter.add_mesh(
                    registered[f"Epicranial aponeurosis.{side}"],
                    color="#dec47d",
                    opacity=0.45,
                )
        else:
            plotter.add_mesh(
                mesh.extract_cells(selected).extract_surface(
                    algorithm="dataset_surface"
                ),
                color="#dba08d",
                opacity=0.7,
            )
            glyphs = pv.PolyData(centers[samples])
            glyphs["fiber"] = (old_fibers if column == 0 else new_fibers)[samples]
            plotter.add_mesh(
                glyphs.glyph(orient="fiber", scale=False, factor=0.007), color="#233e60"
            )
        plotter.add_text(title, font_size=11, color="#222222")
        camera(plotter, skin.center)
        plotter.camera.zoom(1.08)
    plotter.show(screenshot=out / "forehead-reference-comparison.png", auto_close=True)
    cherries.log_metrics(
        {
            key: statistics[key]
            for key in (
                "fit_rms_mm",
                "heldout_bilateral_pair_rms_mm",
                "changed_cells",
                "mean_squared_superior_alignment_before",
                "mean_squared_superior_alignment_after",
            )
        }
    )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
