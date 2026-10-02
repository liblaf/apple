"""Verify dense deformed-frame eigenmode visualization artifacts on CPU."""

# ruff: noqa: PLR0915

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from PIL import Image

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]

GROUP = Path(__file__).resolve().parents[1]
RENDER = GROUP / "data/74-meeting-eigenmodes"
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
BASELINE_SHA256 = "07efff9f6a96ff7d4556df723f6f6386c31c111ad89ac65ebff21ded82050201"
IDENTIFIERS = ("mode-1-principal", "mode-2-residual", "mode-3-residual")
WINDOW = (1800, 1800)
MAX_LENGTH = 0.0045
ZERO_TOL = 1e-8
GAP_TOL = 1e-6


class Config(cherries.BaseConfig):
    output_dir: Path = cherries.output("82-meeting-eigenmodes-verification", mkdir=True)


def record(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def assert_record(reported: dict) -> dict[str, object]:
    actual = record(Path(reported["path"]))
    assert actual["sha256"] == reported["sha256"]
    assert actual["bytes"] == reported["bytes"]
    return actual


def close(actual: np.ndarray, expected: np.ndarray, tolerance: float) -> float:
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    assert error <= tolerance, (error, tolerance)
    return error


def image_record(reported: dict, expected_size: tuple[int, int]) -> dict[str, object]:
    actual = assert_record(reported)
    with Image.open(actual["path"]) as image:
        assert image.format == "PNG"
        assert image.size == expected_size
    return actual


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), out
    summary_path = RENDER / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "completed"
    source = Path(summary["source"]["path"])
    assert_record(summary["source"])
    assert summary["source"]["sha256"] == BASELINE_SHA256
    with np.load(source, allow_pickle=False) as saved:
        assert bool(saved["solver_valid"])
        assert bool(saved["physical_volume_energy"])
        ids = np.asarray(saved["active_ids"], dtype=np.int64)
        rest = np.asarray(saved["rest_points"], dtype=np.float64)
        u = np.asarray(saved["u"], dtype=np.float64)
        b = np.asarray(saved["B"], dtype=np.float64)
        z = np.asarray(saved["Z"], dtype=np.float64)
    source_z_error = close(b @ b.swapaxes(-1, -2) - np.eye(3), z, 2e-13)
    values, axes = np.linalg.eigh(z)
    values, axes = values[:, ::-1].copy(), axes[:, :, ::-1].copy()
    assert np.all(values[:, 0] >= -ZERO_TOL)
    assert np.all(values[:, 2] <= ZERO_TOL)
    reconstruction_error = close(
        np.einsum("nik,nk,njk->nij", axes, values, axes), z, 2e-13
    )
    gaps = (values[:, :-1] - values[:, 1:]) / np.maximum(
        1.0, np.max(np.abs(values), axis=1, keepdims=True)
    )
    unique = np.c_[
        gaps[:, 0] > GAP_TOL,
        np.min(gaps, axis=1) > GAP_TOL,
        gaps[:, 1] > GAP_TOL,
    ]
    magnitude = 1.0 - 1.0 / np.sqrt(1.0 + np.abs(values))
    lengths = MAX_LENGTH * magnitude * unique * (np.abs(values) > ZERO_TOL)

    volume = pv.read(FIXTURE / "volume.vtu")
    assert np.array_equal(np.asarray(volume.points), rest)
    assert np.array_equal(
        np.flatnonzero(np.asarray(volume.cell_data["ActivationMask"], dtype=bool)), ids
    )
    tets = np.asarray(volume.cells, dtype=np.int64).reshape(-1, 5)[:, 1:][ids]
    rest_tets = rest[tets]
    deformed_tets = (rest + u)[tets]
    rest_edges = (rest_tets[:, 1:] - rest_tets[:, :1]).swapaxes(1, 2)
    deformed_edges = (deformed_tets[:, 1:] - deformed_tets[:, :1]).swapaxes(1, 2)
    deformation_gradient = deformed_edges @ np.linalg.inv(rest_edges)
    transported = deformation_gradient @ axes
    transported_norm = np.linalg.norm(transported, axis=1)
    assert np.isfinite(transported_norm).all()
    assert np.all(transported_norm > 0.0)
    spatial_axes = transported / transported_norm[:, None, :]
    centers_rest = rest_tets.mean(axis=1)
    centers_deformed = deformed_tets.mean(axis=1)
    weights = (
        np.asarray(volume.cell_data["Volume"])[ids]
        * np.asarray(volume.cell_data["MuscleFraction"])[ids]
    )
    control_ids = np.asarray(volume.cell_data["ActivationControlId"])[ids]

    npz_path = RENDER / "eigenmodes.npz"
    with np.load(npz_path, allow_pickle=False) as saved:
        assert np.array_equal(saved["global_cell_ids"], ids)
        npz_errors = {
            "centers_rest": close(saved["centers_rest"], centers_rest, 0.0),
            "centers_deformed": close(saved["centers_deformed"], centers_deformed, 0.0),
            "eigenvalues": close(saved["eigenvalues_descending"], values, 0.0),
            "reference_axis_projectors": close(
                np.einsum(
                    "...ik,...jk->...ijk",
                    saved["reference_axes_columns"],
                    saved["reference_axes_columns"],
                ),
                np.einsum("...ik,...jk->...ijk", axes, axes),
                2e-14,
            ),
            "spatial_axis_projectors": close(
                np.einsum(
                    "...ik,...jk->...ijk",
                    saved["spatial_axes_columns"],
                    saved["spatial_axes_columns"],
                ),
                np.einsum("...ik,...jk->...ijk", spatial_axes, spatial_axes),
                2e-14,
            ),
            "deformation_gradient": close(
                saved["deformation_gradient"], deformation_gradient, 0.0
            ),
            "gaps": close(saved["normalized_adjacent_eigenvalue_gaps"], gaps, 0.0),
            "display_magnitude": close(saved["display_magnitude"], magnitude, 0.0),
            "display_length": close(saved["display_length_m"], lengths, 0.0),
            "muscle_volume_weights": close(
                saved["muscle_volume_weights"], weights, 0.0
            ),
        }
        assert np.array_equal(saved["axis_unique"], unique)

    norm2 = np.sum(values**2, axis=0)
    weighted_norm2 = np.sum(weights[:, None] * values**2, axis=0)
    for mode, reported in enumerate(summary["mode_statistics"]):
        assert reported["mode"] == mode + 1
        assert reported["positive_cells"] == int(np.sum(values[:, mode] > ZERO_TOL))
        assert reported["negative_cells"] == int(np.sum(values[:, mode] < -ZERO_TOL))
        assert reported["neutral_cells"] == int(
            np.sum(np.abs(values[:, mode]) <= ZERO_TOL)
        )
        assert reported["nonunique_axes"] == int(np.sum(~unique[:, mode]))
        assert np.isclose(
            reported["squared_frobenius_share_unweighted"],
            norm2[mode] / norm2.sum(),
        )
        assert np.isclose(
            reported["squared_frobenius_share_muscle_volume_weighted"],
            weighted_norm2[mode] / weighted_norm2.sum(),
        )

    glyph_receipts = []
    glyph_paths = []
    for mode, identifier in enumerate(IDENTIFIERS):
        path = RENDER / f"all-active-{identifier}.vtp"
        glyph_paths.append(path)
        glyph = pv.read(path)
        assert glyph.n_cells == glyph.n_points // 2 == len(ids)
        required = {
            "GlobalCellId",
            "ActivationControlId",
            "Z_eigenvalue",
            "ReferenceAxis",
            "SpatialAxis",
            "AxisUnique",
            "DisplayMagnitudePercent",
            "DisplayLengthM",
            "SignRGB",
        }
        assert required.issubset(glyph.cell_data.keys())
        assert np.array_equal(glyph.cell_data["GlobalCellId"], ids)
        assert np.array_equal(glyph.cell_data["ActivationControlId"], control_ids)
        assert np.array_equal(glyph.cell_data["AxisUnique"], unique[:, mode])
        assert np.array_equal(glyph.cell_data["Z_eigenvalue"], values[:, mode])
        assert np.array_equal(glyph.cell_data["ReferenceAxis"], axes[:, :, mode])
        assert np.array_equal(glyph.cell_data["SpatialAxis"], spatial_axes[:, :, mode])
        assert np.array_equal(
            glyph.cell_data["DisplayMagnitudePercent"], 100.0 * magnitude[:, mode]
        )
        assert np.array_equal(glyph.cell_data["DisplayLengthM"], lengths[:, mode])
        sign_rgb = np.where(
            (values[:, mode] > 0.0)[:, None], [178, 24, 43], [33, 102, 172]
        ).astype(np.uint8)
        assert np.array_equal(glyph.cell_data["SignRGB"], sign_rgb)
        lines = np.asarray(glyph.lines).reshape(-1, 3)
        assert np.all(lines[:, 0] == 2)
        endpoints = np.asarray(glyph.points)[lines[:, 1:]]
        geometric_centers = endpoints.mean(axis=1)
        vectors = endpoints[:, 1] - endpoints[:, 0]
        geometric_lengths = np.linalg.norm(vectors, axis=1)
        positive_length = lengths[:, mode] > 0.0
        vector_tolerance = (
            8 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(endpoints))))
        )
        glyph_receipts.append(
            {
                "mode": mode + 1,
                "artifact": record(path),
                "cells": glyph.n_cells,
                "positive_length_cells": int(np.sum(positive_length)),
                "zero_length_cells": int(np.sum(~positive_length)),
                "center_max_abs_error": close(
                    geometric_centers, centers_deformed, 2e-15
                ),
                "length_max_abs_error": close(
                    geometric_lengths, lengths[:, mode], 2e-15
                ),
                "line_vector_max_abs_error": close(
                    vectors,
                    lengths[:, mode, None] * spatial_axes[:, :, mode],
                    vector_tolerance,
                ),
                "line_vector_tolerance": vector_tolerance,
                "sign_rgb_exact": True,
            }
        )

    view_receipts = {}
    for name, reported in summary["views"].items():
        visibility_path = RENDER / f"{name}-visibility.npz"
        with np.load(visibility_path, allow_pickle=False) as saved:
            assert np.array_equal(saved["global_cell_ids"], ids)
            assert np.array_equal(saved["control_ids"], control_ids)
            mask = np.asarray(saved["mask"], dtype=bool)
            assert mask.shape == ids.shape
            assert int(saved["retained_count"]) == int(np.sum(mask))
            assert np.array_equal(saved["window_size"], WINDOW)
        assert reported["common_visible_candidates"] == int(np.sum(mask))
        mode_counts = []
        for mode, panel in enumerate(reported["modes"]):
            expected = int(np.sum(mask & (lengths[:, mode] > 0.0)))
            assert panel["shown_nonzero_unique_lines"] == expected
            image_record(panel["image"], WINDOW)
            mode_counts.append(expected)
        assert reported["mode_2_sign"]["shown_nonzero_unique_lines"] == mode_counts[1]
        image_record(reported["mode_2_sign"]["image"], WINDOW)
        image_record(reported["triptych"], (3 * WINDOW[0], WINDOW[1]))
        view_receipts[name] = {
            "visibility": record(visibility_path),
            "common_visible_candidates": int(np.sum(mask)),
            "shown_lines_by_mode": mode_counts,
            "images": 5,
        }

    assert summary["encoding"]["frame"].startswith("saved deformed full-fit geometry")
    assert summary["encoding"]["magnitude"].startswith("a = 1-1/sqrt(1+abs(z))")
    assert summary["encoding"]["length"].startswith("0.0045 m * a")
    assert summary["encoding"]["sampling"].startswith(
        "all camera-facing region-matched active tetrahedra"
    )
    sources = {}
    for name, reported in summary["sources"].items():
        source_record = assert_record(reported["source"])
        snapshot_record = assert_record(reported["snapshot"])
        assert source_record["sha256"] == snapshot_record["sha256"]
        sources[name] = snapshot_record
    style_reference = assert_record(summary["style_reference"])

    source_dir = out / "sources"
    source_dir.mkdir()
    verifier_snapshot = source_dir / Path(__file__).name
    shutil.copy2(__file__, verifier_snapshot)
    receipt = {
        "status": "passed",
        "scope": "CPU-only reconstruction of dense deformed-frame glyph artifacts; no rendering or mechanics solve",
        "inputs": {
            "summary": record(summary_path),
            "baseline": record(source),
            "eigenmodes": record(npz_path),
            "glyphs": [record(path) for path in glyph_paths],
            "style_reference": style_reference,
            "executed_sources": sources,
        },
        "active_cells": len(ids),
        "source_Z_error": source_z_error,
        "Z_reconstruction_error": reconstruction_error,
        "npz_errors": npz_errors,
        "glyphs": glyph_receipts,
        "views": view_receipts,
        "image_count": sum(item["images"] for item in view_receipts.values()),
        "mode_statistics": summary["mode_statistics"],
        "encoding": summary["encoding"],
        "verifier_source": record(verifier_snapshot),
    }
    receipt_path = out / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    cherries.log_output(receipt_path)
    cherries.log_output(verifier_snapshot)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
