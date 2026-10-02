"""CPU-only verification of the saved three-mode eigenvisualization."""

# ruff: noqa: PLR0915

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from matplotlib import colors

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]

GROUP = Path(__file__).resolve().parents[1]
RENDER = GROUP / "data/71-eigenmodes"
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
ZERO_TOL = 1e-8
GAP_TOL = 1e-6
NORM = colors.SymLogNorm(linthresh=0.01, linscale=0.5, vmin=-40, vmax=40, base=10)
CMAP = colors.LinearSegmentedColormap.from_list(
    "signed_activation", ["#053061", "#4393c3", "#b5b6b4", "#d6604d", "#67001f"]
)


class Config(cherries.BaseConfig):
    output_dir: Path = cherries.output("80-eigenmodes-verification", mkdir=True)


def record(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def rgb(values: np.ndarray) -> np.ndarray:
    return np.round(CMAP(NORM(values))[:, :3] * 255).astype(np.uint8)


def assert_record(reported: dict) -> dict[str, object]:
    actual = record(Path(reported["path"]))
    assert actual["sha256"] == reported["sha256"]
    assert actual["bytes"] == reported["bytes"]
    return actual


def close(actual: np.ndarray, expected: np.ndarray, tolerance: float) -> float:
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    assert error <= tolerance, (error, tolerance)
    return error


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), out
    summary_path = RENDER / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "completed"
    source = Path(summary["source"]["path"])
    assert_record(summary["source"])
    with np.load(source, allow_pickle=False) as saved:
        assert bool(saved["solver_valid"])
        assert bool(saved["physical_volume_energy"])
        ids = np.asarray(saved["active_ids"], dtype=np.int64)
        rest = np.asarray(saved["rest_points"], dtype=np.float64)
        b = np.asarray(saved["B"], dtype=np.float64)
        z = np.asarray(saved["Z"], dtype=np.float64)
    z_source_error = close(b @ b.swapaxes(-1, -2) - np.eye(3), z, tolerance=2e-13)
    values, axes = np.linalg.eigh(z)
    values, axes = values[:, ::-1].copy(), axes[:, :, ::-1].copy()
    reconstruction = np.einsum("nik,nk,njk->nij", axes, values, axes)
    reconstruction_error = close(reconstruction, z, tolerance=2e-13)
    scale = np.maximum(1.0, np.max(np.abs(values), axis=1, keepdims=True))
    gap = (values[:, :-1] - values[:, 1:]) / scale
    unique = np.c_[
        gap[:, 0] > GAP_TOL,
        np.min(gap, axis=1) > GAP_TOL,
        gap[:, 1] > GAP_TOL,
    ]

    mesh = pv.read(FIXTURE / "volume.vtu")
    assert np.array_equal(np.asarray(mesh.points), rest)
    assert np.array_equal(
        np.flatnonzero(np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)), ids
    )
    tets = np.asarray(mesh.cells, dtype=np.int64).reshape(-1, 5)[:, 1:]
    centers = rest[tets[ids]].mean(axis=1)
    volume = (
        np.asarray(mesh.cell_data["Volume"])[ids]
        * np.asarray(mesh.cell_data["MuscleFraction"])[ids]
    )

    npz_path = RENDER / "eigenmodes.npz"
    with np.load(npz_path, allow_pickle=False) as saved:
        assert np.array_equal(saved["global_cell_ids"], ids)
        center_error = close(saved["centers_rest"], centers, tolerance=2e-15)
        value_error = close(saved["eigenvalues_descending"], values, tolerance=0.0)
        axis_projector_error = close(
            np.einsum(
                "...ik,...jk->...ijk",
                saved["reference_axes_columns"],
                saved["reference_axes_columns"],
            ),
            np.einsum("...ik,...jk->...ijk", axes, axes),
            tolerance=2e-14,
        )
        assert np.array_equal(saved["axis_unique"], unique)
        weight_error = close(saved["muscle_volume_weights"], volume, tolerance=0.0)

    cloud = pv.read(RENDER / "all-active-cell-modes.vtp")
    assert cloud.n_points == len(ids)
    cloud_center_error = close(np.asarray(cloud.points), centers, tolerance=2e-15)
    assert np.array_equal(cloud.point_data["GlobalCellId"], ids)
    cloud_errors = {}
    for mode in range(3):
        cloud_errors[f"mode_{mode + 1}_value"] = close(
            cloud.point_data[f"Mode{mode + 1}Eigenvalue"],
            values[:, mode],
            tolerance=0.0,
        )
        cloud_errors[f"mode_{mode + 1}_axis_projector"] = close(
            cloud.point_data[f"Mode{mode + 1}ReferenceAxis"][:, :, None]
            * cloud.point_data[f"Mode{mode + 1}ReferenceAxis"][:, None, :],
            axes[:, :, mode, None] * axes[:, None, :, mode],
            tolerance=2e-14,
        )
    residual_error = close(
        cloud.point_data["ResidualFrobeniusNorm"],
        np.sqrt(values[:, 1] ** 2 + values[:, 2] ** 2),
        tolerance=2e-15,
    )

    mode_norm2 = np.sum(values**2, axis=0)
    weighted_norm2 = np.sum(volume[:, None] * values**2, axis=0)
    for mode, reported in enumerate(summary["mode_statistics"]):
        assert reported["mode"] == mode + 1
        assert reported["positive_cells"] == int(np.sum(values[:, mode] > ZERO_TOL))
        assert reported["negative_cells"] == int(np.sum(values[:, mode] < -ZERO_TOL))
        assert reported["neutral_cells"] == int(
            np.sum(np.abs(values[:, mode]) <= ZERO_TOL)
        )
        assert reported["nonunique_axes"] == int(np.sum(~unique[:, mode]))
        close(
            np.asarray(reported["eigenvalue_percentiles_0_1_10_50_90_99_100"]),
            np.percentile(values[:, mode], [0, 1, 10, 50, 90, 99, 100]),
            tolerance=0.0,
        )
        assert np.isclose(
            reported["squared_frobenius_share_unweighted"],
            mode_norm2[mode] / mode_norm2.sum(),
        )
        assert np.isclose(
            reported["squared_frobenius_share_muscle_volume_weighted"],
            weighted_norm2[mode] / weighted_norm2.sum(),
        )

    view_receipts = {}
    for view, reported in summary["views"].items():
        sample_path = RENDER / f"{view}-sample.npz"
        with np.load(sample_path, allow_pickle=False) as saved:
            sample = np.asarray(saved["active_array_indices"], dtype=np.int64)
            assert np.array_equal(sample, np.unique(sample))
            assert np.array_equal(saved["global_cell_ids"], ids[sample])
            sample_center_error = close(saved["centers"], centers[sample], 0.0)
        assert reported["sample_count"] == len(sample)
        modes = []
        for mode, mode_report in enumerate(reported["modes"]):
            nonzero = np.abs(values[sample, mode]) > ZERO_TOL
            line_indices = sample[nonzero & unique[sample, mode]]
            assert mode_report["common_sample_count"] == len(sample)
            assert mode_report["nonzero_dots"] == int(np.sum(nonzero))
            assert mode_report["direction_lines"] == len(line_indices)
            assert mode_report["degenerate_direction_dots_only"] == int(
                np.sum(nonzero & ~unique[sample, mode])
            )
            assert mode_report["neutral_omitted"] == int(np.sum(~nonzero))
            glyph = pv.read(assert_record(mode_report["line_mesh"])["path"])
            assert glyph.n_cells == len(line_indices)
            lines = np.asarray(glyph.lines).reshape(-1, 3)
            assert np.all(lines[:, 0] == 2)
            endpoints = np.asarray(glyph.points)[lines[:, 1:]]
            glyph_centers = endpoints.mean(axis=1)
            vectors = endpoints[:, 1] - endpoints[:, 0]
            lengths = np.linalg.norm(vectors, axis=1)
            length = float(mode_report["glyph_length_m"])
            direction_tolerance = (
                8
                * np.finfo(np.float64).eps
                * max(
                    1.0,
                    float(np.max(np.abs(endpoints))) if endpoints.size else 1.0,
                )
                / length
            )
            assert np.array_equal(glyph.cell_data["GlobalCellId"], ids[line_indices])
            modes.append(
                {
                    "mode": mode + 1,
                    "lines": len(line_indices),
                    "center_max_abs_error": close(
                        glyph_centers, centers[line_indices], 2e-15
                    ),
                    "length_max_abs_error": close(
                        lengths, np.full(len(lengths), length), 2e-15
                    ),
                    "axis_max_abs_error": close(
                        glyph.cell_data["ReferenceAxis"],
                        axes[line_indices, :, mode],
                        0.0,
                    ),
                    "line_direction_max_abs_error": close(
                        vectors / lengths[:, None],
                        axes[line_indices, :, mode],
                        direction_tolerance,
                    ),
                    "line_direction_tolerance": direction_tolerance,
                    "value_max_abs_error": close(
                        glyph.cell_data["Z_eigenvalue"], values[line_indices, mode], 0.0
                    ),
                    "rgb_exact": bool(
                        np.array_equal(
                            glyph.cell_data["ColorRGB"], rgb(values[line_indices, mode])
                        )
                    ),
                    "line_mesh": record(Path(mode_report["line_mesh"]["path"])),
                    "raw_image": assert_record(mode_report["raw_image"]),
                }
            )
            assert modes[-1]["rgb_exact"]
        figures = [assert_record(item) for item in reported["figures"]]
        view_receipts[view] = {
            "sample": record(sample_path),
            "sample_count": len(sample),
            "sample_center_max_abs_error": sample_center_error,
            "modes": modes,
            "figures": figures,
        }

    sources = {}
    for name, reported in summary["sources"].items():
        source_record = assert_record(reported["source"])
        snapshot_record = assert_record(reported["snapshot"])
        assert source_record["sha256"] == snapshot_record["sha256"]
        sources[name] = snapshot_record
    source_dir = out / "sources"
    source_dir.mkdir()
    verifier_snapshot = source_dir / Path(__file__).name
    shutil.copy2(__file__, verifier_snapshot)
    receipt = {
        "status": "passed",
        "scope": "CPU-only reconstruction and artifact verification; no rendering or mechanics solve",
        "inputs": {
            "summary": record(summary_path),
            "source_activation": record(source),
            "eigenmodes_npz": record(npz_path),
            "all_active_vtp": record(RENDER / "all-active-cell-modes.vtp"),
            "executed_sources": sources,
        },
        "active_cells": len(ids),
        "source_Z_error": z_source_error,
        "Z_reconstruction_error": reconstruction_error,
        "npz_errors": {
            "centers": center_error,
            "eigenvalues": value_error,
            "axis_projectors": axis_projector_error,
            "muscle_volume_weights": weight_error,
        },
        "all_active_vtp_errors": {
            "centers": cloud_center_error,
            "residual_norm": residual_error,
            **cloud_errors,
        },
        "views": view_receipts,
        "image_count": sum(
            len(view["figures"]) + len(view["modes"]) for view in view_receipts.values()
        ),
        "encoding": summary["encoding"],
        "verifier_source": record(verifier_snapshot),
    }
    receipt_path = out / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    cherries.log_output(receipt_path)
    cherries.log_output(verifier_snapshot)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
