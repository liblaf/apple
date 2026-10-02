"""Independently verify complete activation fields and saved visibility masks."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
DATA = GROUP / "data/72-visible-activation-glyphs"
OUTPUT = GROUP / "data/73-visible-glyph-report-verification"
FIXTURE = (
    GROUP / "../../07/face-actuation-diagnosis/data/12-historical-fixture/volume.vtu"
)


class Config(cherries.BaseConfig):
    """Verification uses the final saved gallery."""


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_record(record: dict) -> None:
    path = Path(record["path"])
    assert path.stat().st_size == record["bytes"], path
    assert digest(path) == record["sha256"], path


def verify_fields(summary: dict) -> list[dict]:
    volume = pv.read(FIXTURE)
    active = np.flatnonzero(np.asarray(volume.cell_data["MuscleFraction"]) > 0)
    assert len(active) == 288235
    cells = volume.cells.reshape(-1, 5)
    assert np.all(cells[:, 0] == 4)
    tetrahedra = volume.points[cells[active, 1:]]
    centers = tetrahedra.mean(axis=1)
    names = np.asarray(volume.field_data["MuscleName"]).astype(str)
    checks = []
    for state in summary["states"]:
        for key in (
            "checkpoint",
            "trace",
            "run_summary",
            "config",
            "provenance",
            "settings",
        ):
            verify_record(state[key])
        with np.load(state["checkpoint"]["path"], allow_pickle=False) as saved:
            assert np.array_equal(saved["active_ids"], active)
            assert np.array_equal(saved["rest_points"], volume.points)
            assert str(saved["model"]) == "learned-axis"
            assert bool(saved["solver_valid"])
            assert int(saved["step"]) == state["step"]
            q = saved["q"]
            np.testing.assert_array_equal(saved["C"], np.einsum("ni,nj->nij", q, q))
        norm = np.linalg.norm(q, axis=1)
        axes = q / norm[:, None]
        strength = norm**2
        shortening = strength / (1 + strength)
        points = pv.read(state["samples"]["path"])
        lines = pv.read(state["glyphs"]["path"])
        assert points.n_points == lines.n_lines == lines.n_cells == len(active)
        np.testing.assert_allclose(points.points, centers, rtol=0, atol=1e-14)
        for arrays in (points.point_data, lines.cell_data):
            np.testing.assert_array_equal(arrays["GlobalCellId"], active)
            for name in ("MuscleId", "ActivationControlId", "MuscleFraction"):
                np.testing.assert_array_equal(
                    arrays[name], volume.cell_data[name][active]
                )
            np.testing.assert_array_equal(
                arrays["MuscleLabel"].astype(str),
                names[volume.cell_data["MuscleId"][active]],
            )
            np.testing.assert_allclose(
                arrays["LearnedAxisRest"], axes, rtol=0, atol=1e-14
            )
            np.testing.assert_allclose(
                arrays["LearnedAxisDyadRest"],
                np.einsum("ni,nj->nij", axes, axes).reshape(-1, 9),
                rtol=0,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                arrays["qNormSquared"], strength, rtol=1e-14, atol=0
            )
            np.testing.assert_allclose(
                arrays["CommandedShorteningFraction"], shortening, rtol=1e-14, atol=0
            )
            np.testing.assert_allclose(
                arrays["CommandedShorteningPercent"],
                100 * shortening,
                rtol=1e-14,
                atol=0,
            )
        connectivity = lines.lines.reshape(-1, 3)
        assert np.all(connectivity[:, 0] == 2)
        ends = lines.points[connectivity[:, 1:]]
        np.testing.assert_allclose(ends.mean(axis=1), centers, rtol=0, atol=1e-14)
        expected_delta = 0.003 * shortening[:, None] * axes
        np.testing.assert_allclose(
            ends[:, 1] - ends[:, 0], expected_delta, rtol=0, atol=1e-14
        )
        lengths = np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1)
        np.testing.assert_allclose(lengths, 0.003 * shortening, rtol=0, atol=1e-14)
        assert np.min(np.diff(lengths[np.argsort(shortening)])) >= -1e-14
        checks.append(
            {
                "state": state["id"],
                "step": state["step"],
                "line_count": lines.n_lines,
                "region_count": int(
                    np.unique(lines.cell_data["ActivationControlId"]).size
                ),
                "max_line_vector_error_m": float(
                    np.abs(ends[:, 1] - ends[:, 0] - expected_delta).max()
                ),
                "common_length_scale_m": 0.003,
                "max_length_error_m": float(np.abs(lengths - 0.003 * shortening).max()),
                "status": "passed",
            }
        )
    return checks


def verify_visibility(summary: dict) -> list[dict]:
    """Replay the raster lookup and separately compute the camera projection."""
    volume = pv.read(FIXTURE)
    active = np.flatnonzero(np.asarray(volume.cell_data["MuscleFraction"]) > 0)
    controls = np.asarray(volume.cell_data["ActivationControlId"])[active]
    centers = volume.points[volume.cells.reshape(-1, 5)[active, 1:]].mean(axis=1)
    surface = pv.read(DATA / "context/muscle-regions.vtp")
    boundary_ids = np.unique(surface.cell_data["SourceTetraGlobalCellId"])
    boundary = np.isin(active, boundary_ids)
    cameras_path = (
        GROUP / "../../08/physical-volume-closeups/data/20-regions/summary.json"
    )
    cameras = {
        view["id"]: view["camera"]
        for view in json.loads(cameras_path.read_text())["views"]
    }
    results = []
    for view in ("side-context", "region1-mouth-corner"):
        path = DATA / "visibility" / f"{view}.npz"
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["global_cell_ids"], active)
            np.testing.assert_array_equal(saved["control_ids"], controls)
            mask = saved["mask"]
            inside = saved["projected_inside"]
            pixels = saved["projected_pixel_xy"]
            front = saved["front_control_ids"]
            labels = saved["front_label_image"]
            width, height = saved["window_size"]
            assert (width, height) == (1800, 1800)
            assert labels.shape == (height, width)
            assert set(np.unique(labels)) <= set(range(-1, 103))
            x, y = pixels.T
            np.testing.assert_array_equal(
                inside, (x >= 0) & (x < width) & (y >= 0) & (y < height)
            )
            expected_front = np.full(len(active), -1)
            expected_front[inside] = labels[height - 1 - y[inside], x[inside]]
            np.testing.assert_array_equal(front, expected_front)
            np.testing.assert_array_equal(mask, inside & (front == controls))
            camera = cameras[view]
            forward = np.asarray(camera["focal_point"]) - camera["position"]
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, camera["view_up"])
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            relative = centers - camera["focal_point"]
            projected = np.column_stack(
                (
                    width / 2
                    + relative @ right * height / (2 * camera["parallel_scale"]),
                    height / 2
                    + relative @ up * height / (2 * camera["parallel_scale"]),
                )
            )
            np.testing.assert_array_equal(pixels, projected.astype(np.int32))
            retained = int(mask.sum())
            interior = int((mask & ~boundary).sum())
            boundary_count = int((mask & boundary).sum())
            assert retained == int(saved["retained_count"])
            assert interior == int(saved["retained_interior_count"]) > 0
            assert boundary_count == int(saved["retained_boundary_source_count"])
            assert retained == interior + boundary_count
            for state in summary["states"]:
                assert state["views"][view]["line_count_submitted"] == retained
            results.append(
                {
                    "view": view,
                    "retained_count": retained,
                    "retained_interior_count": interior,
                    "retained_boundary_count": boundary_count,
                    "occluded_by_other_region_count": int(
                        (inside & (front >= 0) & (front != controls)).sum()
                    ),
                    "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                    "archive_sha256": digest(path),
                    "status": "passed",
                }
            )
    return results


def main(cfg: Config) -> None:
    del cfg
    destination = OUTPUT / "activation-glyphs.json"
    assert not destination.exists(), destination
    summary = json.loads((DATA / "summary.json").read_text())
    assert summary["status"] == "completed_activation_glyph_render"
    for collection in (summary["inputs"], summary["outputs"]):
        for record in collection.values():
            verify_record(record)
    for key in ("source", "muscle_context_source"):
        source = summary[key]
        verify_record(source["live_at_generation"])
        verify_record(source["snapshot"])
        assert source["live_at_generation"]["sha256"] == source["snapshot"]["sha256"]
    fields = verify_fields(summary)
    visibility = verify_visibility(summary)
    with zipfile.ZipFile(DATA / "glyph-data.zip") as bundle:
        members = bundle.namelist()
        assert len(members) == len(set(members))
        assert any(name.startswith("visibility/") for name in members)
        for name in members:
            with bundle.open(name) as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == digest(
                    DATA / name
                ), name
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = OUTPUT / Path(__file__).name
    shutil.copyfile(__file__, source)
    receipt = {
        "verified_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "status": "passed",
        "summary_sha256": digest(DATA / "summary.json"),
        "input_count": len(summary["inputs"]),
        "output_count": len(summary["outputs"]),
        "field_checks": fields,
        "visibility_checks": visibility,
        "zip_members": len(members),
        "checks": [
            "source/input/output hashes",
            "every active tetrahedron exactly once",
            "saved controls and muscle metadata",
            "reference centroids",
            "unsigned axes and dyads",
            "common linear length scale with no minimum or cell-size scaling",
            "interior tetrahedra retained for front muscle",
            "saved visibility labels and camera projection",
            "archive bytes",
        ],
        "verifier_sha256": digest(source),
        "runtime": {
            "numpy": np.__version__,
            "pyvista": pv.__version__,
            "vtk": list(pv.vtk_version_info),
        },
    }
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    cherries.log_output(destination)
    cherries.log_output(source)
    cherries.log_metric("verification/states", len(fields))


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
