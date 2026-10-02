"""Independently verify saved-shape activation glyph exports and visibility."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from muscle_glyph_context import build_muscle_region_context
from PIL import Image

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]
GROUP = Path(__file__).resolve().parents[1]
DATA = GROUP / "data/76-deformed-activation-glyphs"
OUTPUT = GROUP / "data/77-deformed-glyph-verification"
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
CAMERAS = ROOT / "exp/2026/09/08/physical-volume-closeups/data/20-regions/summary.json"
DISPLAY_MAX_LENGTH_M = 0.0045
VIEWS = ("side-context", "region1-mouth-corner")


class Config(cherries.BaseConfig):
    """Verify the completed deformed-glyph gallery without recomputing a solve."""


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_record(record: dict[str, Any]) -> None:
    path = Path(record["path"])
    assert path.is_file(), path
    assert path.stat().st_size == record["bytes"], path
    assert digest(path) == record["sha256"], path


def _fixture() -> tuple[
    pv.UnstructuredGrid, pv.PolyData, np.ndarray, np.ndarray, np.ndarray
]:
    volume = pv.read(FIXTURE / "volume.vtu")
    skin = pv.read(FIXTURE / "skin.vtp")
    cells = np.asarray(volume.cells, dtype=np.int64).reshape(-1, 5)
    assert np.all(cells[:, 0] == 4)
    active = np.flatnonzero(
        np.asarray(volume.cell_data["MuscleFraction"], dtype=np.float64) > 0.0
    )
    np.testing.assert_array_equal(
        active, np.flatnonzero(volume.cell_data["ActivationMask"])
    )
    assert len(active) == 288235
    return (
        volume,
        skin,
        cells[:, 1:],
        active,
        np.asarray(volume.points, dtype=np.float64),
    )


def _saved_state(
    state: dict[str, Any], active: np.ndarray, rest: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(state["checkpoint"]["path"], allow_pickle=False) as saved:
        assert int(saved["step"]) == state["step"]
        assert bool(saved["solver_valid"])
        assert str(saved["model"]) == "learned-axis"
        np.testing.assert_array_equal(saved["active_ids"], active)
        np.testing.assert_array_equal(saved["rest_points"], rest)
        q = np.asarray(saved["q"], dtype=np.float64)
        u = np.asarray(saved["u"], dtype=np.float64)
        np.testing.assert_array_equal(saved["C"], np.einsum("ni,nj->nij", q, q))
    assert q.shape == (len(active), 3)
    assert u.shape == rest.shape
    assert np.isfinite(q).all()
    assert np.isfinite(u).all()
    return q, u


def _expected_fields(
    q: np.ndarray, u: np.ndarray, rest: np.ndarray, tets: np.ndarray, active: np.ndarray
) -> dict[str, np.ndarray]:
    rest_tets = rest[tets[active]]
    deformed_tets = (rest + u)[tets[active]]
    rest_centroids = rest_tets.mean(axis=1)
    centroids = deformed_tets.mean(axis=1)
    dm = np.swapaxes(rest_tets[:, 1:] - rest_tets[:, :1], 1, 2)
    ds = np.swapaxes(deformed_tets[:, 1:] - deformed_tets[:, :1], 1, 2)
    # Solve F Dm = Ds as Dm.T F.T = Ds.T.  This is deliberately independent
    # from the renderer's explicit Ds @ inverse(Dm) implementation.
    deformation_gradient = np.swapaxes(
        np.linalg.solve(np.swapaxes(dm, 1, 2), np.swapaxes(ds, 1, 2)), 1, 2
    )
    np.testing.assert_allclose(deformation_gradient @ dm, ds, rtol=1e-13, atol=1e-14)
    q_norm = np.linalg.norm(q, axis=1)
    assert np.all(q_norm > 0.0)
    rest_axis = q / q_norm[:, None]
    transported = np.einsum("nij,nj->ni", deformation_gradient, rest_axis)
    stretch = np.linalg.norm(transported, axis=1)
    assert np.isfinite(stretch).all()
    assert np.all(stretch > 0.0)
    spatial_axis = transported / stretch[:, None]
    strength = q_norm**2
    shortening = strength / (1.0 + strength)
    return {
        "centroids": centroids,
        "rest_centroids": rest_centroids,
        "rest_axis": rest_axis,
        "spatial_axis": spatial_axis,
        "F": deformation_gradient,
        "detF": np.linalg.det(deformation_gradient),
        "stretch": stretch,
        "strength": strength,
        "shortening": shortening,
    }


def _verify_skin(
    state: dict[str, Any], skin: pv.PolyData, rest: np.ndarray, u: np.ndarray
) -> dict[str, Any]:
    saved = pv.read(state["deformed_skin"]["path"])
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    expected = rest[skin_ids] + u[skin_ids]
    assert saved.n_points == skin.n_points
    assert saved.n_cells == skin.n_cells
    np.testing.assert_allclose(saved.points, expected, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        saved.point_data["RestPosition"], rest[skin_ids], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        saved.point_data["Displacement"], u[skin_ids], rtol=0.0, atol=0.0
    )
    assert str(saved.field_data["CoordinateFrame"][0]) == "saved deformed coordinates"
    return {
        "point_count": saved.n_points,
        "cell_count": saved.n_cells,
        "max_point_error_m": float(np.max(np.abs(saved.points - expected))),
        "max_displacement_mm": float(
            1000.0 * np.linalg.norm(u[skin_ids], axis=1).max()
        ),
    }


def _verify_region_surface(
    state: dict[str, Any],
    volume: pv.UnstructuredGrid,
    tets: np.ndarray,
    active: np.ndarray,
    rest: np.ndarray,
    u: np.ndarray,
) -> dict[str, Any]:
    surface = pv.read(state["deformed_muscle_regions"]["path"])
    controls = np.asarray(volume.cell_data["ActivationControlId"], dtype=np.int64)
    active_control = controls[active]
    source_ids = np.asarray(
        surface.cell_data["SourceTetraGlobalCellId"], dtype=np.int64
    )
    surface_control = np.asarray(
        surface.cell_data["ActivationControlId"], dtype=np.int64
    )
    assert surface.n_cells == len(source_ids) == len(surface_control)
    assert set(np.unique(surface_control)) == set(range(103))
    source_positions = np.searchsorted(active, source_ids)
    assert np.all(source_positions < len(active))
    np.testing.assert_array_equal(active[source_positions], source_ids)
    np.testing.assert_array_equal(active_control[source_positions], surface_control)
    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
    assert np.all(faces[:, 0] == 3)
    x = rest + u
    max_vertex_error = 0.0
    for row, source in zip(faces[:, 1:], source_ids, strict=True):
        candidates = x[tets[source]]
        distances = np.linalg.norm(
            surface.points[row, None, :] - candidates[None, :, :], axis=2
        )
        max_vertex_error = max(
            max_vertex_error, float(np.max(np.min(distances, axis=1)))
        )
    assert max_vertex_error <= 1.0e-14
    return {
        "region_count": int(np.unique(surface_control).size),
        "triangle_count": surface.n_cells,
        "max_triangle_vertex_error_m": max_vertex_error,
    }


def _verify_full_fields(
    state: dict[str, Any],
    volume: pv.UnstructuredGrid,
    tets: np.ndarray,
    active: np.ndarray,
    rest: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    q, u = _saved_state(state, active, rest)
    expected = _expected_fields(q, u, rest, tets, active)
    controls = np.asarray(volume.cell_data["ActivationControlId"], dtype=np.int64)[
        active
    ]
    muscle_ids = np.asarray(volume.cell_data["MuscleId"], dtype=np.int32)[active]
    muscle_fraction = np.asarray(volume.cell_data["MuscleFraction"], dtype=np.float64)[
        active
    ]
    names = np.asarray(volume.field_data["MuscleName"]).astype(str)
    samples = pv.read(state["samples"]["path"])
    glyphs = pv.read(state["glyphs"]["path"])
    assert samples.n_points == glyphs.n_cells == glyphs.n_lines == len(active)
    for arrays in (samples.point_data, glyphs.cell_data):
        np.testing.assert_array_equal(arrays["GlobalCellId"], active)
        np.testing.assert_array_equal(arrays["ActivationControlId"], controls)
        np.testing.assert_array_equal(arrays["MuscleId"], muscle_ids)
        np.testing.assert_allclose(
            arrays["MuscleFraction"], muscle_fraction, rtol=0.0, atol=0.0
        )
        np.testing.assert_array_equal(
            arrays["MuscleLabel"].astype(str), names[muscle_ids]
        )
        for array, reference in (
            ("LearnedAxisRest", expected["rest_axis"]),
            ("LearnedAxisSpatial", expected["spatial_axis"]),
            (
                "LearnedAxisDyadRest",
                np.einsum(
                    "ni,nj->nij", expected["rest_axis"], expected["rest_axis"]
                ).reshape(-1, 9),
            ),
            (
                "LearnedAxisDyadSpatial",
                np.einsum(
                    "ni,nj->nij", expected["spatial_axis"], expected["spatial_axis"]
                ).reshape(-1, 9),
            ),
            ("DeformationGradient", expected["F"].reshape(-1, 9)),
            ("DetF", expected["detF"]),
            ("AxisTransportStretch", expected["stretch"]),
            ("RestCentroid", expected["rest_centroids"]),
            ("qNormSquared", expected["strength"]),
            ("CommandedShorteningFraction", expected["shortening"]),
            ("CommandedShorteningPercent", 100.0 * expected["shortening"]),
            ("DisplayLineLengthM", DISPLAY_MAX_LENGTH_M * expected["shortening"]),
        ):
            # The independently solved F is compared to the renderer's explicit
            # inverse form; conditioning in a few inverted cells costs ~2e-13.
            np.testing.assert_allclose(arrays[array], reference, rtol=1e-10, atol=5e-12)
    np.testing.assert_allclose(
        samples.points, expected["centroids"], rtol=0.0, atol=1e-14
    )
    connectivity = np.asarray(glyphs.lines, dtype=np.int64).reshape(-1, 3)
    assert np.all(connectivity[:, 0] == 2)
    ends = glyphs.points[connectivity[:, 1:]]
    np.testing.assert_allclose(
        ends.mean(axis=1), expected["centroids"], rtol=0.0, atol=1e-14
    )
    delta = (
        DISPLAY_MAX_LENGTH_M
        * expected["shortening"][:, None]
        * expected["spatial_axis"]
    )
    np.testing.assert_allclose(ends[:, 1] - ends[:, 0], delta, rtol=0.0, atol=1e-14)
    lengths = np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1)
    np.testing.assert_allclose(
        lengths, DISPLAY_MAX_LENGTH_M * expected["shortening"], rtol=0.0, atol=1e-14
    )
    assert np.min(np.diff(lengths[np.argsort(expected["shortening"])])) >= -1.0e-14
    return (
        {
            "state": state["id"],
            "line_count": glyphs.n_lines,
            "region_count": int(np.unique(controls).size),
            "max_line_vector_error_m": float(
                np.max(np.abs(ends[:, 1] - ends[:, 0] - delta))
            ),
            "max_length_error_m": float(
                np.max(np.abs(lengths - DISPLAY_MAX_LENGTH_M * expected["shortening"]))
            ),
            "max_F_error": float(
                np.max(
                    np.abs(
                        glyphs.cell_data["DeformationGradient"]
                        - expected["F"].reshape(-1, 9)
                    )
                )
            ),
            "active_inverted_count": int(np.count_nonzero(expected["detF"] < 0.0)),
            "status": "passed",
        },
        expected,
        u,
    )


def _verify_visibility(
    state: dict[str, Any],
    volume: pv.UnstructuredGrid,
    active: np.ndarray,
    expected: dict[str, np.ndarray],
    u: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    cameras = {
        entry["id"]: entry["camera"]
        for entry in json.loads(CAMERAS.read_text())["views"]
    }
    deformed = volume.copy(deep=True)
    deformed.points = np.asarray(volume.points, dtype=np.float64) + u
    context = build_muscle_region_context(deformed)
    controls = np.asarray(volume.cell_data["ActivationControlId"], dtype=np.int64)[
        active
    ]
    boundary = np.isin(
        active, np.unique(context.combined.cell_data["SourceTetraGlobalCellId"])
    )
    results = []
    masks: dict[str, np.ndarray] = {}
    for view in VIEWS:
        path = DATA / "visibility" / f"{state['id']}--{view}.npz"
        with np.load(path, allow_pickle=False) as saved:
            width, height = np.asarray(saved["window_size"], dtype=np.int64)
            assert (width, height) == (1800, 1800)
            np.testing.assert_array_equal(saved["global_cell_ids"], active)
            np.testing.assert_array_equal(saved["control_ids"], controls)
            camera = cameras[view]
            forward = np.asarray(camera["focal_point"], dtype=np.float64) - np.asarray(
                camera["position"], dtype=np.float64
            )
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, np.asarray(camera["view_up"], dtype=np.float64))
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            relative = expected["centroids"] - np.asarray(
                camera["focal_point"], dtype=np.float64
            )
            projected = np.column_stack(
                (
                    width / 2
                    + relative @ right * height / (2 * camera["parallel_scale"]),
                    height / 2
                    + relative @ up * height / (2 * camera["parallel_scale"]),
                )
            ).astype(np.int32)
            np.testing.assert_array_equal(saved["projected_pixel_xy"], projected)
            x, y = projected.T
            inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            np.testing.assert_array_equal(saved["projected_inside"], inside)
            labels = saved["front_label_image"]
            assert labels.shape == (height, width)
            assert set(np.unique(labels)) <= set(range(-1, 103))
            front = np.full(len(active), -1, dtype=np.int16)
            front[inside] = labels[height - 1 - y[inside], x[inside]]
            np.testing.assert_array_equal(saved["front_control_ids"], front)
            mask = inside & (front == controls)
            np.testing.assert_array_equal(saved["mask"], mask)
            assert (
                int(mask.sum())
                == int(saved["retained_count"])
                == state["views"][view]["line_count_submitted"]
            )
            assert (
                int((mask & ~boundary).sum())
                == int(saved["retained_interior_count"])
                > 0
            )
            assert int((mask & boundary).sum()) == int(
                saved["retained_boundary_source_count"]
            )
            masks[view] = mask.copy()
        results.append(
            {
                "state": state["id"],
                "view": view,
                "retained_count": int(mask.sum()),
                "retained_interior_count": int((mask & ~boundary).sum()),
                "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                "status": "passed",
            }
        )
    return results, masks


def _verify_state_specific_masks(
    masks: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    """Demonstrate that deformed fields did not reuse the rest-frame masks."""
    rest_data = (
        ROOT
        / "exp/2026/09/09/activation-space-smoothness/data/74-visible-activation-glyphs/visibility"
    )
    checks = []
    for view in VIEWS:
        with np.load(rest_data / f"{view}.npz", allow_pickle=False) as saved:
            rest_mask = saved["mask"]
        hashes = {
            hashlib.sha256(mask[view].tobytes()).hexdigest() for mask in masks.values()
        }
        assert len(hashes) == len(masks)
        for mask in masks.values():
            assert not np.array_equal(mask[view], rest_mask)
        checks.append(
            {
                "view": view,
                "deformed_mask_count": len(hashes),
                "rest_mask_reused": False,
            }
        )
    return checks


def _verify_pngs(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Decode every submitted panel instead of trusting only its recorded hash."""
    checks = []
    for state in summary["states"]:
        for view in VIEWS:
            path = Path(state["views"][view]["file"]["path"])
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                assert image.format == "PNG"
                assert image.size == (1800, 1800)
            checks.append(
                {
                    "state": state["id"],
                    "view": view,
                    "bytes": path.stat().st_size,
                    "status": "passed",
                }
            )
    return checks


def main(cfg: Config) -> None:
    del cfg
    assert not OUTPUT.exists(), OUTPUT
    summary = json.loads((DATA / "summary.json").read_text())
    assert summary["status"] == "completed_activation_glyph_render"
    for collection in (summary["inputs"], summary["outputs"]):
        for record in collection.values():
            verify_record(record)
    for key in ("source", "muscle_context_source", "profile_source"):
        verify_record(summary[key]["live_at_generation"])
        verify_record(summary[key]["snapshot"])
        assert (
            summary[key]["live_at_generation"]["sha256"]
            == summary[key]["snapshot"]["sha256"]
        )
    volume, skin, tets, active, rest = _fixture()
    fields: list[dict[str, Any]] = []
    skins: list[dict[str, Any]] = []
    surfaces: list[dict[str, Any]] = []
    visibility: list[dict[str, Any]] = []
    masks: dict[str, dict[str, np.ndarray]] = {}
    for state in summary["states"]:
        for key in (
            "checkpoint",
            "trace",
            "run_summary",
            "config",
            "provenance",
            "settings",
            "samples",
            "glyphs",
            "deformed_skin",
            "deformed_muscle_regions",
        ):
            verify_record(state[key])
        field, expected, u = _verify_full_fields(state, volume, tets, active, rest)
        fields.append(field)
        skins.append({"state": state["id"], **_verify_skin(state, skin, rest, u)})
        surfaces.append(
            {
                "state": state["id"],
                **_verify_region_surface(state, volume, tets, active, rest, u),
            }
        )
        state_visibility, masks[state["id"]] = _verify_visibility(
            state, volume, active, expected, u
        )
        visibility.extend(state_visibility)
    state_specific_masks = _verify_state_specific_masks(masks)
    pngs = _verify_pngs(summary)
    with zipfile.ZipFile(DATA / "glyph-data.zip") as bundle:
        members = bundle.namelist()
        assert len(members) == len(set(members))
        for name in members:
            path = DATA / name
            assert path.is_file(), path
            with bundle.open(name) as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == digest(
                    path
                ), name
    OUTPUT.mkdir(parents=True)
    source = OUTPUT / Path(__file__).name
    shutil.copyfile(__file__, source)
    receipt = {
        "verified_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "status": "passed",
        "summary_sha256": digest(DATA / "summary.json"),
        "field_checks": fields,
        "skin_checks": skins,
        "surface_checks": surfaces,
        "visibility_checks": visibility,
        "state_specific_mask_checks": state_specific_masks,
        "png_checks": pngs,
        "zip_members": len(members),
        "verifier_sha256": digest(source),
        "checks": [
            "hashes and archive bytes",
            "one line for every active tetrahedron",
            "saved deformed centroids",
            "independent affine F solve and edge mapping",
            "F transport and spatial axes",
            "fixed 4.5 mm shortening scale",
            "exact GlobalPointId skin mapping",
            "deformed region-surface vertices",
            "independent camera projection and saved-label lookup",
            "state-specific masks distinct from rest and each other",
            "retained interior tetrahedra",
            "PNG decoding and 1800x1800 dimensions",
        ],
        "runtime": {
            "numpy": np.__version__,
            "pyvista": pv.__version__,
            "vtk": list(pv.vtk_version_info),
        },
    }
    destination = OUTPUT / "activation-glyphs.json"
    destination.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    cherries.log_output(destination)
    cherries.log_output(source)
    cherries.log_metric("verification/states", len(fields))
    cherries.log_metric("verification/views", len(visibility))


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
