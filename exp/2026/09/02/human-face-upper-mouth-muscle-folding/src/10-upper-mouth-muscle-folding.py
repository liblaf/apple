"""Post-hoc upper-mouth muscle folding receipt from saved face inverse states.

This intentionally reads only the materialized endpoint and VTKHDF history; it
does not instantiate or rerun the inverse/forward physics.
"""

from __future__ import annotations

# ruff: noqa: C901, PLR0915
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pydantic_settings as ps
import pyvista as pv
from scipy.spatial import cKDTree

from liblaf import cherries

CASE = "20-human-face-smile-no-skin-lr3"
MUSCLE_ID = 254
MUSCLE_NAME = "Orbicularis oris001_Head_muscles_0"
STEPS = tuple(range(201))
FPS = 30
RADIUS_M = 0.006
SIZE = (1800, 600)
VIDEO_SIZE = (1200, 900)


class Config(cherries.BaseConfig):
    """Fixed evidence source and reproducible local-section policy."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    source_endpoint: Path = (
        Path(__file__).resolve().parents[4]
        / "06/17/human-face-smile-prestrain-v2/data"
        / f"{CASE}.vtu"
    )
    source_history: Path = (
        Path(__file__).resolve().parents[4]
        / "06/17/human-face-smile-prestrain-v2/data"
        / f"{CASE}-steps.vtkhdf"
    )
    output_dir: Path = cherries.output("10-upper-mouth-muscle-folding", mkdir=True)


def fail(message: str) -> None:
    raise ValueError(message)


def digest(path: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": h.hexdigest(),
    }


def digest_paths(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode())
        h.update(digest(path)["sha256"].encode())
    return h.hexdigest()


def tets(grid: pv.UnstructuredGrid) -> np.ndarray:
    if not np.all(grid.celltypes == pv.CellType.TETRA):
        fail("source mesh is not tetra-only")
    packed = np.asarray(grid.cells)
    if packed.size != grid.n_cells * 5 or not np.all(packed.reshape(-1, 5)[:, 0] == 4):
        fail("source mesh has invalid tetra connectivity")
    return packed.reshape(-1, 5)[:, 1:]


def ainv_det(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != 6:
        fail(f"ActivationInv shape must be (n, 6), got {values.shape}")
    a = np.zeros((values.shape[0], 3, 3), dtype=float)
    a[:, 0, 0], a[:, 1, 1], a[:, 2, 2] = (
        1 + values[:, 0],
        1 + values[:, 1],
        1 + values[:, 2],
    )
    a[:, 0, 1] = a[:, 1, 0] = values[:, 3]
    a[:, 1, 2] = a[:, 2, 1] = values[:, 4]
    a[:, 0, 2] = a[:, 2, 0] = values[:, 5]
    return np.linalg.det(a)


def det_f(
    reference: np.ndarray, deformed: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ref = np.stack(
        [reference[cells[:, i]] - reference[cells[:, 0]] for i in (1, 2, 3)], axis=2
    )
    deformed_edges = np.stack(
        [deformed[cells[:, i]] - deformed[cells[:, 0]] for i in (1, 2, 3)], axis=2
    )
    reference_det = np.linalg.det(ref)
    if np.any(~np.isfinite(reference_det)) or np.any(
        np.abs(reference_det) <= np.finfo(float).tiny
    ):
        fail("selected reference tetrahedron is degenerate")
    return np.linalg.det(deformed_edges) / reference_det, np.abs(reference_det) / 6


def compact(
    points: np.ndarray,
    cells: np.ndarray,
    ids: np.ndarray,
    fields: dict[str, np.ndarray],
) -> pv.UnstructuredGrid:
    used = np.unique(cells[ids].ravel())
    local = np.searchsorted(used, cells[ids])
    packed = np.column_stack((np.full(ids.size, 4), local)).astype(np.int64).ravel()
    result = pv.UnstructuredGrid(
        packed, np.full(ids.size, pv.CellType.TETRA, dtype=np.uint8), points[used]
    )
    result.cell_data["SourceCellId"] = ids.astype(np.int64)
    for name, values in fields.items():
        result.cell_data[name] = values
    return result


def metrics(volume: np.ndarray, f: np.ndarray, a: np.ndarray) -> dict[str, Any]:
    g = f * a
    masks = {
        "f_negative": f < 0,
        "ainv_negative": a < 0,
        "g_negative": g < 0,
        "double_inverted": (f < 0) & (a < 0),
    }
    total = float(volume.sum())
    output: dict[str, Any] = {
        "cells": int(f.size),
        "rest_volume": total,
        "min_det_f": float(f.min()),
        "min_det_ainv": float(a.min()),
        "min_det_g": float(g.min()),
    }
    for key, mask in masks.items():
        output[f"{key}_cells"] = int(mask.sum())
        output[f"{key}_rest_volume_fraction"] = float(volume[mask].sum() / total)
    return output


def onset(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    hits = [r["step"] for r in rows if r[f"{key}_cells"]]
    return {
        "first_onset_step": min(hits) if hits else None,
        "last_present_step": max(hits) if hits else None,
        "persistent_through_last": bool(hits and max(hits) == rows[-1]["step"]),
    }


def hdf_state(
    hdf: h5py.File, step: int, point_ids: np.ndarray, cell_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    po = int(hdf["VTKHDF/Steps/PointDataOffsets/DeformedPoint"][step])
    co = int(hdf["VTKHDF/Steps/CellDataOffsets/ActivationInv"][step])
    points = np.asarray(
        hdf["VTKHDF/PointData/DeformedPoint"][po + point_ids], dtype=float
    )
    activation = np.asarray(
        hdf["VTKHDF/CellData/ActivationInv"][co + cell_ids], dtype=float
    )
    return points, activation


def camera(bounds: tuple[float, float, float, float, float, float]) -> dict[str, Any]:
    span = np.asarray(
        (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    )
    center = np.asarray(
        (
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        )
    )
    return {
        "position": (center + 2.4 * max(span) * np.asarray((1.0, 0.25, 0.6))).tolist(),
        "focal_point": center.tolist(),
        "view_up": [0, 1, 0],
        "parallel_scale": float(0.68 * max(span)),
    }


def set_camera(plot: pv.Plotter, spec: dict[str, Any]) -> None:
    plot.set_background("white")
    plot.enable_parallel_projection()
    (
        plot.camera.position,
        plot.camera.focal_point,
        plot.camera.up,
        plot.camera.parallel_scale,
    ) = spec["position"], spec["focal_point"], spec["view_up"], spec["parallel_scale"]


def panel(
    plot: pv.Plotter,
    ref: pv.UnstructuredGrid,
    state: pv.UnstructuredGrid,
    scalar: str,
    clim: tuple[float, float],
    step: int,
    spec: dict[str, Any],
) -> None:
    set_camera(plot, spec)
    plot.add_mesh(ref, color="#9aa0a6", style="wireframe", line_width=0.8, opacity=0.65)
    plot.add_mesh(
        state,
        scalars=scalar,
        clim=clim,
        cmap="coolwarm",
        show_edges=True,
        edge_color="#202124",
        line_width=0.25,
        scalar_bar_args={"title": scalar, "fmt": "%.2g"},
    )
    bad = state.threshold((0.5, 1.5), scalars="DoubleInverted")
    if bad.n_cells:
        plot.add_mesh(
            bad, color="#ff00b8", show_edges=True, edge_color="black", line_width=1.2
        )
    suffix = " | magenta: F<0 & Ainv<0" if scalar == "DetG" else ""
    plot.add_text(
        f"step {step:03d} / 200 | {scalar}{suffix}\ngray: reference | no exaggeration",
        font_size=12,
        color="black",
        position="upper_left",
    )


def render_static(
    out: Path,
    whole: pv.UnstructuredGrid,
    ref: pv.UnstructuredGrid,
    endpoint: pv.UnstructuredGrid,
    face: pv.UnstructuredGrid,
    spec: dict[str, Any],
    ranges: dict[str, tuple[float, float]],
) -> list[Path]:
    render = out / "render"
    render.mkdir()
    # The global view supplies spatial evidence: orange = all orbicularis oris, magenta = exact local upper-lip section.
    p = pv.Plotter(off_screen=True, window_size=(2600, 1800), lighting="light kit")
    p.set_background("white")
    surface = face.copy(deep=True)
    surface.points = np.asarray(face.point_data["DeformedPoint"], dtype=float)
    surface = surface.extract_surface(algorithm="dataset_surface")
    p.add_mesh(surface, color="#dcc5b0", opacity=0.58, smooth_shading=True)
    p.add_mesh(whole, color="#f39c12", opacity=0.34)
    p.add_mesh(
        endpoint, color="#d91c5c", show_edges=True, edge_color="black", line_width=0.7
    )
    p.add_text(
        "Saved bumpy endpoint: upper mouth / top lip\norange: Orbicularis oris id254, MuscleFraction >= 0.5 | magenta: exact 6 mm upper-lip local section",
        font_size=16,
        color="black",
    )
    p.view_xy(negative=False)
    p.camera.zoom(1.55)
    global_path = render / "upper-mouth-global-context.png"
    p.screenshot(global_path)
    p.close()
    p = pv.Plotter(
        shape=(1, 3), off_screen=True, window_size=SIZE, lighting="light kit"
    )
    for i, name in enumerate(("DetF", "DetAinv", "DetG")):
        p.subplot(0, i)
        panel(p, ref, endpoint, name, ranges[name], 194, spec)
    mechanism = render / "upper-mouth-primary-mechanism.png"
    p.screenshot(mechanism)
    p.close()
    return [global_path, mechanism]


def render_video(
    out: Path,
    frames: list[Path],
    ref: pv.UnstructuredGrid,
    spec: dict[str, Any],
    ranges: dict[str, tuple[float, float]],
) -> tuple[Path, dict[str, Any], list[Path]]:
    png_dir = out / "video" / "frames"
    png_dir.mkdir(parents=True)
    images = []
    for step, path in enumerate(frames):
        state = pv.read(path)
        p = pv.Plotter(off_screen=True, window_size=VIDEO_SIZE, lighting="light kit")
        # A single mobile-legible deformation panel keeps every saved state while
        # avoiding a threefold rendering cost.  DetAinv/DetG remain in each VTU
        # frame and in the static three-panel mechanism figure.
        panel(p, ref, state, "DetF", ranges["DetF"], step, spec)
        target = png_dir / f"frame-{step:03d}.png"
        p.screenshot(target)
        p.close()
        images.append(target)
    video = out / "video" / "upper-mouth-muscle-evolution.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(png_dir / "frame-%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,pix_fmt,r_frame_rate,nb_read_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    facts = json.loads(probe.stdout)
    stream = facts["streams"][0]
    if (
        stream["codec_name"] != "h264"
        or stream["pix_fmt"] != "yuv420p"
        or stream["r_frame_rate"] != "30/1"
        or int(stream["nb_read_frames"]) != len(images)
    ):
        fail(f"video contract failed: {facts}")
    return video, facts, images


def main(config: Config) -> None:
    endpoint_path, history_path = (
        config.source_endpoint.resolve(),
        config.source_history.resolve(),
    )
    if not endpoint_path.is_file() or not history_path.is_file():
        fail("saved source endpoint/history absent")
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError(config.output_dir)
    face = pv.read(endpoint_path)
    if not isinstance(face, pv.UnstructuredGrid):
        fail("endpoint is not UnstructuredGrid")
    names = np.asarray(face.field_data["MuscleName"])
    if names[MUSCLE_ID] != MUSCLE_NAME:
        fail(f"muscle id mapping differs: {names[MUSCLE_ID]!r}")
    reference = np.asarray(face.points, dtype=float)
    deformed = np.asarray(face.point_data["DeformedPoint"], dtype=float)
    cells = tets(face)
    active = np.asarray(face.cell_data["ActivationMask"], dtype=bool)
    muscle = np.asarray(face.cell_data["MuscleId"], dtype=np.int64)
    muscle_fraction = np.asarray(face.cell_data["MuscleFraction"], dtype=float)
    whole_ids = np.flatnonzero(
        active & (muscle == MUSCLE_ID) & (muscle_fraction >= 0.5)
    )
    if whole_ids.size < 1:
        fail("selected muscle has no active cells")
    f_all, volume_all = det_f(reference, deformed, cells)
    a_all = ainv_det(np.asarray(face.cell_data["ActivationInv"], dtype=float))
    centroids = reference[cells].mean(axis=1)
    # Define upper lip directly from the saved IsLip surface marking, not a guessed
    # anatomy coordinate.  The top quartile in the vertical (Y) coordinate is the
    # visible top-lip/philtrum-side band; only sufficiently-muscular source cells
    # within 2 mm of it may nominate the local folding seed.
    lip_points = reference[np.asarray(face.point_data["IsLip"], dtype=bool)]
    upper_lip_quantile = float(np.quantile(lip_points[:, 1], 0.75))
    upper_lip = lip_points[lip_points[:, 1] >= upper_lip_quantile]
    if upper_lip.size == 0:
        fail("saved IsLip marking has no upper-lip points")
    upper_lip_distance = cKDTree(upper_lip).query(centroids)[0]
    upper_lip_candidates = np.flatnonzero(
        (active & (muscle == MUSCLE_ID) & (muscle_fraction >= 0.5))
        & (upper_lip_distance <= 0.002)
    )
    if upper_lip_candidates.size < 2:
        fail("no sufficiently close Orbicularis oris upper-lip candidates")
    seed = int(upper_lip_candidates[np.argmin(f_all[upper_lip_candidates])])
    local_ids = np.flatnonzero(
        (active & (muscle == MUSCLE_ID) & (muscle_fraction >= 0.5))
        & (np.linalg.norm(centroids - centroids[seed], axis=1) <= RADIUS_M)
    )
    if local_ids.size < 2 or seed not in local_ids:
        fail("local section selection failed")
    point_ids = np.unique(cells[local_ids].ravel())
    local_cells = cells[local_ids]
    # Make local connectivity once, in global coordinates, so the history uses the exact same source ids.
    fields = {
        "RestVolume": volume_all[local_ids],
        "DetF": f_all[local_ids],
        "DetAinv": a_all[local_ids],
        "DetG": f_all[local_ids] * a_all[local_ids],
        "DoubleInverted": ((f_all[local_ids] < 0) & (a_all[local_ids] < 0)).astype(
            np.int8
        ),
    }
    ref = compact(reference, cells, local_ids, {"RestVolume": fields["RestVolume"]})
    whole = compact(reference, cells, whole_ids, {})
    endpoint = compact(deformed, cells, local_ids, fields)
    out = config.output_dir
    out.mkdir(parents=True)
    history_dir = out / "history" / "frames"
    history_dir.mkdir(parents=True)
    rows = []
    frame_paths = []
    with h5py.File(history_path, "r") as hdf:
        steps = np.asarray(hdf["VTKHDF/FieldData/inverse_step"], dtype=int)
        if not np.array_equal(steps, np.asarray(STEPS)):
            fail("history does not have exact step 0..200")
        for step in STEPS:
            points, activation = hdf_state(hdf, step, point_ids, local_ids)
            global_points = np.empty((point_ids.size, 3))
            global_points[:] = points
            # cells reference global IDs; reconstruct compact grid directly with local point ids.
            local_conn = np.searchsorted(point_ids, local_cells)
            packed = (
                np.column_stack((np.full(local_ids.size, 4), local_conn))
                .astype(np.int64)
                .ravel()
            )
            grid = pv.UnstructuredGrid(
                packed,
                np.full(local_ids.size, pv.CellType.TETRA, dtype=np.uint8),
                global_points,
            )
            ref_edges = np.stack(
                [
                    reference[local_cells[:, i]] - reference[local_cells[:, 0]]
                    for i in (1, 2, 3)
                ],
                axis=2,
            )
            deformed_edges = np.stack(
                [
                    global_points[np.searchsorted(point_ids, local_cells[:, i])]
                    - global_points[np.searchsorted(point_ids, local_cells[:, 0])]
                    for i in (1, 2, 3)
                ],
                axis=2,
            )
            f = np.linalg.det(deformed_edges) / np.linalg.det(ref_edges)
            a = ainv_det(activation)
            g = f * a
            grid.cell_data["SourceCellId"] = local_ids
            grid.cell_data["RestVolume"] = volume_all[local_ids]
            grid.cell_data["DetF"] = f
            grid.cell_data["DetAinv"] = a
            grid.cell_data["DetG"] = g
            grid.cell_data["DoubleInverted"] = ((f < 0) & (a < 0)).astype(np.int8)
            path = history_dir / f"step-{step:03d}.vtu"
            grid.save(path)
            frame_paths.append(path)
            row = {"step": step, **metrics(volume_all[local_ids], f, a)}
            rows.append(row)
            cherries.set_step(step)
            cherries.log_metrics(
                {
                    f"upper_mouth/{k}": v
                    for k, v in row.items()
                    if isinstance(v, float | int)
                }
            )
    with (out / "history" / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ranges = {
        name: (
            -max(
                abs(min(float(pv.read(f).cell_data[name].min()) for f in frame_paths)),
                abs(max(float(pv.read(f).cell_data[name].max()) for f in frame_paths)),
            ),
            max(
                abs(min(float(pv.read(f).cell_data[name].min()) for f in frame_paths)),
                abs(max(float(pv.read(f).cell_data[name].max()) for f in frame_paths)),
            ),
        )
        for name in ("DetF", "DetAinv", "DetG")
    }
    # Stable camera covers rest and every exact saved history state.
    all_bounds = np.vstack(
        [ref.bounds] + [pv.read(path).bounds for path in frame_paths]
    )
    bounds = (
        all_bounds[:, 0].min(),
        all_bounds[:, 1].max(),
        all_bounds[:, 2].min(),
        all_bounds[:, 3].max(),
        all_bounds[:, 4].min(),
        all_bounds[:, 5].max(),
    )
    spec = camera(tuple(map(float, bounds)))
    static = render_static(out, whole, ref, endpoint, face, spec, ranges)
    video, probe, images = render_video(out, frame_paths, ref, spec, ranges)
    summary = {
        "status": "ok",
        "purpose": "post-hoc saved-state analysis; no physics rerun",
        "source": {"endpoint": digest(endpoint_path), "history": digest(history_path)},
        "anatomy": {
            "muscle_id": MUSCLE_ID,
            "muscle_name": MUSCLE_NAME,
            "selection": "ActivationMask && MuscleId == 254 && MuscleFraction >= 0.5",
            "whole_muscle": metrics(
                volume_all[whole_ids], f_all[whole_ids], a_all[whole_ids]
            ),
            "upper_lip_proximity": {
                "surface_definition": "saved IsLip points with Y at or above its 75th percentile",
                "y_quantile": 0.75,
                "y_threshold": upper_lip_quantile,
                "surface_point_count": int(upper_lip.shape[0]),
                "orbicularis_centroid_distance_m": {
                    "minimum": float(upper_lip_distance[whole_ids].min()),
                    "median": float(np.median(upper_lip_distance[whole_ids])),
                },
                "candidate_distance_limit_m": 0.002,
                "candidate_count": int(upper_lip_candidates.size),
            },
        },
        "local_section": {
            "criterion": "seed is the minimum-DetF active Orbicularis oris tetrahedron with MuscleFraction >= 0.5 and reference centroid <= 0.002 m from the saved upper IsLip surface; section is the same-muscle reference-centroid 0.006 m ball around that seed, capturing the directly adjacent inner material",
            "radius_m": RADIUS_M,
            "seed_source_cell_id": seed,
            "seed_reference_centroid": centroids[seed].tolist(),
            "seed_upper_lip_distance_m": float(upper_lip_distance[seed]),
            "source_cell_ids": local_ids.tolist(),
            "source_cell_count": int(local_ids.size),
            "source_point_count": int(point_ids.size),
            "endpoint": metrics(
                volume_all[local_ids], f_all[local_ids], a_all[local_ids]
            ),
        },
        "history": {
            "frame_count": len(frame_paths),
            "inverse_steps": {"first": 0, "last": 200, "exact_consecutive": True},
            "onset_persistence": {
                key: onset(rows, key)
                for key in (
                    "f_negative",
                    "ainv_negative",
                    "g_negative",
                    "double_inverted",
                )
            },
            "step_194": rows[194],
            "last_step": rows[-1],
        },
        "sign_convention": {
            "f_negative": "DetF < 0",
            "ainv_negative": "DetAinv < 0",
            "g_negative": "DetG = DetF * DetAinv < 0",
            "double_inverted": "DetF < 0 and DetAinv < 0",
        },
        "limits": {
            "inverse_converged": bool(face.field_data["inverse/converged"][0]),
            "forward_failures": int(face.field_data["inverse/forward_fail_count"][0]),
            "causality": "nearest/anatomically located muscle and determinant co-occurrence do not prove this muscle alone caused the visible surface bumpiness",
        },
        "render": {
            "static": [digest(p) for p in static],
            "video": digest(video),
            "video_probe": probe,
            "render_contract": {
                "fps": FPS,
                "source_frame_count": len(frame_paths),
                "video_frame_count": len(images),
                "rendered_video_scalar": "DetF; DetAinv and DetG are retained per VTK frame and shown in static mechanism figure",
                "one_frame_per_saved_inverse_step": True,
                "no_interpolation_or_duplication": True,
                "no_deformation_exaggeration": True,
            },
            "frames_sha256": digest_paths(frame_paths),
            "png_sha256": digest_paths(images),
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    cherries.main(main)
