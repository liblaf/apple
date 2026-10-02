# ruff: noqa: C901, EM101, EM102, FBT003, ICN001, PLR0915, RUF001, TRY003
"""Render the exact saved geometry of material cell 662949 on the CPU."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
CELL_ID = 662_949
FACES_UNORIENTED = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
EDGES = tuple(itertools.combinations(range(4), 2))


class Config(cherries.BaseConfig):
    output: Path = ROOT / "data" / "44-material-tet"
    fixture: Path = ROOT / "data" / "10-fixture" / "volume.vtu"
    baseline: Path = ROOT / "data" / "21-fiber-region-B-v2" / "final.npz"
    stable: Path = ROOT / "data" / "22-replays-v2" / "fat-stable-nu049" / "final.npz"
    neo: Path = ROOT / "data" / "22-replays-v2" / "fat-neo-nu049" / "final.npz"


@dataclass(frozen=True)
class StateSpec:
    key: str
    label: str
    source: Path | None
    color: str
    role: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def oriented_faces(rest_local_mm: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    """Orient the four exact rest faces outwards without changing their vertices."""
    center = rest_local_mm.mean(axis=0)
    result: list[tuple[int, int, int]] = []
    for raw in FACES_UNORIENTED:
        face = list(raw)
        points = rest_local_mm[face]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        if float(normal @ (points.mean(axis=0) - center)) < 0.0:
            face[1], face[2] = face[2], face[1]
        result.append(tuple(face))
    return tuple(result)


def rest_frame(
    rest: np.ndarray, global_ids: np.ndarray, fixed_mask: np.ndarray
) -> dict[str, Any]:
    """Choose a deterministic orthonormal frame from the four rest vertices only."""
    fixed_local = np.flatnonzero(fixed_mask)
    if fixed_local.size != 1:
        raise ValueError(f"cell {CELL_ID} must contain exactly one fixed vertex")
    fixed_index = int(fixed_local[0])
    free = [index for index in range(4) if index != fixed_index]
    origin = rest.mean(axis=0)

    z_axis = rest[free].mean(axis=0) - rest[fixed_index]
    z_axis /= np.linalg.norm(z_axis)
    anchor_index = min(free, key=lambda index: int(global_ids[index]))
    x_axis = rest[anchor_index] - rest[fixed_index]
    x_axis -= float(x_axis @ z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    basis = np.column_stack((x_axis, y_axis, z_axis))
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-14, rtol=0.0):
        raise AssertionError("rest-defined local basis is not orthonormal")
    if not np.isclose(np.linalg.det(basis), 1.0, atol=1e-14, rtol=0.0):
        raise AssertionError("rest-defined local basis is not right handed")

    rotation = basis.T
    global_to_local = np.eye(4)
    global_to_local[:3, :3] = 1000.0 * rotation
    global_to_local[:3, 3] = -1000.0 * rotation @ origin
    local_to_global = np.eye(4)
    local_to_global[:3, :3] = basis / 1000.0
    local_to_global[:3, 3] = origin
    return {
        "origin_global_m": origin,
        "basis_columns_global": basis,
        "global_to_local_affine_m_to_mm": global_to_local,
        "local_to_global_affine_mm_to_m": local_to_global,
        "fixed_vertex_local_index": fixed_index,
        "anchor_vertex_local_index": anchor_index,
        "definition": (
            "origin is the rest tet centroid; +z points from the fixed rest vertex "
            "to the opposite-face centroid; +x is the fixed-to-lowest-global-ID "
            "free-vertex vector projected normal to +z; +y=+z cross +x"
        ),
        "formula": "local_mm = 1000 * basis_columns_global.T @ (global_m - origin_global_m)",
    }


def to_local_mm(global_m: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    return (
        1000.0 * (global_m - frame["origin_global_m"]) @ frame["basis_columns_global"]
    )


def deformation_j(rest: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    rest_edges = np.column_stack(tuple(rest[i] - rest[0] for i in (1, 2, 3)))
    current_edges = np.column_stack(tuple(current[i] - current[0] for i in (1, 2, 3)))
    rest_det = float(np.linalg.det(rest_edges))
    current_det = float(np.linalg.det(current_edges))
    if rest_det <= 0.0:
        raise ValueError(f"cell {CELL_ID} has nonpositive rest orientation")
    return current_det / rest_det, current_det


def vertex_records(
    global_m: np.ndarray,
    local_mm: np.ndarray,
    point_ids: np.ndarray,
    global_ids: np.ndarray,
    original_ids: np.ndarray,
    fixed_mask: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "local_vertex_index": index,
            "mesh_point_id": int(point_ids[index]),
            "global_point_id": int(global_ids[index]),
            "vtk_original_point_id": int(original_ids[index]),
            "is_fixed": bool(fixed_mask[index]),
            "global_position_m": global_m[index].tolist(),
            "local_position_mm": local_mm[index].tolist(),
        }
        for index in range(4)
    ]


def plot_tet(
    ax: Any,
    local_mm: np.ndarray,
    faces: tuple[tuple[int, int, int], ...],
    spec: StateSpec,
    j_value: float,
    global_ids: np.ndarray,
    fixed_index: int,
    limits: tuple[float, float],
) -> None:
    polygons = [local_mm[np.asarray(face)] for face in faces]
    collection = Poly3DCollection(
        polygons,
        facecolors=spec.color,
        edgecolors="none",
        linewidths=0.0,
        alpha=0.24,
    )
    ax.add_collection3d(collection)
    for start, stop in EDGES:
        segment = local_mm[[start, stop]]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color="#172321",
            linewidth=1.8,
            solid_capstyle="round",
        )
    free = [index for index in range(4) if index != fixed_index]
    ax.scatter(
        local_mm[free, 0],
        local_mm[free, 1],
        local_mm[free, 2],
        color="#176b87",
        edgecolor="white",
        linewidth=0.55,
        s=44,
        depthshade=False,
        zorder=5,
    )
    fixed = local_mm[fixed_index]
    ax.scatter(
        [fixed[0]],
        [fixed[1]],
        [fixed[2]],
        marker="*",
        color="#c43a56",
        edgecolor="white",
        linewidth=0.65,
        s=145,
        depthshade=False,
        zorder=6,
    )
    for index, point in enumerate(local_mm):
        suffix = " fixed" if index == fixed_index else ""
        ax.text(
            point[0],
            point[1],
            point[2],
            f"  {int(global_ids[index])}{suffix}",
            color="#172321",
            fontsize=7.2,
            zorder=7,
        )
    ax.set_title(f"{spec.label}\n$J$ = {j_value:.6f}", fontsize=12, pad=10)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_zlim(limits)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_proj_type("ortho")
    ax.view_init(elev=22.0, azim=-62.0, roll=0.0)
    ax.set_xlabel("$x_L$ (mm)", fontsize=8, labelpad=3)
    ax.set_ylabel("$y_L$ (mm)", fontsize=8, labelpad=3)
    ax.set_zlabel("$z_L$ (mm)", fontsize=8, labelpad=3)
    ax.tick_params(labelsize=7, pad=0)
    ax.grid(True, linewidth=0.35, alpha=0.5)


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any], *, compact: bool = False) -> None:
    if compact:
        text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    else:
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def artifact_manifest(output: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact-manifest.json":
            files[str(path.relative_to(output))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    return {"files": files}


def main(cfg: Config) -> None:
    specs = (
        StateSpec("rest", "Rest", None, "#9aa6a1", "reference geometry"),
        StateSpec(
            "baseline_21_v2",
            "Saved baseline (21-v2)",
            cfg.baseline,
            "#d99135",
            "saved inverse endpoint; not recomputed",
        ),
        StateSpec(
            "stable_fat_nu049",
            "Stable fat ν=.49 replay",
            cfg.stable,
            "#2b9ca6",
            "saved fixed-control forward replay; not recomputed",
        ),
        StateSpec(
            "neo_fat_nu049",
            "Neo fat ν=.49 replay",
            cfg.neo,
            "#7e66ad",
            "saved fixed-control forward replay; not recomputed",
        ),
    )
    inputs = [cfg.fixture, *(spec.source for spec in specs if spec.source is not None)]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        cherries.log_input(path)

    fixture = pv.read(cfg.fixture)
    if fixture.n_points != 228_660 or fixture.n_cells != 1_146_517:
        raise ValueError("fixture topology differs from the verified face fixture")
    encoded = np.asarray(fixture.cells, dtype=np.int64).reshape(-1, 5)
    if np.any(encoded[:, 0] != 4):
        raise ValueError("fixture must contain tetrahedra only")
    tetrahedra = encoded[:, 1:]
    point_ids = tetrahedra[CELL_ID]
    rest = np.asarray(fixture.points, dtype=np.float64)[point_ids]
    global_ids = np.asarray(fixture.point_data["GlobalPointId"], dtype=np.int64)[
        point_ids
    ]
    original_ids = np.asarray(
        fixture.point_data["vtkOriginalPointIds"], dtype=np.int64
    )[point_ids]
    fixed_mask = np.asarray(fixture.point_data["IsFixed"], dtype=bool)[point_ids]
    frame = rest_frame(rest, global_ids, fixed_mask)

    state_positions: dict[str, np.ndarray] = {}
    state_inputs: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.source is None:
            state_positions[spec.key] = rest.copy()
            state_inputs[spec.key] = {
                "path": str(cfg.fixture.relative_to(ROOT)),
                "sha256": sha256(cfg.fixture),
                "geometry": "fixture rest coordinates",
            }
            continue
        with np.load(spec.source, allow_pickle=False) as saved:
            if "u" not in saved.files or saved["u"].shape != (fixture.n_points, 3):
                raise ValueError(f"{spec.source}: expected u with fixture point shape")
            state_positions[spec.key] = rest + np.asarray(saved["u"])[point_ids]
        state_inputs[spec.key] = {
            "path": str(spec.source.relative_to(ROOT)),
            "sha256": sha256(spec.source),
            "geometry": "fixture rest coordinates plus saved full-precision displacement",
        }

    local_positions = {
        key: to_local_mm(points, frame) for key, points in state_positions.items()
    }
    faces = oriented_faces(local_positions["rest"])
    rest_det = deformation_j(rest, rest)[1]
    states: list[dict[str, Any]] = []
    for spec in specs:
        points = state_positions[spec.key]
        local = local_positions[spec.key]
        j_value, current_det = deformation_j(rest, points)
        if j_value <= 0.0:
            raise ValueError(f"{spec.key}: inverted target tet")
        states.append(
            {
                "key": spec.key,
                "label": spec.label,
                "role": spec.role,
                "source": state_inputs[spec.key],
                "J_det_Ds_over_det_Dm": j_value,
                "signed_six_volume_m3": current_det,
                "volume_m3": current_det / 6.0,
                "volume_mm3": current_det * 1.0e9 / 6.0,
                "vertices": vertex_records(
                    points,
                    local,
                    point_ids,
                    global_ids,
                    original_ids,
                    fixed_mask,
                ),
                "edge_lengths_mm": [
                    {
                        "local_vertex_indices": [start, stop],
                        "length": float(np.linalg.norm(local[stop] - local[start])),
                    }
                    for start, stop in EDGES
                ],
            }
        )

    all_local = np.concatenate(tuple(local_positions.values()))
    bound_center = float((all_local.min() + all_local.max()) / 2.0)
    bound_radius = float((all_local.max() - all_local.min()) / 2.0 * 1.18)
    limits = (bound_center - bound_radius, bound_center + bound_radius)

    if cfg.output.exists():
        shutil.rmtree(cfg.output)
    cfg.output.mkdir(parents=True)
    fig = plt.figure(figsize=(18.5, 6.2), facecolor="#fbfaf7")
    for panel, (spec, state) in enumerate(zip(specs, states, strict=True), start=1):
        ax = fig.add_subplot(1, 4, panel, projection="3d", facecolor="#fbfaf7")
        plot_tet(
            ax,
            local_positions[spec.key],
            faces,
            spec,
            state["J_det_Ds_over_det_Dm"],
            global_ids,
            frame["fixed_vertex_local_index"],
            limits,
        )
    fig.suptitle(
        "Exact saved geometry of tetrahedral cell 662949",
        fontsize=16,
        fontweight="semibold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.018,
        (
            "Shared rest-defined orthonormal frame and bounds · coordinates in mm · "
            "saved displacement at 1× · exact planar faces and straight edges · "
            "★ fixed vertex 111321"
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#172321",
    )
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.11, top=0.80, wspace=0.08)
    png = cfg.output / "cell-662949-four-states.png"
    pdf = cfg.output / "cell-662949-four-states.pdf"
    fig.savefig(
        png,
        dpi=240,
        facecolor=fig.get_facecolor(),
        metadata={"Software": "44-render-material-tet.py"},
    )
    fig.savefig(
        pdf,
        facecolor=fig.get_facecolor(),
        metadata={
            "Title": "Exact saved geometry of tetrahedral cell 662949",
            "Creator": "44-render-material-tet.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)

    report = {
        "schema_version": 1,
        "artifact_role": "bounded CPU visualization of exact saved tetrahedron geometry",
        "cell": {
            "fixture_cell_id": CELL_ID,
            "vtk_original_cell_id": int(
                np.asarray(fixture.cell_data["vtkOriginalCellIds"])[CELL_ID]
            ),
            "mesh_point_ids_in_cell_order": point_ids.tolist(),
            "global_point_ids_in_cell_order": global_ids.tolist(),
            "vtk_original_point_ids_in_cell_order": original_ids.tolist(),
            "fixed_vertex_local_index": frame["fixed_vertex_local_index"],
            "fixed_global_point_id": int(global_ids[frame["fixed_vertex_local_index"]]),
            "rest_signed_six_volume_m3": rest_det,
            "rest_volume_m3": rest_det / 6.0,
        },
        "coordinate_frame": frame,
        "topology": {
            "triangles_local_vertex_indices_outward": faces,
            "edges_local_vertex_indices": EDGES,
        },
        "render": {
            "camera": {
                "projection": "orthographic",
                "elevation_deg": 22.0,
                "azimuth_deg": -62.0,
                "roll_deg": 0.0,
            },
            "shared_axis_limits_mm": list(limits),
            "geometry_operations": {
                "displacement_scale": 1.0,
                "smoothing": False,
                "synthetic_displacement": False,
                "neighboring_cell_context": False,
            },
        },
        "inputs": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in inputs
        },
        "states": states,
    }
    write_json(cfg.output / "cell-662949-geometry.json", json_ready(report))

    browser = {
        "schema_version": 1,
        "cell_id": CELL_ID,
        "units": "mm",
        "coordinate_frame": "shared_rest_defined_local_orthonormal",
        "global_point_ids_in_cell_order": global_ids.tolist(),
        "fixed_vertex_local_index": frame["fixed_vertex_local_index"],
        "states": [
            {
                "key": spec.key,
                "label": spec.label,
                "J": state["J_det_Ds_over_det_Dm"],
                "vertices": local_positions[spec.key].tolist(),
                "triangles": [list(face) for face in faces],
            }
            for spec, state in zip(specs, states, strict=True)
        ],
    }
    write_json(cfg.output / "cell-662949-browser.json", browser, compact=True)
    source_dir = cfg.output / "sources"
    source_dir.mkdir()
    shutil.copy2(Path(__file__), source_dir / Path(__file__).name)
    write_json(cfg.output / "artifact-manifest.json", artifact_manifest(cfg.output))

    cherries.log_output(cfg.output)
    cherries.log_metrics(
        {
            "tet/cell_id": CELL_ID,
            "tet/baseline_J": states[1]["J_det_Ds_over_det_Dm"],
            "tet/stable_nu049_J": states[2]["J_det_Ds_over_det_Dm"],
            "tet/neo_nu049_J": states[3]["J_det_Ds_over_det_Dm"],
            "tet/solver_runs": 0,
        }
    )
    logger.info("Wrote exact four-state material-tet artifact to %s", cfg.output)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
