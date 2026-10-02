"""Load the historical FEM exterior and static anatomy in its source frame."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[6]
VOLUME_PATH = (
    ROOT
    / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture/volume.vtu"
)
MELON_HEAD = Path(os.environ["APPLE_MELON_HEAD"])
CRANIUM_PATH = MELON_HEAD / "13-cranium.ply"
MANDIBLE_PATH = MELON_HEAD / "13-mandible.ply"
EYES_PATH = (
    ROOT
    / "exp/2026/09/21/joint-activation-material-mandible/data/rigid-eyes-001/eyes.vtp"
)


@dataclass
class ShapeScene:
    """Static meshes and source point mapping for one FEM reference frame."""

    volume: pv.UnstructuredGrid
    rest_points: np.ndarray
    exterior_rest: pv.PolyData
    exterior_point_ids: np.ndarray
    bones: dict[str, pv.PolyData]
    eyes: pv.PolyData


def load_static_context() -> ShapeScene:
    """Load full volume boundary, registered skull parts, and frozen eyes.

    The skull meshes are the exact `13-*.ply` inputs consumed by the TetWild
    volume builder. `eyes.vtp` records an unchanged registered source mesh in
    the same FEM world frame. No pose or alignment transform is applied.
    """
    volume = pv.read(VOLUME_PATH)
    if not isinstance(volume, pv.UnstructuredGrid):
        msg = f"expected UnstructuredGrid at {VOLUME_PATH}"
        raise TypeError(msg)
    rest_points = np.array(volume.points, copy=True)
    rest_points.setflags(write=False)
    exterior = volume.extract_surface(
        algorithm=None, pass_pointid=True, pass_cellid=False
    )
    if "vtkOriginalPointIds" not in exterior.point_data:
        msg = "extract_surface did not preserve original point IDs"
        raise RuntimeError(msg)
    point_ids = np.asarray(
        exterior.point_data.pop("vtkOriginalPointIds"), dtype=np.int64
    )
    if (
        len(point_ids) != exterior.n_points
        or np.any(point_ids < 0)
        or np.any(point_ids >= volume.n_points)
    ):
        msg = "exterior point map is invalid"
        raise ValueError(msg)
    exterior.point_data["GlobalPointId"] = point_ids

    bones = {
        "cranium": pv.read(CRANIUM_PATH),
        "mandible": pv.read(MANDIBLE_PATH),
    }
    eyes = pv.read(EYES_PATH)
    return ShapeScene(volume, rest_points, exterior, point_ids, bones, eyes)


def deformed_exterior(scene: ShapeScene, displacement: np.ndarray) -> pv.PolyData:
    """Return the complete volume exterior after transferring volume `u`.

    Exterior coordinates are evaluated directly as `X + u` at the original
    volume point IDs. No cropping, projection, smoothing, or remeshing occurs.
    """
    u = np.asarray(displacement)
    rest = scene.rest_points
    if u.shape != rest.shape or not np.isfinite(u).all():
        msg = f"displacement must be finite with shape {rest.shape}, got {u.shape}"
        raise ValueError(msg)
    surface = scene.exterior_rest.copy(deep=True)
    surface.points = rest[scene.exterior_point_ids] + u[scene.exterior_point_ids]
    return surface
