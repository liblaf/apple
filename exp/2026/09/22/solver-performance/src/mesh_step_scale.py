# ruff: noqa: EM101, EM102, TRY003
"""Rest-mesh scale for Newton-only step caps.

The scale intentionally uses deformable FEM cells only.  Collision meshes also
contain rigid skull, mandible, and eye geometry, so their edges are not a tissue
step-length reference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import warp as wp

_EDGE_COLUMNS = {
    3: np.asarray(((0, 1), (0, 2), (1, 2)), dtype=np.int64),
    4: np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int64),
}


def _as_numpy(value: Any, *, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        result = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        result = value
    else:
        try:
            result = wp.to_torch(value).detach().cpu().numpy()
        except Exception as error:
            message = f"{name} must be a NumPy, Torch, or Warp array"
            raise TypeError(message) from error
    return np.asarray(result)


def _registry(model: Any) -> dict[str, Any]:
    wrapped = getattr(getattr(model, "warp_model", None), "__wrapped__", None)
    potentials = getattr(wrapped, "potentials", None)
    if not isinstance(potentials, dict) or not potentials:
        raise TypeError("model must expose a nonempty concrete Warp potential registry")
    return potentials


def mean_rest_edge_length(model: Any, rest_points: Any) -> float:
    """Return the positive mean length of unique global tri/tet FEM edges.

    ``rest_points`` is the deformable model's supplied rest geometry in metres.
    It is converted once.  Every registry potential must expose rank-two global
    triangle or tetrahedron ``cells``; other cell arities fail visibly.
    """
    points = _as_numpy(rest_points, name="rest_points")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("rest_points must have shape (points, 3)")
    if not np.isfinite(points).all():
        raise ValueError("rest_points contains non-finite coordinates")
    pieces: list[np.ndarray] = []
    for name, potential in _registry(model).items():
        cells = _as_numpy(
            getattr(potential, "cells", None), name=f"potential {name!r}.cells"
        )
        if cells.ndim != 2 or cells.shape[1] not in _EDGE_COLUMNS:
            raise TypeError(
                f"potential {name!r} must have triangle or tetrahedron cells"
            )
        if not np.issubdtype(cells.dtype, np.integer):
            raise TypeError(f"potential {name!r}.cells must be integer")
        cells = cells.astype(np.int64, copy=False)
        if cells.size and (cells.min() < 0 or cells.max() >= len(points)):
            raise ValueError(f"potential {name!r}.cells indexes outside rest_points")
        pairs = _EDGE_COLUMNS[cells.shape[1]]
        edges = cells[:, pairs].reshape(-1, 2)
        edges.sort(axis=1)
        pieces.append(edges[:, 0] * len(points) + edges[:, 1])
    if not pieces:
        raise ValueError("FEM registry has no cells")
    keys = np.unique(np.concatenate(pieces))
    if not len(keys):
        raise ValueError("FEM registry has no edges")
    lengths = np.linalg.vector_norm(
        points[keys // len(points)] - points[keys % len(points)], axis=1
    )
    mean = float(lengths.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("mean FEM rest edge length must be positive and finite")
    return mean
