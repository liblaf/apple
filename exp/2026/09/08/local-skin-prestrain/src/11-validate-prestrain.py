"""Validate the saved local skin-prestrain encoding without a solve."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

GROUP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GROUP / "src"))
from skin_prestrain import (  # noqa: E402
    DEFAULT_CLOSEUPS,
    DEFAULT_FIXTURE,
    build_skin_prestrain,
)

OUTPUT = GROUP / "data/10-prestrain-field"


def digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    hasher = hashlib.sha256()
    hasher.update(value.dtype.str.encode())
    hasher.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    hasher.update(value.tobytes())
    return hasher.hexdigest()


def main() -> None:
    field = build_skin_prestrain(DEFAULT_FIXTURE, DEFAULT_CLOSEUPS)
    with np.load(OUTPUT / "skin-prestrain.npz", allow_pickle=False) as saved:
        npz = {key: np.asarray(saved[key]) for key in saved.files}
    required = {
        "vertex_weight",
        "w",
        "c",
        "activation_inv",
        "triangles",
        "point_ids",
        "triangle_area",
    }
    if set(npz) != required:
        raise AssertionError(f"unexpected NPZ schema: {sorted(npz)}")
    exact = {
        name: bool(np.array_equal(npz[name], getattr(field, name))) for name in required
    }
    if not all(exact.values()):
        raise AssertionError(f"saved NPZ differs from rebuilt field: {exact}")
    a = npz["activation_inv"]
    c = npz["c"]
    volume = pv.read(DEFAULT_FIXTURE / "volume.vtu")
    skin = pv.read(DEFAULT_FIXTURE / "skin.vtp")
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    target = np.asarray(skin.points) + np.asarray(volume.point_data["Smile"])[ids]
    rois = json.loads(DEFAULT_CLOSEUPS.read_text())["exploratory_target_space_rois"][
        "rois"
    ]
    masks = {
        name: np.logical_and.reduce(
            tuple(
                (target[:, axis] >= record["box_xyz_m"][letter][0])
                & (target[:, axis] <= record["box_xyz_m"][letter][1])
                for axis, letter in enumerate(("x", "y", "z"))
            )
        )
        for name, record in rois.items()
    }
    marked = np.logical_or.reduce(
        tuple(
            masks[name]
            for name in (
                "right_mouth_corner",
                "right_lateral_cheek",
                "right_lower_cheek_jaw",
            )
        )
    )
    protected = masks["right_nose_to_mouth"]
    protected_incident = protected[npz["triangles"]].any(axis=1)
    protected_boundary_triangles = protected_incident & (npz["w"] > 0.0)
    zero = build_skin_prestrain(DEFAULT_FIXTURE, DEFAULT_CLOSEUPS, max_contraction=0.0)
    validation = {
        "status": "passed_cpu_prestrain_field_validation",
        "scope": "Rebuilds only the material-fixed field; no forward solve or geometry update.",
        "npz_matches_rebuild_exactly": exact,
        "shape": {"activation_inv": list(a.shape), "w": list(npz["w"].shape)},
        "bounds": {
            "finite": bool(np.isfinite(a).all() and np.isfinite(c).all()),
            "weight_in_unit_interval": bool(
                np.all((npz["w"] >= 0.0) & (npz["w"] <= 1.0))
            ),
            "contraction_in_requested_interval": bool(np.all((c >= 0.0) & (c <= 0.01))),
        },
        "sign": {
            "diagonal_is_nonnegative_for_contraction": bool(np.all(a[:, :2] >= 0.0)),
            "shear_is_exact_zero": bool(np.all(a[:, 2] == 0.0)),
            "outside_support_is_identity": bool(np.all(a[npz["w"] == 0.0] == 0.0)),
            "outside_marked_vertex_support_is_exact_zero": bool(
                np.all(npz["vertex_weight"][~marked] == 0.0)
            ),
            "protected_vertices_are_exact_zero": bool(
                np.all(npz["vertex_weight"][protected] == 0.0)
            ),
            "protected_incident_triangle_count": int(protected_incident.sum()),
            "protected_incident_triangles_are_exact_zero": bool(
                np.all(npz["w"][protected_incident] == 0.0)
            ),
            "protected_incident_triangles_nonzero_count": int(
                protected_boundary_triangles.sum()
            ),
        },
        "null_scale": {
            "zero_scale_activation_is_exact_identity": bool(
                np.all(zero.activation_inv == 0.0)
            ),
            "zero_scale_contraction_is_exact_zero": bool(np.all(zero.c == 0.0)),
        },
        "hashes": {name: digest(value) for name, value in npz.items()},
    }
    checks = [
        *validation["npz_matches_rebuild_exactly"].values(),
        *validation["bounds"].values(),
        *(value for value in validation["sign"].values() if isinstance(value, bool)),
        *validation["null_scale"].values(),
    ]
    if not all(checks):
        raise AssertionError(f"failed validation: {validation}")
    (OUTPUT / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
