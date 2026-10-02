"""Export the completed normal-loss fixed-axis state as an actual tetrahedral mesh."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from experiment import Profile
from shape_scene import VOLUME_PATH

from liblaf import cherries


class Config(cherries.BaseConfig):
    source: Path = Path("51-visualization-checkpoints-002")
    output: Path = Path("53-paraview-l2-normal-fixed-001")


def main(cfg: Config) -> None:
    source = cherries.input(cfg.source)
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = source / "l2-normal/l2-normal-rankone_fixed/last.npz"
    manifest = json.loads((source / "l2-normal/manifest.json").read_text())
    expected = manifest["stages"][2]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == expected["last_sha256"]
    with np.load(checkpoint, allow_pickle=False) as saved:
        u, b = saved["u"], saved["B"]
        assert int(saved["step"]) == 200
        assert str(saved["mode"]) == "rankone_fixed"
        assert str(saved["activation_model"]) == "strain"
    mesh = pv.read(VOLUME_PATH)
    assert np.all(mesh.celltypes == pv.CellType.TETRA)
    rest = np.asarray(mesh.points).copy()
    tets = mesh.cells_dict[pv.CellType.TETRA]
    active = np.flatnonzero(mesh.cell_data["ActivationMask"])
    deformed = rest + u
    mesh.points = deformed
    for name in (
        "Activation",
        "ActivationInv",
        "ActivationFiber",
        "ActivationFiberConfidence",
    ):
        mesh.cell_data["Fixture" + name] = mesh.cell_data.pop(name)
    mesh.cell_data["ReferenceVolume"] = mesh.cell_data.pop("Volume")
    mesh.point_data["RestPosition"] = rest
    mesh.point_data["Displacement"] = u
    full_b = np.broadcast_to(np.eye(3), (mesh.n_cells, 3, 3)).copy()
    full_b[active] = b
    mesh.cell_data["ActivationInverseMatrix"] = full_b.reshape(-1, 9)
    dm = (rest[tets[:, 1:]] - rest[tets[:, :1]]).swapaxes(1, 2)
    ds = (deformed[tets[:, 1:]] - deformed[tets[:, :1]]).swapaxes(1, 2)
    det_f = np.linalg.det(ds @ np.linalg.inv(dm))
    mesh.cell_data["DetF"] = det_f
    mesh.cell_data["IsInverted"] = (det_f <= 0).astype(np.uint8)
    strength = np.zeros(mesh.n_cells)
    strength[active] = np.linalg.eigvalsh(b)[:, -1] - 1.0
    mesh.cell_data["PrincipalActivationAmplitude"] = strength
    path = out / "l2-normal-fixed-axis.vtu"
    mesh.save(path, binary=True)
    reopened = pv.read(path)
    np.testing.assert_array_equal(reopened.points, deformed)
    np.testing.assert_array_equal(reopened.cells, mesh.cells)
    np.testing.assert_array_equal(
        reopened.cell_data["ActivationInverseMatrix"], full_b.reshape(-1, 9)
    )
    receipt = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest,
        "step": 200,
        "points": mesh.n_points,
        "tetrahedra": mesh.n_cells,
        "inverted_cells": int((det_f <= 0).sum()),
        "geometry": "exact rest + saved displacement; original tetrahedra",
        "actual_activation": "ActivationInverseMatrix is saved B on active cells, identity elsewhere",
        "output": str(path.resolve()),
        "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (out / "export.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
