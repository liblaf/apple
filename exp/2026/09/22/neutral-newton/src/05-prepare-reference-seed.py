"""Prepare a cold seed with CPU geometry only and no equilibrium checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[6]
OLD_GROUP = ROOT / "exp/2026/09/21/joint-activation-material-mandible"
sys.path.insert(0, str(OLD_GROUP / "src"))

from joint_common import ProfileJoint
from reference_seed import prepare_reference_seed

from liblaf import cherries


class Config(cherries.BaseConfig):
    inputs: Path = OLD_GROUP / "data/simple-skin-forward-inputs-001"
    eyes: Path = OLD_GROUP / "data/rigid-eyes-001"
    output_dir: Path = cherries.output("reference-seed-001", mkdir=True)


def main(cfg: Config) -> None:
    with np.load(cfg.inputs / "geometry.npz") as arrays:
        geometry = SimpleNamespace(**{key: arrays[key] for key in arrays.files})
    with np.load(cfg.eyes / "eyes.npz") as arrays:
        eyes = SimpleNamespace(**{key: arrays[key] for key in arrays.files})
    volume = pv.read(cfg.inputs / "prepared/volume.vtu")
    physics = SimpleNamespace(
        points=np.asarray(volume.points),
        tets=np.asarray(volume.cells).reshape(-1, 5)[:, 1:],
        full_skull=SimpleNamespace(geometry=geometry),
        eyes=eyes,
    )
    _, receipt = prepare_reference_seed(physics, cfg.output_dir)
    cherries.log_metrics(receipt["final"])
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
