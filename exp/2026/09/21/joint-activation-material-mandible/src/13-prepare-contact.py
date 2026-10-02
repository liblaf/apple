"""Freeze and inspect the exact FEM bone/soft contact surface map."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import ipctk
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_contact import build_owned_contact
from joint_data import PreparedInputs

from liblaf import cherries

LOG = logging.getLogger(__name__)
COMPLETED = False


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    prepared_dir: Path = GROUP / "data/prepared"
    contact_spec: Path = GROUP / "data/contact/config.json"
    output_dir: Path = GROUP / "data/contact-initial-validation"


def main(cfg: Config) -> None:
    global COMPLETED  # noqa: PLW0603
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    archive_sources(cfg.output_dir)
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    mesh = pv.read(prepared.volume_path)
    fixed = np.unique(
        np.concatenate(
            [
                prepared.arrays[name]
                for name in (
                    "cranium_node_ids",
                    "mandible_node_ids",
                    "historical_fixed_node_ids",
                )
            ]
        )
    )
    started = time.perf_counter()
    contact, receipt = build_owned_contact(
        mesh, fixed, json.loads(cfg.contact_spec.read_text())
    )
    LOG.info("Built exact contact surfaces: %s", receipt)
    displacement = torch.zeros((mesh.n_points, 3), dtype=torch.float64)
    state = contact.state_at(displacement)
    reference = contact.diagnostics(state, displacement)
    intersecting = bool(
        ipctk.has_intersections(
            contact.collision_mesh,
            contact.collision_mesh.rest_positions,
            ipctk.LBVH(),
        )
    )
    result = {
        "schema": "joint-contact-initial-validation-v1",
        "success": reference["contact_numerically_valid"] and not intersecting,
        "contact_spec_sha256": sha256(cfg.contact_spec),
        "input_manifest_sha256": sha256(cfg.prepared_dir / "manifest.json"),
        "surface_map": receipt,
        "reference": reference,
        "initial_intersections": intersecting,
        "seconds": time.perf_counter() - started,
    }
    write_json(cfg.output_dir / "summary.json", result)
    assert result["success"], result
    LOG.info("Contact initialization passed: %s", reference)
    cherries.log_output(cfg.output_dir)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
    if not COMPLETED:
        raise SystemExit(1)
