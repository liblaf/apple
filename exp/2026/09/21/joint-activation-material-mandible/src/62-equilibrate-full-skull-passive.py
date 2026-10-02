"""Equilibrate passive full-skull contact before applying fitted baseline stresses."""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs
from joint_equilibrium import Equilibrium, ForwardConvergenceError, configure_cuda
from joint_fields import BULK_TISSUES, research_informed_material_config
from joint_full_skull_contact import (
    FullSkullJointPhysics,
    load_admitted_initialization,
    load_full_skull_geometry,
)

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    prepared_dir: Path = GROUP / "data/prepared"
    audit: Path = GROUP / "data/full-skull-initialization-audit-001/summary.json"
    admission: Path = (
        GROUP / "data/full-skull-initialization-candidate-002/admission.json"
    )
    max_pncg_steps: int = 5000
    output_dir: Path = GROUP / "data/full-skull-passive-equilibrium-001"


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    configure_cuda()
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    audit = json.loads(cfg.audit.read_text())
    geometry = load_full_skull_geometry(Path(audit["geometry"]["path"]), cfg.audit)
    admission = json.loads(cfg.admission.read_text())
    seed = torch.as_tensor(
        load_admitted_initialization(admission, geometry), device="cuda"
    )
    materials = research_informed_material_config()["materials"]
    skin = materials["skin"]
    contact = {
        "schema": "joint-full-source-bone-contact-v1",
        "enabled": True,
        "surface_selection": "pure-soft-vs-complete-source-bones",
        "attachment_policy": "no-source-triangle-exclusions",
        "friction": "frictionless",
        "dhat_m": 0.0001,
        "stiffness_mpa": 0.01,
    }
    physics = FullSkullJointPhysics(
        prepared.volume_path,
        prepared.skin_path,
        prepared.arrays,
        bulk_young_mpa={name: materials[name]["young_mpa"] for name in BULK_TISSUES},
        bulk_nu={name: materials[name]["poisson"] for name in BULK_TISSUES},
        skin_young_mpa=skin["reference_map"]["young_mpa"],
        skin_nu=skin["poisson"],
        thickness_m=skin["thickness_m"],
        full_skull_geometry=geometry,
        full_skull_admission=admission,
        full_skull_contact_config=contact,
        rtol=1e-3,
        atol=1e-9,
        max_steps=cfg.max_pncg_steps,
        forward_method="pncg",
        adjoint_rtol=1e-7,
    )
    protocol = {
        "schema": "joint-full-skull-passive-equilibration-protocol-v1",
        "purpose": "passive-contact initialization before baseline stress continuation",
        "geometry_sha256": geometry.geometry_sha256,
        "admission_sha256": sha256(cfg.admission),
        "materials": materials,
        "all_baseline_and_activation_stresses": 0,
        "skin_stiffness_multiplier": 1,
        "jaw_pose_rad_m": [0] * 6,
        "phases": [
            {
                "name": "pncg_globalization",
                "rtol": 1e-3,
                "atol": 1e-9,
                "max_steps": cfg.max_pncg_steps,
            },
            {
                "name": "exact_newton_refinement",
                "rtol": 1e-6,
                "atol": 1e-12,
                "linear_rtol": 1e-3,
                "max_newton_steps": 12,
            },
        ],
        "bone_bone_contact": False,
        "final_preparation_complete": False,
        "replaces_failed_performance_benchmark": False,
    }
    write_json(cfg.output_dir / "protocol.json", protocol)
    bulk = torch.zeros((3, 3, 3), device="cuda", dtype=torch.float64)
    resultant = torch.zeros((2, 2), device="cuda", dtype=torch.float64)
    stiffness = torch.ones((), device="cuda", dtype=torch.float64)
    pose = torch.zeros(6, device="cuda", dtype=torch.float64)
    phases = []
    complete = False
    for phase in ("pncg_globalization", "exact_newton_refinement"):
        if phase == "exact_newton_refinement":
            physics.runtime = Equilibrium(
                physics.runtime.forward,
                rtol=1e-6,
                atol=1e-12,
                adjoint_rtol=1e-7,
                forward_method="newton_cg",
                newton_linear_rtol=1e-3,
                newton_max_steps=12,
            )
        LOG.info("Starting %s: zero baseline stress, full source contact", phase)
        started = time.perf_counter()
        try:
            solved = physics.solve(
                bulk, resultant, stiffness, None, pose, seed, seed_pose=pose, key=phase
            )
        except (ForwardConvergenceError, AssertionError) as error:
            # Only an explicit numerical solver failure is a reportable outcome.
            # Any unrelated assertion remains a programming error and propagates.
            receipt = copy.deepcopy(physics.runtime.last_forward)
            if not receipt or receipt.get("success") is not False:
                raise
            phases.append(
                {
                    "phase": phase,
                    "success": False,
                    "wall_seconds": time.perf_counter() - started,
                    "forward": receipt,
                    "failure": str(error),
                }
            )
            write_json(cfg.output_dir / "phases.json", phases)
            break
        metrics = physics.metrics(solved)
        valid = (
            metrics["inverted_tetrahedra"] == 0
            and metrics["detF_min"] >= 0.25
            and metrics["detF_max"] <= 2
            and metrics["surface_motion_rms_mm"] <= 0.25
        )
        path = cfg.output_dir / f"{phase}.npz"
        np.savez_compressed(path, initial_displacement_m=solved.detach().cpu().numpy())
        phases.append(
            {
                "phase": phase,
                "success": bool(valid),
                "wall_seconds": time.perf_counter() - started,
                "forward": copy.deepcopy(physics.runtime.last_forward),
                "metrics": metrics,
                "state": {"path": str(path.resolve()), "sha256": sha256(path)},
            }
        )
        write_json(cfg.output_dir / "phases.json", phases)
        LOG.info("%s finished: %s", phase, phases[-1])
        if not valid:
            break
        seed = solved.detach().clone()
        complete = phase == "exact_newton_refinement"
    summary = {
        "schema": "joint-full-skull-passive-equilibration-v1",
        "success": complete,
        "status": "passive_equilibrium_only"
        if complete
        else "passive_equilibration_failed",
        "phases": phases,
        "protocol_sha256": sha256(cfg.output_dir / "protocol.json"),
        "nonzero_prestress_preparation_complete": False,
        "jaw_domain_validated": False,
        "final_launch_ready": False,
    }
    write_json(cfg.output_dir / "summary.json", summary)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
