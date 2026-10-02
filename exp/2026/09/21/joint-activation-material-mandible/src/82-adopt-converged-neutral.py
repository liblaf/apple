"""Freeze the reviewed loaded face and transfer expression displacement targets."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import ipctk
import numpy as np
import pyvista as pv
import torch
from joint_common import (
    GROUP,
    HISTORICAL,
    ROOT,
    ProfileJoint,
    archive_sources,
    sha256,
    write_json,
)
from joint_data import PreparedInputs, array_sha256
from joint_frozen_neutral import FrozenNeutral, NeutralIncrementProblem, load_script

from liblaf import cherries
from liblaf.apple.forward._problem import ForwardProblem


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/simple-skin-forward-011"
    endpoint_audit: Path = GROUP / "data/simple-forward-terminal-audit-011/summary.json"
    output_dir: Path = GROUP / "data/frozen-neutral-004"


def record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def verify_runtime(neutral: FrozenNeutral) -> dict:  # noqa: PLR0915
    runner = load_script("68-run-simple-skin-forward.py")
    runner.configure_cuda()
    physics, baseline = neutral.build_physics()
    assert torch.equal(
        physics.targets_t,
        torch.as_tensor(neutral.arrays["target_total_displacement_m"]),
    )
    assert torch.equal(
        physics.weights_t,
        torch.as_tensor(neutral.arrays["observation_weight_normalized"]),
    )
    mass = neutral.arrays["active_effective_volume_m3"]
    assert torch.equal(physics.muscle_mass_t, torch.as_tensor(mass / mass.sum()))
    assert np.array_equal(
        physics.arrays["graph_conductance_m"], neutral.arrays["graph_conductance_m"]
    )
    model = physics.runtime.forward.model
    state = physics.runtime.forward.state
    pose = torch.zeros(6)
    model.dof_map.fixed_values = physics.boundary(pose).detach().clone()
    origin = torch.as_tensor(neutral.arrays["neutral_displacement_m"])
    full = physics.full_skull.extend_seed(origin, pose)
    translated = NeutralIncrementProblem(model, full)
    zero = torch.zeros_like(translated.origin)
    translated.update(state, zero)
    energy = translated.fun(state).clone()
    gradient = translated.grad(state).clone()
    force = float(torch.linalg.vector_norm(gradient))
    assert force <= neutral.manifest["force_threshold_code"]
    assert torch.equal(state.u[: len(origin)], origin)

    # Identical coordinates must preserve forces, tangents and collision paths.
    direct = ForwardProblem(model)
    direct.update(state, translated.origin)
    # CUDA reductions may differ by roundoff after rebuilding the same state.
    energy_repeat = direct.fun(state).clone()
    gradient_repeat = direct.grad(state).clone()
    torch.testing.assert_close(energy_repeat, energy, rtol=1e-12, atol=1e-22)
    torch.testing.assert_close(gradient_repeat, gradient, rtol=1e-10, atol=1e-20)
    direction = torch.sin(torch.arange(len(zero), dtype=torch.float64)) * 1e-7
    hvp = direct.hess_prod(state, direction).clone()
    step = direct.max_step_size(state, direction).clone()
    translated.update(state, zero)
    torch.testing.assert_close(
        translated.hess_prod(state, direction), hvp, rtol=1e-10, atol=1e-20
    )
    assert torch.equal(translated.max_step_size(state, direction), step)
    translated.update(state, direction)
    perturbed_gradient = translated.grad(state).clone()
    direct.update(state, translated.origin + direction)
    torch.testing.assert_close(
        direct.grad(state), perturbed_gradient, rtol=1e-10, atol=1e-20
    )
    translated.update(state, zero)

    # Reusing the loaded equilibrium must require no accepted PNCG update.
    optimizer = runner.MonitoredPncg(
        monitor=SimpleNamespace(last_force_norm=None),
        criteria=runner.StrictPncg.ConvergenceCriteria(
            atol_primary=neutral.manifest["force_threshold_code"],
            rtol_primary=0.0,
            max_steps=2,
        ),
    )
    # Peach.minimize always takes a step before checking convergence. Check
    # its primary force criterion at the admitted seed instead of perturbing
    # a neutral the user has already approved.
    opt_state = optimizer.init(translated, state, zero)
    opt_state.convergence_state.grad_norm_first = torch.linalg.vector_norm(gradient)
    stop, result = optimizer.terminate(translated, state, opt_state)
    assert bool(stop)
    assert opt_state.step == 0
    solution = optimizer.postprocess(translated, state, opt_state, result)
    assert bool(solution.success)
    assert torch.equal(state.u[: len(origin)], origin)
    collision = model.collision
    contact = runner.contact_receipt(physics)
    positions = (collision.vertices + state.u[collision.indices]).detach().cpu().numpy()
    intersects = bool(
        ipctk.has_intersections(collision.collision_mesh, positions, ipctk.LBVH())
    )
    assert not intersects
    assert contact["contact_numerically_valid"]

    # Verify the new material interface preserves the frozen field exactly.
    active = torch.zeros((len(physics.ids), 3, 3))
    values = neutral.expression_materials(
        baseline, physics.active_t, skin_multiplier=torch.ones(()), active_stress=active
    )
    for name, fields in baseline.items():
        for key, value in fields.items():
            assert torch.equal(values[name][key], value)
    unit_target_error = max(
        float(
            torch.abs(
                neutral.expression_residual(
                    origin.index_copy(
                        0,
                        physics.observation_t,
                        torch.as_tensor(
                            neutral.arrays["target_total_displacement_m"][index]
                        ),
                    ),
                    index,
                )
            ).max()
        )
        for index in range(len(neutral.manifest["cohort"]["names"]))
    )
    assert unit_target_error == 0.0
    altered = neutral.expression_materials(
        baseline,
        physics.active_t,
        skin_multiplier=torch.tensor(1.01),
        active_stress=active,
    )
    assert torch.equal(
        altered["skin"]["baseline_stress"], baseline["skin"]["baseline_stress"]
    )
    model.set_materials(altered)
    translated.update(state, zero)
    changed_force = float(torch.linalg.vector_norm(translated.grad(state)))
    model.set_materials(baseline)
    translated.update(state, zero)
    restored_force = float(torch.linalg.vector_norm(translated.grad(state)))
    assert abs(restored_force - force) <= 1e-12 * force
    return {
        "success": True,
        "force_norm_N": force * 1e6,
        "force_threshold_N": neutral.manifest["force_threshold_code"] * 1e6,
        "pncg_success_without_state_change": True,
        "pncg_check": "initialized primary force criterion passes at step zero; minimize is not called because it unconditionally takes a first step",
        "coordinate_translation_preserves_energy_gradient_hvp_ccd": True,
        "repeat_energy_absolute_difference_code": abs(float(energy_repeat - energy)),
        "repeat_gradient_max_absolute_difference_code": float(
            torch.abs(gradient_repeat - gradient).max()
        ),
        "soft_bone_intersections": intersects,
        "contact": contact,
        "expression_material_interface_reproduces_baseline_exactly": True,
        "all_transferred_target_residuals_exactly_zero": True,
        "physics_uses_adopted_weights_volumes_graph_and_total_targets": True,
        "runtime_source_hashes_verified": True,
        "one_percent_stiffness_change_force_at_frozen_shape_N": changed_force * 1e6,
        "baseline_stress_trainable_scalars": 0,
        "mechanical_stability_proven": False,
    }


def main(cfg: Config) -> None:  # noqa: PLR0915
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = archive_sources(cfg.output_dir)
    roots = {
        "experiment": GROUP / "src",
        "apple": ROOT / "src/liblaf/apple",
        "tensor-reference": HISTORICAL,
    }
    runtime_sources = {}
    for relative, digest in provenance["sources"].items():
        category, local = relative.split("/", 1)
        if category == "experiment" and not (
            local.startswith("joint_") or local == "68-run-simple-skin-forward.py"
        ):
            continue
        path = roots[category] / local
        assert sha256(path) == digest
        runtime_sources[relative] = record(path)
    runtime_sources["uv.lock"] = record(ROOT / "uv.lock")
    protocol_path = cfg.run_dir / "protocol.json"
    summary_path = cfg.run_dir / "summary.json"
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads(summary_path.read_text())
    endpoint = json.loads(cfg.endpoint_audit.read_text())
    checkpoint = Path(summary["checkpoint"]["path"])
    assert summary["success"]
    assert endpoint["success"]
    assert (
        sha256(protocol_path)
        == summary["protocol_sha256"]
        == endpoint["protocol_sha256"]
    )
    assert sha256(summary_path) == endpoint["summary_sha256"]
    assert (
        sha256(checkpoint)
        == summary["checkpoint"]["sha256"]
        == endpoint["checkpoint"]["sha256"]
    )
    inputs = protocol["inputs"]
    prepared = PreparedInputs.load(
        Path(inputs["prepared_npz"]), Path(inputs["prepared_manifest"])
    )
    with np.load(checkpoint, allow_pickle=False) as archive:
        origin = archive["displacement_m"].copy()
    volume = pv.read(prepared.volume_path)
    skin = pv.read(prepared.skin_path)
    reference = np.asarray(volume.points).copy()
    neutral_points = reference + origin
    cells = np.asarray(volume.cells).reshape(-1, 5)[:, 1:]
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    obs = prepared.arrays["observation_node_ids"]
    assert np.array_equal(ids, obs)
    arrays = {key: value.copy() for key, value in prepared.arrays.items()}
    arrays["neutral_displacement_m"] = origin
    arrays["neutral_points_m"] = neutral_points
    arrays["target_points_m"] = (
        neutral_points[obs][None] + arrays["target_displacement_m"]
    )
    arrays["target_total_displacement_m"] = (
        origin[obs][None] + arrays["target_displacement_m"]
    )
    assert np.array_equal(
        arrays["target_displacement_m"], prepared.arrays["target_displacement_m"]
    )
    assert np.allclose(
        reference[obs][None] + arrays["target_total_displacement_m"],
        arrays["target_points_m"],
        rtol=0,
        atol=1e-15,
    )
    ds = neutral_points[cells[:, 1:]] - neutral_points[cells[:, :1]]
    dm = reference[cells[:, 1:]] - reference[cells[:, :1]]
    volumes = np.linalg.det(ds) / 6
    detf = np.linalg.det(ds) / np.linalg.det(dm)
    assert np.all(volumes > 0)
    arrays["active_effective_volume_m3"] = (
        volumes[arrays["active_cell_ids"]] * arrays["active_muscle_fraction"]
    )
    preparation = load_script("10-prepare-inputs.py")
    gi, gj, conductance, graph = preparation.build_graph(
        neutral_points,
        cells,
        arrays["active_cell_ids"],
        np.asarray(volume.cell_data["MuscleId"]),
        np.asarray(volume.cell_data["MuscleFraction"]),
    )
    assert np.array_equal(gi, arrays["graph_i"])
    assert np.array_equal(gj, arrays["graph_j"])
    arrays["graph_conductance_m"] = conductance
    local_tri = np.asarray(skin.faces).reshape(-1, 4)[:, 1:]
    xyz = neutral_points[ids][local_tri]
    area = (
        np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        / 2
    )
    point_area = np.zeros(len(ids))
    np.add.at(point_area, local_tri.reshape(-1), np.repeat(area / 3, 3))
    assert np.all(point_area > 0)
    arrays["observation_area_weights_m2"] = point_area
    arrays["observation_weight_normalized"] = point_area / point_area.sum()
    np.savez_compressed(cfg.output_dir / "state.npz", **arrays)
    volume.points = neutral_points
    volume.point_data["NeutralDisplacementFromConstitutiveReference_m"] = origin
    volume.cell_data["PhysicalDetF"] = detf
    volume.save(cfg.output_dir / "neutral-volume.vtu")
    skin.points = neutral_points[ids]
    skin.point_data["NeutralDisplacementFromConstitutiveReference_m"] = origin[ids]
    skin.save(cfg.output_dir / "neutral-skin.vtp")
    target_files = []
    for index, name in enumerate(prepared.target_names):
        target = skin.copy()
        target.points = arrays["target_points_m"][index]
        path = cfg.output_dir / f"target-{name}.vtp"
        target.save(path)
        target_files.append(path)
    manifest = {
        "schema": "joint-frozen-neutral-v1",
        "success": False,
        "decision": "User adopted force-converged Run011 geometry after ParaView inspection; preserve expression displacement fields on the new neutral.",
        "coordinate_contract": "x = X_constitutive + u_neutral + v_expression; stored neutral meshes are visualization/observation geometry, never stress-free FEM inputs.",
        "target_policy": "preserve_expression_displacements",
        "baseline_stress_policy": "prescribed skin baseline and bulk loaded equilibrium fixed; zero trainable baseline stress parameters",
        "stiffness_policy": "skin stiffness remains trainable with strong spatial regularization and literature prior; each change requires fresh equilibrium and neutral-drift assessment",
        "activation_frame": "original constitutive reference; six components per active muscle tetrahedron and expression",
        "force_threshold_code": summary["force_threshold"],
        "poisson_ratios": protocol["mechanics"]["poisson_ratios"],
        "cohort": {
            key: prepared.manifest["cohort"][key]
            for key in ("names", "training", "reserved")
        },
        "sources": {
            "protocol": record(protocol_path),
            "summary": record(summary_path),
            "checkpoint": record(checkpoint),
            "endpoint_audit": record(cfg.endpoint_audit),
            "prepared_npz": record(prepared.npz_path),
            "prepared_manifest": record(prepared.manifest_path),
            "constitutive_volume": record(prepared.volume_path),
            "constitutive_skin": record(prepared.skin_path),
            "skin_field": record(Path(inputs["skin_field_path"])),
            "skin_field_manifest": record(Path(inputs["skin_field_manifest_path"])),
            "geometry": record(Path(inputs["geometry"]["geometry"]["geometry_path"])),
            "geometry_audit": record(
                Path(inputs["geometry"]["geometry"]["audit_path"])
            ),
            "admission": record(Path(inputs["admission_path"])),
        },
        "runtime_sources": runtime_sources,
        "artifacts": {
            path.name: record(path)
            for path in [
                cfg.output_dir / "state.npz",
                cfg.output_dir / "neutral-volume.vtu",
                cfg.output_dir / "neutral-skin.vtp",
                *target_files,
            ]
        },
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": array_sha256(value),
            }
            for key, value in arrays.items()
        },
        "regularization_geometry": "same-muscle graph metric, effective muscle volumes and observation area weights recomputed on adopted neutral; connectivity and partitions unchanged",
        "activation_graph": graph,
        "geometry": {
            "nodes": len(neutral_points),
            "tetrahedra": len(cells),
            "inverted_tetrahedra": int(np.count_nonzero(detf <= 0)),
            "detF_min": float(detf.min()),
            "detF_max": float(detf.max()),
        },
        "full_joint_optimization_ready": False,
        "remaining_validation": "full-skull expression and jaw derivatives, converged activation/jaw control and stiffness-dependent neutral drift remain separate preparation tasks",
    }
    neutral = FrozenNeutral(arrays, manifest, cfg.output_dir)
    verification = verify_runtime(neutral)
    write_json(cfg.output_dir / "validation.json", verification)
    manifest["artifacts"]["validation.json"] = record(
        cfg.output_dir / "validation.json"
    )
    manifest["success"] = True
    write_json(cfg.output_dir / "manifest.json", manifest)
    FrozenNeutral.load(cfg.output_dir)
    write_json(
        GROUP / "data/current-neutral.json",
        {
            "schema": "joint-current-neutral-v1",
            "directory": str(cfg.output_dir.resolve()),
            "manifest_sha256": sha256(cfg.output_dir / "manifest.json"),
            "baseline_stress_trainable_scalars": 0,
            "target_policy": manifest["target_policy"],
        },
    )
    cherries.log_metrics(
        {
            "force_norm_N": verification["force_norm_N"],
            "inverted_tetrahedra": manifest["geometry"]["inverted_tetrahedra"],
        }
    )
    for name in ("manifest.json", "validation.json", "provenance.json"):
        cherries.log_output(cfg.output_dir / name)
    cherries.log_output(GROUP / "data/current-neutral.json")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
