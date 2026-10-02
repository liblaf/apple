"""Compare damped adjoints at the same saved failed MouthOpen trial."""

# ruff: noqa: E402, PLR0915
from __future__ import annotations

import copy
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import ipctk
import numpy as np
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
sys.path[:0] = [str(GROUP / "src"), str(SOLVERS), str(JOINT)]
from joint_common import ProfileJoint, sha256, write_json
from joint_equilibrium import configure_cuda
from mesh_step_scale import mean_rest_edge_length
from mouthopen_gradient_check import check_joint_pullback
from mouthopen_runtime import install_mouthopen_hybrid_runtime
from neutral_active_strain import install_active_strain
from reference_rebase import build_rebased_physics

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    trial_dir: Path = GROUP / "data/inverse-mouthopen-rigid-trial-001"
    reference_dir: Path = GROUP / "data/reference-clearance-002"
    output_dir: Path = GROUP / "data/damped-adjoint-probe-001"
    relative_shifts: tuple[float, ...] = (0.001, 0.01, 0.1)
    wall_seconds_per_shift: float = 90.0


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verify(item: dict) -> Path:
    path = Path(item["path"])
    assert record(path) == item
    return path


def main(cfg: Config) -> None:
    output = cfg.output_dir.resolve()
    assert not output.exists(), output
    assert all(value > 0 for value in cfg.relative_shifts)
    output.mkdir(parents=True)
    trial = cfg.trial_dir.resolve()
    protocol = json.loads((trial / "protocol.json").read_text())
    summary = json.loads((trial / "summary.json").read_text())
    source = json.loads(verify(protocol["source_run"]["protocol"]).read_text())
    assert (
        record(cfg.reference_dir / "reference-clearance.npz")
        == source["sources"]["reference_repair"]
    )
    endpoint = verify(summary["endpoint"])
    target_path = verify(protocol["sources"]["blendshapes"])
    inputs = {
        "endpoint": record(endpoint),
        "trial_summary": record(trial / "summary.json"),
        "trial_protocol": record(trial / "protocol.json"),
        "blendshapes": record(target_path),
    }
    source_dir = output / "sources"
    source_dir.mkdir()
    for path in (
        Path(__file__),
        GROUP / "src/mouthopen_runtime.py",
        GROUP / "src/mouthopen_gradient_check.py",
        JOINT / "joint_equilibrium.py",
    ):
        shutil.copy2(path, source_dir / path.name)
    write_json(
        output / "protocol.json",
        {
            "config": cfg.model_dump(mode="json"),
            "inputs": inputs,
            "definition": "(H_ff + lambda I)p = -L_f, lambda = relative_shift * mean(abs(diag(H_ff)))",
            "scope": "Fixed saved trial. Physical forward unchanged. Shifted adjoint is an approximate implicit gradient.",
            "source_sha256": {path.name: sha256(path) for path in source_dir.iterdir()},
        },
    )
    configure_cuda()
    ipctk.set_num_threads(4)
    physics, _ = build_rebased_physics(cfg.reference_dir, inverse=True)
    model = physics.runtime.forward.model
    baseline, _, _ = install_active_strain(model)
    model.set_materials(baseline)
    kappa = float(source["ipc_stiffness_mpa"])
    collision = model.collision
    potential = collision.potential
    collision.potential = ipctk.BarrierPotential(
        type(potential.barrier)(), potential.dhat, kappa, collision.use_physical_barrier
    )
    physics.contact_definition["config"]["stiffness_mpa"] = kappa
    runtime = install_mouthopen_hybrid_runtime(
        physics,
        max_step_norm_m=0.5 * mean_rest_edge_length(model, physics.points),
        fixed_stiffness_mpa=kappa,
        newton_max_steps=3000,
    )
    with np.load(endpoint, allow_pickle=False) as data:
        q_saved = torch.as_tensor(data["activation_inv"].copy())
        pose_saved = torch.as_tensor(data["pose_rad_m"].copy())
        seed = torch.as_tensor(data["displacement_m"].copy())
        np.testing.assert_array_equal(
            data["active_cell_ids"], physics.base.active_t.cpu().numpy()
        )
    with np.load(target_path, allow_pickle=False) as data:
        ids = data["skin_global_ids"].copy()
        triangles = data["skin_triangles"]
        neutral = data["new_neutral_points_m"]
        target = data["target_points_m"][
            list(data["expression_names"]).index("MouthOpen")
        ]
    xyz = neutral[triangles]
    area = (
        np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        / 2
    )
    weights = np.zeros(len(ids))
    np.add.at(weights, triangles.ravel(), np.repeat(area / 3, 3))
    weights /= weights.sum()
    scale2 = float((weights[:, None] * (target - neutral) ** 2).sum())
    weights_t = torch.as_tensor(weights)
    target_u = torch.as_tensor(target - physics.points[ids])
    ids_t = torch.as_tensor(ids, dtype=torch.int64)
    pose_scale = torch.tensor([math.pi / 18] * 3 + [0.01] * 3)

    def materials(q: torch.Tensor) -> dict:
        result = {name: dict(fields) for name, fields in baseline.items()}
        result["muscle"]["activation_inv"] = baseline["muscle"][
            "activation_inv"
        ].index_copy(0, physics.base.active_t, q)
        return result

    def physical_pose(z: torch.Tensor) -> torch.Tensor:
        return z * pose_scale

    def objective(u: torch.Tensor) -> torch.Tensor:
        return (weights_t[:, None] * (u[ids_t] - target_u).square()).sum() / scale2

    rows = []
    first_gradient = None
    for index, shift in enumerate(cfg.relative_shifts):
        LOG.info("Damped adjoint variant %d: relative shift %.6g", index, shift)
        runtime.adjoint_relative_shift = shift
        runtime.warm_adjoints.clear()
        runtime.last_adjoint = {}
        runtime.last_sparse_adjoint = {}
        started = time.perf_counter()
        runtime.deadline = started + cfg.wall_seconds_per_shift
        q = q_saved.detach().clone().requires_grad_()
        z = (pose_saved / pose_scale).detach().clone().requires_grad_()
        row = {"relative_shift": shift}
        try:
            u = runtime.solve(
                materials(q), physics.boundary(physical_pose(z)), seed, key="MouthOpen"
            )
            loss = objective(u)
            gradients = torch.autograd.grad(loss, (q, z))
            row["solve_seconds"] = time.perf_counter() - started
            assert all(bool(torch.isfinite(value).all()) for value in gradients)
            row["loss"] = float(loss)
            assert abs(row["loss"] - summary["final"]["loss"]) < 1e-12
            row["forward_displacement_change_max_m"] = float(
                (u.detach() - seed).abs().max()
            )
            row["adjoint"] = copy.deepcopy(runtime.last_adjoint)
            row["sparse_adjoint"] = copy.deepcopy(runtime.last_sparse_adjoint)
            runtime.deadline = None
            check = check_joint_pullback(
                physics,
                runtime,
                materials,
                physical_pose,
                q,
                z,
                u,
                grads=gradients,
                objective=objective,
            )
            q_error = min(value["relative_error"] for value in check["q"].values())
            pose_error = max(
                min(
                    values[f"epsilon_{epsilon:.0e}"]["relative_error"]
                    for epsilon in (1e-4, 1e-5)
                )
                for values in check["jaw_coordinates"].values()
            )
            assert max(q_error, pose_error) < 1e-3, (q_error, pose_error)
            row["fixed_state_pullback_error"] = {
                "q": q_error,
                "maximum_pose_coordinate": pose_error,
            }
            write_json(output / f"gradient-check-{index:02d}.json", check)
            combined = torch.cat([value.detach().flatten() for value in gradients])
            if first_gradient is None:
                first_gradient = combined.clone()
            row["gradient_cosine_to_first_success"] = float(
                torch.dot(combined, first_gradient)
                / (
                    torch.linalg.vector_norm(combined)
                    * torch.linalg.vector_norm(first_gradient)
                )
            )
            row["q_gradient_norm"] = float(torch.linalg.vector_norm(gradients[0]))
            row["pose_gradient_normalized"] = gradients[1].detach().cpu().tolist()
            saved = output / f"adjoint-{index:02d}.pt"
            torch.save(
                {
                    "p_free": runtime.warm_adjoints["MouthOpen"].cpu(),
                    "gradient_q": gradients[0].cpu(),
                    "gradient_pose_normalized": gradients[1].cpu(),
                },
                saved,
            )
            row["saved_adjoint"] = record(saved)
            row["success"] = True
        except Exception as error:
            row.update(
                success=False,
                failure={"type": type(error).__name__, "message": str(error)},
                sparse_adjoint=copy.deepcopy(runtime.last_sparse_adjoint),
            )
            LOG.exception("Damped adjoint variant failed")
        row["seconds"] = time.perf_counter() - started
        row["forward"] = copy.deepcopy(runtime.last_forward)
        rows.append(row)
        write_json(
            output / "summary.json",
            {
                "status": "running",
                "variants": rows,
                "source_geometry": summary["final"]["geometry"],
                "inverse_converged": False,
            },
        )
    result = json.loads((output / "summary.json").read_text())
    result["status"] = "completed"
    successful = [row["relative_shift"] for row in rows if row["success"]]
    result["smallest_successful_tested_relative_shift"] = (
        min(successful) if successful else None
    )
    write_json(output / "summary.json", result)
    cherries.log_output(output)
    cherries.log_metrics(
        {"probe/successful_variants": len(successful), "probe/variants": len(rows)}
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
