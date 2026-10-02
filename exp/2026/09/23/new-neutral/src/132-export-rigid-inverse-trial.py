# ruff: noqa: PLR0915
"""Export a forward-verified inverse trial after its exact adjoint fails."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
sys.path.insert(0, str(JOINT))
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402


class Config(cherries.BaseConfig):
    source_dir: Path = GROUP / "data/inverse-mouthopen-rigid-002"
    output_dir: Path = GROUP / "data/inverse-mouthopen-rigid-trial-001"
    terminal_log: Path = GROUP / "tmp/inverse-mouthopen-rigid-002-terminal.log"


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def metrics(
    *, endpoint: dict, rendering: Path, blendshapes: Path, old_pose: np.ndarray
) -> tuple[dict, float, float]:
    with np.load(rendering, allow_pickle=False) as data:
        reference = data["full_reference_points_m"]
        ids = data["skin_global_ids"]
        triangles = data["skin_triangles"]
    with np.load(blendshapes, allow_pickle=False) as data:
        index = [str(name) for name in data["expression_names"]].index("MouthOpen")
        neutral = data["new_neutral_points_m"]
        target = data["target_points_m"][index]
    u = endpoint["displacement_m"].numpy()
    q = endpoint["activation_inv"].numpy()
    pose = endpoint["pose_rad_m"].numpy()
    area = 0.5 * np.linalg.norm(
        np.cross(
            neutral[triangles[:, 1]] - neutral[triangles[:, 0]],
            neutral[triangles[:, 2]] - neutral[triangles[:, 0]],
        ),
        axis=1,
    )
    weights = np.zeros(len(ids))
    np.add.at(weights, triangles.ravel(), np.repeat(area / 3.0, 3))
    weights /= weights.sum()
    error = reference[ids] + u[ids] - target
    scale2 = float((weights[:, None] * (target - neutral) ** 2).sum())
    loss = float((weights[:, None] * error**2).sum() / scale2)
    relative_rotation = (
        Rotation.from_rotvec(pose[:3]) * Rotation.from_rotvec(old_pose[:3]).inv()
    )
    return (
        {
            "loss": loss,
            "fit_rms_mm": float(np.sqrt((weights[:, None] * error**2).sum()) * 1000),
            "activation_rms": float(np.sqrt(np.mean(q**2))),
            "activation_max_abs": float(np.abs(q).max()),
            "activation_min": float(q.min()),
            "activation_max": float(q.max()),
            "activation_nonzero": int(np.count_nonzero(q)),
            "pose_rad_m": pose.tolist(),
            "pose_rotation_degrees": float(np.degrees(np.linalg.norm(pose[:3]))),
            "pose_translation_mm": float(np.linalg.norm(pose[3:]) * 1000),
            "relative_rotation_degrees": float(
                np.degrees(relative_rotation.magnitude())
            ),
            "relative_translation_m": float(np.linalg.norm(pose[3:] - old_pose[3:])),
            "objective_scale_squared_m2": scale2,
        },
        loss,
        scale2,
    )


def main(cfg: Config) -> None:
    source, output = cfg.source_dir.resolve(), cfg.output_dir.resolve()
    assert source.is_dir()
    assert not output.exists(), output
    protocol_path = source / "protocol.json"
    source_protocol = json.loads(protocol_path.read_text())
    rows = read_rows(source / "trials.jsonl")
    failure_path = source / "failure.json"
    failure = json.loads(failure_path.read_text())
    assert failure["status"] == "accepted_gradient_failed"
    trial = next(row for row in rows if row["iteration"] == 1 and row["trial"] == 0)
    assert trial["success"]
    assert trial["accepted"]
    attempt = source / "initialization/attempt-00000"
    endpoint_path = attempt / "endpoint.pt"
    final_path = attempt / "step-001/final.pt"
    endpoint = torch.load(endpoint_path, map_location="cpu", weights_only=False)
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    assert set(endpoint) == {"activation_inv", "pose_rad_m", "displacement_m"}
    for name in endpoint:
        assert torch.equal(endpoint[name], final[name]), name
    assert all(
        value.dtype == torch.float64 and value.device.type == "cpu"
        for value in endpoint.values()
    )
    rendering_source = Path(source_protocol["rendering"]["archive"]["path"])
    blendshape = Path(source_protocol["sources"]["blendshapes"]["path"])
    assert sha256(rendering_source) == source_protocol["rendering"]["archive"]["sha256"]
    assert sha256(blendshape) == source_protocol["sources"]["blendshapes"]["sha256"]
    old_pose = np.asarray(
        json.loads((source / "summary.json").read_text())["initial"]["pose_rad_m"],
        dtype=np.float64,
    )
    values, loss, _ = metrics(
        endpoint=endpoint,
        rendering=rendering_source,
        blendshapes=blendshape,
        old_pose=old_pose,
    )
    assert loss == float(trial["loss"])
    initial = json.loads((source / "summary.json").read_text())["initial"]
    init_receipt = json.loads((attempt / "summary.json").read_text())
    step = init_receipt["steps"][-1]
    assert step["status"] == "force_contact_converged"
    assert init_receipt["last_completed"]["checkpoint"] == record(endpoint_path)
    forward = trial["forward"]
    assert forward["success"]
    assert all(forward["terminal_gates"].values())
    assert values["relative_rotation_degrees"] < 1.0
    assert values["relative_translation_m"] < 0.001
    output.mkdir(parents=True)
    rendering = output / "rendering.npz"
    shutil.copy2(rendering_source, rendering)
    active_ids = np.load(source / "endpoint.npz", allow_pickle=False)["active_cell_ids"]
    np.savez_compressed(
        output / "endpoint.npz",
        displacement_m=endpoint["displacement_m"].numpy(),
        activation_inv=endpoint["activation_inv"].numpy(),
        active_cell_ids=active_ids,
        pose_rad_m=endpoint["pose_rad_m"].numpy(),
    )
    geometry = init_receipt["last_completed"]["geometry"]
    candidate = {
        "iteration": 1,
        "trial": 0,
        "label": "forward_verified_armijo_trial_not_optimizer_accepted",
        **values,
        "force_norm_n": float(forward["grad_norm"] * 1e6),
        "force_threshold_n": float(forward["force_threshold"] * 1e6),
        "forward_converged": True,
        "contact_valid": True,
        "geometry": geometry,
        "valid_forward": False,
        "inverse_converged": False,
        "forward": forward,
    }
    initial_row = {**initial, "label": "source_optimizer_iteration_zero"}
    progress = output / "progress.jsonl"
    progress.write_text(
        "\n".join(json.dumps(row, allow_nan=False) for row in (initial_row, candidate))
        + "\n"
    )
    adjoint_failure = {
        **failure,
        "source_failure": record(failure_path),
        "source_terminal_log": record(cfg.terminal_log),
    }
    exported_protocol = {
        "schema": "new-neutral-mouthopen-rigid6-inverse-v1",
        "export_schema": "new-neutral-mouthopen-rigid6-forward-verified-trial-v1",
        "status": "forward_verified_trial_adjoint_failed",
        "expression_name": source_protocol["expression_name"],
        "expression_index": source_protocol["expression_index"],
        "source_run": {
            "protocol": record(protocol_path),
            "summary": record(source / "summary.json"),
            "trials": record(source / "trials.jsonl"),
            "attempt_summary": record(attempt / "summary.json"),
            "attempt_endpoint": record(endpoint_path),
            "attempt_final": record(final_path),
        },
        "rendering": {"archive": record(rendering)},
        "sources": {"blendshapes": source_protocol["sources"]["blendshapes"]},
        "parameterization": source_protocol["parameterization"],
        "objective": source_protocol["objective"],
        "force_contract": source_protocol["force_contract"],
        "trial": {"iteration": 1, "trial": 0, "armijo_accepted": True},
        "adjoint_failure": adjoint_failure,
    }
    write_json(output / "protocol.json", exported_protocol)
    summary = {
        "status": "forward_verified_trial_adjoint_failed",
        "trial_label": "forward_verified_armijo_trial_not_optimizer_accepted",
        "initial": initial_row,
        "final": candidate,
        "inverse_converged": False,
        "accepted_optimizer_iterations": 0,
        "adjoint_failure": adjoint_failure,
        "endpoint": record(output / "endpoint.npz"),
    }
    write_json(output / "summary.json", summary)
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "trial/loss": loss,
            "trial/fit_rms_mm": values["fit_rms_mm"],
            "trial/force_n": candidate["force_norm_n"],
            "trial/relative_rotation_degrees": values["relative_rotation_degrees"],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
