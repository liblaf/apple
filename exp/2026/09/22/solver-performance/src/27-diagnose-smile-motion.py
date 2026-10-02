"""Read saved endpoints to quantify Smile motion without a physics solve."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
PARENT = GROUP.parent.parent / "21/joint-activation-material-mandible"


class Config(cherries.BaseConfig):
    fit_dir: Path = GROUP / "data/smile-fit-adam03-unconditional-004"
    inputs: Path = PARENT / "data/expression-inputs-002/state.npz"
    output: Path = Path("smile-shape-diagnosis-001/motion-audit.json")


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main(cfg: Config) -> None:
    torch.set_num_threads(4)
    cfg.output = cherries.output(cfg.output, mkdir=True)
    summary = json.loads((cfg.fit_dir / "summary.json").read_text())
    expected = summary["frozen_inputs"]["inputs_state"]["sha256"]
    assert sha256(cfg.inputs) == expected
    index = summary["source_expression_index"]
    with np.load(cfg.inputs, allow_pickle=False) as arrays:
        ids = arrays["observation_node_ids"]
        weights = arrays["observation_weight_normalized"]
        target = arrays["target_total_displacement_m"][index]
    assert np.isclose(weights.sum(), 1)

    def dot(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(weights[:, None] * a * b))

    def rms(a: np.ndarray) -> float:
        return float(np.sqrt(dot(a, a)))

    results = {}
    for arm, record in summary["arms"].items():
        directory = cfg.fit_dir / "arms" / arm / "expressions/Smile"
        start = torch.load(
            directory / "initial.pt", map_location="cpu", weights_only=False
        )
        end = torch.load(
            directory / "latest.pt", map_location="cpu", weights_only=False
        )
        assert sha256(directory / "latest.pt") == record["latest_checkpoint"]["sha256"]
        initial = start["displacement_m"].numpy()[ids]
        final = end["displacement_m"].numpy()[ids]
        wanted = target - initial
        motion = final - initial
        residual = final - target
        trace = [
            json.loads(line)
            for line in (directory / "trace.jsonl").read_text().splitlines()
        ]
        results[arm] = {
            "motion_from_initial_weighted_rms_mm": 1000 * rms(motion),
            "motion_from_initial_max_node_mm": float(
                1000 * np.linalg.norm(motion, axis=1).max()
            ),
            "target_from_initial_weighted_rms_mm": 1000 * rms(wanted),
            "target_direction_amplitude_fraction": dot(motion, wanted)
            / dot(wanted, wanted),
            "weighted_motion_target_cosine": dot(motion, wanted)
            / (rms(motion) * rms(wanted)),
            "final_fit_rms_mm": 1000 * rms(residual),
            "best_fit_rms_mm": min(row["fit_rms_mm"] for row in trace),
            "best_fit_step": min(trace, key=lambda row: row["fit_rms_mm"])[
                "accepted_steps"
            ],
            "zero_step_forward_evaluations_after_initial": sum(
                row["forward"].get("steps", -1) == 0 for row in trace[1:]
            ),
            "max_abs_recorded_primal_objective_correction": max(
                abs(row["primal_objective_correction_estimate"]) for row in trace
            ),
            "checkpoint_sha256": record["latest_checkpoint"]["sha256"],
        }
        assert np.isclose(
            results[arm]["final_fit_rms_mm"], end["metrics"]["fit_rms_mm"]
        )
    receipt = {
        "schema": "smile-saved-endpoint-motion-diagnosis-v1",
        "scope": "CPU post-processing only; no forward solve, adjoint or optimizer update",
        "inputs_sha256": expected,
        "arms": results,
        "success": True,
    }
    cfg.output.write_text(json.dumps(receipt, indent=2) + "\n")
    cherries.log_metrics(
        {
            arm: {
                key: value
                for key, value in result.items()
                if isinstance(value, (int, float))
            }
            for arm, result in results.items()
        }
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    cherries.main(main)
