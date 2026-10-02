"""Freeze face regularization from a discarded fit-only pilot, before endpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import torch
from experiment_profile import ProfileCometNoCommit
from tensor_controls import project

from liblaf import cherries

HERE = Path(__file__).resolve().parent.parent
COMPLETED = False


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    pilot: Path = HERE / "data/13-face-pilot-v2"
    output_dir: Path = cherries.output("14-frozen-face-settings", mkdir=True)
    probe_step: int = 3
    smooth_gradient_ratio: float = 0.25
    rank_gradient_ratio: float = 0.15


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(value: torch.Tensor) -> float:
    return float(value.square().mean().sqrt())


def main(cfg: Config) -> None:
    global COMPLETED  # noqa: PLW0603
    torch.set_num_threads(4)
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir())
    pilot_summary = json.loads((cfg.pilot / "summary.json").read_text())
    assert pilot_summary["status"] == "completed_fixed_budget"
    assert pilot_summary["primary_endpoint"]["step"] == cfg.probe_step
    state_path = cfg.pilot / "optimizer-latest.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    assert state["step"] == cfg.probe_step
    pilot_cfg = state["config"]
    assert pilot_cfg["model"] == "tensor"
    assert pilot_cfg["rank_weight"] == pilot_cfg["smoothness_weight"] == 0
    grad_path = cfg.pilot / f"gradients-{cfg.probe_step:04d}.npz"
    with np.load(grad_path) as loaded:
        gradients = {key: torch.from_numpy(loaded[key].copy()) for key in loaded}
    norms = {key: rms(value) for key, value in gradients.items()}
    assert all(np.isfinite(value) and value > 0 for value in norms.values())
    smooth_weight = cfg.smooth_gradient_ratio * norms["fit"] / norms["smoothness"]
    rank_weight = cfg.rank_gradient_ratio * norms["fit"] / norms["rank"]
    arms = {
        "psd": {"smoothness_weight": 0.0, "rank_weight": 0.0},
        "psd-smooth": {"smoothness_weight": smooth_weight, "rank_weight": 0.0},
        "psd-smooth-rank": {
            "smoothness_weight": smooth_weight,
            "rank_weight": rank_weight,
        },
    }
    updates: dict[str, torch.Tensor] = {}
    receipts = {}
    for name, weights in arms.items():
        q = torch.nn.Parameter(state["q"].clone())
        optimizer = torch.optim.Adam(
            [q], lr=pilot_cfg["learning_rate"], eps=pilot_cfg["adam_eps"]
        )
        # Deep-copy because Adam mutates the supplied moment tensors in place.
        optimizer.load_state_dict(
            torch.load(state_path, map_location="cpu", weights_only=False)["optimizer"]
        )
        regularizer_gradient = (
            weights["smoothness_weight"] * gradients["smoothness"]
            + weights["rank_weight"] * gradients["rank"]
        )
        q.grad = gradients["fit"] + regularizer_gradient
        optimizer.step()
        before_projection = q.detach().clone()
        projection = project(
            q, pilot_cfg["stress_cap_mpa"] / pilot_cfg["stress_reference_mpa"]
        )
        update = q.detach() - state["q"]
        updates[name] = update
        baseline = updates["psd"]
        receipts[name] = {
            **weights,
            "magnitude_weight": 0.0,
            "regularizer_gradient_rms": rms(regularizer_gradient),
            "regularizer_to_fit_gradient_rms_ratio": rms(regularizer_gradient)
            / norms["fit"],
            "unprojected_update_rms": rms(before_projection - state["q"]),
            "actual_update_rms": rms(update),
            "update_difference_from_fit_only_rms": rms(update - baseline),
            "update_cosine_with_fit_only": float(
                (update * baseline).sum()
                / (update.square().sum() * baseline.square().sum()).sqrt()
            ),
            **projection,
        }
    sources = out / "sources"
    sources.mkdir()
    for name in (Path(__file__).name, "tensor_controls.py", "experiment_profile.py"):
        (sources / name).write_bytes((HERE / "src" / name).read_bytes())
    frozen = {
        "status": "frozen_before_final_face_runs",
        "pilot": str(cfg.pilot.resolve()),
        "probe_step": cfg.probe_step,
        "selection": {
            "rule": "lambda = declared gradient RMS ratio times fit-gradient RMS divided by unweighted penalty-gradient RMS at the fixed pilot state",
            "smooth_gradient_ratio": cfg.smooth_gradient_ratio,
            "rank_gradient_ratio": cfg.rank_gradient_ratio,
            "no_final_geometry_or_arm_ranking_used": True,
            "learning_rate": pilot_cfg["learning_rate"],
            "adam_eps": pilot_cfg["adam_eps"],
            "stress_reference_mpa": pilot_cfg["stress_reference_mpa"],
            "stress_cap_mpa": pilot_cfg["stress_cap_mpa"],
            "smooth_length_m": pilot_cfg["smooth_length_m"],
            "final_steps": 64,
        },
        "gradient_rms": norms,
        "arms": receipts,
        "initialization_policy": "All final arms restart from zero controls, rest displacement, and zero Adam moments. The pilot is discarded.",
        "comparison_limit": "The Raw6 reference has a different actuation energy and coordinate scale; it is not an isolated PSD-constraint ablation.",
        "provenance": {
            str(path.resolve()): sha256(path)
            for path in (
                state_path,
                grad_path,
                cfg.pilot / "summary.json",
                *sources.glob("*.py"),
            )
        },
    }
    (out / "summary.json").write_text(
        json.dumps(frozen, indent=2, allow_nan=False) + "\n"
    )
    cherries.log_output(out / "summary.json")
    cherries.log_metrics(
        {"weights": {"smoothness": smooth_weight, "rank": rank_weight}}
    )
    print(json.dumps({"arms": receipts, "selection": frozen["selection"]}), flush=True)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
