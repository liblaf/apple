"""Calibrate strain smoothness from a saved L2 fit without another physics solve."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from pathlib import Path

import numpy as np
from experiment import Profile

from liblaf import cherries


class Config(cherries.BaseConfig):
    source: Path = Path("41-l2-unrestricted-active-strain-lr0p05-nosmooth-001")
    checkpoint: Path | None = None
    output: Path = Path("45-active-strain-smoothness-001")
    target_gradient_ratio: float = 0.1
    recent_steps: int = 10


def receipt(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "path": str(path.resolve()),
        "sha256": digest,
    }


def main(cfg: Config) -> None:
    assert 0 < cfg.target_gradient_ratio < 1
    assert cfg.recent_steps > 0
    source = Path("data") / cfg.source
    stage = source / "l2-symmetric6"
    protocol = json.loads((source / "protocol.json").read_text())
    assert protocol["activation_model"] == "strain"
    assert protocol["objective"] == {
        "position_weight": 1.0,
        "normal_weight": 0.0,
        "smooth_weight": 0.0,
    }
    # last.npz is atomically replaced by the runner. Keep this exact file image.
    checkpoint = cfg.checkpoint if cfg.checkpoint is not None else stage / "last.npz"
    blob = checkpoint.read_bytes()
    with np.load(io.BytesIO(blob), allow_pickle=False) as state:
        assert str(state["activation_model"]) == "strain"
        assert str(state["mode"]) == "symmetric6"
        step = int(state["step"])
        strain = state["S"].copy()
        controls_count = state["q"].size
    trace = (stage / "trace.csv").read_text()
    rows = list(csv.DictReader(io.StringIO(trace)))
    row = next(row for row in rows if int(row["step"]) == step)
    solver = next(
        item
        for line in (stage / "solver-receipts.jsonl").read_text().splitlines()
        if (item := json.loads(line))["step"] == step
    )
    # Independently differentiate the saved within-muscle graph on CPU.
    with np.load(source / "mesh.npz", allow_pickle=False) as mesh:
        i, j, conductance = mesh["edge_i"], mesh["edge_j"], mesh["edge_weight"]
        factor = float(mesh["regularizer_factor"])
        mass = mesh["active_volume_weights"]
        delta = strain[i] - strain[j]
        edge_gradient = 2 * factor * conductance[:, None, None] * delta
        gradient = np.zeros_like(strain)
        np.add.at(gradient, i, edge_gradient)
        np.add.at(gradient, j, -edge_gradient)
        raw_loss = float(factor * np.sum(conductance * np.sum(delta**2, axis=(1, 2))))
        smooth_norm = float(np.sqrt(np.sum(np.sum(gradient**2, axis=(1, 2)) / mass)))
    np.testing.assert_allclose(
        raw_loss, float(row["activation_smoothness"]), rtol=1e-12
    )
    np.testing.assert_allclose(
        smooth_norm, float(row["smoothness_gradient_dual_norm"]), rtol=1e-12
    )
    l2_norm = float(row["l2_gradient_dual_norm"])
    assert 0 < l2_norm < np.inf
    assert smooth_norm > 0
    exact_weight = cfg.target_gradient_ratio * l2_norm / smooth_norm
    weight = float(f"{exact_weight:.2g}")
    recent = [row for row in rows if int(row["step"]) <= step][-cfg.recent_steps :]
    recent_ratios = [
        weight
        * float(row["smoothness_gradient_dual_norm"])
        / float(row["l2_gradient_dual_norm"])
        for row in recent
    ]
    balance = {
        "l2_gradient_dual_norm": l2_norm,
        "smoothness_gradient_dual_norm": smooth_norm,
        "weighted_smoothness_gradient_dual_norm": weight * smooth_norm,
        "smoothness_to_l2_gradient_ratio": weight * smooth_norm / l2_norm,
    }
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "calibration-state.npz").write_bytes(blob)
    (out / "source-trace.csv").write_text(trace)
    result = {
        "schema": "smile-active-strain-gradient-calibration-v1",
        "selected_weight": weight,
        "exact_weight": exact_weight,
        "target_gradient_ratio": cfg.target_gradient_ratio,
        "calibration_step": step,
        "activation_model": "strain",
        "gradient_metric": "symmetric Mandel tensor covectors; dual normalized effective-volume norm",
        "formula": "weight = target_ratio * norm(grad L2) / norm(grad smoothness)",
        "calibration_balance": balance,
        "cpu_graph_verification": "raw smoothness and its gradient norm match trace within rtol=1e-12",
        "calibration_loss": {
            "l2": float(row["position_contribution"]),
            "raw_smoothness": raw_loss,
            "weighted_smoothness": weight * raw_loss,
            "weighted_smoothness_to_l2": weight
            * raw_loss
            / float(row["position_contribution"]),
        },
        "euclidean_control_gradient_ratio": weight
        * float(np.linalg.norm(gradient))
        / (float(row["gradient_rms"]) * np.sqrt(controls_count)),
        "recent_gradient_ratio": {
            "first_step": int(recent[0]["step"]),
            "last_step": step,
            "minimum": min(recent_ratios),
            "median": float(np.median(recent_ratios)),
            "maximum": max(recent_ratios),
        },
        "calibration_state": receipt(out / "calibration-state.npz"),
        "source_protocol": receipt(source / "protocol.json"),
        "mesh": receipt(source / "mesh.npz"),
        "analysis_source": receipt(Path(__file__)),
        "source_trace_row": row,
        "solver_receipt": solver,
        "selection": "fixed coefficient rounded to two significant digits at a late unregularized fit state",
        "transfer_limit": "Uses the optimizer's finite approximate L2 gradient. The ratio is global and state-dependent; it is not an Adam update ratio or a guaranteed final ratio after regularization.",
        "fit_modified": False,
    }
    (out / "calibration.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    cherries.log_input(source / "protocol.json")
    cherries.set_step(step)
    cherries.log_metrics({"selected_smooth_weight": weight, **balance})
    logging.getLogger(__name__).info(
        "Step %d: smooth weight %.2g, gradient ratio %.6f, recent range [%.6f, %.6f]",
        step,
        weight,
        balance["smoothness_to_l2_gradient_ratio"],
        min(recent_ratios),
        max(recent_ratios),
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
