"""Render one completed COMP07 hybrid Smile pilot without implying a comparison."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch

from liblaf import cherries

EXPERIMENT = Path(__file__).resolve().parent.parent
SOURCE_GROUP = EXPERIMENT.parent.parent / "21/joint-activation-material-mandible"
PILOT = EXPERIMENT / "data/smile-hybrid-comp07-production-003"
CHECKPOINT = PILOT / "arms/hybrid_diag/expressions/Smile/latest.pt"


def load_visuals() -> Any:
    path = Path(__file__).with_name("21-compare-smile.py")
    spec = importlib.util.spec_from_file_location("smile_preview_visuals", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


visuals = load_visuals()


class Config(cherries.BaseConfig):
    pilot_dir: Path = PILOT
    inputs_dir: Path = SOURCE_GROUP / "data/expression-inputs-002"
    output_dir: Path = EXPERIMENT / "data/smile-hybrid-preview-001"


def render_preview(
    output: Path,
    target: pv.PolyData,
    neutral: pv.PolyData,
    hybrid: pv.PolyData,
    cranium: pv.PolyData,
    mandibles: tuple[pv.PolyData, pv.PolyData, pv.PolyData],
    eyes: pv.PolyData,
    state: dict[str, Any],
) -> str:
    plotter = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(2100, 760))
    plotter.set_background("#f7f7f5")
    plotter.enable_anti_aliasing("ssaa")
    setting = visuals.camera(target, "front")
    rms = float(state["metrics"]["fit_rms_mm"])
    step = int(state["accepted_steps"])
    jaw_deg = float(state["jaw_normalized"][0]) * 10
    items = (
        (
            "Smile target",
            target,
            mandibles[0],
            "target jaw pose unavailable; mandible at neutral reference",
        ),
        ("Frozen neutral", neutral, mandibles[1], "frozen input state"),
        (
            "Hybrid pilot endpoint",
            hybrid,
            mandibles[2],
            f"COMP07 RTX 3090 · step {step} · RMS {rms:.9f} mm · jaw {jaw_deg:.3f}°",
        ),
    )
    for index, (title, mesh, mandible, detail) in enumerate(items):
        plotter.subplot(0, index)
        visuals.add_clay_context(plotter, cranium, mandible, eyes)
        plotter.add_mesh(mesh, color="#d5a18a", smooth_shading=True)
        plotter.add_text(
            f"HYBRID PILOT PREVIEW — NOT A MATCHED OLD/NEW RESULT\n{title}\nfront · actual geometry, no motion amplification\n{detail}",
            position="upper_left",
            font_size=11,
            color="#202124",
        )
        visuals.set_camera(plotter, setting)
    name = "smile-hybrid-pilot-preview-front.png"
    plotter.show(screenshot=output / name, auto_close=True)
    return name


def main(cfg: Config) -> None:
    arm_dir = cfg.pilot_dir / "arms/hybrid_diag"
    arm = json.loads((arm_dir / "arm-summary.json").read_text())
    assert arm["arm"] == "hybrid_diag"
    assert arm["terminal"] == "iteration_budget_reached"
    checkpoint = arm_dir / "expressions/Smile/latest.pt"
    assert checkpoint.is_file()
    assert visuals.sha256(checkpoint) == arm["latest_checkpoint"]["sha256"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["accepted_steps"] == 20
    assert abs(float(state["metrics"]["fit_rms_mm"]) - 5.11076185437991) < 1e-12
    inputs = visuals.EyeExpressionInputs.load(cfg.inputs_dir)
    smile_index = inputs.expression_names.index("Smile")
    parent = Path(inputs.manifest["parent_frozen_neutral"]["directory"])
    frozen = visuals.read_json(parent / "manifest.json")
    prepared = visuals.read_json(Path(frozen["sources"]["prepared_manifest"]["path"]))
    skin = pv.read(Path(prepared["fixture"]["skin_path"])).triangulate()
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    assert np.array_equal(
        ids, np.asarray(inputs.arrays["observation_node_ids"], dtype=np.int64)
    )
    neutral_skin = skin.copy(deep=True)
    neutral_skin.points = np.asarray(
        inputs.arrays["neutral_points_m"], dtype=np.float64
    )[ids]
    target_increment = np.asarray(
        inputs.arrays["expression_displacement_m"], dtype=np.float64
    )[smile_index]
    origin = np.asarray(inputs.arrays["neutral_displacement_m"], dtype=np.float64)
    displacement = np.asarray(state["displacement_m"], dtype=np.float64)
    assert displacement.shape == origin.shape
    target = visuals.surface(neutral_skin, target_increment, target_increment)
    neutral = visuals.surface(
        neutral_skin, target_increment, np.zeros_like(target_increment)
    )
    hybrid = visuals.surface(
        neutral_skin, target_increment, displacement[ids] - origin[ids]
    )
    cranium, neutral_mandible, eyes, pivot = visuals.load_context(inputs)
    axis = np.asarray(inputs.arrays["mandible_frame_world"], dtype=np.float64)[:, 0]
    hybrid_mandible = visuals.transformed_mandible(
        neutral_mandible, pivot, axis, state["jaw_normalized"]
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    image = render_preview(
        cfg.output_dir,
        target,
        neutral,
        hybrid,
        cranium,
        (neutral_mandible, neutral_mandible, hybrid_mandible),
        eyes,
        state,
    )
    receipt = {
        "schema": "smile-hybrid-pilot-preview-v1",
        "success": True,
        "scope": "Completed COMP07 hybrid endpoint only; a preview, not a matched old-versus-new solver comparison.",
        "pilot": {
            "platform": "COMP07 RTX 3090",
            "terminal": arm["terminal"],
            "accepted_updates": int(state["accepted_steps"]),
            "fit_rms_mm": float(state["metrics"]["fit_rms_mm"]),
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": visuals.sha256(checkpoint),
            },
        },
        "inputs_manifest": {
            "path": str((cfg.inputs_dir / "manifest.json").resolve()),
            "sha256": visuals.sha256(cfg.inputs_dir / "manifest.json"),
        },
        "figure": image,
    }
    (cfg.output_dir / "summary.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main)
