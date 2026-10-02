"""Render three saved baseline surfaces with the frozen study-81 oblique view."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from PIL import Image

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]
GROUP = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
CAMERAS = ROOT / "exp/2026/09/08/physical-volume-closeups/data/20-regions/summary.json"
RENDERER = GROUP / "src/81-render-idea-shapes.py"
OUTPUT = GROUP / "data/97-baseline-oblique"
BACKGROUND = "#f4f2ed"
SHAPE_COLOR = "#aeb7ba"
STATES = (
    {
        "id": "old",
        "caption": "Old baseline · update 194",
        "step": 194,
        "path": ROOT
        / "exp/2026/09/07/face-actuation-diagnosis/data/11-historical-no-skin/final.npz",
        "sha256": "cc70d8221e5267395158ea4069c53033ac805df3e07464f38e3b09e2931a4dfb",
    },
    {
        "id": "corrected",
        "caption": "Corrected baseline · update 200",
        "step": 200,
        "path": Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
        / "exp/2026/09/08/physical-volume-baseline/data/20-baseline/final.npz",
        "sha256": "21b7e546f04c5566629a1d3634659ec0c5738df215699ba23e24eb7ac856abd7",
    },
    {
        "id": "psd",
        "caption": "PSD active stress · update 1024",
        "step": 1024,
        "path": ROOT / "exp/2026/09/07/tensor-active-stress/data/102-fit1024/final.npz",
        "sha256": "1ddeffc6eb1cf7110d37dd714a7806814841791e1b6a4b9e763a513438adeb95",
    },
)


class Config(cherries.BaseConfig):
    output_dir: Path = cherries.output("97-baseline-oblique", mkdir=True)


def record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
    }


def load_renderer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("study81_shape_renderer", RENDERER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.BACKGROUND == BACKGROUND
    assert module.SHAPE_COLOR == SHAPE_COLOR
    return module


def main(cfg: Config) -> None:
    out = cfg.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    renderer = load_renderer()

    volume_path = FIXTURE / "volume.vtu"
    skin_path = FIXTURE / "skin.vtp"
    volume = pv.read(volume_path)
    skin = pv.read(skin_path)
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    assert len(np.unique(skin_ids)) == len(skin_ids)
    assert np.array_equal(
        np.asarray(skin.points, dtype=np.float64),
        np.asarray(volume.points, dtype=np.float64)[skin_ids],
    )
    active_ids = np.flatnonzero(
        np.asarray(volume.cell_data["MuscleFraction"], dtype=np.float64) > 0
    )
    camera_receipt = json.loads(CAMERAS.read_text())
    views = {view["id"]: view for view in camera_receipt["views"]}
    camera = views["side-context"]["camera"]

    state_records = []
    panels = []
    for state in STATES:
        checkpoint = Path(state["path"])
        checkpoint_record = record(checkpoint)
        assert checkpoint_record["sha256"] == state["sha256"]
        with np.load(checkpoint, allow_pickle=False) as saved:
            assert int(saved["step"]) == state["step"]
            if "solver_valid" in saved.files:
                assert bool(saved["solver_valid"])
            assert np.array_equal(saved["active_ids"], active_ids)
            if "rest_points" in saved.files:
                rest = np.asarray(saved["rest_points"], dtype=np.float64)
                assert np.array_equal(rest, np.asarray(volume.points, dtype=np.float64))
            else:
                rest = np.asarray(volume.points, dtype=np.float64)
            displacement = np.asarray(saved["u"], dtype=np.float64)
            assert displacement.shape == rest.shape
            assert np.isfinite(displacement).all()
        deformed = skin.copy(deep=True)
        deformed.points = (rest + displacement)[skin_ids]
        output = out / f"{state['id']}.png"
        renderer._render(deformed, camera, state["caption"], SHAPE_COLOR, output)
        with Image.open(output) as image:
            assert image.size == (1800, 1800)
            panels.append(image.convert("RGB"))
        state_records.append(
            {
                "id": state["id"],
                "caption": state["caption"],
                "step": state["step"],
                "checkpoint": checkpoint_record,
                "output": record(output),
            }
        )

    two_state = Image.new("RGB", (3600, 1800))
    two_state.paste(panels[0], (0, 0))
    two_state.paste(panels[2], (1800, 0))
    two_state_path = out / "old-vs-psd.png"
    two_state.save(two_state_path)

    three_state = Image.new("RGB", (5400, 1800))
    for index, panel in enumerate(panels):
        three_state.paste(panel, (1800 * index, 0))
    three_state_path = out / "comparison.png"
    three_state.save(three_state_path)

    summary = {
        "status": "completed_saved_state_render",
        "scope": "Pure rendering of three existing saved endpoint surfaces; no solve, fitting, geometry smoothing, decimation, interpolation, or displacement amplification",
        "state_order": [state["id"] for state in STATES],
        "coordinates": "Exact saved nodal coordinates x = X + u, mapped to the frozen skin by GlobalPointId",
        "states": state_records,
        "camera": camera,
        "style": {
            "renderer": "Exact imported src/81-render-idea-shapes.py::_render",
            "window_size": [1800, 1800],
            "background": BACKGROUND,
            "shape_color": SHAPE_COLOR,
            "smooth_shading": False,
            "ambient": 0.20,
            "diffuse": 0.80,
            "specular": 0.0,
            "deformation_scale": 1.0,
        },
        "composites": {
            "old-vs-psd": {
                "order": ["old", "psd"],
                "operation": "Unscaled juxtaposition of the two rendered 1800-square panels",
                **record(two_state_path),
            },
            "comparison": {
                "order": ["old", "corrected", "psd"],
                "operation": "Unscaled juxtaposition of the three rendered 1800-square panels",
                **record(three_state_path),
            },
        },
        "inputs": {
            "fixture_volume": record(volume_path),
            "fixture_skin": record(skin_path),
            "camera_receipt": record(CAMERAS),
            "study81_renderer": record(RENDERER),
            "executed_source": record(Path(__file__)),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    cherries.log_metrics({"report/states": 3, "report/images": 5})


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
