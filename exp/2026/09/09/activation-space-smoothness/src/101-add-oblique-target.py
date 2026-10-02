"""Add the supplied target to the existing oblique baseline comparisons."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from PIL import Image

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
BASELINE = GROUP / "data/97-baseline-oblique"
TARGET = GROUP / "data/81-idea-shapes/skins/target.vtp"
RENDERER = GROUP / "src/81-render-idea-shapes.py"


class Config(cherries.BaseConfig):
    output_dir: Path = cherries.output("101-oblique-target", mkdir=True)


def record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def main(cfg: Config) -> None:
    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)
    assert not any(output.iterdir()), output
    shape_summary_path = GROUP / "data/81-idea-shapes/summary.json"
    shape_summary = json.loads(shape_summary_path.read_text())
    baseline_summary_path = BASELINE / "summary.json"
    baseline_summary = json.loads(baseline_summary_path.read_text())
    assert record(TARGET)["sha256"] == shape_summary["target"]["skin"]["sha256"]
    assert (
        record(RENDERER)["sha256"]
        == baseline_summary["inputs"]["study81_renderer"]["sha256"]
    )
    spec = importlib.util.spec_from_file_location("study81_shape_renderer", RENDERER)
    assert spec is not None and spec.loader is not None
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    target_mesh = pv.read(TARGET)
    target_path = output / "target.png"
    renderer._render(
        target_mesh,
        baseline_summary["camera"],
        "Target smile",
        renderer.SHAPE_COLOR,
        target_path,
    )
    sources = [
        record(TARGET),
        record(RENDERER),
        record(shape_summary_path),
        record(baseline_summary_path),
        record(Path(__file__)),
    ]
    panels = {}
    with Image.open(target_path) as picture:
        assert picture.size == (1800, 1800)
        panels["target"] = picture.convert("RGB")
    for state in baseline_summary["states"]:
        path = Path(state["output"]["path"])
        receipt = record(path)
        assert receipt["sha256"] == state["output"]["sha256"]
        with Image.open(path) as picture:
            assert picture.size == (1800, 1800)
            panels[state["id"]] = picture.convert("RGB")
        sources.append(receipt)
    composites = {}
    for name, order in (
        ("old-vs-psd", ["target", "old", "psd"]),
        ("comparison", ["target", "old", "corrected", "psd"]),
    ):
        composite = Image.new("RGB", (1800 * len(order), 1800))
        for index, state in enumerate(order):
            composite.paste(panels[state], (1800 * index, 0))
        path = output / f"{name}.png"
        composite.save(path)
        composites[name] = {
            "order": order,
            "operation": "Unscaled juxtaposition; existing result panels preserved pixel-for-pixel",
            **record(path),
        }
    summary = {
        "status": "completed_target_render_and_packaging",
        "scope": "Render the supplied target in the established camera and append unchanged result panels; no fitting or geometry modification",
        "target_coordinates": "Frozen fixture skin + volume Smile field at GlobalPointId; exact verified study-81 target mesh",
        "target": {
            "mesh": record(TARGET),
            "image": record(target_path),
            "label": "Target smile",
        },
        "camera": baseline_summary["camera"],
        "style": baseline_summary["style"],
        "composites": composites,
        "sources": sources,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    cherries.log_metrics(
        {"report/target_views": 1, "report/comparisons": 2, "report/images": 3}
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
