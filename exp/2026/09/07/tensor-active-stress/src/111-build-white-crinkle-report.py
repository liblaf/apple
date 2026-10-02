#!/usr/bin/env python3
"""Assemble the source110 white-skin and fixed-cohort crinkle report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"
INPUT = DATA / "110-white-crinkle/summary.json"
LOG = ROOT / "logs/110-white-crinkle-v2-terminal.log"
QA = ROOT / "tmp/110-pdf-qa/visual-qa.json"
REPORT = DOCS / "111-white-crinkle-report.md"
MANIFEST = ROOT / "tmp/111-publication/manifest.json"
RECEIPT = ROOT / "tmp/111-publication/assembly-receipt.json"
FIGURES = (
    "skin-front",
    "skin-mouth",
    "skin-mouth-edges",
    "muscle-crinkle",
    "muscle-mouth",
    "plane-context",
)


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def record(path: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def checked(value: object, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise TypeError(label)
    path = Path(value["path"])
    if record(path) != value:
        raise ValueError(label)
    return path


def rel(path: Path) -> str:
    return "../" + path.resolve().relative_to(ROOT).as_posix()


def metric(case: dict[str, Any], key: str) -> float | int:
    value = case.get("metrics", {}).get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{key} metric")
    return value


def figure(summary: dict[str, Any], name: str) -> tuple[Path, Path]:
    value = summary.get("figures", {}).get(name)
    if not isinstance(value, dict):
        raise ValueError(name)
    png, pdf = (
        checked(value.get("png"), name + " PNG"),
        checked(value.get("pdf"), name + " PDF"),
    )
    if png.suffix != ".png" or pdf.suffix != ".pdf":
        raise ValueError(name + " suffix")
    return png, pdf


def capture(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    urls = re.findall(r"Experiment is live on comet\.com (https://\S+)", text)
    commands = re.findall(r"cherries/cmd\s+: ([^\n]+)", text)
    starts = re.findall(r"cherries/start_time\s+: ([^\n]+)", text)
    ends = re.findall(r"cherries/end_time\s+: ([^\n]+)", text)
    if len(urls) != 1 or len(commands) != 1 or len(starts) != 1 or len(ends) != 1:
        raise ValueError("source110 terminal capture differs")
    return {
        "command": commands[0].strip(),
        "comet_url": urls[0],
        "wall_time": str(
            datetime.fromisoformat(ends[0].strip())
            - datetime.fromisoformat(starts[0].strip())
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=INPUT)
    p.add_argument("--log", type=Path, default=LOG)
    p.add_argument("--qa", type=Path, default=QA)
    p.add_argument("--visual-observation", action="append", required=True)
    args = p.parse_args()
    if any(x.exists() for x in (REPORT, MANIFEST, RECEIPT)):
        raise FileExistsError("111 outputs must be new")
    source = args.input.resolve()
    summary = read(source)
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "completed_postprocessing"
    ):
        raise ValueError("source110 completion differs")
    cases = summary.get("cases")
    if not isinstance(cases, dict) or set(cases) != {"baseline", "current"}:
        raise ValueError("source110 case keys differ")
    baseline, current = cases["baseline"], cases["current"]
    for c, step in ((baseline, 194), (current, 1024)):
        if (
            not isinstance(c, dict)
            or c.get("step") != step
            or not isinstance(c.get("label"), str)
        ):
            raise ValueError("case identity differs")
        checked(c.get("vtu"), c["label"] + " VTU")
        checked(c.get("npz"), c["label"] + " NPZ")
        for key in (
            "fit_rms_mm",
            "motion_rms_mm",
            "detF_min",
            "detF_max",
            "inverted_tetrahedra",
        ):
            metric(c, key)
    figures = {name: figure(summary, name) for name in FIGURES}
    crinkle = summary.get("crinkle")
    if (
        not isinstance(crinkle, dict)
        or crinkle.get("same_source_cell_ids_all_states") is not True
        or crinkle.get("reclip_deformed_state") is not False
    ):
        raise ValueError("fixed cohort proof differs")
    for key in (
        "rule",
        "retained_halfspace",
        "expected_muscle_tetrahedra",
        "source_cell_ids",
        "muscle_selector",
        "camera",
        "mouth_camera",
    ):
        if key not in crinkle:
            raise ValueError("crinkle lacks " + key)
    checked(crinkle["source_cell_ids"], "source cell IDs")
    capture_data = capture(args.log.resolve())
    qa_path = args.qa.resolve()
    qa = read(qa_path)
    if (
        qa.get("status") != "passed"
        or qa.get("source_summary") != record(source)
        or qa.get("browser_interaction_performed") is not False
    ):
        raise ValueError("source110 visual QA differs")
    checks = qa.get("checks", {})
    if not isinstance(checks, dict) or not all(checks.values()):
        raise ValueError("source110 visual QA failed")
    observation = " ".join(args.visual_observation)
    baseline_display = "June saved no-skin baseline"
    report = f"""# White skin and fixed-cohort muscle section

The current result has visibly smoother skin and much less severe local muscle distortion around the mouth. Surface fit RMS is higher: **1.610 mm**, compared with **0.654 mm** for the bumpy baseline. Some skin waviness and target-shape mismatch remain.

The panels compare **{baseline_display} · global step 194** and **{current["label"]} · global step 1024** using actual saved geometry. They do not have equal optimizer budgets or the same activation formulation. This is therefore a visual record, not a matched causal comparison.

Within each comparison, panels share their recorded camera and true geometric scale. White skin gives exterior context. The target is observed `IsFace` skin only; it has no invented interior tissue state.

## Saved-state diagnostics

| State | Fit RMS (mm) | Motion RMS (mm) | detF min / max | Inverted tetrahedra |
| --- | ---: | ---: | ---: | ---: |
| {baseline_display} · step 194 | {metric(baseline, "fit_rms_mm"):.6f} | {metric(baseline, "motion_rms_mm"):.6f} | {metric(baseline, "detF_min"):.6f} / {metric(baseline, "detF_max"):.6f} | {metric(baseline, "inverted_tetrahedra")} |
| {current["label"]} · step 1024 | {metric(current, "fit_rms_mm"):.6f} | {metric(current, "motion_rms_mm"):.6f} | {metric(current, "detF_min"):.6f} / {metric(current, "detF_max"):.6f} | {metric(current, "inverted_tetrahedra")} |

## White exterior skin

![White skin front]({rel(figures["skin-front"][0])})

![White skin mouth closeup]({rel(figures["skin-mouth"][0])})

[Front PNG]({rel(figures["skin-front"][0])}) · [Front PDF]({rel(figures["skin-front"][1])}) · [Mouth PNG]({rel(figures["skin-mouth"][0])}) · [Mouth PDF]({rel(figures["skin-mouth"][1])}) · [Mouth-with-edges PNG]({rel(figures["skin-mouth-edges"][0])}) · [PDF]({rel(figures["skin-mouth-edges"][1])})

## Fixed-rest-cohort interior muscle

The retained half-space is **{crinkle["retained_halfspace"]}**. The cohort contains **{crinkle["expected_muscle_tetrahedra"]:,}** muscle-bearing tetrahedra (`MuscleFraction > 0`). Its rule is: `{crinkle["rule"]}`. The same saved source-cell IDs are extracted in each state; deformed states are never reclipped. Muscle selection is `{crinkle["muscle_selector"]}`. The retained cohort has **{metric(baseline, "crinkle_muscle_inverted_tetrahedra")}** inverted muscle-bearing tetrahedra in the June state and **{metric(current, "crinkle_muscle_inverted_tetrahedra")}** in the current state; no inversion rejection or repair was applied.

![Fixed-cohort muscle crinkle]({rel(figures["muscle-crinkle"][0])})

![Interior muscle mouth closeup]({rel(figures["muscle-mouth"][0])})

[Crinkle PNG]({rel(figures["muscle-crinkle"][0])}) · [Crinkle PDF]({rel(figures["muscle-crinkle"][1])}) · [Mouth PNG]({rel(figures["muscle-mouth"][0])}) · [Mouth PDF]({rel(figures["muscle-mouth"][1])})

![Rest-space plane context]({rel(figures["plane-context"][0])})

[Plane context PNG]({rel(figures["plane-context"][0])}) · [Plane context PDF]({rel(figures["plane-context"][1])})

## Visual observations

{observation}

Six PNGs and six rendered PDF pages were visually checked; no browser interaction test was performed.

The jagged cut boundary is formed by exposed faces of complete tetrahedra, so part of its outline reflects mesh discretization.

## Reproduction

Run from `{ROOT}` with the saved input files present. Choose a new output directory when rerunning so the completed figures remain intact:

```bash
CHERRIES_NAME=110-white-crinkle-v2 \\
CHERRIES_TAGS=tensor,optimizer,learning-rate,full-field \\
COMET_AUTO_LOG_GIT_METADATA=false \\
COMET_AUTO_LOG_GIT_PATCH=false \\
COMET_AUTO_LOG_ENV_DETAILS=false \\
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
{capture_data["command"]}
```

[Source110 terminal capture]({rel(args.log.resolve())}) records the exact command `{capture_data["command"]}`, completed in **{capture_data["wall_time"]}**, and its [Comet run]({capture_data["comet_url"]}). Downloads contain this report, source110 receipt, manifest, renderer source copies, terminal capture, and the figure archive. They omit the large VTU/NPZ states; rerendering requires the manifest-pinned saved files already in this experiment checkout. Publication verification is recorded separately.
"""
    REPORT.write_text(report, encoding="utf-8")
    source_dir = source.parent / "sources"
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    repro = [
        source,
        args.log.resolve(),
        qa_path,
        Path(summary["manifest"]["path"]),
        Path(summary["source"]["path"]),
        *sorted(source_dir.glob("*.py")),
        REPORT,
        Path(__file__),
    ]
    for path in repro:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = {
        "schema_version": 1,
        "report": record(REPORT),
        "source110": record(source),
        "figures": {
            n: {"png": record(v[0]), "pdf": record(v[1])} for n, v in figures.items()
        },
        "reproduction": [record(x) for x in repro],
        "crinkle": crinkle,
        "capture": capture_data,
        "visual_qa": record(qa_path),
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    RECEIPT.write_text(
        json.dumps(
            {
                "status": "completed_report_assembly",
                "report": record(REPORT),
                "manifest": record(MANIFEST),
                "source110": record(source),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(REPORT), "manifest": str(MANIFEST)}, indent=2))


if __name__ == "__main__":
    main()
