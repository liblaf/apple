"""Plot one raw saved-objective curve per available focused 2D experiment.

This render-only analysis reads trace CSV files written by earlier inverse
solves.  It imports no physics code, evaluates no model, and makes no new
optimization step.  The curves stay separate because L1, L2, and Linf raw
objectives have different units/scales.
"""

from __future__ import annotations

# ruff: noqa: EM101, EM102, FBT003, TRY003
import csv
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
from PIL import Image, ImageDraw, ImageFont

mpl.use("Agg")
import matplotlib.pyplot as plt
import pydantic_settings as ps

from liblaf import cherries

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Case:
    name: str
    label: str
    trace: Path
    output_name: str


class Config(cherries.BaseConfig):
    """Exact CSV inputs; directories are intentionally not Cherries inputs."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    h010_root: Path = Path(
        "../../../08/31/unreachable-pork-factor-study/data/10-pork-2d"
    )
    h020_root: Path = Path("data/20-canonical-h020")
    output_dir: Path = Path("data/71-focused-existing-loss-curves")
    case: str | None = None
    replace_existing: bool = False


def digest(path: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            hasher.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hasher.hexdigest(),
    }


def read_trace(path: Path) -> tuple[list[int], list[float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"step", "objective"}.issubset(rows[0]):
        raise ValueError(f"trace lacks step/objective columns: {path}")
    steps = [int(row["step"]) for row in rows]
    objectives = [float(row["objective"]) for row in rows]
    if steps != list(range(len(steps))) or not all(
        math.isfinite(value) and value >= 0 for value in objectives
    ):
        raise ValueError(f"invalid saved objective history: {path}")
    return steps, objectives


def render(case: Case, output: Path) -> dict[str, Any]:
    steps, objectives = read_trace(case.trace)
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.plot(steps, objectives, color="#27364a", linewidth=1.2)
    axis.set_yscale("log")
    axis.set_xlabel("saved optimization state")
    axis.set_ylabel("raw saved objective (log scale)")
    axis.grid(True, alpha=0.24)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    image = Image.open(output).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 28)
    draw.text((760, 15), case.label, fill=(0, 0, 0), font=font)
    draw.text((120, 65), f"initial = {objectives[0]:.6g}", fill=(0, 0, 0), font=font)
    draw.text((1240, 965), f"final = {objectives[-1]:.6g}", fill=(0, 0, 0), font=font)
    image.save(output)
    if output.stat().st_size <= 20_000:
        raise ValueError(f"empty loss curve: {output}")
    return {
        "case": case.name,
        "label": case.label,
        "source": digest(case.trace),
        "curve": digest(output),
        "states": len(steps),
        "initial_raw_objective": objectives[0],
        "final_raw_objective": objectives[-1],
        "y_scale": "log",
        "comparable_only_within_case": True,
    }


def main(config: Config) -> None:
    if (
        config.output_dir.exists()
        and any(config.output_dir.iterdir())
        and not (config.case is not None and config.replace_existing)
    ):
        raise FileExistsError(f"loss-curve output must be empty: {config.output_dir}")
    cases = (
        Case(
            "h010-direct-nu49",
            "h=.10 | nu=.49 | independent | L2 objective",
            config.h010_root / "height-high/trace.csv",
            "h010-direct-nu49-loss.png",
        ),
        Case(
            "h020-shared-nu49",
            "h=.20 | nu=.49 | shared | L2 objective",
            config.h020_root / "h020-shared/trace.csv",
            "h020-shared-nu49-loss.png",
        ),
        Case(
            "h005-l2",
            "h=.05 | L2 objective",
            config.h010_root / "baseline/trace.csv",
            "h005-l2-loss.png",
        ),
        Case(
            "h005-l1",
            "h=.05 | L1 objective",
            config.h010_root / "loss-l1/trace.csv",
            "h005-l1-loss.png",
        ),
        Case(
            "h005-linf",
            "h=.05 | Linf objective",
            config.h010_root / "loss-linf/trace.csv",
            "h005-linf-loss.png",
        ),
    )
    if any(not case.trace.is_file() for case in cases):
        raise FileNotFoundError("one or more saved trace CSVs are missing")
    selected = tuple(case for case in cases if config.case in {None, case.name})
    if not selected:
        raise ValueError(f"unknown loss-curve case: {config.case}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results = [render(case, config.output_dir / case.output_name) for case in selected]
    receipt = config.output_dir / "receipt.json"
    previous = json.loads(receipt.read_text()) if receipt.is_file() else {"results": []}
    changed = {item["case"] for item in results}
    retained = [item for item in previous["results"] if item["case"] not in changed]
    receipt.write_text(
        json.dumps(
            {
                "status": "ok",
                "physics_runs": 0,
                "losses_are_separate": True,
                "results": [*retained, *results],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for result in results:
        cherries.log_asset(Path(result["curve"]["path"]))
    cherries.log_asset(receipt)
    logger.info(
        "Wrote %s separate saved-objective curves to %s",
        len(results),
        config.output_dir,
    )


if __name__ == "__main__":
    cherries.main(main)
