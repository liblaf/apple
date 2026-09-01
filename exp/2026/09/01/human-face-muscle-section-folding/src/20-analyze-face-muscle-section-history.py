"""Strict per-step determinant-sign receipt for the exported id64 material slab."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv

from liblaf import cherries

REQUIRED = ("RestVolume", "DetF", "DetAinv", "DetG", "SourceCellId")
FRAME = re.compile(r"step-(\d+)\.vtu")


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    history_dir: Path = Path("data/15-face-muscle-section-history")
    output_dir: Path = cherries.output(
        "20-face-muscle-section-history-analysis", mkdir=True
    )
    expected_step: int = 194


def fail(message: str) -> None:
    raise ValueError(message)


def fraction(mask: np.ndarray, volume: np.ndarray) -> float:
    total = float(volume.sum())
    if not math.isfinite(total) or total <= 0.0:
        fail("rest-volume total must be finite and positive")
    return float(volume[mask].sum() / total)


def row(path: Path, step: int) -> dict[str, Any]:
    grid = pv.read(path)
    if not isinstance(grid, pv.UnstructuredGrid) or grid.n_cells == 0:
        fail(f"{path} is not a nonempty unstructured grid")
    if any(name not in grid.cell_data for name in REQUIRED):
        fail(f"{path} does not contain all required arrays {REQUIRED}")
    values = {name: np.asarray(grid.cell_data[name]) for name in REQUIRED}
    if any(value.shape != (grid.n_cells,) for value in values.values()):
        fail(f"{path} contains non-scalar cell data")
    if not all(
        np.all(np.isfinite(values[name]))
        for name in ("RestVolume", "DetF", "DetAinv", "DetG")
    ):
        fail(f"{path} contains non-finite determinant data")
    if np.any(values["RestVolume"] <= 0.0):
        fail(f"{path} contains non-positive rest volume")
    det_f, det_a, det_g, volume = (
        values["DetF"],
        values["DetAinv"],
        values["DetG"],
        values["RestVolume"],
    )
    masks = {
        "f_negative": det_f < 0.0,
        "ainv_negative": det_a < 0.0,
        "g_negative": det_g < 0.0,
        "double_inverted": (det_f < 0.0) & (det_a < 0.0),
    }
    result: dict[str, Any] = {
        "step": step,
        "cells": grid.n_cells,
        "rest_volume": float(volume.sum()),
        "min_det_f": float(det_f.min()),
        "min_det_ainv": float(det_a.min()),
        "min_det_g": float(det_g.min()),
    }
    for name, mask in masks.items():
        result[f"{name}_cells"] = int(mask.sum())
        result[f"{name}_rest_volume_fraction"] = fraction(mask, volume)
    return result


def onset_and_persistence(
    rows: list[dict[str, Any]], key: str
) -> dict[str, int | None]:
    steps = [int(item["step"]) for item in rows if int(item[f"{key}_cells"]) > 0]
    if not steps:
        return {
            "first_onset_step": None,
            "last_present_step": None,
            "persistent_through_last": False,
        }
    return {
        "first_onset_step": min(steps),
        "last_present_step": max(steps),
        "persistent_through_last": max(steps) == int(rows[-1]["step"]),
    }


def main(config: Config) -> None:
    frame_dir = config.history_dir.resolve() / "frames"
    paths: list[tuple[int, Path]] = []
    for path in frame_dir.glob("step-*.vtu"):
        match = FRAME.fullmatch(path.name)
        if match is None:
            fail(f"unexpected history frame name {path.name}")
        paths.append((int(match.group(1)), path))
    paths.sort()
    if not paths:
        fail(f"no history frames under {frame_dir}")
    steps = [step for step, _ in paths]
    if steps != list(range(steps[-1] + 1)):
        fail(
            f"history steps must be consecutive from zero, got {steps[:3]}...{steps[-3:]}"
        )
    if config.expected_step not in steps:
        fail(f"required confirmation step {config.expected_step} is absent")
    rows = [row(path, step) for step, path in paths]
    if (
        len({item["cells"] for item in rows}) != 1
        or len({item["rest_volume"] for item in rows}) != 1
    ):
        fail("section topology/rest volume changes over history")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "history_dir": str(config.history_dir.resolve()),
        "frame_count": len(rows),
        "steps": {
            "first": 0,
            "last": steps[-1],
            "required_confirmation": config.expected_step,
        },
        "section_cells": rows[0]["cells"],
        "section_rest_volume": rows[0]["rest_volume"],
        "sign_convention": {
            "f_negative": "DetF < 0",
            "ainv_negative": "DetAinv < 0",
            "g_negative": "DetG < 0",
            "double_inverted": "DetF < 0 and DetAinv < 0",
        },
        "onset_persistence": {
            key: onset_and_persistence(rows, key)
            for key in ("f_negative", "ainv_negative", "g_negative", "double_inverted")
        },
        "step_194": next(item for item in rows if item["step"] == config.expected_step),
        "last_step": rows[-1],
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    cherries.main(main)
