from __future__ import annotations

import csv
import importlib.util
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pydantic_settings as ps

from liblaf import cherries

logger = logging.getLogger(__name__)


def load_base_module() -> Any:
    path = Path(__file__).with_name("20-toy-unreachable-inverse.py")
    spec = importlib.util.spec_from_file_location("toy_unreachable_inverse", path)
    if spec is None or spec.loader is None:
        msg = f"could not load base toy experiment from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def label_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def signed_label(value: float) -> str:
    prefix = "p" if value >= 0.0 else "m"
    return prefix + label_float(abs(value))


@dataclass(frozen=True)
class SweepCase:
    resolution: Any
    mode: Literal["stretch", "squash"]
    target_y: float
    nu: float

    @property
    def stem(self) -> str:
        return (
            f"30-toy-{self.mode}-{self.resolution.name}"
            f"-dy-{signed_label(self.target_y)}-nu-{label_float(self.nu)}"
        )


class Config(BASE.Config):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    output_summary: Path = cherries.output("30-toy-target-magnitude-sweep-summary.json")
    output_csv: Path = cherries.output("30-toy-target-magnitude-sweep-cases.csv")
    output_table: Path = cherries.output("30-toy-target-magnitude-sweep-table.md")

    resolutions: tuple[str, ...] = ("coarse",)
    target_magnitudes: tuple[float, ...] = (0.005, 0.01, 0.02, 0.04)
    nus: tuple[float, ...] = (0.49, 0.30)
    inverse_max_steps: int = 120
    series_stride: int = 20


def selected_cases(cfg: Config) -> list[SweepCase]:
    cases: list[SweepCase] = []
    for nu in cfg.nus:
        for resolution_name in cfg.resolutions:
            if resolution_name not in BASE.RESOLUTION_SPECS:
                msg = (
                    f"unknown resolution {resolution_name!r}; "
                    f"choose from {sorted(BASE.RESOLUTION_SPECS)}"
                )
                raise ValueError(msg)
            resolution = BASE.RESOLUTION_SPECS[resolution_name]
            for magnitude in cfg.target_magnitudes:
                for mode in cfg.modes:
                    target_y = magnitude if mode == "stretch" else -magnitude
                    cases.append(
                        SweepCase(
                            resolution=resolution,
                            mode=mode,
                            target_y=target_y,
                            nu=nu,
                        )
                    )
    return cases


def solve_case(case: SweepCase, cfg: Config) -> dict[str, Any]:
    old_nu = cfg.nu
    cfg.nu = case.nu
    try:
        row = BASE.solve_case(case, cfg)
    finally:
        cfg.nu = old_nu
    row["nu"] = float(case.nu)
    row["target_magnitude"] = float(abs(case.target_y))
    target_volume = float(row["target/volume/rel_change"])
    inverse_volume = float(row["inverse/volume/rel_change"])
    row["inverse/volume/rel_change_fraction_of_target"] = (
        inverse_volume / target_volume if target_volume != 0.0 else math.nan
    )
    row["inverse/volume/abs_rel_change_fraction_of_target_signed"] = (
        float(row["inverse/volume/abs_rel_change"]) / target_volume
        if target_volume != 0.0
        else math.nan
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    excluded = {"trace", "y_levels"}
    keys = sorted({key for row in rows for key in row if key not in excluded})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def format_float(value: Any) -> str:
    if not isinstance(value, int | float):
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{float(value):.6g}"


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| case | nu | target y | target volume change | inverse volume change | inverse / target volume | best error RMS | best error / target RMS | top y std | top edge RMS | best step | convergence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    format_float(row["nu"]),
                    format_float(row["target_y"]),
                    format_float(row["target/volume/rel_change"]),
                    format_float(row["inverse/volume/rel_change"]),
                    format_float(row["inverse/volume/rel_change_fraction_of_target"]),
                    format_float(row["best/error_rms"]),
                    format_float(row["best/error_rms_fraction_of_target"]),
                    format_float(row["inverse/top_y/std"]),
                    format_float(row["inverse/top_y/edge_rms"]),
                    format_float(row["best/step"]),
                    str(row["convergence/status"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"nu={row['nu']:g},mode={row['mode']},resolution={row['resolution']}"
        group = grouped.setdefault(
            key,
            {
                "nu": row["nu"],
                "mode": row["mode"],
                "resolution": row["resolution"],
                "target_magnitudes": [],
                "target_volume_rel_change": [],
                "inverse_volume_rel_change": [],
                "inverse_volume_fraction_of_target": [],
                "best_error_rms": [],
                "top_y_std": [],
                "top_edge_rms": [],
            },
        )
        group["target_magnitudes"].append(row["target_magnitude"])
        group["target_volume_rel_change"].append(row["target/volume/rel_change"])
        group["inverse_volume_rel_change"].append(row["inverse/volume/rel_change"])
        group["inverse_volume_fraction_of_target"].append(
            row["inverse/volume/rel_change_fraction_of_target"]
        )
        group["best_error_rms"].append(row["best/error_rms"])
        group["top_y_std"].append(row["inverse/top_y/std"])
        group["top_edge_rms"].append(row["inverse/top_y/edge_rms"])
    path.write_text(
        json.dumps(
            {"cases": rows, "groups": list(grouped.values())},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main(cfg: Config) -> None:
    BASE.configure_runtime()
    rows: list[dict[str, Any]] = []
    for case in selected_cases(cfg):
        rows.append(solve_case(case, cfg))
    write_summary(cfg.output_summary, rows)
    write_csv(cfg.output_csv, rows)
    write_table(cfg.output_table, rows)
    logger.info("Wrote %s", cfg.output_summary)
    logger.info("Wrote %s", cfg.output_csv)
    logger.info("Wrote %s", cfg.output_table)


if __name__ == "__main__":
    cherries.main(main)
