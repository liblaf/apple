"""Record the selected 2 mm / 5 degree face shape-loss calibration."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
import pydantic
import pydantic_settings as ps
import pyvista as pv
from experiment import Profile

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("110-shape-loss-config")
    position_rms_tolerance_mm: float = pydantic.Field(default=2.0, gt=0)
    normal_angle_tolerance_deg: float = pydantic.Field(default=5.0, gt=0, lt=180)


def main(cfg: Config) -> None:
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    baseline_path = cherries.input("90-beta1/protocol.json")
    baseline = json.loads(baseline_path.read_text())
    fixture = baseline["fixture"]["skin.vtp"]
    skin_path = Path(fixture["path"])
    digest = hashlib.sha256(skin_path.read_bytes()).hexdigest()
    assert digest == fixture["sha256"]
    skin = pv.read(skin_path)
    diagonal_mm = float(1000 * np.linalg.norm(np.ptp(skin.points, axis=0)))

    theta = math.radians(cfg.normal_angle_tolerance_deg)
    normal_at_tolerance = 4 * math.sin(theta / 2) ** 2
    position_at_tolerance_mm2 = cfg.position_rms_tolerance_mm**2 / 3
    l_ref_mm = math.sqrt(position_at_tolerance_mm2 / normal_at_tolerance)
    normalized_position_at_tolerance = position_at_tolerance_mm2 / l_ref_mm**2
    assert math.isclose(
        normalized_position_at_tolerance, normal_at_tolerance, rel_tol=1e-14
    )
    initial_position_mm2 = baseline["normalization"]["L20"]
    initial_normal = baseline["normalization"]["N0"]
    initial_position = initial_position_mm2 / l_ref_mm**2

    selected = {
        "objective": "position_component_mse_mm2 / l_ref_mm**2 + normal_weight * normal_chord_squared",
        "l_ref_mm": l_ref_mm,
        "normal_weight": 1.0,
        "length_mode": "fixed_physical_length",
        "position_convention": "reference-area-weighted vector position MSE divided by 3, in mm^2",
        "normal_convention": "reference-area-weighted mean squared chord distance of corresponding oriented unit triangle normals",
        "calibration": {
            "position_vector_rms_mm": cfg.position_rms_tolerance_mm,
            "normal_reference_angle_deg": cfg.normal_angle_tolerance_deg,
        },
    }
    receipt = {
        "calibration_passed": True,
        "optimization_run": False,
        "position_contribution_at_tolerance": normalized_position_at_tolerance,
        "normal_contribution_at_tolerance": normal_at_tolerance,
        "reference_bbox_diagonal_mm": diagonal_mm,
        "equivalent_bbox_fraction_for_this_face": l_ref_mm / diagonal_mm,
        "initial_position_component_mse_mm2": initial_position_mm2,
        "initial_normal_chord_squared": initial_normal,
        "initial_position_contribution": initial_position,
        "initial_normal_contribution": initial_normal,
        "initial_position_to_normal_ratio": initial_position / initial_normal,
        "equivalent_previous_beta_for_data_terms": (
            l_ref_mm**2 * initial_normal / initial_position_mm2
        ),
        "normalization_source": str(baseline_path.resolve()),
        "normalization_source_sha256": hashlib.sha256(
            baseline_path.read_bytes()
        ).hexdigest(),
        "skin_fixture": fixture,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for name, value in (("loss-config.json", selected), ("calibration.json", receipt)):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    cherries.log_metrics(
        {
            "l_ref_mm": l_ref_mm,
            "position_at_tolerance": normalized_position_at_tolerance,
            "normal_at_tolerance": normal_at_tolerance,
            "initial_position_to_normal_ratio": initial_position / initial_normal,
        }
    )
    LOG.info(
        "Selected l_ref=%.12g mm; equal contributions=%.12g",
        l_ref_mm,
        normal_at_tolerance,
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
