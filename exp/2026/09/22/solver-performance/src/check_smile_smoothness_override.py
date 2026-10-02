"""CPU-only wiring check for the Smile harness smoothness ablation switch."""

from __future__ import annotations

import sys
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

SOURCE = Path(__file__).with_name("20-fit-smile.py")
loader = SourceFileLoader("check_smile_smoothness_harness", str(SOURCE))
spec = spec_from_loader(loader.name, loader)
assert spec is not None
harness = module_from_spec(spec)
sys.modules[loader.name] = harness
loader.exec_module(harness)

calibration = {"strong_weight": 5.066584049455902}

default_cfg = harness.Config(_cli_parse_args=False)
default = harness.smoothness_weight_receipt(default_cfg, calibration)
assert default == {
    "calibrated_strong_weight": calibration["strong_weight"],
    "override": None,
    "effective_weight": calibration["strong_weight"],
    "mode": "inherited_calibration",
}

zero_cfg = harness.Config(_cli_parse_args=False, smoothness_weight=0.0)
zero = harness.smoothness_weight_receipt(zero_cfg, calibration)
assert zero == {
    "calibrated_strong_weight": calibration["strong_weight"],
    "override": 0.0,
    "effective_weight": 0.0,
    "mode": "explicit_override",
}
cli_cfg = harness.Config(_cli_parse_args=["--smoothness-weight", "0"])
assert cli_cfg.smoothness_weight == 0.0

runner = harness.load_fitter_module()
# Avoid physics construction: this is nevertheless the exact production
# Fitter class whose smooth_weight field run_arm later binds.
fitter = object.__new__(runner.Fitter)
harness.bind_smoothness_weight(fitter, default)
assert fitter.smooth_weight == calibration["strong_weight"]
harness.bind_smoothness_weight(fitter, zero)
assert fitter.smooth_weight == 0.0

print("smoothness override wiring passed")
