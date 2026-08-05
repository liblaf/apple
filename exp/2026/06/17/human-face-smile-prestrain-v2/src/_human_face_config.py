from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pydantic_settings as ps
import torch
import warp as wp

from liblaf import cherries

logger = logging.getLogger(__name__)

SOURCE_MESH = Path(
    "/home/liblaf/github/liblaf/melon/exp/2026/05/27/head/data/62-tetmesh-3191k.vtu"
)
APONEUROSIS_FRACTION = "AponeurosisFraction"
FAT_FRACTION = "FatFraction"
MUSCLE_FRACTION = "MuscleFraction"
FRACTION_SUM = "FractionSum"
ACTIVE_FRACTION = "ActiveFraction"
TARGET_FINITE = "TargetFinite"
SMILE_LOSS_MASK = "SmileLossMask"
SMILE_TARGET = "Smile"
IS_FACE = "IsFace"
IS_FIXED = "IsFixed"
IN_FACE_CONVEX = "InFaceConvex"
VTK_ORIGINAL_CELL_IDS = "vtkOriginalCellIds"
VTK_ORIGINAL_POINT_IDS = "vtkOriginalPointIds"
SETUP_SKIN_ESTIMATED_PRESTRAIN = "skin-estimated-prestrain"
SETUP_SKIN_ESTIMATED_PLUS_TIGHTENING = "skin-estimated-plus-tightening"
SETUP_SKIN_NO_PRESTRAIN = "skin-no-prestrain"
SETUP_NO_SKIN = "no-skin"
FAT_E = 0.003
FAT_NU = 0.49
MUSCLE_E = 0.030
MUSCLE_NU = 0.49
APONEUROSIS_E = 0.10
APONEUROSIS_NU = 0.35
SKIN_E = 0.20
SKIN_NU = 0.49
SKIN_THICKNESS = 0.001
SKIN_EXTRA_TIGHTENING = 0.02
LOSS_SCALE = 1.0e6
ADAM_EPS = 1.0e-8 * LOSS_SCALE
ACTIVE_FRACTION_TOL = 1.0e-6
FORWARD_RTOL = 5.0e-4
FORWARD_ATOL = 1.0e-10
FORWARD_MAX_STEPS = 5000
ADJOINT_RTOL = 5.0e-4
ADJOINT_ATOL = 0.0
ADJOINT_MAXITER = 10_000
INVERSE_PATIENCE = 20
DEFAULT_TIME_BUDGET_HOURS = 10.0
DEFAULT_TIME_RESERVE_MINUTES = 45.0


@dataclass(frozen=True)
class InverseCase:
    target: Literal["smile"]
    lr: float
    setup: Literal[
        "skin-estimated-prestrain",
        "skin-estimated-plus-tightening",
        "skin-no-prestrain",
        "no-skin",
    ]
    label: str = ""

    @property
    def skin_enabled(self) -> bool:
        return self.setup != SETUP_NO_SKIN

    @property
    def skin_prestrain_enabled(self) -> bool:
        return self.setup in {
            SETUP_SKIN_ESTIMATED_PRESTRAIN,
            SETUP_SKIN_ESTIMATED_PLUS_TIGHTENING,
        }

    @property
    def skin_constant_tightening(self) -> float:
        if self.setup == SETUP_SKIN_ESTIMATED_PLUS_TIGHTENING:
            return SKIN_EXTRA_TIGHTENING
        return 0.0

    @property
    def setup_label(self) -> str:
        return self.setup

    @property
    def stem(self) -> str:
        stem = f"20-human-face-{self.target}-{self.setup_label}-{label_lr(self.lr)}"
        if self.label:
            stem = f"{stem}-{self.label}"
        return stem


class PrepareConfig(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(SOURCE_MESH)
    output_mesh: Path = cherries.output("10-human-face-prepared.vtu", mkdir=True)
    output_summary: Path = cherries.output(
        "10-human-face-prepared-summary.json", mkdir=True
    )
    output_skin_prestrain: Path = cherries.output(
        "10-smile-isface-skin-estimated-prestrain.vtp", mkdir=True
    )
    output_skin_plus_tightening: Path = cherries.output(
        "10-smile-isface-skin-estimated-plus-tightening.vtp", mkdir=True
    )
    area_ratio_floor: float = 0.1
    extra_tightening: float = SKIN_EXTRA_TIGHTENING


class InverseConfig(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input("10-human-face-prepared.vtu")
    output_summary: Path = cherries.output("20-inverse-summary.json", mkdir=True)
    output_table: Path = cherries.output("20-inverse-table.md", mkdir=True)
    live_plot_dir: Path = Path("figs/live")

    target: str = "smile"
    case_set: str = "required"
    case_label: str = ""
    initial_activation_mesh: Path | None = None
    use_initial_displacement: bool = False
    inverse_lr: float = 1.0
    loss_scale: float = LOSS_SCALE
    adam_eps: float = ADAM_EPS
    inverse_max_steps: int = 200
    mandatory_baseline_steps: int = 200
    segment_steps: int = 12
    live_snapshot_interval: int = 8
    area_ratio_floor: float = 0.1
    diagnostic_min_delta_rel: float = 1.0e-3
    flat_log_slope_tol: float = 5.0e-3
    aggressive_lr_factor: float = 2.0
    slow_lr_factor: float = 1.5
    lr_shrink_factor: float = 0.5
    max_lr: float = 1.0
    min_lr: float = 0.00375
    loss_deterioration_rel: float = 1.0e-2
    time_budget_hours: float = DEFAULT_TIME_BUDGET_HOURS
    reserve_minutes: float = DEFAULT_TIME_RESERVE_MINUTES
    step_time_budget_s: float = 180.0
    require_convergence: bool = False
    require_solver_success: bool = True


def configure_runtime() -> None:
    if not torch.cuda.is_available():
        msg = "This experiment uses Warp kernels through Torch and needs CUDA."
        raise RuntimeError(msg)
    logging.getLogger("liblaf.apple.forward._forward").setLevel(logging.WARNING)
    logging.getLogger("liblaf.apple.inverse._diff_forward").setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message=r"The \.grad attribute of a Tensor that is not a leaf Tensor.*",
        category=UserWarning,
    )
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cuda")
    wp.config.mode = "release"
    wp.init()


def label_lr(lr: float) -> str:
    text = f"{lr:g}"
    if text.startswith("0."):
        text = text[2:]
    elif text.startswith("-0."):
        text = f"m{text[3:]}"
    else:
        text = text.replace("-", "m").replace(".", "p")
    return f"lr{text}"


def selected_cases(cfg: InverseConfig) -> list[InverseCase]:
    targets = ["smile"] if cfg.target == "all" else split_csv(cfg.target)
    invalid = sorted(set(targets) - {"smile"})
    if invalid:
        msg = f"unknown target(s) {invalid}; expected smile or all"
        raise ValueError(msg)
    return [
        InverseCase(
            target=target,  # pyright: ignore[reportArgumentType]
            lr=cfg.inverse_lr,
            setup=setup,
            label=cfg.case_label,
        )
        for target in targets
        for setup in skin_variants(cfg.case_set)
    ]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def skin_variants(case_set: str) -> list[str]:
    aliases = {
        "all": [
            SETUP_SKIN_NO_PRESTRAIN,
            SETUP_NO_SKIN,
            SETUP_SKIN_ESTIMATED_PRESTRAIN,
            SETUP_SKIN_ESTIMATED_PLUS_TIGHTENING,
        ],
        "required": [
            SETUP_SKIN_ESTIMATED_PLUS_TIGHTENING,
            SETUP_SKIN_NO_PRESTRAIN,
            SETUP_NO_SKIN,
        ],
        SETUP_SKIN_ESTIMATED_PRESTRAIN: [SETUP_SKIN_ESTIMATED_PRESTRAIN],
        "skin-prestrain": [SETUP_SKIN_ESTIMATED_PRESTRAIN],
        "estimated-prestrain": [SETUP_SKIN_ESTIMATED_PRESTRAIN],
        SETUP_SKIN_ESTIMATED_PLUS_TIGHTENING: [SETUP_SKIN_ESTIMATED_PLUS_TIGHTENING],
        "skin-prestrain-tightening": [SETUP_SKIN_ESTIMATED_PLUS_TIGHTENING],
        "estimated-plus-tightening": [SETUP_SKIN_ESTIMATED_PLUS_TIGHTENING],
        SETUP_SKIN_NO_PRESTRAIN: [SETUP_SKIN_NO_PRESTRAIN],
        "skin": [SETUP_SKIN_NO_PRESTRAIN],
        "skin-pre0": [SETUP_SKIN_NO_PRESTRAIN],
        SETUP_NO_SKIN: [SETUP_NO_SKIN],
    }
    if case_set in aliases:
        return aliases[case_set]
    variants: list[str] = []
    for item in split_csv(case_set):
        if item not in aliases or len(aliases[item]) != 1:
            msg = (
                f"unknown case set {item!r}; expected required, "
                "skin-no-prestrain, no-skin, skin-estimated-prestrain, "
                "skin-estimated-plus-tightening, or a comma list"
            )
            raise ValueError(msg)
        variants.extend(aliases[item])
    return variants
