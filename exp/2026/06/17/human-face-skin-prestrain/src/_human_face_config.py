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
FAT_E = 0.003
FAT_NU = 0.49
MUSCLE_E = 0.030
MUSCLE_NU = 0.49
APONEUROSIS_E = 0.10
APONEUROSIS_NU = 0.35
SKIN_E = 0.20
SKIN_NU = 0.49
SKIN_THICKNESS = 0.001
ACTIVE_FRACTION_TOL = 1.0e-6
FORWARD_RTOL = 5.0e-4
FORWARD_ATOL = 1.0e-10
FORWARD_MAX_STEPS = 5000
ADJOINT_RTOL = 5.0e-4
ADJOINT_ATOL = 0.0
ADJOINT_MAXITER = 10_000
INVERSE_PATIENCE = 20


@dataclass(frozen=True)
class InverseCase:
    target: Literal["smile"]
    lr: float
    skin_enabled: bool
    skin_prestrain: float
    label: str = ""

    @property
    def setup_label(self) -> str:
        if not self.skin_enabled:
            return "no-skin"
        return f"skin-pre{self.skin_prestrain * 100.0:g}pct".replace(".", "p")

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


class InverseConfig(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input("10-human-face-prepared.vtu")
    output_summary: Path = cherries.output("20-inverse-summary.json", mkdir=True)
    output_table: Path = cherries.output("20-inverse-table.md", mkdir=True)

    target: str = "smile"
    case_set: str = "skin-no-prestrain,no-skin"
    case_label: str = ""
    initial_activation_mesh: Path | None = None
    use_initial_displacement: bool = False
    inverse_lr: float = 0.03
    inverse_max_steps: int = 300
    inverse_loss_min_delta: float = 2.0e-8
    require_convergence: bool = True
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
            skin_enabled=skin_enabled,
            skin_prestrain=skin_prestrain,
            label=cfg.case_label,
        )
        for target in targets
        for skin_enabled, skin_prestrain in skin_variants(cfg.case_set)
    ]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def skin_variants(case_set: str) -> list[tuple[bool, float]]:
    aliases = {
        "all": [(False, 0.0), (True, 0.0), (True, 0.05), (True, 0.10)],
        "required": [(True, 0.0), (False, 0.0)],
        "no-skin": [(False, 0.0)],
        "skin": [(True, 0.0)],
        "skin-no-prestrain": [(True, 0.0)],
        "skin-pre0": [(True, 0.0)],
        "skin-pre5": [(True, 0.05)],
        "skin-pre5pct": [(True, 0.05)],
        "skin-pre10": [(True, 0.10)],
        "skin-pre10pct": [(True, 0.10)],
    }
    if case_set in aliases:
        return aliases[case_set]
    variants: list[tuple[bool, float]] = []
    for item in split_csv(case_set):
        if item not in aliases or len(aliases[item]) != 1:
            msg = (
                f"unknown case set {item!r}; expected all, no-skin, "
                "skin-no-prestrain, skin-pre5, skin-pre10, or a comma list"
            )
            raise ValueError(msg)
        variants.extend(aliases[item])
    return variants
