from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import torch
import warp as wp

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

# Reference modules only use this name in postponed annotations. The concrete
# Cherries config is declared by the material-sweep entrypoint.
type InverseConfig = Any


@dataclass(frozen=True)
class InverseCase:
    target: Literal["smile"]
    lr: float
    setup: Literal[
        "skin-estimated-prestrain",
        "skin-no-prestrain",
        "no-skin",
    ]
    label: str = ""

    @property
    def skin_enabled(self) -> bool:
        return self.setup != SETUP_NO_SKIN

    @property
    def skin_prestrain_enabled(self) -> bool:
        return self.setup == SETUP_SKIN_ESTIMATED_PRESTRAIN

    @property
    def skin_constant_tightening(self) -> float:
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
