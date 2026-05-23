from typing import Any

import torch
import warp as wp


def warp_default_dtype() -> Any:
    return wp.dtype_from_torch(torch.get_default_dtype())
