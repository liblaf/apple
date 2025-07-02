import os
import sys
from pathlib import Path


def data(path: str | os.PathLike) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    path = entrypoint().parent / "data" / path
    return path


def entrypoint() -> Path:
    return Path(sys.argv[0])
