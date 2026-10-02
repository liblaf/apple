"""Export compact, read-only shape snapshots for an SSH live mirror."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np


def write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def export(source: Path, output: Path, pid: int, host: str) -> None:  # noqa: C901
    output.mkdir(parents=True, exist_ok=True)
    alive = Path(f"/proc/{pid}/cmdline").exists()
    if alive:
        alive = (
            b"46-run-active-strain-chain.py"
            in Path(f"/proc/{pid}/cmdline").read_bytes()
        )
    write_bytes(
        output / "remote-status.json",
        json.dumps(
            {
                "host": host,
                "pid": pid,
                "process_alive": alive,
                "checked_at": dt.datetime.now(dt.UTC).isoformat(),
            }
        ).encode(),
    )
    if not (source / "chain-status.json").exists():
        return
    if not (output / "mesh.npz").exists():
        with np.load(source / "mesh.npz", allow_pickle=False) as mesh:
            save_npz(
                output / "mesh.npz",
                **{
                    key: mesh[key]
                    for key in (
                        "skin_ids",
                        "rest_points",
                        "target_displacement_skin",
                        "triangles",
                    )
                },
            )
    with np.load(output / "mesh.npz", allow_pickle=False) as mesh:
        ids = mesh["skin_ids"]
        rest = mesh["rest_points"][ids]
    for name in ("protocol.json", "chain-status.json", "summary.json"):
        if (source / name).exists():
            write_bytes(output / name, (source / name).read_bytes())
    chain = json.loads((output / "chain-status.json").read_text())
    for stage in chain["stages"]:
        current, target = source / stage["stage_dir"], output / stage["stage_dir"]
        if (
            not (current / "summary.json").exists()
            or not (current / "last.npz").exists()
        ):
            continue
        summary_blob = (current / "summary.json").read_bytes()
        summary = json.loads(summary_blob)
        trace = (current / "trace.csv").read_bytes()
        rows = list(csv.DictReader(io.StringIO(trace.decode())))
        with np.load(current / "last.npz", allow_pickle=False) as state:
            step = int(state["step"])
            if (
                step != summary["last_step"]
                or not rows
                or int(rows[-1]["step"]) != step
            ):
                continue  # The runner publishes these files consecutively.
            target.mkdir(parents=True, exist_ok=True)
            previous = target / "live-fit.npz"
            previous_step = -1
            if previous.exists():
                with np.load(previous, allow_pickle=False) as old:
                    previous_step = int(old["step"])
            if previous_step != step:
                save_npz(
                    previous,
                    fit_positions=rest + state["u"][ids],
                    step=np.asarray(step),
                )
        write_bytes(target / "summary.json", summary_blob)
        write_bytes(target / "trace.csv", trace)
        if (current / "gradient-balance.json").exists():
            write_bytes(
                target / "gradient-balance.json",
                (current / "gradient-balance.json").read_bytes(),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--skip-mesh", action="store_true")
    args = parser.parse_args()
    export(args.source, args.output, args.pid, args.host)
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
        for path in sorted(args.output.rglob("*")):
            if args.skip_mesh and path == args.output / "mesh.npz":
                continue
            if path.is_file() and not path.name.endswith(".tmp"):
                archive.add(path, arcname=path.relative_to(args.output))


if __name__ == "__main__":
    main()
