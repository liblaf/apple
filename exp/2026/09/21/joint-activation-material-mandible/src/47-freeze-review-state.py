"""Freeze one coherent neutral checkpoint and its trace for related figures."""

from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path

import torch
from joint_common import GROUP, ProfileJoint, sha256, write_json

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    run_dir: Path
    output_dir: Path = GROUP / "data/review-snapshots"


def main(cfg: Config) -> None:
    checkpoint_path = cfg.run_dir / "terminal.pt"
    payload = checkpoint_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    assert checkpoint["stage"] == "neutral"
    trace_payload = (cfg.run_dir / "trace.json").read_bytes()
    trace = json.loads(trace_payload)
    trace = [row for row in trace if row["update"] <= checkpoint["update"]]
    assert trace[-1]["update"] == checkpoint["update"]
    assert trace[-1]["shared"] == checkpoint["shared_coefficients"].tolist()
    target = cfg.output_dir / (
        f"{cfg.run_dir.name}-u{checkpoint['update']:04d}-{digest[:8]}"
    )
    target.mkdir(parents=True, exist_ok=False)
    (target / "terminal.pt").write_bytes(payload)
    write_json(target / "trace.json", trace)
    write_json(
        target / "snapshot.json",
        {
            "schema": "joint-neutral-review-snapshot-v1",
            "source_run": str(cfg.run_dir.resolve()),
            "source_checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": digest,
            "source_trace_sha256": hashlib.sha256(trace_payload).hexdigest(),
            "snapshot_trace_sha256": sha256(target / "trace.json"),
            "update": checkpoint["update"],
            "scope": "immutable visualization input; convergence unchanged",
        },
    )
    LOG.info("Frozen review state: %s", target)
    cherries.log_output(target)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
