"""Verify saved ablation states, solver receipts, and source snapshots on CPU."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

GROUP = Path(__file__).resolve().parents[1]
CASES = ("no-skin", "skin-zero", "skin-local-1pct")
CANON = (
    Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
    / "exp/2026/09/08/physical-volume-baseline/data/20-baseline"
)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def file_record(path):
    return dict(
        path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest(path)
    )


def main():
    out = GROUP / "data/50-verification"
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir())
    q0 = np.load(CANON / "final.npz")["q"]
    field_hash = digest(GROUP / "data/10-prestrain-field/skin-prestrain.npz")
    cases = {}
    physics_hashes = set()
    material_hashes = set()
    for stage in ("forward", "refit"):
        for case in CASES:
            directory = (
                GROUP
                / "data"
                / f"{'20' if stage == 'forward' else '30'}-{stage}-{case}"
            )
            summary = json.loads((directory / "summary.json").read_text())
            expected_status = (
                "completed_fixed_activation_forward"
                if stage == "forward"
                else "completed_200_updates_not_stationarity_certified"
            )
            assert summary["status"] == expected_status
            end = 0 if stage == "forward" else 200
            with (directory / "trace.csv").open(newline="") as stream:
                trace = list(csv.DictReader(stream))
            assert [int(row["step"]) for row in trace] == list(range(end + 1))
            assert summary["last_evaluated_step"] == end
            for row in trace:
                assert row["forward_success"] == "True"
                assert row["adjoint_success"] == (
                    "True" if stage == "refit" else "False"
                )
                numeric = {
                    key: float(value)
                    for key, value in row.items()
                    if value not in ("", "True", "False")
                }
                assert all(math.isfinite(value) for value in numeric.values())
                assert math.isclose(
                    numeric["objective_mm2"] * 3,
                    numeric["fit_rms_mm"] ** 2,
                    rel_tol=1e-11,
                )
            best = min(trace, key=lambda row: float(row["objective_mm2"]))
            assert int(best["step"]) == summary["best_step"]
            receipts = [
                json.loads(line)
                for line in (directory / "solver-receipts.jsonl")
                .read_text()
                .splitlines()
            ]
            assert [r["step"] for r in receipts] == list(range(end + 1))
            assert all(r["forward"]["success"] for r in receipts)
            if stage == "refit":
                assert all(r["adjoint"]["success"] for r in receipts)
            provenance = json.loads((directory / "provenance.json").read_text())
            assert provenance["field"]["sha256"] == field_hash
            assert (
                provenance["optimizer"]["moments"]
                == "fresh zero moments for every case; no cached gradient from old physics"
            )
            snapshots = 0
            for module, record in provenance["sources"].items():
                assert digest(Path(record["snapshot"])) == record["sha256"]
                if module == "local_physics":
                    physics_hashes.add(record["sha256"])
                if module == "volume_preserving_active":
                    material_hashes.add(record["sha256"])
                snapshots += 1
            origin = np.load(directory / "step-0000.npz")
            assert np.array_equal(origin["q"], q0)
            final = np.load(directory / "final.npz")
            last = np.load(directory / "last.npz")
            assert int(final["step"]) == int(best["step"])
            assert int(last["step"]) == end
            assert bool(final["solver_valid"]) and bool(last["solver_valid"])
            for state in (origin, final, last):
                assert np.isfinite(state["u"]).all() and np.isfinite(state["q"]).all()
            for step in range(end + 1):
                surface = np.load(directory / f"surface-{step:04d}.npz")
                assert int(surface["step"]) == step
                assert surface["u"].shape == (15299, 3)
                assert np.isfinite(surface["u"]).all()
                assert len(np.unique(surface["point_ids"])) == 15299
                if step == int(final["step"]):
                    assert np.array_equal(
                        surface["u"], final["u"][surface["point_ids"]]
                    )
                if step % 10 == 0:
                    state = np.load(directory / f"step-{step:04d}.npz")
                    assert int(state["step"]) == step
                    assert np.array_equal(
                        surface["u"], state["u"][surface["point_ids"]]
                    )
            if stage == "refit":
                optimizer = torch.load(
                    directory / "optimizer-latest.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                assert optimizer["step"] == end
                assert optimizer["case"] == case
                assert optimizer["field_sha256"] == field_hash
                assert np.array_equal(optimizer["q"].numpy(), last["q"])
                assert np.array_equal(optimizer["u"], last["u"])
                assert torch.isfinite(optimizer["gradient"]).all()
                states = optimizer["optimizer"]["state"]
                assert len(states) == 1
                assert int(next(iter(states.values()))["step"]) == end
                group = optimizer["optimizer"]["param_groups"][0]
                assert group["lr"] == 0.3 and group["eps"] == 0.01
                assert group["betas"] == (0.9, 0.999)
            artifacts = {
                name: file_record(directory / name)
                for name in (
                    "summary.json",
                    "trace.csv",
                    "provenance.json",
                    "solver-receipts.jsonl",
                    "final.npz",
                    "last.npz",
                    "step-0000.npz",
                )
            }
            cases[f"{stage}/{case}"] = dict(
                status="passed",
                evaluations=len(trace),
                source_snapshots_verified=snapshots,
                surface_states_verified=end + 1,
                best_step=int(best["step"]),
                artifacts=artifacts,
            )
    assert len(physics_hashes) == len(material_hashes) == 1
    result = dict(
        status="passed",
        scope="CPU-only independent artifact verification; no physics solve or optimization",
        cases=cases,
        local_physics_sha256=next(iter(physics_hashes)),
        volume_material_sha256=next(iter(material_hashes)),
        fixed_field_sha256=field_hash,
        source=file_record(Path(__file__)),
    )
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("status", "fixed_field_sha256")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
