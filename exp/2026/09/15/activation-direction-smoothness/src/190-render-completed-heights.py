"""Verify and plot four height endpoints after completing the Adam budget."""

# ruff: noqa: SLF001

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import scipy.linalg as la
import study

from liblaf import cherries

slide = importlib.import_module("170-render-free-heights")
render = slide.render
GROUP = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
FIELDS = (
    "step",
    "normalized_loss",
    "objective_normalized",
    "roughness",
    "tensor_neighbor_rms",
    "min_J",
    "force_residual_inf",
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("190-completed-height-figures")
    dpi: int = 240


def load(height: float) -> tuple:
    source = (
        GROUP
        / "data"
        / (
            "tune-w0"
            if height == 0.05
            else f"180-free-height-continuation/h{round(height * 1000):03d}"
        )
    )
    name = f"h{round(height * 1000):03d}-unconstrained-w0"
    folder = source / name
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["accepted_iterations"] == 1200
    assert summary["height"] == height
    with (folder / "trace.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    trace = {key: np.asarray([float(row[key]) for row in rows]) for key in FIELDS}
    failed = (
        []
        if height == 0.05
        else [row for row in rows if row["forward_converged"] == "False"]
    )
    np.testing.assert_array_equal(trace["step"], np.arange(int(trace["step"][0]), 1201))
    for name in ("summary.json", "trace.csv", "checkpoint.npz", "history.npz"):
        cherries.log_input(folder / name)
    case = render.Case(
        source,
        summary,
        trace,
        render._load_npz(folder / "history.npz"),
        render._load_npz(folder / "checkpoint.npz"),
    )
    if height != 0.05:
        assert len(failed) == sum(
            count
            for key, count in summary["forward_status_counts"].items()
            if key != "converged"
        )
        assert all(
            (rows[i]["forward_seed_was_reset"] == "True")
            == (rows[i - 1]["forward_converged"] == "False")
            for i in range(1, len(rows))
        )
        assert summary["final_forward_converged"] == (
            rows[-1]["forward_converged"] == "True"
        )
    return case, failed


def main(cfg: Config) -> None:
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    mesh = study.ph.build_mesh(100, 10)
    cases, records, failures = [], {}, {}
    for height in slide.HEIGHTS:
        case, failed = load(height)
        cases.append(case)
        failures[str(height)] = failed
        checkpoint = case.checkpoint
        assert int(checkpoint["step"]) == 1200
        np.testing.assert_array_equal(checkpoint["u"], case.history["u"][-1])
        np.testing.assert_array_equal(mesh.p, case.history["points"])
        np.testing.assert_array_equal(mesh.tri, case.history["triangles"])
        np.testing.assert_array_equal(
            study.matrices(mesh, checkpoint["controls"], case.mode), checkpoint["B"]
        )
        u = checkpoint["u"].ravel()[mesh.free]
        _, residual, H, J = study.ph.assemble(mesh, u, checkpoint["B"])
        loss, _, _ = study.ph.loss(mesh, u, height, "l2")
        np.testing.assert_allclose(
            loss / height**2, case.summary["final"]["normalized_loss"], rtol=1e-12
        )
        np.testing.assert_allclose(
            J.min(), case.summary["final"]["min_J"], rtol=1e-10, atol=1e-13
        )
        force = float(np.linalg.norm(residual, np.inf))
        converged = force <= 1e-10
        assert converged == case.summary.get("final_forward_converged", True)
        eigenvalue = float(
            la.eigh(
                ((H + H.T) * 0.5).toarray(), subset_by_index=[0, 0], eigvals_only=True
            )[0]
        )
        if "final_smallest_hessian_eigenvalue" in case.summary:
            np.testing.assert_allclose(
                eigenvalue,
                case.summary["final_smallest_hessian_eigenvalue"],
                rtol=1e-7,
                atol=1e-12,
            )
        records[height] = {
            "height": height,
            "step": 1200,
            "normalized_L2": loss / height**2,
            "fit_rms": float(np.sqrt(loss)),
            "tensor_neighbor_rms": case.summary["final"]["tensor_neighbor_rms"],
            "failed_solves": len(failed),
            "rest_seeded_solves": case.summary.get("forward_resets_used", 0),
            "final_forward_converged": converged,
            "force_residual_inf": force,
            "min_physical_J": float(J.min()),
            "inverted_cells": int(np.count_nonzero(J <= 0)),
            "minimum_hessian_eigenvalue": eigenvalue,
            "checkpoint": str(case.source / case.name / "checkpoint.npz"),
            "checkpoint_sha256": hashlib.sha256(
                (case.source / case.name / "checkpoint.npz").read_bytes()
            ).hexdigest(),
        }
        np.savez_compressed(
            output / f"{case.name}-glyphs.npz", **render._activation_geometry(case)
        )
        LOG.info("Verified h=%.2f: %s", height, records[height])
    # Repeating the established h=.20 continuation must preserve its endpoint.
    previous = render._load_npz(
        GROUP / "data/140-reset-continuation/h200-unconstrained-w0/checkpoint.npz"
    )
    for key in ("controls", "moment", "variance", "u", "B"):
        np.testing.assert_array_equal(cases[-1].checkpoint[key], previous[key])
    checks = slide.plot(cases, output, cfg.dpi, completion=records)
    checks["h200_reproduces_previous_reset_endpoint_exactly"] = True
    for name, value in (
        ("comparison.json", list(records.values())),
        ("forward-failures.json", failures),
        ("delivery-checks.json", checks),
    ):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    snapshot = output / "source"
    snapshot.mkdir()
    for source in (Path(__file__), Path(slide.__file__), Path(render.__file__)):
        shutil.copy2(source, snapshot / source.name)


if __name__ == "__main__":
    cherries.main(main, profile=render.ProfileFigures)
