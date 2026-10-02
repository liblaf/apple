"""Check every saved deformation and the final report's local asset links."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import study
from liblaf.cherries import core, plugins, profiles
from PIL import Image

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]


class ProfileArtifacts(profiles.Profile):
    def init(self):
        run = core.run
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    output: Path = Path("50-artifact-checks")


def main(cfg: Config):
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    mesh = study.ph.build_mesh(100, 10)
    results = []
    for directory in ("tune-w0", "tune-w001", "tune-w01", "tune-w1"):
        folder = GROUP / "data" / directory
        for case in json.loads((folder / "summary.json").read_text()):
            path = folder / case["name"]
            history = np.load(path / "history.npz")
            assert np.array_equal(history["points"], mesh.p)
            assert np.array_equal(history["triangles"], mesh.tri)
            assert np.array_equal(history["muscle"], mesh.muscle)
            assert np.array_equal(history["top"], mesh.top)
            u = history["u"]
            F = np.einsum("feia,eib->feab", (mesh.p[None] + u)[:, mesh.tri], mesh.grad)
            J = np.linalg.det(F)
            assert np.all(J > 0), case["name"]
            fixed = mesh.lookup.reshape(-1, 2) < 0
            fixed_error = float(np.abs(u[:, fixed]).max())
            assert fixed_error == 0
            target = np.column_stack(
                (
                    np.zeros(len(mesh.top)),
                    4
                    * case["height"]
                    * mesh.p[mesh.top, 0]
                    * (1 - mesh.p[mesh.top, 0]),
                )
            )
            losses = np.mean(np.sum((u[:, mesh.top] - target) ** 2, axis=2), axis=1)
            with (path / "trace.csv").open() as f:
                trace = list(csv.DictReader(f))
            steps = history["steps"]
            assert len(steps) == len(u)
            expected_loss = np.array(
                [float(trace[int(step)]["raw_loss"]) for step in steps]
            )
            expected_J = np.array([float(trace[int(step)]["min_J"]) for step in steps])
            np.testing.assert_allclose(losses, expected_loss, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(
                J.min(axis=1), expected_J, rtol=1e-12, atol=1e-14
            )
            results.append(
                {
                    "case": case["name"],
                    "saved_frames": len(u),
                    "min_J_all_frames": float(J.min()),
                    "fixed_error": fixed_error,
                    "loss_max_abs_error": float(np.max(np.abs(losses - expected_loss))),
                }
            )
    assert len(results) == 32
    report = GROUP / "docs/10-results.md"
    checked_links = []
    for target in re.findall(r"\]\(([^)]+)\)", report.read_text()):
        if target.startswith(("http:", "https:", "#")):
            continue
        target_path = (report.parent / target.split("#", 1)[0]).resolve()
        assert target_path.is_file(), target_path
        checked_links.append(str(target_path))
    gallery = GROUP / "data/30-figures/index.html"
    for target in re.findall(r'(?:src|href)="([^"]+)"', gallery.read_text()):
        if target.startswith(("http:", "https:", "#")):
            continue
        target_path = (gallery.parent / target).resolve()
        assert target_path.is_file(), target_path
    images = []
    for path in sorted(gallery.parent.glob("*.png")):
        with Image.open(path) as picture:
            images.append({"file": path.name, "size": picture.size})
            picture.verify()
    assert len(images) == 6
    receipt = {
        "cases": results,
        "case_count": len(results),
        "saved_frame_count": sum(c["saved_frames"] for c in results),
        "report_links": checked_links,
        "png_images": images,
        "gallery": "all local href/src targets exist",
    }
    (output / "checks.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k not in {"cases", "report_links"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileArtifacts)
