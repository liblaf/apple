"""One-shot, runtime-only refresh of accepted expression-fit review assets."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from joint_common import GROUP, write_json


def stable_bytes(path: Path) -> bytes:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return payload


def write_snapshot(path: Path, payload: bytes) -> None:
    """Publish one immutable status payload for the renderer to bind."""
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    assert path.read_bytes() == payload


def main() -> None:
    fit_dir = GROUP / "data/expression-fitting-008"
    inputs_dir = GROUP / "data/expression-inputs-002"
    runtime_dir = Path("/run/user/1000/apple-expression-fit-review")
    status_path = fit_dir / "status.json"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    if not status_path.is_file():
        write_json(runtime_dir / "status.json", {"status": "no_fit_status"})
        return
    status_bytes = stable_bytes(status_path)
    status = json.loads(status_bytes)
    assert status["schema"] == "joint-fixed-material-expression-fitting-v1"
    digest = hashlib.sha256(status_bytes).hexdigest()
    snapshot_path = runtime_dir / f"status-{digest}.json"
    if snapshot_path.is_file():
        assert snapshot_path.read_bytes() == status_bytes
    else:
        write_snapshot(snapshot_path, status_bytes)
    renderer_digest = hashlib.sha256(
        (GROUP / "src/94-render-expression-fitting.py").read_bytes()
    ).hexdigest()
    visual_dir = (
        GROUP
        / "data"
        / f"expression-fitting-visuals-{digest[:12]}-{renderer_digest[:8]}"
    )
    summary_path = visual_dir / "summary.json"
    rendered = False
    if not summary_path.is_file():
        env = dict(os.environ)
        env.update(
            {
                "CHERRIES_NAME": "Expression fit review refresh",
                "CHERRIES_TAGS": "expression-fit,review,refresh",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            }
        )
        subprocess.run(
            [
                "uv",
                "run",
                "--frozen",
                "python",
                "src/94-render-expression-fitting.py",
                "--inputs-dir",
                str(inputs_dir),
                "--fit-dir",
                str(fit_dir),
                "--output-dir",
                str(visual_dir),
                "--status-snapshot",
                str(snapshot_path),
            ],
            cwd=GROUP,
            env=env,
            check=True,
        )
        rendered = True
    rendered_summary = json.loads(summary_path.read_text())
    assert rendered_summary["fit_status"]["captured_sha256"] == digest
    assert Path(rendered_summary["fit_status"]["path"]) == snapshot_path.resolve()
    subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "python",
            "src/43-build-review.py",
            "--expression-fit-dir",
            str(fit_dir),
            "--expression-fit-visual-dir",
            str(visual_dir),
            "--archive-review",
            "false",
        ],
        cwd=GROUP,
        check=True,
    )
    write_json(
        runtime_dir / "status.json",
        {
            "status": "refreshed",
            "fit_status_sha256": digest,
            "fit_status_snapshot": str(snapshot_path),
            "fit_phase": status.get("phase"),
            "fit_running": status["running"],
            "rendered_new_assets": rendered,
            "visual_dir": str(visual_dir),
        },
    )


if __name__ == "__main__":
    main()
