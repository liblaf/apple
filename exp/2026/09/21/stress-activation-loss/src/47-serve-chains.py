"""Serve read-only mirrored active-strain chain progress on a private network.

The server reads local mirrors only. A synchronizer writes one ``chain-status.json``
per chain and may write ``remote-status.json`` with remote process evidence.
Run this foreground process or a transient service; do not install a persistent unit.
"""
# ruff: noqa: C901, FBT001, PLR0911

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import gzip
import io
import json
import logging
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

GROUP = Path(__file__).resolve().parents[1]
REPO = GROUP.parents[4]
REFERENCE = REPO / "exp/2026/09/07/tensor-active-stress/data/107-learning-rate-render"
LOG = logging.getLogger(__name__)
STALE_SECONDS = 45.0


def iso(value: float) -> str:
    return dt.datetime.fromtimestamp(value, dt.UTC).isoformat()


def read_json(path: Path) -> dict:
    with path.open() as stream:
        value = json.load(stream)
    assert isinstance(value, dict)
    return value


def history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for row in csv.DictReader(io.StringIO(path.read_text())):
        try:
            item = {
                "step": int(float(row["step"])),
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "objective": float(row["objective"]),
                "l2": float(row.get("position_contribution", "nan")),
                "normal": float(row.get("normal_contribution", "nan")),
                "smooth": float(row.get("regularizer_contribution", "nan")),
                "rms_mm": float(row.get("fit_rms_mm", "nan")),
            }
        except (KeyError, TypeError, ValueError):
            continue
        item["solver_valid"] = row.get("solver_valid", "false").lower() == "true"
        rows.append(item)
    return rows


def estimate(
    rows: list[dict], budget: int | None, alive: bool, summary: dict, completed: bool
) -> dict:
    if completed or str(summary.get("status", "")).startswith("completed"):
        return {
            "status": "completed",
            "remaining_seconds": 0.0,
            "remaining_attempts": 0,
        }
    if budget is None or not rows:
        return {"status": "unavailable", "remaining_seconds": None}
    estimate_rows = list(rows)
    attempted = summary.get("attempted_steps")
    elapsed = summary.get("elapsed_seconds")
    if (
        isinstance(attempted, (int, float))
        and isinstance(elapsed, (int, float))
        and attempted > estimate_rows[-1]["step"]
        and elapsed >= estimate_rows[-1]["elapsed_seconds"]
    ):
        estimate_rows.append({"step": attempted, "elapsed_seconds": elapsed})
    last = estimate_rows[-1]
    remaining = max(0, budget - last["step"])
    if remaining == 0:
        return {"status": "completed", "remaining_seconds": 0.0}
    if not alive:
        return {"status": "stopped", "remaining_seconds": None}
    points = estimate_rows[-10:]
    if len(points) < 2 or points[-1]["step"] <= points[0]["step"]:
        return {"status": "estimating", "remaining_seconds": None}
    pace = (points[-1]["elapsed_seconds"] - points[0]["elapsed_seconds"]) / (
        points[-1]["step"] - points[0]["step"]
    )
    if not np.isfinite(pace) or pace <= 0:
        return {"status": "unavailable", "remaining_seconds": None}
    return {
        "status": "running",
        "seconds_per_attempt": pace,
        "remaining_seconds": remaining * pace,
        "remaining_attempts": remaining,
        "estimated_finish_time": iso(time.time() + remaining * pace),
    }


def checkpoint(stage_dir: Path) -> dict:
    path = stage_dir / "live-fit.npz"
    if not path.exists():
        path = stage_dir / "last.npz"
    if not path.exists():
        return {"available": False}
    try:
        with np.load(path, allow_pickle=False) as state:
            step = int(state["step"])
        return {
            "available": True,
            "file": path.name,
            "step": step,
            "mtime_ns": path.stat().st_mtime_ns,
        }
    except (KeyError, OSError, ValueError):
        return {"available": False}


class Chains:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.last_checked = 0.0
        self.cached: dict[str, dict] = {}

    def refresh(self) -> None:
        if time.monotonic() - self.last_checked < 2.0:
            return
        self.last_checked = time.monotonic()
        chains: dict[str, dict] = {}
        for path in sorted(self.root.glob("*/chain-status.json")):
            try:
                state = read_json(path)
                chain_id = str(state.get("chain_id", path.parent.name))
                state["chain_id"] = chain_id
                state["path"] = path.parent.name
                state["_root"] = path.parent
                chains[chain_id] = state
            except (OSError, json.JSONDecodeError, AssertionError) as error:
                LOG.warning("Skipping incomplete chain status %s: %s", path, error)
        self.cached = chains

    def status(self) -> dict:
        self.refresh()
        now = time.time()
        chains = []
        for chain in self.cached.values():
            root = chain["_root"]
            remote_path = root / "remote-status.json"
            if not remote_path.exists():
                remote_path = self.root / "remote-status.json"
            try:
                remote = read_json(remote_path) if remote_path.exists() else {}
            except (OSError, json.JSONDecodeError, AssertionError) as error:
                LOG.warning(
                    "Ignoring incomplete remote status %s: %s", remote_path, error
                )
                remote = {}
            checked = remote.get("checked_at")
            checked_time = (
                dt.datetime.fromisoformat(checked).timestamp()
                if isinstance(checked, str)
                else 0.0
            )
            stale = not checked_time or now - checked_time > STALE_SECONDS
            alive = bool(remote.get("process_alive")) and not stale
            stages = []
            for stage in chain.get("stages", []):
                if not isinstance(stage, dict):
                    continue
                stage_dir = root / str(stage.get("stage_dir", ""))
                rows = history(stage_dir / "trace.csv")
                summary = (
                    read_json(stage_dir / "summary.json")
                    if (stage_dir / "summary.json").exists()
                    else {}
                )
                budget = (
                    stage.get("budget")
                    or summary.get("budget")
                    or chain.get("protocol_steps")
                )
                budget = int(budget) if isinstance(budget, (int, float)) else None
                saved = checkpoint(stage_dir)
                last_step = summary.get("last_step", stage.get("last_step"))
                if last_step is None and rows:
                    last_step = rows[-1]["step"]
                if saved.get("step") is not None:
                    last_step = saved["step"]
                stages.append(
                    {
                        **stage,
                        "last_step": last_step,
                        "history": rows,
                        "summary": summary,
                        "eta": estimate(
                            rows,
                            budget,
                            alive,
                            summary,
                            stage.get("status") == "completed",
                        ),
                        "checkpoint": saved,
                        "checkpoint_available": saved["available"],
                    }
                )
            current_index = chain.get("current_stage")
            current = next(
                (item for item in stages if item.get("status") == "running"), None
            )
            if current is None:
                current = next(
                    (item for item in stages if item.get("index") == current_index),
                    None,
                )
            current_eta = (
                current["eta"]
                if current is not None
                else {"status": "unavailable", "remaining_seconds": None}
            )
            future = [item for item in stages if item.get("status") == "queued"]
            chain_remaining = (
                0.0
                if str(chain.get("status", "")).startswith("completed")
                else current_eta.get("remaining_seconds")
            )
            if chain_remaining is not None and current_eta.get("seconds_per_attempt"):
                pace = current_eta["seconds_per_attempt"]
                for item in future:
                    budget = item.get("budget")
                    if isinstance(budget, (int, float)):
                        chain_remaining += budget * pace
            chains.append(
                {
                    **{
                        key: value
                        for key, value in chain.items()
                        if not key.startswith("_")
                    },
                    "remote": {**remote, "stale": stale, "alive": alive},
                    "stages": stages,
                    "eta": current_eta,
                    "chain_remaining_seconds": chain_remaining,
                }
            )
        return {
            "schema": "active-strain-live-chains-v1",
            "server_time": iso(now),
            "chains": chains,
        }

    def geometry(self, chain_id: str, index: int) -> dict:
        self.refresh()
        chain = self.cached[chain_id]
        stage = next(
            item for item in chain.get("stages", []) if int(item["index"]) == index
        )
        root = chain["_root"]
        stage_dir = root / str(stage["stage_dir"])
        with np.load(root / "mesh.npz", allow_pickle=False) as mesh:
            skin_ids = mesh["skin_ids"]
            rest = mesh["rest_points"][skin_ids]
            target = rest + mesh["target_displacement_skin"]
            compact = stage_dir / "live-fit.npz"
            if compact.exists():
                with np.load(compact, allow_pickle=False) as state:
                    fit = state["fit_positions"]
                    step = int(state["step"])
            else:
                with np.load(stage_dir / "last.npz", allow_pickle=False) as state:
                    fit = rest + state["u"][skin_ids]
                    step = int(state["step"])
            assert fit.shape == rest.shape
            return {
                "step": step,
                "indices": mesh["triangles"].ravel().tolist(),
                "rest_positions": rest.ravel().tolist(),
                "target_positions": target.ravel().tolist(),
                "fit_positions": fit.ravel().tolist(),
                "errors_mm": (1000 * np.linalg.norm(fit - target, axis=1)).tolist(),
            }


def handler_for(data: Chains) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            try:
                if url.path in ("/", "/index.html"):
                    self.respond(
                        (GROUP / "src/live-chains.html").read_bytes(),
                        "text/html; charset=utf-8",
                    )
                elif url.path in (
                    "/vendor/three.module.js",
                    "/vendor/three.core.js",
                    "/vendor/OrbitControls.js",
                ):
                    self.respond(
                        (REFERENCE / url.path.lstrip("/")).read_bytes(),
                        "text/javascript; charset=utf-8",
                    )
                elif url.path == "/api/manifest":
                    self.respond_json(data.status())
                elif url.path == "/api/geometry":
                    self.respond_json(
                        data.geometry(query["chain"][0], int(query["stage"][0]))
                    )
                elif url.path == "/favicon.ico":
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (
                KeyError,
                IndexError,
                OSError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, str(error))

        def respond_json(self, value: dict) -> None:
            self.respond(
                json.dumps(value, separators=(",", ":"), allow_nan=False).encode(),
                "application/json",
            )

        def respond(self, body: bytes, content_type: str) -> None:
            compressed = "gzip" in self.headers.get("Accept-Encoding", "")
            if compressed:
                body = gzip.compress(body, compresslevel=1)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if compressed:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            if len(args) > 1 and str(args[1]) not in ("200", "204"):
                LOG.info(fmt, *args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Local mirror containing one directory per chain",
    )
    parser.add_argument("--bind", required=True, help="Private/Tailnet address only")
    parser.add_argument("--port", type=int, default=8775)
    cfg = parser.parse_args()
    assert cfg.root.is_dir()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    server = ThreadingHTTPServer(
        (cfg.bind, cfg.port), handler_for(Chains(cfg.root.resolve()))
    )
    LOG.info("Serving read-only chain mirrors at http://%s:%d", cfg.bind, cfg.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
