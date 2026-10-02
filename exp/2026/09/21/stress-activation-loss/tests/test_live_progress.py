"""CPU checks for wall-time estimates without importing the numerical solver."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[6]
    / "exp/2026/09/21/stress-activation-loss/src/43-serve-live.py"
)
spec = importlib.util.spec_from_file_location("live_progress", SOURCE)
assert spec is not None
assert spec.loader is not None
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


def estimate(rows: list[dict], *, attempted: int, age: float = 0, **kwargs) -> dict:
    summary = {
        "status": "running",
        "budget": 30,
        "attempted_steps": attempted,
        "elapsed_seconds": kwargs.pop("elapsed_seconds", 0),
    }
    return live.estimate_remaining(summary, rows, 1000, 1000 + age, **kwargs)


def test_recent_pace_excludes_initial_solve_and_accounts_for_current_attempt():
    rows = [
        {"segment": 0, "step": step, "elapsed_seconds": 500 + step * 20}
        for step in range(16)
    ]
    rows[0]["elapsed_seconds"] = 0  # Slow setup lies outside the recent window.
    result = estimate(rows, attempted=15, age=7, process_alive=True)
    assert result["window_attempts"] == 10
    assert result["seconds_per_attempt"] == 20
    assert result["remaining_seconds"] == 293
    assert result["estimated_finish_time"] == live.timestamp(1300)
    assert not result["overdue"]


def test_resume_clock_and_skipped_attempts():
    rows = [
        {"segment": 0, "step": 5, "elapsed_seconds": 9000},
        {"segment": 1, "step": 6, "elapsed_seconds": 10},
        {"segment": 1, "step": 8, "elapsed_seconds": 50},
    ]
    result = estimate(rows, attempted=10, elapsed_seconds=90, process_alive=True)
    assert result["window_attempts"] == 4
    assert result["remaining_attempts"] == 20
    assert result["seconds_per_attempt"] == 20
    assert result["remaining_seconds"] == 400


def test_overdue_final_attempt_keeps_nonzero_estimate():
    rows = [
        {"segment": 0, "step": 28, "elapsed_seconds": 10},
        {"segment": 0, "step": 29, "elapsed_seconds": 30},
    ]
    result = estimate(rows, attempted=29, age=60, process_alive=True)
    assert result["overdue"]
    assert result["remaining_seconds"] > 0
    assert result["current_attempt_seconds"] == 60


@pytest.mark.parametrize(
    ("attempted", "alive", "status", "seconds"),
    [
        (30, False, "completed", 0),
        (12, False, "stopped", None),
        (0, True, "estimating", None),
    ],
)
def test_terminal_and_insufficient_history(
    *, attempted: int, alive: bool, status: str, seconds: float | None
) -> None:
    result = estimate([], attempted=attempted, process_alive=alive)
    assert result["status"] == status
    assert result["remaining_seconds"] == seconds
    assert result["estimated_finish_time"] is None
