"""Replay the matched benchmark with an explicitly selected IPC thread count."""

import os
import runpy
from pathlib import Path

import ipctk

threads = int(os.environ["APPLE_IPC_THREADS"])
assert threads > 0
ipctk.set_num_threads(threads)
assert ipctk.get_num_threads() == threads
print(f"IPC threads: {threads}", flush=True)
runpy.run_path(str(Path(__file__).with_name("10-benchmark.py")), run_name="__main__")
