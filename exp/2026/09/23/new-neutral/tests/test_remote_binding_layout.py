# ruff: noqa: PT009, PT027
"""CPU-only contract tests for the remote collector `data/` mirror layout."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

GROUP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GROUP / "src"))

from smile_recovery_lineage import resolve_data_binding  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RemoteBindingLayout(unittest.TestCase):
    def test_maps_remote_data_suffix_to_collector_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mirror = Path(directory) / "remote-smile-recovery-006"
            target = mirror / "inverse-smile-coupled-006" / "rendering.npz"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"layout-fixture")
            binding = {
                "path": "/root/shared/new-neutral/data/inverse-smile-coupled-006/rendering.npz",
                "sha256": sha256(target),
            }
            self.assertEqual(resolve_data_binding(binding, mirror), target)

    def test_explicit_local_data_root_supplies_omitted_immutable_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collector = root / "collector"
            local_data = root / "local-data"
            target = local_data / "blendshapes-005" / "blendshapes.npz"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"immutable-input")
            binding = {
                "path": "/remote/new-neutral/data/blendshapes-005/blendshapes.npz",
                "sha256": sha256(target),
            }
            self.assertEqual(
                resolve_data_binding(binding, collector, local_data), target
            )

    def test_frozen_sources_resolve_locally_and_missing_endpoint_fails(self) -> None:
        run = (
            GROUP
            / "data/recovery006-lineage-preflight-inputs/inverse-smile-coupled-006"
        )
        protocol = json.loads((run / "protocol.json").read_text())
        for binding in protocol["sources"].values():
            self.assertTrue(resolve_data_binding(binding, GROUP / "data").is_file())
        collector = GROUP / "data/remote-smile-recovery-006"
        final_run = collector / "inverse-smile-coupled-006"
        final_protocol = json.loads((final_run / "protocol.json").read_text())
        final_summary = json.loads((final_run / "summary.json").read_text())
        self.assertTrue(
            resolve_data_binding(final_summary["endpoint"], collector).is_file()
        )
        self.assertTrue(
            resolve_data_binding(
                final_protocol["rendering"]["archive"], collector
            ).is_file()
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(AssertionError),
        ):
            resolve_data_binding(final_summary["endpoint"], Path(directory))


if __name__ == "__main__":
    unittest.main()
