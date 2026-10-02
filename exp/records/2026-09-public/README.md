# Public September 2026 experiment records

This is a compact, public-facing record of selected September experiments. With
the original allow-listed experiment outputs available locally, run
`python exp/records/2026-09-public/package_records.py` from the repository root
to rebuild `bundles/`. Those inputs are not all included in a public checkout.

Each `*.tar.gz` contains a small allow-listed group of numeric result tables,
receipts that passed the text scan, and synthetic or profile plots. The
selection deliberately excludes copied source trees, runtime archives, raw
meshes, checkpoints, third-party anatomy, meeting material, and model-face
renders.

`selected-manifest.json` gives every copied input's original repository path,
exact bundle member path, SHA-256 digest, and size. `excluded-manifest.json`
records allow-listed inputs rejected for a local path, private or Tailnet
address, credential marker, decoding, or size rule. Rejected JSON receipts are
represented by a separately scanned derived JSON record that retains public
numeric and scalar evidence while omitting paths, hosts, commands, snapshots,
and credential fields. `derived-sanitized-manifest.json` records that source
digest and transformation. `coverage-inventory.json` states the total data
scope for every September study. `checksums.sha256` covers the bundles.

This package documents partial and unconverged outcomes. It is an audit record,
not a claim that every included inverse run converged or that the raw data have
been released.
