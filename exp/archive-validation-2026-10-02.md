# September archive validation on 2026-10-02

This publication preserves historical experiment code, technical reports, and
selected result records. It does not rerun the numerical experiments or promote
an unconverged checkpoint to a converged result.

## Source preservation

The September source snapshots have file-specific Ruff allowances in
`pyproject.toml`. They include historical style, complexity, and correctness
diagnostics. An allowance records a known limitation; it is not a repair.
The public path cleanup was separately checked for syntax errors, undefined
names, duplicate root assignments, and import ordering. Scientific methods and
recorded numerical outcomes were retained.

Local source paths now derive from the checkout root. Inputs in a historical
checkout or the separate anatomy repository require the configuration described
in [the archive README](README.md). Formatting and path normalization can change
source hashes; original run hashes remain historical provenance.

## Experiment tests

Historical tests are retained under the corresponding experiment's `tests/`
directory. They are outside the default `tests`, `src`, and `benches` paths.
An initial check with local input artifacts available passed 44 tests and failed
two. Both failures were in the joint-direction tolerance contract: its dynamic
module loader is incompatible with the current Python 3.14 dataclass behavior,
and one assertion expects an older complementarity expression. These failures
are retained as part of the historical record.

The active-strain chain and remote-binding tests require saved experiment data.
The shape-scene test requires external anatomy inputs. The live-progress test
requires a private serving component that is excluded from publication. Supply
the appropriate historical inputs and environment before running those tests.

The core and portable experiment test suite is validated separately from these
archive-only tests. A passing portable suite does not validate every archived
script or establish full-face convergence.

## Result artifacts

[The result package](records/2026-09-public/README.md) contains selected numeric
records and scientific plots. Its manifests bind each bundle member to its
source path and SHA-256 digest. Derived JSON records explicitly document the
removal of private operational fields and retain the original source digest.
Bundle membership, member hashes, and archive checksums are verified during
packaging. Raw meshes, checkpoints, and third-party anatomy are not part of this
package.
