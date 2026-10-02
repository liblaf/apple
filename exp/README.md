# Experiment archive

Experiments are organized as `YYYY/mm/dd/<study>/src/` for scripts and `docs/`
for dated protocols, findings, and validation records. The September 2026
publication contains selected source and technical reports from these studies.

This is a partial archive. Selected numeric records and scientific plots are
bundled with source hashes in [September records](records/2026-09-public/README.md).
Generated inputs, meshes, checkpoints, logs, runtime snapshots, rendered reports,
and presentation assets are otherwise kept separately.
Some scripts and reports require those artifacts or historical environments;
links to local data and omitted material may therefore be unavailable in a
checkout. Private operational scripts and personal planning documents are also
excluded.

Reports describe the state and checks recorded on their own dates. A saved
checkpoint, rendered animation, or low force residual alone does not establish
inverse convergence or geometric validity. Consult each study's protocol and
reported limitations before reusing a result. Publishing this archive does not
represent a new execution of the experiments.

## Historical source and validation

Historical scripts and experiment-local tests are retained even when they have
known lint diagnostics or require unpublished input data. The repository's Ruff
configuration lists narrow, file-specific allowances for those snapshots; they
do not apply to the core library. These allowances preserve the historical
implementation and do not certify that an archived script runs successfully.

Experiment-local tests live under each study's `tests/` directory. Some require
saved datasets, private runtime components, or their original Python environment.
They are outside the default test paths. Run a study's tests explicitly after
providing its inputs; the source and test records remain useful when those
requirements are unavailable.

Private preview addresses have been removed from published reports. Local paths
use repository-relative references or named configuration variables. Set
`APPLE_HISTORICAL_WORKTREE` when a script needs an older checkout, and
`APPLE_MELON_HEAD` when it needs externally supplied head-model inputs. Report-only
placeholders such as `APPLE_ROOT`, `CHERRIES_ROOT`, and `EXPERIMENT_WORKSPACE`
refer to the corresponding local checkout or workspace. A historical-worktree
reference does not imply that its sources or data are in this checkout.

Formatting and path normalization can change source hashes. Digests recorded in
historical receipts refer to the original run inputs; they are not replaced by
hashes of the published source. Recreate validation receipts before rerunning
an experiment with changed source.
