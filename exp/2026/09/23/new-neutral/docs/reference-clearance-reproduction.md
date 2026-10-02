# Reference-clearance reproduction

The commands require a checkout with the large frozen input fixtures already
available at `exp/2026/09/21/joint-activation-material-mandible/data/`:
`frozen-neutral-004/`, `rigid-eyes-001/`, and the constitutive volume and
geometry records referenced by the frozen-neutral manifest. They also require
the solver-performance and neutral-newton source directories used by the
scripts. A source-only checkout without those local assets cannot reproduce the
repair.

Run the Cherries cells from the experiment group. Each script uses
`ProfileJoint`, which records sources and disables Git commits.

```bash
cd exp/2026/09/23/new-neutral
CHERRIES_NAME="Reference surface clearance seed" CHERRIES_TAGS="new-neutral,reference-clearance,ipc-seed" \
  uv run --project ${APPLE_ROOT} python src/50-repair-reference.py
CHERRIES_NAME="Reference IPC-volume alternation" CHERRIES_TAGS="new-neutral,reference-clearance,ipc-volume-alternation" \
  uv run --project ${APPLE_ROOT} python src/51-alternate-reference-repair.py
CHERRIES_NAME="Reference clearance finalization" CHERRIES_TAGS="new-neutral,reference-clearance,finalization" \
  uv run --project ${APPLE_ROOT} python src/53-finalize-reference.py
```

Cell 50 writes the first exact-IPC surface candidate under
`data/reference-clearance-001/attempt-01/`. It may intentionally fail its
positive-volume gate after preserving that candidate and its IPC trace; retain
the generated directory and continue with cell 51.

Cell 51 alternates exact IPC stencil projection toward 100.1 micrometres with
fixed-node signed-volume repair. It accepts the requested 100 micrometre
clearance only after rebuilding and checking the complete IPC collision set,
then writes `data/reference-clearance-002/selected-candidate.npz` and its
alternation receipt.

Cell 53 verifies the selected candidate again, writes the final three-array
archive and meshes, and records the achieved clearance separately from the
100.1 micrometre projection target. The final receipt requires zero inverted
tetrahedra and fixed coordinates unchanged.

`data/reference-clearance-002` is the validated historical artifact. It was
packaged by the archived `53-finalize-reference.py` from a manually generated
`alternation-06.npz`; its finalization run is recorded at
<https://www.comet.com/liblaf/apple/d3b85b6ae4da47f6b3c5742f54eee917>. The new
cell 51 records the same exact alternation procedure for future runs, but has
only received static compilation, Ruff, and source review; it was not run
against the large fixtures while preserving the validated artifact.
