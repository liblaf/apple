# Smile collision construction audit

## Purpose

Verify that the paired Smile inverse-fit harness constructs one contact-on
model with the complete source cranium, complete source mandible, and both
registered source eyeballs. This is a construction and shared-neutral geometry
gate only; it does not solve an equilibrium.

## Command

Executed on `V100 host` with `CUDA_VISIBLE_DEVICES=1`:

```bash
cd ${APPLE_HISTORICAL_WORKTREE}/exp/2026/09/22/solver-performance
CUDA_VISIBLE_DEVICES=1 \
CHERRIES_NAME='Smile collision construction audit' \
CHERRIES_TAGS='smile,collision,bones,eyes,construction' \
../../../../../.venv/bin/python src/18-audit-smile-collision.py \
  --source-root ${APPLE_HISTORICAL_WORKTREE} \
  --output-dir data/smile-collision-audit-001
```

Comet reporting could not start because this compute host has no `COMET_API_KEY`.
The Cherries log and copied result are retained locally.

## Results

The audit passed. The assembled IPC map has 62,594 vertices: 34,245 soft
surface vertices, 17,575 cranium vertices, 9,476 mandible vertices, and 1,298
eye vertices. It retains all 35,162 cranium, 18,948 mandible, and 2,560 eye
triangles, with zero excluded source-bone triangles.

At the shared neutral seed, IPC found 5,195 active soft-rigid pairs, a minimum
active distance of 17.066746 micrometres, and no intersections. This exceeds
the 10 nm numerical CCD buffer. Cranium and eye displacements were exactly
zero. The audit also probes a nonzero mandible rotation and verifies that only
the prescribed mandible obstacle moves.

## Outputs

The machine-readable receipt is
[`summary.json`](../data/smile-collision-audit-001/summary.json). The execution
log is [`18-audit-smile-collision.log`](../logs/18-audit-smile-collision.log).

## Reproducibility

The audit script is [`18-audit-smile-collision.py`](../src/18-audit-smile-collision.py)
and its fail-fast model/state checks are in
[`smile_collision.py`](../src/smile_collision.py). The remote model uses the
same hash-bound expression inputs as the paired fit; no collision geometry,
material field, or solver state was edited.
