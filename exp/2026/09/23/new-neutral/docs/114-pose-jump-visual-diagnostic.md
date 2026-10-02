# Mandible jump and no-collision relaxation

## Purpose and scope

Test the requested initialization sequence from the saved MouthOpen state at
4.1293984 degrees: move the jaw to the chin estimate of 11.5587603 degrees,
carry free FEM vertices within `d_hat = 0.1 mm` of the currently posed mandible,
relax without contact, project out remaining penetrations, and restore contact
for a final equilibrium. The original reference, active-strain formulation,
zero muscle activation, and fixed skin prestrain are retained.

The direct carry moves 173 free FEM vertices and applies all new prescribed jaw
boundary values. The second variant extends the exact carry increment
harmonically to other free vertices. It preserves the same 173 carry values
and prescribed boundary values.

## Observed results

| Stage | Direct carry (`pose-jump-001`) | Harmonic carry (`pose-jump-002`) |
| --- | ---: | ---: |
| Inverted tetrahedra immediately after carry | 13,275 | 1,097 |
| No-contact solve | Failed 600 s budget, last logged force about 0.106 N | 177.01 s, 0.00969189 N |
| Inversions after converged no-contact solve | No saved terminal geometry | 299 |
| Push-out | Not reached | 20.43 s |
| Contact-on correction | Not reached | 34.93 s, 0.00992066 N |
| Final inverted tetrahedra | No final state | 282 |
| Total proposal time | Incomplete | 242.74 s |

The force threshold is `1e-8` in solver units, or 0.01 N. The harmonic final
state passes this force gate, the no-surface-intersection check, and the
10 nm contact gap buffer. Its minimum active contact distance is 56.9483 um.
It is **not a physically valid forward state** because tetrahedra remain
inverted. There is no completed matched continuation timing baseline, so
these data do not establish a speedup over continuation.

The original failed direct trial did not save its terminal relaxation state.
To answer the visual question, `pose-jump-004` repeats the exact direct
procedure with accepted-state snapshots every 10 seconds and a 120 s
relaxation budget. Its saved last accepted state, recorded at 120.1206 s,
has force **1.95178145 N**, 482 inverted tetrahedra, minimum `det(F)`
-73.85234, and surface motion RMS 6.54817 mm. It is unconverged, with contact
disabled and no push-out applied. The force is not monotone; this is the last
accepted state, not the minimum-force state. The failure callback uses a copy
of the last accepted state, never a rejected in-place Newton trial.

The repeat includes snapshot and force-history overhead and uses a shorter
budget than the original direct trial. It is a visual diagnostic, not a
matched timing comparison. `pose-jump-003` was a setup failure caused by the
wrapper's unregistered imported module; it produced no solve. The module
registration was corrected before `pose-jump-004`.

## Geometry limit

A separate CPU audit found 71 of the harmonic final inversions have all four
vertices prescribed. Every such tetrahedron mixes fixed cranium vertices
with rotating mandible vertices. These cells cannot be repaired by moving
free vertices or by surface push-out. The source already had eight such
inversions. See `115-fixed-boundary-pose-audit.md` and
`data/pose-jump-002/fixed-tet-audit.json`. Joint inverse optimization has not
started, and none of these states is an inverse fit.

## Visual assets

`data/review-repaired-reference-005/pose-jump/` contains identical-camera,
true-scale front and side comparisons, a translucent anatomy comparison,
the direct repeat's force curve, and `receipt.json` binding the exact saved
checkpoints by SHA256. The columns are the source state, the direct repeat
at 120 s, and the converged harmonic no-contact state. The contact-on
harmonic result is described separately and is not substituted into the
no-contact comparison.

Tailnet review: <PRIVATE_PREVIEW_URL>. The existing transient
preview service is reused.

## Commands and evidence

Working directory: `exp/2026/09/23/new-neutral`. Python is the repository's
`.venv/bin/python`; each command uses normal Cherries `ProfileJoint`, with
Git commits disabled.

```bash
CHERRIES_NAME='MouthOpen harmonic carry pose jump test' \
CHERRIES_TAGS='new-neutral,mouthopen,pose-jump,harmonic-initializer,hybrid' \
.venv/bin/python src/113-test-harmonic-pose-jump.py

CHERRIES_NAME='Direct pose jump saved no collision iterate' \
CHERRIES_TAGS='new-neutral,mouthopen,pose-jump,direct,partial' \
.venv/bin/python src/115-test-direct-pose-jump-snapshots.py

CHERRIES_NAME='Direct and smoothed pose jump visual comparison' \
CHERRIES_TAGS='new-neutral,mouthopen,pose-jump,review' \
.venv/bin/python src/114-review-pose-jump.py
```

Comet runs:

- Original direct: <https://www.comet.com/liblaf/apple/499a2bafd5cb47108b1d46654f802384>
- Harmonic: <https://www.comet.com/liblaf/apple/b793e34d71e047abbaf40426d2a60716>
- Direct repeat: <https://www.comet.com/liblaf/apple/b9e78a6bb8794a7593924ccd4263fef9>

The terminal logs are under `tmp/pose-jump-{001,002,004}-terminal.log` and
`tmp/pose-jump-review-terminal.log`; physics runs preserve source snapshots
under their output directories. Summary metrics come from their saved
`summary.json`, `direct-relaxed-force.jsonl`, and geometry audit, not from
intended behavior. All runs use the repaired reference and the same frozen
`inverse-mouthopen-003/initialization.pt`. GPU: RTX 4090; IPC threads: 4.
