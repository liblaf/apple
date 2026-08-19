# Fixed-activation prestrain replay analysis

This is a fixed-muscle-activation forward diagnostic. It is not an inverse
result and does not authorize an inverse experiment.

## Terminal comparison

| checkpoint | target RMS (mm) | error/target | dihedral RMS (deg) | normal Laplacian (mm) | inverted tets | folded skin triangles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact baseline | 2.72095 | 0.512406 | 13.329 | 0.217061 | 47 | 25 |
| c020 continuation | 2.77878 | 0.523298 | 6.38689 | 0.180379 | 46 | 16 |
| c020 direct | 2.77865 | 0.523273 | 6.47688 | 0.180112 | 46 | 18 |

## Advisory decision

- Outcome: `stop-c020-replay-or-branch-failure-do-not-escalate`
- Reason: c=.02 did not reproduce the canonical alpha-0 roughness/branch within the declared tolerances; c=.05 must not be run until this is resolved
- Meaningful smoothing: True
- Fidelity acceptable: True
- Alpha-0 replay stable: True
- Alpha-0 roughness stable: True
- Continuation/direct branch stable: False
- The effect-size thresholds are advisory deterministic engineering
  thresholds, not statistical claims.
- Quality counts and matched fixed views require human visual review;
  small imperceptible inversions or folds are not automatic vetoes.
- A c=.05 forward probe requires separate explicit approval. This analyzer
  cannot launch it.

## Quality warnings

- continuation: minimum det(F) is lower than the exact baseline
- direct: minimum det(F) is lower than the exact baseline

## Required fixed-view review

Record whether smoothing is visible in each matched geometry and normal-
residual view, and separately record any new visible artifact:

- [ ] front smoothing visible
- [ ] 30 degree smoothing visible
- [ ] mouth smoothing visible
- [ ] eye-cheek (+x) smoothing visible
- [ ] no new visible artifact in any fixed view

Final resolution rules:

- c=.02 is sufficient only when the quantitative rule passes, smoothing
  is visible in at least three of four views, and no new artifact appears.
- A stable, fit-safe but quantitatively or visibly weak c=.02 result
  conditionally requires a separately approved c=.05 forward probe.
- Any replay, branch, solver, fixed-boundary, fit, or visible-artifact
  failure stops escalation; do not run c=.05.
