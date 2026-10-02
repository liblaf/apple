# No-contact Newton inner-CG budget probe

This one-step diagnostic used the same accepted partial state, fixed pose,
materials, and no-contact model for two safeguarded Newton corrections. It
tested the hypothesis that the 1,000-iteration inner CG ceiling forces a
larger regularization shift. It does not change the continuation algorithm
or establish a force-converged forward state.

Run from the experiment directory:

```bash
cd exp/2026/09/23/new-neutral
CHERRIES_NAME='Matched no-contact Newton inner CG budget probe' \
CHERRIES_TAGS='mouthopen,newton,cg-budget,no-contact' \
uv run python -u src/124-probe-no-contact-newton.py \
  --input-checkpoint data/pose-rigid-resume-001/newton-probe-fixture.pt
```

Both variants used `linear_rtol=1e-3`, `atol=1e-8`, maximum coordinate step
`0.895431 mm`, reuse shift policy, initial shift ratio `1.0`, signed-mean
shift scale, and collision disabled. The initial force was
`1.450749e-8` and initial energy was `3.139401862e-6`.

With a 1,000-step limit, CG rejected the initial `0.000420104` shift for its
iteration budget and the Newton retry accepted shift `0.00420104` after 503
CG iterations. Energy changed to `3.139401812e-6` and force to
`1.450452e-8`.

With a 3,000-step limit, CG accepted the initial `0.000420104` shift after
1,601 iterations, with no regularization retry. Energy changed to
`3.139401363e-6`, while force changed to `1.766699e-8`. The larger energy
drop does not imply a force improvement: this is one accepted energy step,
not a full Newton convergence run. The 1,000-step total also includes first
variant setup and compilation, so these wall times do not support a speed
comparison.

The receipt records regularization retries, PCG data, forces, energies,
endpoints, source hashes, and fixed inputs at
`data/no-contact-newton-probe-001/receipt.json`. Comet:
<https://www.comet.com/liblaf/apple/2222611bdf0b41aab238ba733d728106>.
