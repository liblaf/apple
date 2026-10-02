# Saved neutral energy curve

The repaired-reference active-strain run records 433 energy samples: the
initial state, 332 PNCG steps, and 100 Newton steps. Total material plus IPC
energy starts at 2.543597407 J and ends at 2.471214861 J. Raw MPa·m³ values are
multiplied by 1e6. The terminal trace energy matches the saved summary to
floating-point precision.

The plot marks the barrier-stiffness increase at PNCG step 193, from 0.1693 to
0.3386 MPa, and disconnects the curve across that objective change. The right
panel enlarges the Newton energy range. Samples are taken after accepted
updates and their adaptive stiffness update. The solver remains unconverged:
0.159060385 N residual against 0.01 N required. Energy reduction alone does not
establish force convergence.

Tailnet plot (private preview omitted),
[PNG](../data/review-repaired-reference-001/energy/energy-curve.png),
[SVG](../data/review-repaired-reference-001/energy/energy-curve.svg),
[PDF](../data/review-repaired-reference-001/energy/energy-curve.pdf), and
[CSV samples](../data/review-repaired-reference-001/energy/energy-samples.csv).
The plot receipt binds the original trace and summary by SHA-256. All seven
new or changed HTTP assets matched their local hashes in
`data/http-verification-energy.json`. Visual inspection and Ruff passed.
The existing transient server serves the new files without restarting.

Run from `exp/2026/09/23/new-neutral`, with a fresh review output containing
the saved run's review receipt when reproducing:

```bash
CHERRIES_NAME='Repaired neutral energy curve' \
CHERRIES_TAGS='new-neutral,active-strain,energy,review' \
.venv/bin/python -u src/70-plot-energy.py
```

No physics evaluation or solver run was needed. The Cherries profile disables
automatic Git commits. [Comet run](https://www.comet.com/liblaf/apple/4613cef0c16f43489fbb049f6f67ff84)
completed with `energy/samples=433`, `energy/initial_j=2.5435974066747624`, and
`energy/terminal_j=2.471214861151565`; source HEAD was
`d56fa1b553b287b22b2cf7bb82d46117e34ed6bb` with experiment edits archived locally.
