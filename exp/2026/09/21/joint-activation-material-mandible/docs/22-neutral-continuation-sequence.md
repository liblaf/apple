# Neutral continuation sequence

`src/22-run-neutral-continuation.py` runs the remaining 25%, 50%, and 100%
skin-prestress stages from an already converged contact-enabled neutral
checkpoint. Each stage must meet the existing stationarity, objective stability,
shape, oral-geometry, and contact gates before the next begins. The final target
is the declared 80.6 N/m literature proxy, not a measurement on this anatomy.

The sequence uses the validated Newton-CG inner solver. Select the outer method
explicitly. It allows up to 200 accepted updates, 1,000 forward evaluations, and
four hours per stage; exhausting any budget without convergence stops the
sequence. It never substitutes a lower prestress target.

Successful stages produce contact-force/gap maps, neutral overlays and internal
sections, stress/stiffness maps, and convergence plots. Exact child commands,
logs, exit codes, and checkpoint hashes are saved in a fresh output directory.
Sequence 001 failed before physics because the float-fraction CLI validator
rejected its string argument; that parser was corrected. Sequence 002 reached
stationarity at 25% but failed the surface budget (`0.275980 > 0.25 mm`) and
stopped without launching 50% or 100%. Both failures remain preserved.

The wrapper now continues either an admitted constant20 or Spatial80 parent.
It uses the declared skin-coordinate index and carries the exact recorded
basis, audit, CPU/derivative receipt paths and `0.5 * 100` spatial roughness
weight. A child may not change basis. The current unconverged Spatial80
checkpoint was checked and correctly rejected by the convergence gate; its
separate basis-argument extraction passed.

```bash
CHERRIES_NAME='Neutral prestress continuation' \
CHERRIES_TAGS='joint-inverse,neutral,contact,continuation' \
uv run --frozen python src/22-run-neutral-continuation.py \
  --initial-checkpoint PATH_TO_CONVERGED_10_PERCENT_CHECKPOINT \
  --output-dir data/neutral-continuation-001 \
  --outer-method bfgs
```

The final summary names the converged 100% checkpoint to supply to
`src/50-run-prepared-joint-sequence.py`. Higher-fraction runs that stop remain
available for diagnosis or an explicitly declared continuation; this wrapper
does not retry failures.
