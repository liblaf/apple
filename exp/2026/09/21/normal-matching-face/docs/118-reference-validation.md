# Reference-length loss validation

The [selected configuration](../data/110-shape-loss-config/loss-config.json)
uses l_ref = 13.236093032531715 mm and direct normal weight one. The
[fit protocol](115-reference-fit-protocol.md) defines fresh neutral Raw6 fits
with objective `P/l_ref_mm**2 + N + eta*R`. Effective Adam epsilon is
5.707952862381702e-5, learning rate .3, and eta is zero off or
1.8346203690062914e-5 on. These epsilon/eta values consistently divide the old
values by l_ref²; the numerical physics sources are unchanged.

Full implicit derivative validation passed the unchanged 2% finite-difference
gate and exited zero after normal Cherries shutdown. Across eight recorded
checks, the largest relative discrepancy was 0.018896418908775043 (1.88964%)
for normal-only and 0.0003708819630345545 (0.0370882%) for combined position,
normal and activation smoothness. The combined unsmoothed derivative follows
by subtracting the previously verified direct q-only R derivative; this is
not a separate full-face finite-difference run.

Both one-update smoke branches then exited zero with zero inverted tetrahedra.
The neutral objective was 0.07379003473354348, composed of position
0.04940857010416178 and normal 0.0243814646293817. After one update the
position RMS was about 5.02835 mm and normal RMS about 8.73646 degrees.

The independent CPU smoke audit passed and exited zero before the main fits
launched. The first Adam updates matched within 2.7755575615628914e-17 off
and 2.0816681711721685e-17 on. Each branch had two accepted solve receipts.
The audit checked 99 current/snapshot source records covering 98 unique files,
three fixture records, and five preflight records. The prior beta-1 run has
97 source records covering 96 unique files, all present unchanged in the new
source set. Duplicate records are Python's `__main__` and `__mp_main__` aliases
for the corresponding entrypoint. Physics/material/tolerance and mesh-size
invariants also match the prior experiment.

- [Accepted derivative receipt](../data/121-reference-validation/gradient-validation.json)
- [Smoke summaries](../data/123-reference-smoke/summary.json)
- [Accepted CPU audit](../data/124-reference-smoke-verification/checks.json)
- [Derivative Comet run](https://www.comet.com/liblaf/apple/b225f500a7fc439b87059668ec55ab18)

Two setup attempts are preserved for provenance. The first derivative attempt
was deliberately interrupted with exit 130 when a final lint annotation
changed the archived entrypoint hash; the runner's audit-receipt plumbing was
also finalized before restarting validation. No fitting states came from that
attempt. See [interruption receipt](../data/119-reference-validation-prelint/interruption.json),
[log](../logs/119-reference-validation-prelint.log), and its
[Comet record](https://www.comet.com/liblaf/apple/09039be4f3ac4010827d60e2e87c5404).
The first CPU smoke audit failed an incorrect assertion that the 97 module
records represented 97 unique paths. Correcting that auditor assumption did
not alter the numerical sources or smoke states. Its [failed receipt](../data/124-reference-smoke-verification-source-record-count-error/checks.json)
and [log](../logs/124-reference-smoke-verification-source-record-count-error.log)
remain available. The accepted audit distinguishes record and unique-file
counts; the final verifier additionally asserts alias-hash consistency.

Working directory: `exp/2026/09/21/normal-matching-face`. Exact derivative and
smoke commands are in the frozen protocol. The CPU smoke audit was run with
DEBUG=1, a human-readable Cherries name and tags, and CPU thread caps of four:

```bash
DEBUG=1 CHERRIES_NAME='Reference-length face smoke verification' CHERRIES_TAGS='face,3d,raw6,reference-loss,smoke,verification,cpu' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/135-verify-reference.py --comparison-dir 123-reference-smoke --output 124-reference-smoke-verification
```

Preserved numerical logs are `logs/121-reference-validation.log` and
`logs/123-reference-smoke.log`. These checks establish consistency of the
objective, derivatives, and first update; they do not establish convergence or
mechanical validity of later fitted states.
