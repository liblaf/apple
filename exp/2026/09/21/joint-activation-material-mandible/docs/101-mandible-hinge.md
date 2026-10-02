# One mandible hinge angle per expression

Run `expression-fitting-003` uses one scalar angle per expression; all three
translation components are exactly zero. The unchanged rigid boundary backend
receives the six-vector `(theta * axis, 0, 0, 0)` and transforms the FEM mandible
supports and full source mandible obstacle together. Cranium and eyes stay fixed.

The axis points from registered mandible landmark 1 to landmark 9:
`[0.9999739766184107, -0.006463428415849528, 0.0032047120113181536]`.
The pivot is their midpoint, in metres:
`[1.406947255, 2.19754493, 0.0029887008700000004]`.
Positive rotation lowers the anterior mandible in this model's world +Y-up frame.
The source landmarks are unnamed; this remains a provisional hinge, not an
independently validated anatomical joint axis.

## Opening range and evidence

The adopted neutral defines zero. The optimizer permits 0–40 degrees of opening.
This is an exploratory model bound, not a measured range for this subject, and
the neutral's dental occlusion has not been independently verified. The previous
±10-degree bound was inherited from the six-coordinate optimizer; it was not a
physiological opening range.

An observational study of 60 healthy dental students reported mean maximal-opening
mandibular rotation of 39.1 ± 5.9 degrees for men and 36.3 ± 4.3 degrees for women,
with condylar movement of 20.5 ± 4.0 mm and 18.1 ± 2.5 mm respectively.
[Primary study](https://pubmed.ncbi.nlm.nih.gov/8765386/).
These data motivate avoiding a 10-degree opening cap; they do not validate
40 degrees about a fixed condylar axis. Another healthy-subject motion study
observed both rotation and translation in every subject's opening/closing motion.
[Mapelli et al.](https://pubmed.ncbi.nlm.nih.gov/19173245/).
This requested rotation-only model excludes joint sliding. Soft-rigid contact
remains enabled; rigid-rigid and soft-soft contact remain absent.

## Numerical change and validation

The optimized scalar is `q = theta / 10 degrees`, bounded `[0, 4]`. Keeping the
10-degree normalization preserves the former optimizer's angle step scale.
The prior is `jaw_weight * q**2 / 6`, exactly the old six-coordinate prior
restricted to this unit hinge axis. Projected-gradient stopping uses the new bounds.
Physical six-vector exports remain compatible with the existing renderer;
new metrics also record the signed opening angle. Legacy six-coordinate
checkpoints are rejected by shape assertions instead of being resumed.

Script `101-validate-mandible-hinge.py` passed finite-angle rigid-distance and
fixed-axis checks at 0, 5, 20, and 40 degrees. Autograd displacement derivatives
matched centered differences to at most `1.49e-11` metres per normalized unit.
It also checked zero translation, opening direction, prior equivalence, scalar
input shape, and the lower-bound projected gradient. The receipt is
`data/mandible-hinge-validation-001/summary.json`; it binds the runner and input
manifest hashes. Ruff and Python compilation passed. This validates the scalar
map; it does not claim new full-equilibrium finite-difference validation.

The existing physical solver/implicit-derivative validations remain unchanged.
The smoothness coefficient reuses the 36 activation gradients at identical zero
physical jaw pose from run 001; restricting the jaw's trainable subspace does not
change those activation gradients. Historical six-coordinate jaw-gradient rows
are explicitly labeled as outside the reused calibration scope.

## Run transition

The previous fit was deliberately interrupted with its accepted checkpoints
preserved; `expression-fitting-002/external-interruption.json` records the reason.
It had accepted one update each for MouthOpen, BrowDownLeft, and NeckCompression.
Run 003 starts fresh and does not transfer their unconstrained jaw poses.

The replacement transient service started at 2026-09-21 21:59:38 Asia/Shanghai.
[Comet run](https://www.comet.com/liblaf/apple/5ad3ef62e2d546f0980a6ba251a742e0).
The live publisher now follows run 003 and reports 36 total jaw variables.
The existing tailnet server remains active at
live progress (private preview omitted).
Startup and fitting are not convergence evidence.

Live verification found run 003 fitting MouthOpen with a one-element jaw state
and a one-element adjoint gradient (`-1.73615536` in normalized-angle units).
The initial forward solve passed; physical translation exports were exactly
zero. The runner hash matches both its protocol and the hinge validation receipt.
The tailnet page returned HTTP 200 and displayed 62,258,796 total variables,
including 36 jaw coordinates and zero material coordinates. At this check the
initial checkpoint had zero accepted outer updates; the first trial was running.

## First accepted update and poor initial fit

Run 003 then accepted the full first MouthOpen proposal. Its opening angle is
0.030 degrees, translation is exactly zero, and active-stress tensor RMS is
0.0482 kPa. The surface error changed from 7.630657 to 7.609062 mm, only a 0.283%
RMS reduction. This remains essentially the starting face, not a converged fit.
The PNCG equilibrium took 2,443 iterations and 296.1 seconds, passed the force
threshold and contact checks, and had zero inverted tetrahedra. The expression's
projected gradient remains far from stationary.

For comparison, run 002's first MouthOpen proposal was reduced to a quarter step
after full and half jaw-boundary motions failed CCD. Its accepted error was
7.623010 mm. Restricting the jaw removes those rejected directions in the first
run-003 trial, but the normalized Adam step still proposes just 0.03 degrees.
One step per target per round, across 36 targets, further slows visible progress
for any one expression. These observations establish an early-iteration and
throughput problem; they do not establish that the anatomy cannot fit the target.
The live page now shows accepted update counts, initial/current RMS, relative
improvement and hinge angle alongside the checkpoint figures.
