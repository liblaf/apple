# Fresh face fits with the selected reference length

Run two independent neutral-start unrestricted Raw6 face fits for 200 Adam
updates, with activation smoothing off and on. Load the selected
`data/110-shape-loss-config/loss-config.json`: fixed physical reference length
13.236093032531715 mm and normal weight one. Use the literal dimensionless loss
`J = P/l_ref_mm**2 + N + eta*R`. P is the reference-area component position MSE
in mm², N the reference-area mean squared oriented unit-normal chord, and R
the existing same-muscle tensor variation with 5 mm smoothing length.

The selected calibration equates 2 mm vector position RMS to a 5 degree
reference normal deviation. At neutral P/l_ref² is 0.04940857010416178 and N
is 0.0243814646293817. Multiplying the data objective by l_ref² gives the old
relative weighting beta 0.49346630712002737; this is only a weighting identity.
The fixed physical length preserves the absolute calibration on this face.

Scale Adam epsilon and the smoothing coefficient consistently with the new
dimensionless objective: epsilon is `0.01/l_ref_mm**2`; eta is zero off and
`0.003214147722027223/l_ref_mm**2` on. Adam learning rate stays .3 and moment
betas stay .9/.999, with no schedule. This preserves the nominal Adam update
under an overall loss rescaling from a corresponding dimensional objective;
finite solver tolerances still prevent claims of bitwise trajectory identity.

Start every branch with q=0, B=I, u=0, zero Adam moments and zero adjoint initial
guess. Retain the same surface support, passive materials, constraints, physical
volume energy, target, Raw6 controls, and forward/adjoint tolerances. No fitted
state, determinant barrier, contact, or activation projection is added.

Use new `reference_study.py` and `120-run-reference.py`, reusing the frozen
`10-run.py` branch loop, gradient check, checkpoint and source-archive helpers.
The helper field named beta is a direct normal coefficient in this new study;
the old `L20/N0` multiplier is not applied. Protocol metadata records this
meaning. The old source files and completed outputs remain unchanged.

First run full implicit finite differences for normal-only and combined with
smoothness, using the existing 2% gate and tightened forward tolerances only
during validation. The unsmoothed combined derivative is covered by removing
the independently checked direct q-only R derivative. Next run one-update
neutral smoke fits and the independent CPU audit. Main fitting starts only
after those pass. Preserve failures rather than relaxing tolerances or silently
restarting. Independent final CPU verification checks source/fixture hashes,
selected configuration and preflight lineage, initial/Adam state, accepted
solve receipts and endpoint geometry/objective decomposition.

Compare at equal updates with the prior beta 0/.05/.25/1 cases. Beta 0/.05
paused at update 100 and resumed under an audited finite-tolerance replay;
beta .25, beta 1 and these new fits run continuously from neutral. Report
position and normal RMS, target-relative surface-gradient and high-pass
residual, activation variation, motion, physical det(F), inversion counts and
own-objective convergence diagnostics separately. Include the latest common
saved inversion-free checkpoint. Completing 200 updates is not convergence or
mechanical stability. Objective and absolute gradient magnitudes across
different normalizations must not be used as fit-quality rankings.

At setup the shared RTX 4090 had approximately 20 GiB free, with another
eye-neutral forward job already running. Leave it and desktop apps untouched;
do not interpret wall-clock differences against prior runs as algorithm speed.

Working directory: `exp/2026/09/21/normal-matching-face`. Use the repository
Python interpreter, named/tagged Cherries runs and CPU thread caps of four.

```bash
CHERRIES_NAME='Face reference loss: implicit derivative validation' CHERRIES_TAGS='face,raw6,normal-matching,reference-length,gradient-validation' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/120-run-reference.py --output 121-reference-validation --validation-only true
DEBUG=1 CHERRIES_NAME='Face reference loss: neutral one-update smoke' CHERRIES_TAGS='face,raw6,normal-matching,reference-length,smoke' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/120-run-reference.py --output 123-reference-smoke --steps 1
CHERRIES_NAME='Face reference loss: fresh neutral fits' CHERRIES_TAGS='face,raw6,normal-matching,reference-length,neutral,smoothness' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/120-run-reference.py
```

Verification, analysis and render commands and outcomes will be recorded in
the validation/results reports once executed; this protocol stays frozen as
an input to the validation and main source/lineage gates.
