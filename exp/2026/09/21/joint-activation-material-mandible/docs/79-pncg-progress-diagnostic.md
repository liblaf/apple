# PNCG progress and contact-conditioning diagnostic

This CPU-only analysis compares two stopped continuations of the same positive
IPC model and exact force gate. It reads accepted-step traces and terminal
receipts; it does not evaluate the mechanics model or estimate a Hessian
condition number.

Run 006 used a 10 micrometre displacement cap. All 3,343 accepted steps hit
that cap, every Armijo search accepted its first trial, all raw directional
curvatures were positive, and every sampled CCD fraction was one. The raw
quadratic model proposed infinity-norm displacements from 0.618 to 6.754 mm
(median 2.199 mm). The exact force fell only from `2.565e-7` to `2.048e-7`
MPa m2. Thus neither contact, backtracking, nor negative curvature limited this
segment: the displacement cap did.

Run 007 raised the cap to 0.5 mm. It initially reduced force much faster, but
then repeatedly approached contact at near-zero separation. The sampled gap
fell to `8.015e-15 m`, CCD history fell to `2.884e-4`, 74 directions had
negative raw curvature, damping reached its configured `1e-3` ceiling, and a
CCD-limited step took 10.79 seconds. Although total energy remained monotone
and standard IPC stayed positive, the terminal exact force was `3.430`
MPa m2. The run was correctly interrupted after 331 accepted steps. Both runs
ended with zero inverted tetrahedra.

The PNCG implementation explains both regimes. Dai--Kou directions assume a
useful line search along each direction; taking roughly 0.5% of the raw Newton
displacement in run 006 provides little effective conjugacy. In run 007, the
larger trials entered the singular barrier layer. Damping is added as
`factor * mean(abs(Hdiag)) * ||p||^2`, but a run initialized at `1e-6` has a
hard default maximum of `1e-3`. Moreover, a negative-curvature trial accepted
at Armijo step zero does not raise damping because the update multiplier is
`(0 + 1)^2 = 1`.

The evidence supports an intermediate 0.1 mm cap, initial damping `1e-3`, and
frequent declared restarts while retaining PNCG, standard IPC, and the exact
force gate. This is a numerical continuation choice, not a constitutive-model
change.

A separate synthetic check evaluated a proposed 10 nm CCD-only clearance.
Eight repeated approaches reached 4.12 nm with zero clearance and stopped at
11.45 nm with `min_distance=1e-8 m`. The barrier `dmin` remained zero, so
`dhat`, stiffness, energy, gradient, and Hessian are unchanged. At the stopped
006 checkpoint the actual gap was 17.922 micrometres, so this proposed buffer
was inactive. The buffer still changes the numerical feasible set for states
that attempt to cross 10 nm and must be declared and validated before use.

```bash
CHERRIES_NAME='PNCG cap and contact-conditioning trace diagnostic' \
CHERRIES_TAGS='joint-inverse,pncg,contact,diagnostic,cpu' \
uv run python src/79-diagnose-pncg-progress.py \
  --output-dir data/pncg-progress-diagnostic-001
```

The [receipt](../data/pncg-progress-diagnostic-001/summary.json) has SHA-256
`ea8ab3806e89120b8d1e01d34ed9e1b53f242072352237e4189a6eff931215e2`.
The script SHA-256 is
`d1c3da31cbd7ca86b9dfaca333b46e9884e013f7910172133c15bcdab569b1a9`.
The [Comet run](https://www.comet.com/liblaf/apple/dc298fc157634a82b1787010a4cbf58b)
completed normally.
