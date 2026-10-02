# Mixed positional and gradient-space loss on the 3D face

Test the user's combined loss using the same learned-axis contraction-only
activation, corrected physical-volume energy, mesh, target skin correspondence,
materials, boundary conditions, and forward/adjoint solvers as the preceding
learned-axis experiment. Old numerical sources and results remain unchanged.

Every branch starts with `s=0`, `B=I+snnT=I`, and zero displacement. Strength
and unit axis are learned (three physical degrees of freedom per active cell),
with strength clamped nonnegative and axes normalized after Adam. All branches
share the exact neutral-gradient-derived axes archived by the preceding study.
These axes are effective target-derived initialization, not anatomical fibers
or loss-specific optimal axes. Axis derivatives vanish at zero strength.

Let `L2` be the existing area-weighted positional component MSE in mm² and
`Lg` the existing area-weighted rest-surface displacement-gradient residual
squared Frobenius norm. Recompute `L20=L2(0)` and `Lg0=Lg(0)` once from the
neutral displacement, freeze them, and set `K=L20`. The mixed data objective is

```text
D_beta = K * (L2/L20 + beta * Lg/Lg0) / (1 + beta)
L = D_beta + lambda * R
```

The pure endpoints are `D_L2=L2` and `D_gradient=K*Lg/Lg0`. Every objective
has the same neutral value; this does not guarantee equal Adam updates. Record
the physical Frobenius RMS of each projected initial Adam update and pairwise
directional cosines. The new pure-gradient scale is about 116.77; the prior
200-update gradient-only run used 99.57, so it is historical context rather
than a strictly matched control for this normalized family.

Before pilots, verify composite algebra/autograd on CPU and full implicit
strength/axis finite differences at beta=1 with the previously accepted tight
equilibrium settings. Use training tolerances for all optimization runs.

Run five matched 20-update no-smoothness pilots: pure L2, pure gradient, and
beta=0.25,1,4. Among the three mixed pilots with no physical inversions, choose
the beta minimizing `max(L2/L20, Lg/Lg0)` at update 20. Break exact ties by smaller
beta. This is an explicit balanced training diagnostic, not held-out validation
or a claim of optimal weighting. Record the pure-loss pilots as matched
20-update baselines; do not present them as 200-update comparisons.

For the selected beta, start a fresh eight-update no-smoothness probe to set
the coefficient scale by the ratio of Euclidean strength-plus-axis gradient
norms of data and regularizer. Recheck the full implicit derivative for the
selected objective. Then restart four 20-update smoothness pilots at zero and
0.1,1,10 times this coefficient. Select the smallest positive tested coefficient
with at least 75% reduction in squared activation-tensor roughness relative to
off and zero physical inversions; report the fit cost. Fail visibly if none
qualifies, then explicitly extend the bracket if needed.

The unchanged sign-invariant regularizer is `R=ell²/V_active * sum w_ij
||B_i-B_j||_F²`, ell=5 mm, on same-muscle face-sharing cell pairs. It constrains
activation variation; gradient matching constrains target-relative surface
derivatives. These are separate mechanisms.

The primary final comparison uses the selected beta for two fresh neutral-start
branches: mixed loss with smoothness off and on, 200 Adam updates each. Adam
lr=0.3, eps=0.01, betas=(0.9,0.999); both halve lr every 100 updates. Preserve
per-evaluation scalar traces and solver receipts and full state checkpoints.
Two consecutive 25-update checks starting at 100 require own-objective span
below 0.1% and physical rank-one projected-gradient RMS below 1% of initial.
Budget exhaustion and solver success are not inverse convergence or stability.

Report position RMS, gradient RMS, motion, regional target-relative high-pass
residuals, activation roughness and strengths, axis changes, physical det(F),
inversions, and convergence. Inspect comparable unexaggerated surface images.
Run an independent CPU audit of final objective accounting, activation
constraints, shared initialization/schedule, and signed volume ratios.
