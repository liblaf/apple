# Adding contraction-only activation to the Adam comparison

The added control space allows contraction along arbitrary principal directions while prohibiting active extension. It sits between the existing x-fiber-only case and unconstrained symmetric activation. All cases retain Adam, the raw MSE objective, the original learning-rate schedule and the corrected physical-volume law `J=det(F)`.

## Constraint and implementation

The optimized matrix is the inverse active strain, **B=A⁻¹**. For a symmetric pure-stretch activation, enforce

$$
B=I+S,\quad S=S^T\succeq0.
$$

Consequently, B has eigenvalues at least 1, and A has principal stretches in `(0,1]`. For finite controls this excludes active extension, singularity and reflection. No upper bound on B is imposed, so arbitrarily strong contraction remains possible. This constrains the activation tensor; the actual deformation F may still expand through elastic coupling and the near-incompressible response. In particular, F=A is not generally a stress-free state of this corrected energy.

| Comparison | Constraint | Controls per muscle triangle |
| --- | --- | ---: |
| x-fiber contraction | B=diag(1+a,1), a≥0 | 1 |
| Contraction only, free directions | B=I+S, S symmetric PSD | 3 |
| Free symmetric B | B=I+S, S symmetric and otherwise unrestricted | 3 |

After each ordinary Adam proposal, diagonalize S and clamp its negative eigenvalues to zero:

```python
eigenvalues, eigenvectors = np.linalg.eigh(S)
S = (eigenvectors * np.maximum(eigenvalues, 0)[..., None, :]) @ eigenvectors.swapaxes(-1, -2)
B = np.eye(2) + S
```

This is the nearest feasible tensor in the matrix Frobenius norm. [Higham's spectral projection result](https://nhigham.com/2021/01/26/what-is-the-nearest-positive-semidefinite-matrix/) gives the corresponding eigenvalue-clipping rule. It is implemented in [controls2d.py](../src/controls2d.py:14).

The raw parameter coordinates remain `(Sxx,Syy,Sxy)`, so the Frobenius norm counts the off-diagonal component twice; this projection is not the Euclidean projection in that packed vector. Adam uses the same raw-coordinate gradients and moments as the free case. Only the proposed control is projected. For the separate stationarity diagnostic, the symmetric matrix gradient uses off-diagonal entry `(dL/dSxy)/2`, followed by a unit-step Frobenius gradient mapping. No loss-based outer rejection or gradient clipping is added.

Off-diagonal entries are allowed because the contraction directions may rotate relative to x/y. Elementwise positivity would not suffice: a symmetric matrix with diagonal entries 1 and off-diagonal entries 2 has a negative eigenvalue. Likewise, merely making B positive definite would not suffice, since positive eigenvalues below 1 would permit active extension through A=B⁻¹.

For the corrected material,

$$
W(F,B)-W(F,I)=\tfrac\mu2\operatorname{tr}\big[F^TF(BB^T-I)\big],
\qquad \Delta P=\mu F(BB^T-I).
$$

The new constraint also makes the effective tensor `BBᵀ-I` positive semidefinite. This is an algebraic property of the same active-strain energy, not a switch to another constitutive law. It does not establish global convexity, an inversion barrier or realistic anatomy.

## Measured three-way comparison

All six cases were run from identity activation on the same 100×10 mesh, with the original bottom/side fixation and parabolic targets. Adam settings remain initial learning rate 0.03, decay 0.99 per update, betas `(0.9,0.999)`, epsilon `1e-8`, raw top-node vector MSE and a budget of 1,200 updates. Physical determinant terms remain `det(F)`, not `det(F @ B)`. Forward tolerance remains `1e-10`; no new regularizer or activation amplitude cap was introduced.

The four x/free displacement and control histories reproduce the previous Adam results **exactly**, including the free case's failed proposal for the larger target. Thus adding the new mode did not change the original numerical trajectories.

![Three activation spaces with the same Adam settings](../data/110-contraction-figures/contraction-loss.png)

| Target / control | Last solved update | Final MSE / h² | Final RMS / L | MSE reduction from identity | Loss increases |
| --- | ---: | ---: | ---: | ---: | ---: |
| h=0.05 / x-fiber | 1200 | 0.50709744 | 0.03560539 | 5.87% | 0 |
| h=0.05 / contraction only | 1200 | 0.47832548 | 0.03458054 | 11.21% | 0 |
| h=0.05 / free B | 1200 | 0.44119758 | 0.03321135 | 18.10% | 7 |
| h=0.20 / x-fiber | 1200 | 0.49075139 | 0.14010730 | 8.90% | 0 |
| h=0.20 / contraction only | 1200 | 0.48787780 | 0.13969650 | 9.44% | 0 |
| h=0.20 / free B | 262 | 0.44228121 | 0.13300845 | 17.90% | 10 |

Normalization by h² is only for display and comparative diagnostics. The optimizer still uses raw MSE. The initial normalized loss is 0.53872053 in all cases. A plotted loss rise exceeds `max(1e-12,1e-10*abs(previous_normalized_loss))`. Length units L are not calibrated to mm here.

Both contraction-only cases finished 1,200 updates without a forward failure or measured loss increase. Their normalized Frobenius gradient mappings remain 2.296e-4 and 1.228e-4, so these are budget endpoints, not demonstrated inverse optima. The final learning rate is about 1.75e-7. Free B for h=0.20 failed its update-263 forward proposal; its last solved update is 262 and its animation holds that state explicitly.

![Final shapes for h=0.05](../data/110-contraction-figures/contraction-final-h050.png)

![Final shapes for h=0.20](../data/110-contraction-figures/contraction-final-h200.png)

The contraction-only case improves the saved fit over x-fiber contraction, with a modest gain for h=0.20. It preserves non-extending, invertible activation while free B still fits better using inadmissible activation modes. This is a comparison of finite Adam trajectories; the larger control space does not by itself guarantee better finite-step optimization. The new constraint also does not remove the visible element-scale variation in the deformed muscle band.

## Verification and limits

The independent [audit JSON](../data/120-contraction-checks/contraction-checks.json) checks all 400 muscle elements in each of the 1,201 saved states for both contraction-only cases:

- Minimum eigenvalue and singular value of B remain 1 to about 1e-15 roundoff, hence the maximum singular value of A remains 1 to the same tolerance. Minimum det(B) is 1; no nonpositive determinants occur.
- Minimum eigenvalue of `BBᵀ-I` is no lower than -2.67e-15, consistent with positive semidefiniteness to roundoff.
- The rotated-tensor projection check has maximum error 2.22e-16. Identity and normal-cone gradient checks pass, including the packed shear factor of two.
- The new mode's implicit gradient agrees with a centered finite difference at a strictly feasible rotated control with relative error **3.91e-10**.
- Across all six endpoints, force residuals are at most 6.72e-14, fixed-boundary error is exactly zero and boundary/integrated-area disagreement is at most 5.56e-17. All endpoint smallest algebraic Hessian eigenvalues are positive. Both baseline trajectories and source snapshots pass their regression checks.

For contraction-only h=0.05/h=0.20, minimum physical J is 0.58257/0.24587; the smallest Hessian eigenvalue is 5.021e-5/4.664e-5. Maximum singular values of B are 4.3506/5.4304, so the smallest active principal stretches are approximately 0.230/0.184. These active contractions are distinct from actual tissue strains. Local positive J and a positive endpoint Hessian are not proofs of global geometric validity or inverse convergence.

For the next meeting, add this middle control space to the activation-definition slide, use the three-way loss figure and equal-scale shapes, and report the fit/constraint tradeoff together. Keep the fixed-boundary target-area conflict and decaying-learning-rate limitation from the [Adam report](70-adam-results.md); the constraint does not resolve either issue.

## Reproducibility and artifacts

The Cherries computation, renderer and verifier all exited normally with code 0. This run retained complete last-valid Adam checkpoint state and used reduced-frequency remote metrics. Automatic Git commits were disabled. Existing results were preserved; no commit or push was made.

Working directory: `exp/2026/09/14/fiber-contraction-parabola`.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export COMET_AUTO_LOG_ENV_DETAILS=false COMET_AUTO_LOG_GIT_PATCH=false

CHERRIES_NAME='Adam comparison: x-fiber, contraction-only, free' \
CHERRIES_TAGS='2d,parabola,adam,contraction-only,active-strain' \
uv run python src/70-run-adam.py \
  --modes x_contraction,contraction_only,unconstrained \
  --output 100-adam-contraction
```

The figure and audit entrypoints are [110-render-contraction.py](../src/110-render-contraction.py) and [120-verify-contraction.py](../src/120-verify-contraction.py). Their defaults read `100-adam-contraction`. Existing output directories are not overwritten.

- [Protocol and source hashes](../data/100-adam-contraction/protocol.json), [six-case summary](../data/100-adam-contraction/summary.json), and [figure statistics](../data/110-contraction-figures/contraction-figure-manifest.json).
- [Loss PDF](../data/110-contraction-figures/contraction-loss.pdf), [h=0.05 animation](../data/110-contraction-figures/contraction-history-h050.mp4), and [h=0.20 animation](../data/110-contraction-figures/contraction-history-h200.mp4).
- Comet: [six-case computation](https://www.comet.com/liblaf/apple/dfd10516476849cc8a09aaddf9dace13), [figures](https://www.comet.com/liblaf/apple/cb78ac025d0c42c8b85f6ba42740ca53), and [independent audit](https://www.comet.com/liblaf/apple/6edb09c3c52541b1ab029cc9e1d5b9ed). Corresponding logs and Comet summary blocks are under the experiment's `logs/` directory.

Figures were visually checked. Both videos contain 121 frames at 15 fps (8.067 s), using actual saved updates with no interpolation. A stopped case is held and labeled. The h=0.05/h=0.20 video resolutions are 2100×360/2100×510. Focused Ruff checks and `git diff --check` passed.
