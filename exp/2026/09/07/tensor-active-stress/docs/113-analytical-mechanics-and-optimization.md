# Analytical mechanics, Adam scaling, and PSD parameterization

2026-09-08. This analysis reads the existing source, saved tensors, and saved Adam moments. Small CPU algebra checks were performed; no face forward solve, adjoint solve, optimization, or change to the physical implementation was made.

The analysis sharpens the [previous explanation](112-why-psd-result-is-smoother.md): retaining a passive energy written in terms of actual deformation does not by itself prove smoothing. At small strain, additive active stress is equivalent to a constrained preferred strain. A stronger distinction in our nonlinear implementation is a uniform lower bound on the muscle contribution's rank-one deformation stiffness. The historical Raw6 parameterization does not maintain that bound uniformly over its controls.

## 1. What follows from the constitutive equations

### Exact energy and the linearized modulus

Write `A` for the historical field named `Ainv`, without assuming that it is invertible or physically admissible. The two muscle energies are

$$
W_{\mathrm{old}}(F,A)=W_0(FA),\qquad
W_{\mathrm{new}}(F,Q)=W_0(F)+\tfrac12 Q:(F^TF-I),\quad Q\succeq0,
$$

where the implemented stable Neo-Hookean density is

$$
W_0(F)=\tfrac\mu2(\|F\|_F^2-3)-\mu(J-1)+\tfrac{\lambda_0}2(J-1)^2,
\qquad J=\det F.
$$

These are the actual formulas in [`_stable_neo_hookean_active.py`](../../../../../../src/liblaf/apple/warp/fem/_stable_neo_hookean_active.py), lines 20–32, and [`tensor_active.py`](../src/tensor_active.py), lines 34–62.

For `F = I + H`, set `e = sym(H)`. Expanding the determinant through second order gives

$$
W_0(I+H)=\mu\|e\|_F^2+\tfrac{\lambda_0-\mu}{2}(\operatorname{tr}e)^2+O(\|H\|^3).
$$

Consequently, the linear elasticity tensor is

$$
\mathbb C:e=2\mu e+\lambda_{\mathrm{lin}}\operatorname{tr}(e)I,
\qquad \lambda_{\mathrm{lin}}=\lambda_0-\mu.
$$

This distinction matters: the code's stored `lambda` is not the linear Lamé coefficient for this energy convention. The frozen muscle constants give `mu = 0.0100671141 MPa`, `lambda_0 = 0.4932885906 MPa`, `lambda_lin = 0.4832214765 MPa`, and bulk modulus `K = lambda_0 - mu/3 = 0.4899328859 MPa`. They are positive and unchanged between these activation models.

### At small strain, the two sources are equivalent before constraints

Let `A = I + S`, with symmetric `S`, and linearize jointly around zero strain and zero activation. The quadratic energies are

$$
W_{\mathrm{old}}^{(2)}=\tfrac12(e+S):\mathbb C:(e+S),
\qquad
W_{\mathrm{new}}^{(2)}=\tfrac12e:\mathbb C:e+Q:e.
$$

Setting `Q = C:S` makes their derivatives with respect to displacement identical. The remaining term `0.5 S:C:S` is independent of displacement and does not change the forward equilibrium. Equivalently,

$$
W_{\mathrm{new}}^{(2)}
=\tfrac12(e+\mathbb C^{-1}:Q):\mathbb C:(e+\mathbb C^{-1}:Q)
-\tfrac12 Q:\mathbb C^{-1}:Q.
$$

Thus the new model also has an equivalent preferred strain, `e_a = -C^-1:Q`, in this limit. Under the correspondence `Q = C:S`, PSD restricts the equivalent `S` to the cone `{S: C:S is PSD}`; arbitrary symmetric historical `S` need not lie in that cone. The displacement-independent constants are not added to the inverse objective, which remains the surface data loss.

For a homogeneous free material at linearized equilibrium,

$$
2\mu e+\lambda_{\mathrm{lin}}\operatorname{tr}(e)I+Q=0,
\qquad
\operatorname{tr}e=-\frac{\operatorname{tr}Q}{3K}\leq0.
$$

PSD does not prohibit transverse expansion or shear. It constrains the combination of deviatoric strain and volume change needed to produce them. Because this material is nearly incompressible, small volume contraction can coexist with substantial deviatoric deformation.

### There is no additional spatial filter merely from the model name

In homogeneous linear elasticity, force balance is `div(C:e + Q) = 0`. For a nonzero Fourier wavevector `k`, using the convention `exp(i k.x)`,

$$
\left[\mu|k|^2I+(\lambda_{\mathrm{lin}}+\mu)kk^T\right]\widehat u
=i\widehat Q\,k.
$$

Therefore the stress-to-displacement response has order `|u_hat| = O(|Q_hat|/|k|)`. The historical linearized model has the same transfer with `Q_hat = C:S_hat`. It is not a comparison between a filtered and an unfiltered displacement field. The stress input is differentiated once in force balance; applying the familiar `1/|k|^2` body-force response directly to `Q` would miss that derivative.

This calculation assumes constant coefficients and excludes the zero mode. Finite boundaries, tissue mixtures, and nonuniform geometry alter the transfer. In the incompressible limit, the hydrostatic part of active stress can be absorbed into pressure in the interior, subject to the boundary conditions. PSD therefore does not remove inverse nonuniqueness or impose smoothness of the activation field.

### A genuine nonlinear distinction: a uniform rank-one stiffness bound

Consider a rank-one deformation perturbation `H = a b^T`. The determinant `det(F + tH)` is affine in `t`, so `D^2 det(F)[H,H] = 0`. Direct differentiation gives

$$
D^2W_{\mathrm{new}}(F,Q)[H,H]
=\operatorname{tr}\!\left(H(\mu I+Q)H^T\right)
+\lambda_0\left(\operatorname{cof}F:H\right)^2
\geq\mu\|H\|_F^2.
$$

The result holds for fixed PSD `Q`, positive `mu`, and nonnegative `lambda_0`. For the historical term,

$$
D^2W_{\mathrm{old}}(F,A)[H,H]
=\mu\|HA\|_F^2
+\lambda_0\left(\operatorname{cof}(FA):HA\right)^2
\geq\mu\sigma_{\min}(A)^2\|H\|_F^2.
$$

The old parameterization permits `sigma_min(A)` to approach zero. It can therefore create extremely weak local deformation directions. The new PSD model preserves the passive `mu` floor. This is a stronger analytical explanation for reduced localized distortion than saying that active stress is generally smoother.

The saved historical field supports the relevance of this mechanism. Among its 288,235 active cells, `sigma_min(A)` has minimum **0.000509585** and 1st percentile **0.335035**. The corresponding stiffness-bound factors `sigma_min(A)^2` are **2.60e-7** and **0.112248**, relative to the passive `mu` floor. **2.294%** of active cells have a factor below **0.25**. The current `mu I + Q` has minimum eigenvalue at least `mu`, up to floating-point roundoff.

These statements concern the intrinsic muscle contribution. The shared material fractions and other tissues also contribute to the assembled mechanics. Rank-one stiffness is not full-Hessian convexity, global uniqueness, an inversion barrier, or a guarantee of a smooth inverse solution. PSD `Q` can have null directions; it need not add stiffness above `mu` in every direction.

The previous saved-state calculation supplies complementary evidence: physical volume-distortion RMS was **0.431684** in the baseline, but the compensated elastic `det(FA)` distortion was only **0.144994**; the current physical value is **0.022861**. This supports activation-dependent preferred volume as another relevant difference, without establishing a causal fraction of the visual improvement.

For the inverse problem, the local sensitivity is

$$
\frac{\partial u}{\partial\theta}
=-\left(\frac{\partial r}{\partial u}\right)^{-1}
\frac{\partial r}{\partial\theta},
$$

where `r(u, theta) = 0` is force balance. Both the deformation tangent and the activation-to-force derivative change with formulation. A positive added tangent at fixed activation alone cannot prove that optimization over activation will produce a smoother surface. The current endpoint also has less motion and higher data error, as documented in the [visual comparison](111-white-crinkle-report.md).

The literature likewise distinguishes constitutive properties from the labels “active stress” and “active strain”: suitable active-strain maps can preserve rank-one convexity, and equivalent uniaxial calibration does not establish equivalent shear response. See [Ambrosi and Pezzuto](https://staff.polito.it/davide.ambrosi/Papers/jelast.pdf) and [Giantesio, Musesti, and Riccobelli](https://arxiv.org/abs/1709.04977). The formulas and bounds above are derived for the implemented energy, not asserted for all models in those classes.

## 2. Why Adam still depended on the base learning rate

Adam forms moving averages of the gradient and squared gradient, applies bias correction, and takes the coordinatewise update

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\qquad
\Delta\theta_t=-\alpha\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
$$

The moments adapt the direction and its coordinate scales. The external `alpha` remains a multiplicative step-size choice; Adam does not perform a line search that selects it from trial objective values. This is the algorithm in the [original Adam paper](https://arxiv.org/abs/1412.6980) and [PyTorch documentation](https://docs.pytorch.org/docs/main/generated/torch.optim.Adam.html).

### In this run, epsilon dominated the adaptive denominator

The saved optimizer uses `betas = (0.9, 0.999)` and `epsilon = 0.01`. Reading its moments on CPU gives:

| Moment diagnostic | Step 512 | Step 1024 |
| --- | ---: | ---: |
| Median `sqrt(v_hat)` | 2.21610e-6 | 1.50547e-6 |
| 99th percentile `sqrt(v_hat)` | 1.50213e-4 | 1.05615e-4 |
| Maximum `sqrt(v_hat)` | 8.23756e-4 | 5.48282e-4 |
| Fraction with `sqrt(v_hat) < epsilon` | 100% | 100% |
| RMS `m_hat` | 2.07352e-5 | 1.05209e-5 |

Consequently,

$$
\Delta\theta_t\approx-\frac{\alpha}{\epsilon}\widehat m_t.
$$

Replacing the actual denominator by `epsilon` changes the step-512 unprojected direction by only **2.216% in relative RMS**. Numerically, this run is close to momentum gradient descent in the chosen coordinates, with `alpha/epsilon = 30` or `60`, rather than strongly normalized Adam. The moment magnitude falls during continuation, so a fixed `alpha` produces smaller later steps.

This also explains why approximate gradient-scale invariance is weak here. Under positive rescaling `g -> c g`, Adam's direction becomes

$$
\frac{c\widehat m}{c\sqrt{\widehat v}+\epsilon}
=\frac{\widehat m}{\sqrt{\widehat v}+\epsilon/c}.
$$

Fixed epsilon makes parameter units and objective normalization consequential. The source deliberately retained the historical epsilon for the matched experiments; these observations do not establish that a smaller epsilon would improve the run. A previous matched-first-step smaller-epsilon candidate did not meet its selection threshold, as recorded in [the refined optimizer report](98-refined-optimizer-report.md).

### What the later learning-rate probe actually established

Both probe arms started from identical `Q`, displacement, Adam moments, and counter, with the same cached first gradient. Only the base rate changed. Moment restoration and the rate replacement are explicit in [`102-learning-rate-continuation.py`](../src/102-learning-rate-continuation.py), lines 292–331.

| Probe quantity | LR 0.3 | LR 0.6 |
| --- | ---: | ---: |
| First accepted physical stress-update RMS | 1.41128e-5 MPa | 2.82228e-5 MPa |
| Surface fit RMS after 32 updates | 2.270270 mm | 2.229375 mm |
| Fit-RMS decrease from 2.313335 mm | 0.043065 mm | 0.083960 mm |

The first physical-update ratio is **1.999808**. Over 32 updates, LR 0.6 travels **1.973164 times** the accumulated physical stress path and achieves about **1.95 times** the fit reduction. This is consistent with moving roughly twice as far in the same update budget. It is not evidence that LR 0.3 stopped working or that LR 0.6 is more efficient per unit physical path. The full receipts are linked in [the learning-rate report](108-learning-rate-report.md).

Thus “we needed a larger learning rate in the latter half” is too strong. We demonstrated a local advantage per fixed update budget and selected that arm. We did not establish a generally necessary increasing schedule, the best global rate, or convergence of the inverse problem.

## 3. Avoiding PSD clipping through parameterization

Yes, a factorization can enforce PSD by construction. It changes the optimization geometry, however, and exact zero activation is a special boundary point.

### A simple six-parameter factor

Use an unconstrained lower-triangular `L` with six entries per cell and

$$
Q=Q_{\mathrm{ref}}LL^T.
$$

No PSD clipping is needed. All PSD tensors, including rank-deficient ones, are representable. The diagonal does not need a softplus for PSD: allowing signed diagonal entries is valid. A full 3-by-3 factor `B` also works but uses nine coordinates for six physical degrees of freedom. Factor formulations are standard; see [Bhojanapalli, Kyrillidis, and Sanghavi](https://proceedings.mlr.press/v49/bhojanapalli16.html). Their convergence results have assumptions on the objective and initialization that are not established for this inverse-physics problem.

Let `G = sym(d loss/dQ)` under the Frobenius pairing. For the full factor,

$$
\nabla_B\mathcal L=2Q_{\mathrm{ref}}GB.
$$

For a lower-triangular factor, keep the lower-triangular entries of `2 Q_ref G L`. At `L = 0`, every factor gradient is zero, regardless of whether the gradient in physical `Q` contains a useful feasible direction. Adam cannot leave this exact-zero initialization with zero moments. The current run starts exactly at zero in [`20-face-inverse.py`](../src/20-face-inverse.py), lines 327–329.

A new factor experiment should therefore start from a small full-rank state, for example `L = sqrt(delta) I`, giving `Q = delta Q_ref I`. That is a small nonzero baseline, not the existing exact-zero baseline. The scale should be checked against the resulting physical deformation and solve accuracy. At rank-deficient factors, `z^T DQ[L][Delta L] z = 0` for every `z` in `ker(L^T)`: positive stress in these null directions appears only at second order. Arbitrary saved Adam moments cannot simply be reused in the new coordinates.

### The exact-zero obstruction is general

Suppose a differentiable map `phi` from unconstrained real parameters into PSD matrices satisfies `phi(theta_0) = 0`. For any vectors `z` and parameter direction `h`,

$$
f(t)=z^T\phi(\theta_0+th)z\geq0,\qquad f(0)=0.
$$

The scalar function has a minimum at zero, so its derivative vanishes. Hence `z^T Dphi(theta_0)[h] z = 0` for every `z`, which forces

$$
D\phi(\theta_0)=0.
$$

No smooth unconstrained PSD parameterization can simultaneously represent exact zero at finite parameters and provide nonzero first-order sensitivity there. One must accept a nonzero start, constrained parameters, or a nonsmooth/boundary-aware operation. This is not specific to the square factor.

### Comparison of common choices

| Parameterization | Exact zero / rank deficiency at finite parameters? | Main optimization issue |
| --- | --- | --- |
| Direct six coordinates with projection | Yes | Outward step components are discarded; boundary is nonsmooth |
| Plain lower-triangular `Q = Q_ref L L^T` | Yes | Zero factor has zero gradient; near-null directions have weak sensitivity |
| Symmetric square `Q = Q_ref S^2` | Yes | Same zero trap; opposite-sign eigenvalues of `S` create additional derivative degeneracy |
| Cholesky with softplus-positive diagonal | No; SPD only | Approaching zero requires increasingly negative parameters and weak gradients |
| Matrix exponential `Q = Q_ref exp(S)` | No; SPD only | Near-zero gradients vanish; large positive eigenvalues amplify sensitivity |
| Spectral softplus of symmetric `S` | No; SPD only | Near-zero gradients vanish; a naive eigenvector-backprop implementation needs care at repeated eigenvalues |

The matrix-function softplus is smooth mathematically. Differentiating an implementation expressed through arbitrary eigenvectors is a separate numerical issue; [PyTorch documents eigenvector-gradient limitations near repeated eigenvalues](https://docs.pytorch.org/docs/main/generated/torch.linalg.eigh.html).

For scalar dimensionless `q` near zero,

$$
q=b^2:\quad\frac{dq}{db}=2\sqrt q\ \text{on the positive branch},
\qquad
q=e^s:\quad\frac{dq}{ds}=q,
\qquad
q=\operatorname{softplus}(s):\quad\frac{dq}{ds}\sim q.
$$

With ordinary gradient descent, these lead respectively to induced rates in dimensionless `q` proportional to `4 q` and `q^2` multiplying `-d loss/dq`. Thus removing clipping can introduce weak physical progress near zero. In an epsilon-dominated Adam regime, changing coordinates does not automatically remove this problem. In general, `Delta Q = Dphi(theta) Delta theta + O(||Delta theta||^2)`; Adam's nominal learning rate does not specify a fixed physical stress step after a nonlinear parameterization.

### PSD and the upper bound are separate constraints

Plain factors, exponentials, and softplus do not enforce the old upper stress cap. If a smooth factor experiment must retain an upper bound, one possibility is

$$
M=LL^T,\qquad Q=Q_{\max}M(I+M)^{-1}.
$$

Its eigenvalues are `Q_max m/(1+m)`, in `[0, Q_max)`. It represents zero and rank deficiency but has the same zero-factor trap, and gradients weaken near the upper bound. Exact equality at the upper cap is reached only as a limit. No claim is made that the unused cap would remain unused after changing parameterization.

### What to retain and what to test

If exact zero initialization and direct access to the full PSD boundary are required, retain the current direct coordinates and projection. They avoid the zero-gradient trap. For a symmetric candidate, lower eigenvalue clipping is the Frobenius nearest-PSD projection; see [Higham](https://nhigham.com/wp-content/uploads/2023/10/high88d.pdf). The current two-sided clamp is the Frobenius projection onto the spectral box `0 <= Q <= Q_max I`, by spectral separability of that norm. The current orthonormal six-coordinate map makes that metric exact in [`tensor_controls.py`](../src/tensor_controls.py), lines 13–75. This acts on activation tensors, not on rendered geometry.

If avoiding clipping is the priority and a small nonzero initial stress is acceptable, the simplest useful experiment is the plain lower-triangular factor. Compare it with direct coordinates from the **same** small physical `Q`, with fresh optimizer moments in both arms and calibrated initial physical `Delta Q` scales. Keep the full tensor field, physical model, and intended cap policy explicit. Compare fit, motion, distortion, and physical update path; equal numerical Adam learning rates would not be a fair coordinate comparison. No such optimization comparison has been run here.

## Checks and reproducibility

The two exact rank-one Hessian identities were checked using CPU float64 autodifferentiation on 12 random old/new tensor examples; maximum absolute discrepancy was `3.55e-15`. Joint small-strain expansions showed decreasing remainder at decreasing perturbation size. A direct autograd check verified the factor gradient is zero at `B = 0` despite a nonzero physical matrix gradient. These are algebra checks, not evidence of face-level convergence or causal visual smoothing.

The saved field statistics use `face-actuation-diagnosis/data/11-historical-no-skin/final.npz` (`Ainv`) and `tensor-active-stress/data/102-fit1024/final.npz` (`Q`), with identical active-cell IDs already verified in the [saved-state comparison](111-white-crinkle-report.md). The stiffness-factor quantiles are cell-count quantiles, not volume-weighted values. They follow from `min(abs(eigvalsh(Ainv)), axis=1)**2` and `1 + min(eigvalsh(Q), axis=1)/mu`.

The following CPU snippet reproduces the Adam measurements from the repository root. It reads only checkpoints generated by these experiments.

```python
from pathlib import Path
import torch

base = Path("exp/2026/09/07/tensor-active-stress/data")
for name in ("92-fit512", "102-fit1024"):
    saved = torch.load(base / name / "optimizer-latest.pt",
                       map_location="cpu", weights_only=False)
    optimizer = saved["optimizer"]
    group = optimizer["param_groups"][0]
    state = optimizer["state"][group["params"][0]]
    t = int(state["step"])
    b1, b2 = group["betas"]
    m = state["exp_avg"] / (1 - b1**t)
    v = (state["exp_avg_sq"] / (1 - b2**t)).sqrt()
    direction = m / (v + group["eps"])
    approximation = m / group["eps"]
    quantiles = torch.quantile(v.flatten(), v.new_tensor((0.5, 0.99, 1.0)))
    rms = lambda x: float(x.square().mean().sqrt())
    print(name, group["lr"], group["eps"], quantiles.tolist(),
          float((v < group["eps"]).double().mean()), rms(m),
          rms(direction - approximation) / rms(direction))
```
