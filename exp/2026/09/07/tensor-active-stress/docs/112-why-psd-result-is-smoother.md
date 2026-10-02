# Why the PSD active-stress result looks smoother

Research and saved-state audit, 2026-09-08. No forward solve, adjoint solve, or optimization was run for this investigation.

The strongest explanation is a change in the admissible mechanics. The historical Raw6 model can change each active tetrahedron's preferred deformation, including its preferred volume. The new model retains the passive energy of the actual deformation and adds a positive-semidefinite quadratic stress contribution. This removes some ways of fitting the target through local distortion and adds deformation stiffness. The saved states support this mechanism, but do not identify how much of the visual difference it causes independently of fit, motion, and optimization history.

There is still no spatial smoothness penalty in the selected PSD run. A restriction on the constitutive model can favor more regular geometry without making neighboring activation tensors similar.

## 1. What changed in the equations

The historical muscle term is

$$
W_{\mathrm{old}}(F,A_{\mathrm{inv}})=W_0(G),\qquad G=FA_{\mathrm{inv}}.
$$

Here, the field named `Ainv = I + symmetric(q)` has six independent parameters per active tetrahedron. Its name does not guarantee that it is invertible, positive-definite, or orientation-preserving. The implementation evaluates both the distortion invariant and determinant on `G`: [`_stable_neo_hookean_active.py`](../../../../../../src/liblaf/apple/warp/fem/_stable_neo_hookean_active.py), lines 20–32. In particular, its volumetric term uses

$$
J_G=\det G=\det F\,\det A_{\mathrm{inv}},
$$

so a large physical volume change can be partly compensated by activation. This is an intended capability of a change of preferred configuration, but unrestricted independent tetrahedral controls give the inverse problem considerable freedom to use it.

An exact local example makes this freedom explicit. For an orientation-preserving deformation with polar decomposition `F = R U`, choosing the symmetric positive-definite `Ainv = U^-1` gives `G = R`. The historical muscle energy and stress are then zero, even if the physical stretch `U` is far from identity. This is a statement about the muscle contribution in that cell. The surrounding mesh, fixed constraints, and nonmuscle material fractions still exert forces; it does not establish zero total face energy. It also does not mean the saved optimizer actually canceled all strain this way.

The new muscle term is

$$
W_{\mathrm{new}}(F,Q)=W_0(F)+\tfrac12 Q:(F^T F-I),\qquad Q\succeq0.
$$

The passive term continues to evaluate the actual `F`, including its determinant. The active first Piola stress is `F Q`: [`tensor_active.py`](../src/tensor_active.py), lines 34–62. Thus activation cannot change the argument of the passive volume penalty from `det(F)` to a compensated determinant. It can still drive deformation against that resistance, including compression; actual volume preservation is not a hard constraint.

## 2. What the saved meshes show

The comparison uses the June no-skin saved step 194 and PSD step 1024 from the [white-skin and muscle-section comparison](111-white-crinkle-report.md). Their input provenance and topology checks are recorded in [`110-white-crinkle/summary.json`](../data/110-white-crinkle/summary.json).

The following new CPU calculations use the same 288,235 active tetrahedra in both files. RMS weights are rest tetrahedral volume multiplied by `MuscleFraction`, normalized over that cohort. The determinant is dimensionless.

| Quantity on active tetrahedra | Historical Raw6 | Current PSD |
| --- | ---: | ---: |
| RMS of actual `det(F) - 1` | 0.4316839084 | 0.0228614211 |
| RMS of historical elastic `det(F Ainv) - 1` | 0.1449939302 | Not applicable |
| Cells with actual `det(F) < 0` | 110 | 0 |
| Cells with `det(F) < 0` but `det(F Ainv) > 0` | 75 | Not applicable |

The baseline's compensated volumetric deviation is about one third of its physical volumetric deviation. The correlation between `log(abs(det(F)))` and `log(abs(det(Ainv)))` is **−0.7576975**. These observations support volumetric compensation in the actual saved baseline. They do not prove that it caused every visible bump.

The 75 cells with opposite physical and elastic orientation illustrate a more extreme use of the unrestricted parameterization: a negative determinant of `Ainv` makes an inverted physical cell have a positive elastic determinant. This does not mean its total energy is small or its strain is harmless.

In fact, the volume-weighted RMS of `||singular_values(F) - 1||` is **0.591174** in the baseline, while the corresponding value for `G` is **0.584666**. Their similarity rules out claiming that the baseline erased all elastic distortion. The current physical value is **0.275661**.

The counts above concern active tetrahedra only. Across the whole volume, the corresponding inversion counts are **142** and **0**, as recorded in the existing visual comparison.

## 3. Why this particular PSD stress can resist local variations

For a deformation perturbation `H`, holding `Q` fixed, direct differentiation gives

$$
D^2 W_{\mathrm{act}}(F)[H,H]
=\operatorname{tr}(H Q H^T)
=\|H Q^{1/2}\|_F^2\geq0.
$$

The code implements this exact tangent in [`tensor_active.py`](../src/tensor_active.py), lines 101–125. With `H = grad(v)`, its integrated contribution is a nonnegative, direction-dependent cost for displacement gradients. For a sinusoidal displacement perturbation, this tangent contribution scales with squared wavenumber where the wave direction has positive stiffness in `Q`. It is therefore a plausible mechanism for making short-scale deformation harder to generate.

This is a statement about the forward deformation tangent at fixed activation. It does **not** prove that the reduced inverse objective is convex, that every high-frequency response decreases, or that the resulting geometry is globally smooth. Rank-deficient `Q` contributes no extra stiffness in its null directions. There is no term here comparing `Q` in neighboring tetrahedra.

Both formulations already share an elastic medium that transmits activation to the surface. Saying that elasticity filters small-scale forces is therefore insufficient to explain their difference. The relevant changes are the preferred configuration, the retained passive resistance, and the restricted active stress.

## 4. The PSD constraint mattered; the upper cap did not

The selected lineage contains no smoothness, magnitude, or rank penalty. Its recorded regularizer gradients are zero, and its full objective equals the data objective. But each Adam update is followed by spectral projection: [`102-learning-rate-continuation.py`](../src/102-learning-rate-continuation.py), lines 509–515.

The lower PSD constraint clipped negative proposed eigenvalues on every one of the 1,024 updates. On the final update, **38.9975%** of the proposed eigenvalues were negative. The upper cap never clipped an eigenvalue along this lineage. Thus “the cap did not bind” must not be read as “the projection did nothing.”

For a physical deformation with positive determinant, the active Cauchy stress is `F Q F^T / det(F)`, also PSD. This allows nonnegative principal active tensions. It restricts the constitutive choices, while still permitting induced expansion, shear, and complex shape changes through mechanics and boundary constraints. It is not a six-to-one reduction of the control field: six tensor coordinates remain in every active cell.

The endpoint's absence of inversions is an observation. The stable Neo-Hookean energy has no infinite determinant barrier, and PSD activation does not impose `det(F) > 0`.

## 5. What the literature supports

There is no general theorem that active stress is smoother or more stable than active strain. Ambrosi and Pezzuto analyze constitutive conditions separately: a suitable fixed active-strain map can preserve rank-one convexity of the passive model, while an active-stress law must be checked for its own properties. Their distinction between activation and compatibility also explains why changing a local preferred configuration changes the mechanical problem. See [*Active stress vs. active strain in mechanobiology: constitutive issues*](https://staff.polito.it/davide.ambrosi/Papers/jelast.pdf), especially Sections 1, 4, and 5.

Giantesio, Musesti, and Riccobelli show that active stress and active strain generally predict different shear responses even when calibrated to agree in uniaxial deformation. Agreement on a contraction magnitude therefore does not make the two formulations mechanically equivalent. See [*A comparison between active strain and active stress in transversely isotropic hyperelastic materials*](https://arxiv.org/abs/1709.04977).

Our stiffness formula above is a direct derivation for the implemented quadratic PSD term. It should not be attributed to every model called “active stress.” Conversely, the unconstrained Raw6 inverse field is much less restricted than the commonly studied volume-preserving, fiber-directed active-strain maps.

Finite iteration count can also act as regularization in inverse problems; this is established for spectral methods such as Landweber iteration in [Blanchard, Hoffmann, and Reiß, *Optimal adaptation for early stopping in statistical inverse problems*](https://arxiv.org/abs/1606.07702). That theory does not directly establish a spectral smoothing law for this nonlinear projected-Adam run. Optimizer parameterization and stopping remain possible contributors, not a demonstrated explanation of this endpoint.

## 6. Why visual smoothness does not establish smoother activation

The current result has higher surface fit error and less total motion:

| Saved endpoint | Surface fit RMS | Surface motion RMS |
| --- | ---: | ---: |
| Historical Raw6, step 194 | 0.654339 mm | 4.974140 mm |
| PSD, step 1024 | 1.610246 mm | 4.125968 mm |

The current motion is about **17.1% smaller**. Some apparent regularity may therefore come from fitting less of the target deformation. The runs also have different iteration budgets, parameter units, and learning-rate histories; neither endpoint is an established inverse optimum.

Within the PSD trajectory itself, the activation variation diagnostic increased to **2.509 times** its step-512 value. From selected step 544 to step 1024, the full-face 5-mm normal high-pass RMS rose from **0.207779 mm** to **0.243397 mm**, although its ratio to total normal displacement decreased. These observations are recorded in [the learning-rate report](108-learning-rate-report.md), lines 84–107. They directly contradict interpreting the result as evidence that the control field became smoother during continued fitting.

The defensible conclusion is that **this constrained constitutive model produced substantially less distorted visible geometry without an explicit spatial penalty**. The strongest supported mechanism is that it retains passive resistance to actual deformation and removes some of the Raw6 model's local preferred-shape freedom. Its PSD tangent supplies an additional plausible resistance to small-scale deformation. The causal shares of these effects and the remaining fit/motion difference have not been measured.

## 7. Small controlled tests that would separate the explanations

Start with a simple muscle block and passive covering layer, retaining independent per-tetrahedron controls. Keep the target, rest mesh, passive constants, boundary constraints, objective, and stopping checks fixed. Compare fit-versus-roughness and motion-versus-roughness curves rather than only equal iteration numbers.

1. Compare the current quadratic stress with and without the PSD restriction, retaining the same coordinates and numerical scaling. This isolates the effect of the lower spectral constraint. An unconstrained result may fail or distort; preserve that outcome as evidence.
2. Compare unrestricted Raw6 against a positive-definite, volume-preserving active-strain field. This tests whether unrestricted changes of preferred volume explain much of the baseline distortion.
3. At fixed saved activation, inspect forward responses to spatial perturbations at several wavelengths and the associated tangent quadratic forms. This separates added deformation stiffness from the optimizer's ability to change activation to counteract it.

None of these counterfactual solves was run in this investigation.

## Reproducing the saved-state determinant calculation

Run the following from the repository root using the existing `.venv/bin/python`. It only reads saved data and prints measurements.

```python
from pathlib import Path
import numpy as np
import pyvista as pv

b = Path("exp/2026/09/07")
g = b / "tensor-active-stress"
ref = pv.read(g / "data/21-psd/step-0000.vtu")
tets = np.asarray(ref.cells).reshape(-1, 5)[:, 1:]
with np.load(b / "face-actuation-diagnosis/data/11-historical-no-skin/final.npz") as z:
    x, old_u, ids, A = z["rest_points"], z["u"], z["active_ids"], z["Ainv"]
with np.load(g / "data/102-fit1024/final.npz") as z:
    new_u = z["u"]
    assert np.array_equal(ids, z["active_ids"])
t = tets[ids]
dm = (x[t[:, 1:]] - x[t[:, :1]]).transpose(0, 2, 1)
inv = np.linalg.inv(dm)
w = np.linalg.det(dm) / 6 * np.asarray(ref.cell_data["MuscleFraction"])[ids]
w /= w.sum()

def deformation_gradient(u):
    y = x + u
    return (y[t[:, 1:]] - y[t[:, :1]]).transpose(0, 2, 1) @ inv

F = deformation_gradient(old_u)
G = F @ A
H = deformation_gradient(new_u)
j, a, gj, hj = (np.linalg.det(v) for v in (F, A, G, H))
assert np.allclose(gj, j * a, rtol=1e-10, atol=1e-10)
for label, value in (("old JF", j), ("old JG", gj), ("new JF", hj)):
    print(label, "RMS from one", np.sqrt(w @ ((value - 1) ** 2)),
          "negative cells", np.count_nonzero(value < 0))
print("inverted F but positive G", np.count_nonzero((j < 0) & (gj > 0)))
print("log-absolute-determinant correlation",
      np.corrcoef(np.log(np.abs(j)), np.log(np.abs(a)))[0, 1])
for label, value in (("old F", F), ("old G", G), ("new F", H)):
    deviation = np.linalg.norm(np.linalg.svd(value, compute_uv=False) - 1, axis=1)
    print(label, "stretch deviation RMS", np.sqrt(w @ (deviation ** 2)))
```
