# One cold neutral forward with Newton-CG

The single forward solve reached its 100-iteration limit without converging and ended with 130 inverted tetrahedra. This endpoint is **not a usable neutral equilibrium**. It is saved for diagnosis; the selected neutral was not changed.

| Measurement | Result |
| --- | ---: |
| Accepted Newton steps | 100 |
| Forward-loop time | 30.103 s |
| Initial gradient norm | 5.44961865 × 10⁻⁵ |
| Final gradient norm | 7.48156954 × 10⁻⁷ |
| Effective gradient threshold | 5.44961865 × 10⁻⁸ |
| Final gradient / threshold | 13.729 |
| Initial energy | 2.61494665 × 10⁻⁸ |
| Final energy | −2.83304064 × 10⁻⁸ |
| Inverted tetrahedra | 130 |
| Minimum det(F) | −3.737767 |
| Soft–rigid intersections | 0 |

Gradient units are MPa·m² and energy units are MPa·m³; multiply by 10⁶ for N and J respectively. Forward-loop time excludes model construction, geometric preparation, diagonal validation and endpoint review. The stored energy decreases at every accepted step, but this does not establish an admissible equilibrium.

![Energy and gradient over accepted Newton iterations](../data/convergence-plots-002/energy-gradient.png)

## Starting state and model

The solve starts from the original constitutive reference with a geometric eye-contact repair. The uncorrected reference had 3,864 soft–eye triangle intersection pairs and 133 nodes inside the eyes. Eye-surface projection followed by graph-harmonic extension produced a seed with zero inversions, minimum det(F) 0.476865, and no soft–rigid intersections. It used no previous equilibrium and performed no forward solve. The observed skin displacement was exactly zero; the largest internal repair displacement was 0.8140 mm. See the [seed receipt](../data/reference-seed-002/summary.json).

The materials are fat E = 11.2 kPa, passive muscle E = 12 kPa, aponeurosis E = 1,693 kPa, and skin E = 127.382–257.861 kPa; all use ν = 0.49. Skin thickness is 1 mm and prescribed membrane tension is 30.400–80.830 N/m. Bulk activation and additive baseline stress are zero. The mandible's one rotation coordinate is prescribed to zero in this neutral forward. Contact includes the complete cranium, mandible and both fixed source eyeballs.

The bulk energy is the requested polynomial Stable Neo-Hookean law with the symmetric active term ½ Q:(FᵀF − I). Its λ coefficient is λ_classical + μ to preserve the stated small-strain E and ν. Skin uses the exact plane-stress reduction. Historical material and geometry inputs were hash verified; the separate [runtime binding](../data/forward-002/model-binding.json) records solver import migrations, generated version metadata and dependency-resolution changes without modifying the historical manifest.

## Solver and verification

The forward uses Newton-CG only, atol = 10⁻⁸, rtol = 10⁻³, and at most 100 accepted steps. The stopping condition is ‖g‖₂ ≤ max(atol, rtol × ‖g₀‖₂). Neither the absolute nor relative tolerance was met.

The coordinate displacement cap is half the mean unique FEM rest-edge length: 0.895845 mm. Armijo uses coefficient 10⁻⁴, backtracking factor 0.5, and eight attempts. Each Newton iteration starts with zero shift, then mean(abs(diag(H))), increasing by 10 with at most eight attempts. PCG uses abs(diag(H + shift·I)), relative tolerance 10⁻³, and at most 1,000 iterations. CCD retains the complete obstacle geometry, a 10 nm minimum gap and a 0.9 safety factor. Adam's requested learning rate of 1.0 is recorded; no inverse update was performed.

The run installs exact bulk and IPC Hessian diagonals locally so that Jacobi and the shift scale use the same physical Hessian as the Hessian-vector products. Eight assembled diagonal entries, including the minimum and maximum, agree with coordinate Hessian-vector products to a maximum absolute error of 1.09 × 10⁻¹⁹. All 100 accepted steps used a positive shift after rejecting the unshifted system. The saved PCG residuals and shift sequences obey the requested policy.

The [saved-run audit](../data/forward-002/audit.json) verified 332 archived source-file hashes, input/output bindings, all 101 trace samples and the failure classification. Independent CPU review reproduced 130 inversions and found zero triangle intersection pairs against each rigid obstacle. Bone/eye collision feasibility does not certify positive tetrahedral orientation.

There was exactly **one actual forward solve**, in `forward-002`. The earlier `forward-001` launch stopped during setup with zero Newton steps: exact equality rejected the zero-jaw transform's 1.73 × 10⁻¹⁸ m floating-point roundoff. The next launch uses a scale-derived float64 bound and audits the projected state. The [setup-failure receipt](../data/forward-001/setup-failure.json) records that distinction.

## Artifacts and reproduction

The [front comparison](../data/forward-002/review/neutral-comparison-front.png), [side comparison](../data/forward-002/review/neutral-comparison-side.png), [volume mesh](../data/forward-002/review/neutral-volume.vtu) and [skin mesh](../data/forward-002/review/neutral-skin.vtp) show the failed endpoint for inspection. The [review receipt](../data/forward-002/review/summary.json), [forward summary](../data/forward-002/summary.json), [protocol](../data/forward-002/protocol.json) and [full trace](../data/forward-002/trace.jsonl) retain the numerical evidence.

The recorded forward command was `python src/10-run-neutral.py --output-dir data/forward-002`. To reproduce in a new output directory from the experiment directory:

```bash
cd exp/2026/09/22/neutral-newton
CHERRIES_NAME="Cold reference neutral in one Newton-CG forward" \
CHERRIES_TAGS=neutral,newton,cold-forward \
uv run python src/10-run-neutral.py \
  --seed-dir data/reference-seed-002 --output-dir data/forward-reproduction
```

The plotted curves are also available as [SVG](../data/convergence-plots-002/energy-gradient.svg) and [PDF](../data/convergence-plots-002/energy-gradient.pdf). Their saved-data command was `python src/30-plot-convergence.py --output-dir data/convergence-plots-002`; use a new output directory for reproduction. Both panels include all 101 accepted states, including initialization.

This report and its figures consume saved states; no additional forward solve was run for them. Comet records: [geometric seed](https://www.comet.com/liblaf/apple/c03c411469eb4f22bbd546514d018a01), [forward](https://www.comet.com/liblaf/apple/00edd7837763440e8e0c26891915661e), [endpoint review](https://www.comet.com/liblaf/apple/35b2739853ce49c690f5e1cbedc3139f), [final curves](https://www.comet.com/liblaf/apple/e6950d6ffc204fa09b7771f658a8d48a).
