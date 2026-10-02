# Spatial baseline derivatives with full-face bone contact

Run `spatial-face-gradient-validation-002` passed all 16 directional checks
using 33 full-face Newton-CG equilibria. Maximum relative error was **0.2494%**
for the total objective and **0.2498%** for deformation loss alone, below the
unchanged 2% threshold. The adjoint relative residual was `9.933e-8`.

The frozen basis uses 4 fat, 4 aponeurosis and 5 muscle anchors. Each has six
symmetric stress coordinates; skin baseline and stiffness bring the shared
count to 80. The seed was the stationary constant-basis 25% checkpoint, with
a declared `0.02` nonuniform offset and a `0.01` inward log-stiffness offset.
This tests an interior, genuinely nonconstant field rather than only constant
embedding. It is derivative evidence, not a converged neutral preparation.

Three individual anchors, three tissue contrast directions, skin baseline and
skin stiffness were tested at steps `0.003` and `0.001`. Each plus/minus solve
started from the same frozen converged base equilibrium. Deformation-loss
derivatives were checked separately so the strong smoothness term cannot hide
an incorrect physics derivative. The objective retains the existing neutral
surface/muscle terms, `0.001 * prior_total`, and
`0.5 * 100 * mean_t R_smooth,t`.

All checked forward states passed contact validation. The smallest active gap
was `0.0489742 mm`; the largest final forward force norm was `7.490e-13` in
solver units. These are checks of the declared FEM cranium and moving-mandible
contact model, not validation of anatomical source-bone registration.

Run from this experiment directory:

```bash
CHERRIES_NAME='Spatial baseline full-face contact derivatives v2' \
CHERRIES_TAGS=joint-inverse,spatial-baseline,full-face,contact,derivatives \
uv run --frozen python src/25-validate-spatial-face-gradients.py \
  --cpu-validation data/spatial-fields-validation-cpu-v8/summary.json \
  --output-dir data/spatial-face-gradient-validation-002
```

[Comet run](https://www.comet.com/liblaf/apple/fb89de6859f34f19b6ba1d79522db80d),
[receipt](../data/spatial-face-gradient-validation-002/summary.json),
[individual checks](../data/spatial-face-gradient-validation-002/checks.json),
and [log](../data/spatial-face-gradient-validation-002.log) preserve the result.
The run finished successfully, including Cherries/Comet shutdown. Exact input,
basis, parent checkpoint, CPU-validation and implementation hashes are recorded;
the relevant source hashes were checked again before publishing success.
The dirty working tree was archived and no Git commit was made.

The preceding run 001 failed before its first equilibrium because omitted
`device` arguments placed coefficients on CUDA and NumPy-derived basis buffers
on CPU. The class now resolves the default device consistently. CPU/GPU
validation v8 passed 19 checks, including the no-explicit-device reconstruction
regression. Run 001 and its failure note remain preserved; it supplies no
derivative or feasibility evidence.
