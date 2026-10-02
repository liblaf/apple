# Simple heterogeneous-skin forward

## Purpose

This experiment attempted the requested direct forward solve with complete-source
soft-versus-bone IPC, canonical passive bulk materials, zero bulk baseline stress,
no activation, a fixed jaw, and the prescribed heterogeneous skin field. The skin
field contains per-triangle Young modulus, Poisson ratio, thickness, and isotropic
signed stress resultant derived from Flynn et al. Table 3. The transfer is a smooth
manual-anchor interpolation and is not a measured registered material map for this
subject.

The solver is strict PNCG only. It has no Newton step, adjoint, inverse solve, or
fallback. Both runs used `rtol=1e-5`, `atol=1e-11`, 5,000 maximum accepted steps,
40 Armijo backtracks, and a 30-minute callback-checked wall budget. The second run
also used a conservative tetrahedral no-sign-change step bound.

## Inputs

- Rebased mechanics inputs: `data/simple-skin-forward-inputs-001/prepared/`
- Complete source geometry and zero-displacement admission:
  `data/simple-skin-forward-inputs-001/{geometry.npz,geometry-audit.json,admission.json}`
- Skin field: `data/simple-skin-forward-inputs-001/skin-field.npz`, SHA-256
  `2f57719152d7ab48989fc3891eb7f63d170fcd643f83e80346acda381ed220cb`
- Corrected field manifest:
  `data/simple-skin-forward-inputs-001/skin-field-manifest.json`, SHA-256
  `3af1785f1a1988d87400b1add60950f5274689834070ebb7b9d8a397888aa38f`

The field spans 29,899 skin triangles in exact global-FEM triangle order. Its
Young modulus range is 0.127382–0.257861 MPa and its isotropic resultant range is
30.4002–80.8297 N/m at 1 mm thickness. The driver converts N/m to MPa·m with a
factor of `1e-6` and uses the corrected polynomial-SNH Lamé convention
`lambda_code = lambda_classical + mu`.

## Run 001: unguarded volume

Comet: <https://www.comet.com/liblaf/apple/a91150c02e00406788b39711abba66c5>

The run persisted step 0, step 1, and every 50 accepted steps. The energy decreased
on every persisted accepted row, but this did not preserve tetrahedron orientation.
Checkpoint inspection gave:

| Step | min J | max J | Inverted tets |
| ---: | ---: | ---: | ---: |
| 0 | 1.000000 | 1.000000 | 0 |
| 1 | 0.896392 | 1.108460 | 0 |
| 50 | -1.425107 | 3.247231 | 13 |
| 100 | -2.029693 | 2.887530 | 30 |
| 200 | -1.139701 | 3.660943 | 59 |
| 350 | -3.715015 | 4.135335 | 80 |

After accepted step 387, the next full-source TightInclusion CCD call did not
return to Python. The 60-second faulthandler traceback located the main thread in
`collision.max_step_size`. SIGINT and SIGTERM could not unwind that native call;
the owned process was then killed. The last persisted displacement is checkpoint
350; it is inverted and is not admitted for restart. Unsaved steps 351–387 are trace evidence only. The external interruption
receipt is `data/simple-skin-forward-001/external-interruption.json`, and the
checkpoint audit is `data/simple-skin-forward-001/checkpoint-volume-audit.json`.

## Run 002: positive-determinant path guard

Comet: <https://www.comet.com/liblaf/apple/1bd7343193554c31a22f317ef3b40dac>

For each proposed PNCG step, the driver formed the current deformation gradient
`F` and increment `G` for every tetrahedron, then imposed
`t ||F^-1 G||_F <= 0.8` before running IPC CCD. This sufficient bound keeps every
linear trial path nonsingular; an analytic crossing-tetrahedron test returned the
expected fraction 0.4 and stayed positive throughout 101 sampled path points.

The bound prevented inversion but did not preserve useful element quality. The
minimum determinant decreased to `4.16090e-6` by accepted step 47. The next Armijo
search exhausted all 40 backtracks at alpha `3.34177e-19`. The accepted terminal
state had:

- free-force norm 6.79026 N, versus threshold 0.000082474 N;
- 0 inverted tetrahedra, but `J` range `[4.16090e-6, 2.17744]`;
- valid IPC state, 6,042 stored active pairs, and minimum active gap `0.0511600 µm`;
- tissue gradient norm 2.87519 N and contact gradient norm 6.15694 N.

The full receipt is `data/simple-skin-forward-002/summary.json`. The run failed its
force criterion and is not an equilibrium result.

## Limiting tetrahedron

The limiting cell is FEM cell 658,378 (source cell 1,418,748), with vertices
91,242, 91,571, 92,843, and 93,223. It is a pure-fat cell with the prescribed
passive fat parameters `E=0.0112 MPa`, `nu=0.46`, and zero baseline stress.
Reference diagnostics are:

- volume `5.60572e-15 m³`;
- edge lengths 31.1–71.5 µm;
- reference edge-matrix condition number 12.18;
- scaled Jacobian 0.2264, aspect ratio 3.525, radius ratio 4.820.

All four vertices are unfixed, fall inside the `IsTeeth` proximity mask, and are
included in the soft collision vertex set. `IsTeeth` means distance within 2 mm
of teeth; it is not a tooth material label. At the failed terminal state, three
vertices had direct
contact-gradient norms of 0.042–0.059 N, while their tissue-gradient norms were
only 0.05–0.10 mN. The model gradient unit is MPa·m², so conversion to newtons
multiplies the stored values by `1e6`. Globally, the contact and tissue gradient
norms were 6.157 N and 2.875 N, respectively.

The cell lies 0.010267 mm from the source mandible. It was part of the local
candidate-001-to-002 repair footprint: its determinant was 0.1781 before repair
and 0.3675 in the repaired geometry before that geometry became the new reference.
All four nodes moved during repair, by at most 0.011981 mm. The rebased run still
starts at exactly `J=1`; these values describe mesh history and proximity, not an
initial strain in the new forward model.

This evidence localizes a strong direct contact contribution to the collapse of a
fat element near the teeth. It does not by itself establish that the anatomy label
or collision membership is wrong. A separate hash-bound IPC query found no
soft-versus-source-bone intersection at rest or at terminal step 47; see
`data/simple-skin-forward-002/terminal-soft-bone-intersection-audit.json`. The
replayable force and quality receipt is
`data/simple-skin-forward-002/limiting-tet-diagnostic.json`.

## Outcome

Neither run produced a converged forward solution. Run 001 showed that energy
decrease and soft-bone CCD do not prevent volume inversion. Run 002 showed that a
sign-only volume guard can instead approach a nearly flat element and terminate
with a nonzero force. No determinant box, mesh edit, material change, load
continuation, or alternative solver has been introduced. A further run should
follow an evidence-based correction of the bone-adjacent fat mesh or contact neighborhood.
