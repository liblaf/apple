# Raw6 step-5 intermediate audit

## Decision

Raw6 step 5 has a different compression mechanism from FiberRegion B. Its
minimum `det(F)` is 0.445741 in cell 548491, an unfixed, pure-muscle Nasalis
transverse element with a nonidentity per-cell Raw6 active map. All five cells
below 0.5 carry selected activation; four are muscle-dominant and one is a
fat-dominant mixed Zygomaticus major cell. Three touch historical fixed points,
but the worst and third-worst cells do not.

The broader tail points in the same direction. Of 13,126 cells below
`det(F)=0.8`, 12,570 (95.76%) carry selected activation, 9,891 are
muscle-dominant, and only 1,792 (13.65%) touch any fixed vertex. This is no
longer primarily passive fat collapsing against the fixed mandible. It is a
distributed active-field deformation, with additional fixed-interface cases
inside the tail.

The better fit is already accompanied by implausible Raw6 activation. The
active maps remain positive definite, but they are unbounded and strongly
nonisochoric: `det(A_inv)` spans 0.3633-2.1362, has muscle-weighted median
0.8780, and lies outside 0.9-1.1 over 86.16% of active muscle-fraction volume.
This checkpoint is an intermediate equilibrium at optimization step 5, not a
final result or a safe trajectory endpoint.

## Immutable checkpoint

The audited file is
[`step-0005.vtu`](../data/23-raw6-fat049/step-0005.vtu), SHA-256
`3e139d973808de13ea2a7d3962cf362115ae754eb1dba420a010163671b1dde5`, size
68,590,014 bytes. The audit did not read `latest.npz` or rerun the solver.

At this checkpoint the surface fit is 3.776603 mm RMS against the 5.095908 mm
RMS target. Motion is 1.891555 mm RMS and target projection is 0.294273. The
outer KKT value saved in `trace.csv` is 3.03315 against the requested 0.001, so
the state is far from an optimization stopping condition.

The full CPU audit is
[`data/43-raw6-step5-audit`](../data/43-raw6-step5-audit/summary.json). It
recomputes every deformation gradient, checks the saved determinants, runs IPC
on the complete boundary and visible skin, and produces the same cotangent
high-pass maps used for the earlier endpoints. The additional
[`raw6-active-tissue.json`](../data/43-raw6-step5-audit/raw6-active-tissue.json)
separates selected-cell active maps, deformation `F`, and elastic map
`G = F A_inv`.

## Lowest-determinant cells

| rank | cell | `det(F)` | `det(A_inv)` | `det(G)` | dominant tissue | region | fixed vertices |
| ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 1 | 548491 | 0.445741 | 1.961610 | 0.874370 | 100% muscle | Nasalis transverse | 0 |
| 2 | 562596 | 0.474383 | 2.117817 | 1.004656 | 60.94% fat, 38.96% muscle | left Zygomaticus major | 2 |
| 3 | 658114 | 0.481396 | 1.753119 | 0.843944 | 100% muscle | Depressor septi | 0 |
| 4 | 611222 | 0.493961 | 2.070417 | 1.022704 | 100% muscle | left Levator labii superioris | 2 |
| 5 | 620612 | 0.496228 | 1.158123 | 0.574693 | 100% muscle | Depressor septi | 2 |

The minimum cell's deformation principal stretches are
`[0.3462, 1.0495, 1.2268]`. Its active-map eigenvalues are
`[1.0709, 1.2444, 1.4719]`, and its elastic principal stretches are
`[0.4633, 1.2223, 1.5440]`. The contraction is therefore not explained away by
the active rest map; the muscle energy still sees substantial elastic
compression and extension.

The previous FiberRegion minimum, fixed fat-dominant cell 662949, now has
`det(F)=0.814760`. Its Raw6 map is nonidentity with `det(A_inv)=1.576246`.
Passive fixed-socket cell 691951 has `det(F)=1.010709`. A different passive,
fully fat cell appears at `det(F)=0.522116`, but only after twelve active cells
in the ordered tail. These checks rule out recurrence of the earlier single
fixed-socket minimum as the leading step-5 mechanism.

The low tail is already broad:

| threshold | cells | active | muscle / fat / aponeurosis dominant | fixed-incident |
| --- | ---: | ---: | ---: | ---: |
| `det(F)<0.5` | 5 | 5 (100%) | 4 / 1 / 0 | 3 (60.0%) |
| `det(F)<0.75` | 7,206 | 7,027 (97.52%) | 5,990 / 1,081 / 135 | 956 (13.27%) |
| `det(F)<0.8` | 13,126 | 12,570 (95.76%) | 9,891 / 2,882 / 353 | 1,792 (13.65%) |
| `det(F)<0.9` | 35,999 | 31,308 (86.97%) | 19,349 / 15,048 / 1,602 | 5,475 (15.21%) |

No tetrahedron is inverted, but five steps have produced thousands of strongly
compressed cells rather than one isolated outlier.

## Raw6 activation and tissue stretch

Raw6 optimizes six independent symmetric components per selected tetrahedron.
It applies no coordinate bound, matrix exponential, magnitude penalty, or
spatial smoothness term. Positive definiteness is checked only after a trial is
solved. For comparison, the bounded scalar models at their 35% shortening cap
have transverse eigenvalue `sqrt(0.65)=0.8062`, axial eigenvalue
`1/0.65=1.5385`, and unit determinant.

The following quantiles use rest volume times muscle fraction over the 120,020
selected cells:

| field | minimum | q1% | q5% | median | q95% | q99% | maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| minimum eigenvalue of `A_inv` | 0.3900 | 0.4642 | 0.5079 | 0.7068 | 1.0477 | 1.1359 | 1.2704 |
| maximum eigenvalue of `A_inv` | 0.7880 | 0.9087 | 0.9920 | 1.2671 | 1.4700 | 1.5081 | 1.6280 |
| `det(A_inv)` | 0.3633 | 0.4515 | 0.4915 | 0.8780 | 1.7158 | 1.8829 | 2.1362 |
| minimum elastic stretch of `G` | 0.2930 | 0.5203 | 0.5752 | 0.7678 | 0.9659 | 1.0267 | 1.2352 |
| maximum elastic stretch of `G` | 0.7304 | 0.8907 | 0.9686 | 1.1973 | 1.4286 | 1.5521 | 2.8998 |
| `det(G)` | 0.2054 | 0.4554 | 0.5407 | 0.9224 | 1.3475 | 1.5432 | 2.8570 |

By active muscle-fraction volume, 68.38% has a minimum `A_inv` eigenvalue below
the scalar model's 0.8062 transverse limit, and 36.79% falls below 0.65. In
addition, 6.27% has `det(A_inv)<0.5`; 0.21% exceeds determinant 2. The problem
is mostly multi-axial contraction and active volume change rather than excessive
axial contraction: only 0.138% exceeds the scalar axial reference of 1.5385.

The determinant extremes are not harmless coordinates. The lowest active
`det(G)` is 0.205424 in cell 474209, which is 99.02% fat and only 0.098% muscle
but still receives its own Raw6 control. Its `det(A_inv)=0.392463` is weakly
penalized because the active material fraction is tiny. This is a concrete
per-cell identifiability defect in the unregularized Raw6 reference.

Physical deformation is strongest in muscle-dominant cells, although fat also
has a long tail:

| dominant tissue | min `det(F)` | volume q1% `det(F)` | median | q99% | minimum principal stretch, q1% |
| --- | ---: | ---: | ---: | ---: | ---: |
| muscle | 0.4457 | 0.7067 | 0.9999 | 1.4713 | 0.6154 |
| fat | 0.4744 | 0.8678 | 0.9999 | 1.0818 | 0.6885 |
| aponeurosis | 0.6151 | 0.7939 | 0.9997 | 1.1071 | 0.8064 |

## Surface high-pass map

The map shows localized lower-lip and lip-corner extrema plus bilateral
fine-scale variation below the eyes. The strongest 2 mm response is +2.96077
mm at a `LipBottom` vertex. At 5 mm the positive maximum is +3.13211 mm on
`LipBottom`; at 10 mm the largest magnitude is -3.59865 mm on `LipTop`.

| scale | Raw6 face RMS | Raw6 mouth RMS | stable `nu=0.49` face RMS | stable mouth RMS |
| --- | ---: | ---: | ---: | ---: |
| 2 mm | 0.06181 | 0.12095 | 0.01242 | 0.03138 |
| 5 mm | 0.14672 | 0.28827 | 0.03324 | 0.08162 |
| 10 mm | 0.25880 | 0.51501 | 0.07473 | 0.18029 |

All values are area-weighted scalar normal-motion RMS in millimetres. The Raw6
2 mm face RMS and q99 are both about five times the fixed-control stable-fat
endpoint; its 2 mm absolute q99 is 0.2221 mm versus 0.0441 mm. Some of this
structure follows the supplied target because the 2 mm residual high-pass RMS
falls to 0.08766 mm from 0.10687 mm. The high-pass map alone cannot label the
motion an artifact. Together with discontinuous, unbounded per-cell active maps
and the broad active-cell determinant tail, it is evidence that Raw6 buys fit
with spatially rough and physically unconstrained actuation.

![Raw6 step-5 full-face high-pass maps](../data/43-raw6-step5-audit/23-raw6-fat049/face-highpass-maps.png)

![Raw6 step-5 mouth high-pass maps](../data/43-raw6-step5-audit/23-raw6-fat049/mouth-highpass-maps.png)

IPC finds no exact intersections at this checkpoint on either the complete
tetrahedral boundary or visible skin. This is a static endpoint check and does
not establish clearance or continuous safety between optimization steps.

## Reproduction

The bounded audit ran on CPU only and completed in 9.93 seconds:

```bash
DEBUG=1 uv run python src/40-audit-face-results.py \
  --result-dirs data/23-raw6-fat049 \
  --endpoint-name step-0005.vtu \
  --output-dir data/43-raw6-step5-audit
```

The result should be compared again only after the running Raw6 optimization
produces a declared final endpoint. Later checkpoints may move or worsen the
limiting cells; this report does not extrapolate beyond exact step 5.
