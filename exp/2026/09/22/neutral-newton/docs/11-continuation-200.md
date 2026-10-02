# Newton continuation to 200 accepted iterations

The saved 100-iteration endpoint was replayed exactly and continued to a total
of 200 accepted Newton-CG iterations. The continuation did **not** converge and
its endpoint remains diagnostic only: it has 62 inverted tetrahedra.

## Command and continuity contract

The continuation forward command, run from this experiment group, was:

```bash
CHERRIES_NAME='Neutral Newton continuation to 200 iterations' \
CHERRIES_TAGS='neutral,newton-cg,continuation,200-iterations,bone-contact,eye-contact' \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYVISTA_OFF_SCREEN=true \
.venv/bin/python src/10-run-neutral.py \
  --resume-dir data/forward-002 --max-steps 200 --output-dir data/forward-003
```

`forward-003` loads the hash-verified `forward-002/terminal.npz`. It replayed
the saved terminal gradient exactly (7.48156954 × 10⁻⁷ MPa·m²) and energy to
6.62 × 10⁻²⁴ MPa·m³. The original cold-start gradient norm
(5.44961865 × 10⁻⁵ MPa·m²) and effective threshold
(5.44961865 × 10⁻⁸ MPa·m²) were preserved. The first 101 trace records,
iterations 0–100, are byte-identical to the parent trace; only iterations
101–200 were appended.

The cold parent took 30.1030412 s and the added 100-step segment took
39.5514012 s, for 69.6544424 s cumulative forward-loop time. These durations
exclude model construction, the exact-diagonal validation, and saved-state
review/rendering.

The material, contact, mandible, activation, exact-diagonal, and Newton-CG
settings match the parent protocol. Every continuation step reset its shift to
zero before trying `mean(abs(diag(H)))`; its recorded PCG relative residuals
are at most 9.9983 × 10⁻⁴. The parent-source comparison confirms the numerical
implementation is unchanged except for resume handling in this driver.

## Result compared with the 100-iteration endpoint

| Measurement | Iteration 100 | Iteration 200 |
| --- | ---: | ---: |
| Gradient norm (MPa·m²) | 7.48156954 × 10⁻⁷ | 4.97125274 × 10⁻⁷ |
| Gradient / original threshold | 13.729× | 9.122× |
| Energy (MPa·m³) | −2.83304064 × 10⁻⁸ | −4.12474045 × 10⁻⁸ |
| Inverted tetrahedra | 130 | 62 |
| Minimum det(F) | −3.737767 | −1.511613 |
| Soft–rigid intersections | 0 | 0 |

The extra 100 iterations reduced the gradient by 1.505× and decreased the
saved energy at every accepted state, but neither tolerance was met. The
independent CPU review also found no triangle intersections with the cranium,
mandible, or eyes, and no soft-boundary vertices or nonrigid FEM nodes inside
either eye. Those contact checks do not make the endpoint admissible because
the tetrahedra remain inverted.

## Artifacts

The [forward receipt](../data/forward-003/summary.json),
[protocol](../data/forward-003/protocol.json), and [full trace](../data/forward-003/trace.jsonl)
retain the continuation evidence. The [saved-state CPU review](../data/forward-003/review/summary.json)
and [energy/gradient plot receipt](../data/convergence-plots-200/summary.json)
performed zero forward solves. The final curve is available as
![Energy and gradient through 200 accepted Newton iterations](../data/convergence-plots-200/energy-gradient.png)

The forward Comet record is [here](https://www.comet.com/liblaf/apple/d9ceef12e3e4452baee711977968da57).
The saved-state review and curve-render records are
[here](https://www.comet.com/liblaf/apple/a7d47ff1a77d41298edfd19ab11f5a6f)
and [here](https://www.comet.com/liblaf/apple/337d5b84ade54f69af2ed22a0b9c0445).
