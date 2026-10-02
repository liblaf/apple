# Smile inverse-fit solver comparison

`src/21-compare-smile.py` is a post-processing step for the matched Smile
inverse-fit arms written by `src/20-fit-smile.py`. It accepts only a successful
`smile-solver-comparison-v1` receipt with `original` and `hybrid_diag` arms.

It renders the Smile target, original PNCG endpoint, and hybrid endpoint using
one front orthographic camera and one motion scale. The context meshes are the
fixed source cranium and mandible plus the registered fixed eyeballs used as
collision obstacles. A separate two-panel residual figure shares one error
scale, and a convergence figure plots objective and fit RMS against accepted
iteration and accumulated fit-step time.

It also compares the Frobenius norm of the fitted additive active Cauchy stress
on active tetrahedra. Each tet uses the current six-coordinate symmetric stress
parameterization; this is not an active-strain visualization.

Run from this experiment group after both arms are complete:

```bash
CHERRIES_NAME="Smile solver comparison visuals" CHERRIES_TAGS="smile,inverse-physics,solver,visualization" \
python src/21-compare-smile.py --fit-dir data/smile-fit-001 --output-dir data/smile-fit-visuals-001
```

No result is reported in this document until the completed fit receipt,
checkpoints, traces, timing records, and generated images have been inspected.
