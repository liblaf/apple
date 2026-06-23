# Human Face Smile Inverse Activation

## Purpose

This experiment compares Smile inverse activation on the InFaceConvex human-face
tet mesh with two setups:

- `skin-pre0pct`: tetra mesh plus Koiter skin surface, no skin prestrain.
- `no-skin`: tetra mesh only.

The inverse variable is per-muscle-tet `ActivationInv` with 6 DoF per active
tet and no range clamping. The loss is point-to-point L2 on finite `Smile`
displacement over the `IsFace` region.

## Script Structure

The original all-in-one inverse script was split into focused modules:

- `src/_human_face_config.py`: constants, Cherries configs, case selection.
- `src/_human_face_mesh.py`: mesh extraction, orientation, required fields.
- `src/_human_face_forward.py`: material setup, skin surface, forward/adjoint solvers.
- `src/_human_face_runtime.py`: case paths, differentiable forward wrapper, runtime state.
- `src/_human_face_loop.py`: inverse loop, step metrics, VTKHDF frame writing.
- `src/_human_face_case.py`: Smile target setup, summaries, result files.
- `src/_human_face_output.py`: result meshes, bumpiness metrics, Markdown tables.
- `src/_human_face_targets.py`: Smile target mask.
- `src/40-combine-summaries.py`: reproducible combined summary/table generation.

The public entrypoints are thin: `10-prepare-human-face.py`,
`20-inverse-human-face.py`, `30-plot-loss-curves.py`, and
`40-combine-summaries.py`.

## Commands

Run directory:

```bash
cd /home/liblaf/github/liblaf/apple/exp/2026/06/17/human-face-skin-prestrain
```

Final inverse runs:

```bash
DEBUG=1 CHERRIES_NAME="Human face InFaceConvex Smile skin vs no skin zero activation d5e8" CHERRIES_TAGS="human-face,smile,in-face-convex,skin,no-skin,zero-activation,d5e-8,2026-06-17" uv run python src/20-inverse-human-face.py --target smile --case-set required --inverse-lr 0.03 --inverse-max-steps 80 --inverse-loss-min-delta 5e-8
DEBUG=1 CHERRIES_NAME="Human face InFaceConvex Smile no skin zero activation d5e8 max100" CHERRIES_TAGS="human-face,smile,in-face-convex,no-skin,zero-activation,d5e-8,max100,2026-06-17" uv run python src/20-inverse-human-face.py --target smile --case-set no-skin --inverse-lr 0.03 --inverse-max-steps 100 --inverse-loss-min-delta 5e-8 --output-summary 20-inverse-no-skin-summary.json --output-table 20-inverse-no-skin-table.md
DEBUG=1 CHERRIES_NAME="Human face InFaceConvex Smile combined summaries" CHERRIES_TAGS="human-face,smile,summary,in-face-convex,skin,no-skin,2026-06-17" uv run python src/40-combine-summaries.py
DEBUG=1 CHERRIES_NAME="Human face InFaceConvex Smile loss plots" CHERRIES_TAGS="human-face,smile,plots,in-face-convex,skin,no-skin,2026-06-17" uv run python src/30-plot-loss-curves.py
```

`DEBUG=1` was used to keep these long runs local. The Cherries Local plugin
emitted known run-log copy warnings after outputs had already been written.

## Model And Solver

- Mesh source: `/home/liblaf/github/liblaf/melon/exp/2026/05/27/head/data/62-tetmesh-3191k.vtu`.
- Simulation subset: `InFaceConvex`.
- Prepared mesh: 228660 points, 1146517 tets, 288235 active muscle tets.
- Optimized variables: 1729410 activation DoFs.
- Loss points: 15302 finite Smile points in `IsFace`.
- Tetra fractions: aponeurosis `E=0.10 MPa, nu=0.35`; fat `E=0.003 MPa, nu=0.49`; muscle `E=0.030 MPa, nu=0.49`.
- Skin: `E=0.20 MPa, nu=0.49, thickness=0.001`, no prestrain in this comparison.
- Forward: PNCG, `rtol=5e-4`, `atol=1e-10`, `max_steps=5000`.
- Adjoint: CG then MinRes fallback, `rtol=5e-4`, `atol=0`.
- Inverse: Adam, `lr=0.03`, `inverse_loss_min_delta=5e-8`, patience 20.

Forward and adjoint success were logged for every inverse step. The final
summaries include forward step count, forward relative gradient norm, adjoint
solver choice, and adjoint residuals.

## Outputs

- Combined summary: `data/20-inverse-summary.json`
- Comparison table: `data/20-inverse-table.md`
- Skin result: `data/20-human-face-smile-skin-pre0pct-lr03.vtu`
- Skin history: `data/20-human-face-smile-skin-pre0pct-lr03-steps.vtkhdf`
- No-skin result: `data/20-human-face-smile-no-skin-lr03.vtu`
- No-skin history: `data/20-human-face-smile-no-skin-lr03-steps.vtkhdf`
- Loss plots:
  - `figs/30-loss-curves/20-human-face-smile-skin-pre0pct-lr03-log-loss.png`
  - `figs/30-loss-curves/20-human-face-smile-no-skin-lr03-log-loss.png`
  - `figs/30-loss-curves/loss-comparison-log-y.png`

VTKHDF frame checks:

- Skin: 21 frames, steps 0 to 20, final loss `8.6407552099803e-06`.
- No-skin: 85 frames, steps 0 to 84, final loss `3.1477635276403862e-06`.

## Results

| setup | stop | evals | best step | best loss | error RMS | disp edge RMS | disp lap RMS | residual edge RMS | residual lap RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| skin-pre0pct | loss_plateau_20_steps | 21 | 20 | 8.64076e-06 | 0.00509139 | 3.64082e-05 | 1.89731e-05 | 0.000367773 | 8.68798e-05 |
| no-skin | loss_plateau_20_steps | 85 | 84 | 3.14776e-06 | 0.00307299 | 0.000287485 | 0.000116964 | 0.000350093 | 0.000137095 |

Both final cases converged by the configured 20-step plateau criterion and had
successful final forward and adjoint solves. The final forward step counts were
1172 for skin and 1419 for no-skin; final adjoint relative residuals were
`4.866e-4` and `4.856e-4`, respectively.

## Analysis

No-skin fits the Smile target better: final loss is about 2.7x lower and error
RMS is about 40 percent lower than the skin case.

Skin is much smoother in displacement. Compared with skin, no-skin has about
7.9x larger displacement edge RMS and about 6.2x larger displacement Laplacian
RMS. Residual edge RMS is slightly lower without skin, but residual Laplacian
RMS is higher, so the no-skin solution buys target fit with visibly bumpier
displacements.

The skin case converged quickly because loss decreases were already below the
chosen minimum-progress threshold. The no-skin case continued improving and
needed 85 evaluations before the 20-step plateau condition was satisfied.

## Limitations

- This report covers only the human-face Smile target.
- Skin prestrain 5 percent and 10 percent variants are implemented in case
  selection but were not part of this reduced comparison.
- The Cherries Local plugin emitted run-log copy warnings in debug mode; output
  files were verified directly with JSON and HDF5 reads.
