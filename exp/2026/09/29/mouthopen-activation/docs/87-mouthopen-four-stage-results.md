# MouthOpen activation parameterization chain

The four activation parameterizations used `η = 7.2e-06` (10× the earlier `7.2e-7` setting) and `β = 1`. They received 200 optimizer attempts each; this fixed budget does not establish optimizer convergence. The full tensor field drove each saved geometry. Glyph amplitude is the singular-value amplitude `a = σmax(B) - 1`; the figure shows its largest positive principal mode.

![Four-stage MouthOpen comparison](../data/85-mouthopen-four-stage-002/mouthopen-four-stage-preview.png)

[Full-resolution figure](../data/85-mouthopen-four-stage-002/mouthopen-four-stage.png) · [render manifest](../data/85-mouthopen-four-stage-002/manifest.json) · [independent CPU analysis](../data/80-four-stage-analysis-003/analysis.json)

| Parameterization | DoF/cell | Attempts | Updates | Skips | Position RMS (mm) | Normal RMS (°) | Amplitude roughness | Direction roughness | Full-S smoothness/L2 gradient ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Symmetric, unrestricted | 6 | 200 | 199 | 1 | 1.251 | 2.528 | 22.444182 | 28.094885 | 0.27488 |
| PSD, contraction only | 6 | 200 | 191 | 9 | 0.880 | 1.921 | 29.960041 | 31.745061 | 0.33852 |
| Rank one, fixed axis | 1 | 200 | 200 | 0 | 0.816 | 1.914 | 38.422825 | 63.357794 | 0.51338 |
| Rank one, learned axis | 3 | 200 | 200 | 0 | 0.661 | 1.633 | 30.504724 | 45.173894 | 0.49811 |

The spectral roughness terms sum to the direct weighted Frobenius roughness; the CPU audit checked that identity for each endpoint. For unrestricted symmetric activation, amplitude roughness uses signed eigenvalues. For PSD and rank-one stages it uses nonnegative eigenvalues; rank-one directional roughness uses the squared-axis alignment identity.

The gradient ratio is `η ||∇S R|| / ||∇S L2||` in the symmetric full-tensor space, using dual effective-active-volume weighting; the normal-loss term is excluded. It is an endpoint diagnostic. Its independent forward solve may slightly change displacement, as reported per stage in `report.json`.

## Solver and orientation diagnostics

| Stage | Primal free-force residual range | Relative adjoint residual range | Inverted cells | Minimum det(F) | Inverted rest-volume fraction | Boundary self-intersection audit |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| symmetric6 | 3.16e-13–9.49e-11 | 8.68e-08–1e-07 | 222 | -16.90066 | 0.000050 | Intersections detected |
| psd6 | 1.6e-13–1e-10 | 9.23e-08–1e-07 | 327 | -16.88932 | 0.000079 | Intersections detected |
| rankone_fixed | 1.06e-13–9.88e-11 | 9.36e-08–1e-07 | 367 | -16.90616 | 0.000092 | Intersections detected |
| rankone_learned | 1.02e-13–9.95e-11 | 9.24e-08–1e-07 | 476 | -16.89245 | 0.000119 | Intersections detected |

The saved primal and adjoint receipts passed the numerical solver gates. Each accepted endpoint also has a self-intersection audit of the complete extracted FEM boundary; that audit does not test separate bone obstacles, containment, or contact forces. Inversions and boundary intersections remain geometric diagnostics, so the contact-off chain does not validate mechanical realism.

The target is a transferred MouthOpen blendshape, with jaw motion prescribed from the chin-derived rigid pose. The chain retained the `IsFixed` constraints, pruned fixture, and recorded material model. Stage 1 starts at zero activation. Stage 2 PSD-projects stage 1; stage 3 keeps the largest nonnegative eigenmode of stage 2; stage 4 carries stage 3's rank-one tensor into learned-axis controls. Each stage carries its predecessor's displacement and starts fresh Adam. In the learned-axis stage, zero-amplitude axes are released using the saved total tensor gradient.

## Run records

Chain protocol: `188f3c8eac2476e2606279897917bd70c654288bea9c831529d6f4ddee21b838` · chain status: `fb2349dddcf96fe60f868b50c3d0186557ebe49e94d157c9beb0c9fc42d46fb6` · source manifest: `b6b735991de27119410dd80963f1c84ba672e790828e15aa12cb61e9c15120c3`.

Declared input receipts:

- `summary.json`: `34a028a7752c6e5000c257180404048b283970f178f5c6ff344eca4ecc886dfc`
- `final.npz`: `0d39dc53676daf8dcc092376282fd7ce97bddab83159ff0609cbc17d3803c09c`
- `prepared.npz`: `dfb638064460dc46c7e00121737a91ed296440ec85dbaeb804b2cd8cc95b15eb`
- `volume.vtu`: `f25db0dd4b7b6174df4c343bad78fcf3f922e820b3bc997f90eb8716c7dd61af`
- `skin.vtp`: `59fc7819540aaa362b68cad1864784f85c60e306390ef984b1a27241b0e8ce5f`

Completed Cherries commands and URLs are included only when present in the corresponding logs:

| Run | Command | Comet URL | Start–end | Log SHA-256 |
| --- | --- | --- | --- | --- |
| 70 numerical chain | `.venv/bin/python -u src/70-run-four-stage-mouthopen.py` | [Comet](https://www.comet.com/liblaf/apple/8f7e2191fa4f438db595947d8a001a59) | `2026-09-29 10:55:07.348715+08:00 – 2026-09-29 14:34:10.358473+08:00` | `b357226400d6a91f67dc311e94b711a05ab998ffddd8d53ae13dbeb5fb585a64` |
| 80 CPU analysis | `.venv/bin/python -u src/80-analyze-four-stage-mouthopen.py --output 80-four-stage-analysis-003 --chain data/70-mouthopen-four-stage` | [Comet](https://www.comet.com/liblaf/apple/efb9572a400f44c5b7cf9d039220ca23) | `2026-09-29 14:38:30.174354+08:00 – 2026-09-29 14:38:40.821722+08:00` | `19655d3f69dbc879a2546708f466aa1d7f644ce113f245d52a1437b08cc4e5ec` |
| 85 renderer | `.venv/bin/python -u src/85-render-four-stage-mouthopen.py --output 85-mouthopen-four-stage-002 --source 70-mouthopen-four-stage` | [Comet](https://www.comet.com/liblaf/apple/053ac34891aa4cc681302256d386f207) | `2026-09-29 14:40:15.374797+08:00 – 2026-09-29 14:40:33.102121+08:00` | `00059aca8ae8ce203e29f46eb8821d244c27b8de28142a11407e97bc8d60fdad` |

Stage summaries, endpoint checkpoints, source manifests, solver receipts, gradient components, and figure/source hashes are retained in the linked output directories and `report.json`.
