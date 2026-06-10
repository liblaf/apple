# Per-Tetra Volume and IsFace Area Change Diagnostics

## Question

Compute per-tetra volume change and per-`IsFace`-triangle area change for the
corrected `Expression000` post-inverse relaxation runs, and save the results as
ParaView-readable meshes.

## Setup

- Script:
  `exp/2026/06/10/surface-prestrain/src/30-volume-area-change-diagnostics.py`
- Run directory:
  `exp/2026/06/10/surface-prestrain`
- Run:
  `COMET_AUTO_LOG_GIT_PATCH=false COMET_AUTO_LOG_GIT_METADATA=false CHERRIES_NAME='per-tet volume and IsFace area change diagnostics' CHERRIES_TAGS='surface-prestrain,Expression000,volume-change,area-change,isface,515k,3152k,diagnostics' uv run python src/30-volume-area-change-diagnostics.py`
- Comet:
  <https://www.comet.com/liblaf/apple/5339901e08104124bbbff8fd2ac4e5c6>

The script processes the corrected `Expression000` result VTUs from:

- 1% 2D Stable Neo-Hookean surface prestrain
- 4% 2D Stable Neo-Hookean surface prestrain
- 10% metric-penalty surface prestrain

Each case includes three displacement fields when present:

- `TargetDisplacement`
- `PreviousInverseDisplacement`
- `PrestrainDisplacement`

## Outputs

Volume-change VTUs:

- `data/30-volume-change-snh1-515k-expression000-smas1.vtu`
- `data/30-volume-change-snh1-3152k-expression000-smas100.vtu`
- `data/30-volume-change-snh4-515k-expression000-smas1.vtu`
- `data/30-volume-change-snh4-3152k-expression000-smas100.vtu`
- `data/30-volume-change-metric10-515k-expression000-smas1.vtu`
- `data/30-volume-change-metric10-3152k-expression000-smas100.vtu`

IsFace area-change VTPs:

- `data/30-area-change-snh1-515k-expression000-smas1-isface.vtp`
- `data/30-area-change-snh1-3152k-expression000-smas100-isface.vtp`
- `data/30-area-change-snh4-515k-expression000-smas1-isface.vtp`
- `data/30-area-change-snh4-3152k-expression000-smas100-isface.vtp`
- `data/30-area-change-metric10-515k-expression000-smas1-isface.vtp`
- `data/30-area-change-metric10-3152k-expression000-smas100-isface.vtp`

Tables:

- `data/30-volume-area-change-diagnostics-summary.json`
- `data/30-volume-area-change-diagnostics-cases.csv`
- `data/30-volume-area-change-diagnostics-table.md`

## Arrays

Each volume VTU contains per-tetra arrays:

- `RestVolume`
- `TargetVolume`, `TargetVolumeRelChange`, `TargetSignedVolumeRatio`,
  `TargetInvertedTet`
- `PreviousInverseVolume`, `PreviousInverseVolumeRelChange`,
  `PreviousInverseSignedVolumeRatio`, `PreviousInverseInvertedTet`
- `PrestrainVolume`, `PrestrainVolumeRelChange`,
  `PrestrainSignedVolumeRatio`, `PrestrainInvertedTet`

Each IsFace VTP contains only triangles whose three original vertices are
inside `IsFace`, with per-triangle arrays:

- `RestArea`
- `TargetArea`, `TargetAreaRelChange`
- `PreviousInverseArea`, `PreviousInverseAreaRelChange`
- `PrestrainArea`, `PrestrainAreaRelChange`

## Results

Prestrain displacement summary:

| case | volume rel RMS | volume total rel | area rel RMS | area total rel | inverted tets |
| --- | ---: | ---: | ---: | ---: | ---: |
| snh1-515k-expression000-smas1 | 0.102783 | -0.00439100 | 0.0190175 | -0.0171915 | 27 |
| snh1-3152k-expression000-smas100 | 0.105349 | -0.00174070 | 0.0210653 | -0.00782910 | 80 |
| snh4-515k-expression000-smas1 | 0.103121 | -0.00636579 | 0.0672774 | -0.0650184 | 27 |
| snh4-3152k-expression000-smas100 | 0.106621 | -0.00340579 | 0.0502484 | -0.0437491 | 79 |
| metric10-515k-expression000-smas1 | 0.103649 | -0.00721689 | 0.0920243 | -0.0727772 | 27 |
| metric10-3152k-expression000-smas100 | 0.106224 | -0.00269853 | 0.0643233 | -0.0284744 | 84 |

Previous inverse baseline:

| mesh | volume rel RMS | volume total rel | area rel RMS | area total rel | inverted tets |
| --- | ---: | ---: | ---: | ---: | ---: |
| 515k-expression000-smas1 | 0.101750 | -0.00346684 | 0.145620 | 0.000866861 | 27 |
| 3152k-expression000-smas100 | 0.103203 | -0.000705460 | 0.143108 | 0.00341153 | 54 |

Target displacement on `IsFace`:

| mesh | IsFace area rel RMS | IsFace area total rel |
| --- | ---: | ---: |
| 515k | 0.169806 | -0.000551773 |
| 3152k | 0.148114 | -0.000261091 |

## Interpretation

The saved VTUs/VTPs now provide the requested per-element ParaView fields. The
surface-prestrain relaxations dramatically reduce the `IsFace` area-change RMS
relative to the previous inverse displacement. For example, 3152k goes from
`0.143108` area RMS in the previous inverse to `0.0210653` for SNH 1%,
`0.0502484` for SNH 4%, and `0.0643233` for metric 10%.

The per-tetra volume RMS changes much less across post-inverse variants:

- 515k previous inverse: `0.101750`
- 515k prestrain variants: `0.102783` to `0.103649`
- 3152k previous inverse: `0.103203`
- 3152k prestrain variants: `0.105349` to `0.106621`

So the surface relaxations mostly change the IsFace surface area behavior while
leaving bulk tetra volume-change RMS in roughly the same range.

Important caveat: `TargetDisplacement` is only valid on the face target region,
so the target-derived per-tetra volume arrays are diagnostic fields only and
should not be interpreted as physical volume change over the whole mesh. The
target-derived IsFace triangle area arrays are the meaningful target-side area
diagnostic.

## Validation

- `uv run python -m py_compile src/30-volume-area-change-diagnostics.py`
- Smoke run:
  `DEBUG=1 CHERRIES_NAME='volume area diagnostics 515k smoke' CHERRIES_TAGS='surface-prestrain,volume-change,area-change,smoke,515k' uv run python src/30-volume-area-change-diagnostics.py --cases '["metric10-515k-expression000-smas1"]'`
- Full Cherries run:
  `COMET_AUTO_LOG_GIT_PATCH=false COMET_AUTO_LOG_GIT_METADATA=false CHERRIES_NAME='per-tet volume and IsFace area change diagnostics' CHERRIES_TAGS='surface-prestrain,Expression000,volume-change,area-change,isface,515k,3152k,diagnostics' uv run python src/30-volume-area-change-diagnostics.py`
- Read back all six new VTUs and all six new VTPs with PyVista and verified
  required volume/area arrays were present.
