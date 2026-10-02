# Visible muscle activation glyphs

Each active muscle tetrahedron has one centered line. The six complete 3D fields each contain all 288,235 active cells across 103 activation regions. The gallery contains 12 standalone PNGs, with an overview and a mouth-corner view for each saved state. No inverse fit or mechanical solve was run to produce these images.

## Line meaning and length

The saved learned vector is q = v. The model uses C = vvᵀ and B = I + C = A⁻¹. Writing s = ‖v‖², the preferred axial stretch in A is 1/(1+s), and the commanded shortening fraction is a = s/(1+s).

The line is centered at the reference tetrahedron centroid and follows the unoriented axis n = v/‖v‖. Its endpoints are x̄ ± 0.00225 a n, in metres. Equivalently:

**Line length = 4.5 mm × commanded shortening fraction.**

This common scale applies to every cell and every state. It is 50% longer than the preceding 3 mm display scale. Greater contraction always means a longer line: a 50% command gives a 2.25 mm line, and a 25% command gives a 1.125 mm line. There is no minimum length, cell-size scaling, or per-state normalization. Lines may cross their tetrahedron boundaries because this is a display scale, not a physical fiber length or displacement.

Color uses the same linear 0–100% viridis scale in every panel. Lines have a 1 px stroke in 1,800 × 1,800 images. Very weak commands remain tiny; the rate-0.3 endpoint median is 0.1032% shortening. The exact VTP values remain available even when a line is smaller than a pixel.

The line's two ends have equal meaning because v and −v produce the same tensor. These are learned control axes in the reference anatomy, not prescribed anatomical fibers or observed tissue strain. MuscleFraction is exported but is not multiplied into the displayed shortening command.

## Visibility and muscle shape

Muscle shape is conveyed by the distribution and direction of the tetrahedron glyphs. No muscle surfaces, boundary curves, or silhouette overlays are drawn. The reference skin is shown faintly at opacity 0.06 for face context.

For each frozen orthographic camera, the renderer first computes an opaque muscle-region ID image with lighting and antialiasing disabled. A tetrahedron is retained when the frontmost region label at its projected center equals its own ActivationControlId. This retains all depths of the front muscle and hides different muscles behind it. The rule uses only reference geometry and camera, so the masks are identical across all six states. There is no spatial sampling and no outer-layer-only selection.

| View | Retained tetrahedra | Retained interior tetrahedra | Retained boundary-source tetrahedra | Hidden behind another region |
| --- | ---: | ---: | ---: | ---: |
| Overview | 123,552 | 76,574 | 46,978 | 164,501 |
| Mouth corner | 10,310 | 5,680 | 4,630 | 30,333 |

“Interior” means the tetrahedron contributes no triangle to its own region's extracted surface. The overview has 182 centers outside its viewport; the mouth view has 247,592. A centroid-based rule can leave a line crossing a region boundary near its endpoints, and overlapping lines within the same visible muscle remain present.

![Rate-0.3 endpoint activation overview](../data/74-visible-activation-glyphs/geometry/rate03-on/side-context.png)

![Rate-0.3 endpoint activation mouth-corner view](../data/74-visible-activation-glyphs/geometry/rate03-on/region1-mouth-corner.png)

## Saved states

Fit, motion, and inversion counts describe the saved mechanical states. Glyphs and the faint skin remain in reference coordinates. These states have different budgets and fits and are descriptive views, not an additional matched smoothness test.

| State | Update | Learning rate | Fit RMS (mm) | Motion RMS (mm) | Inverted tetrahedra |
| --- | ---: | ---: | ---: | ---: | ---: |
| rate03-on | 256 | 0.300000 | 2.876797 | 4.047514 | 73 |
| original-off | 16 | 7.857986 | 3.600656 | 4.472303 | 1,035 |
| original-on | 128 | 7.857986 | 0.681094 | 5.179578 | 94 |
| quarter-off | 64 | 1.964496 | 2.587802 | 4.358874 | 409 |
| quarter-on | 64 | 1.964496 | 5.152102 | 7.308119 | 2,189 |
| rate03-on-128 | 128 | 0.300000 | 4.382659 | 1.745661 | 1 |

The original unsmoothed run's latest full checkpoint is update 16. Its last accepted surface is update 28, for which no full NPZ was saved; the gallery does not substitute failed controls or surface-only data.

## Downloads and verification

The report gallery (private preview omitted) provides each PNG separately. The [ParaView ZIP](../data/74-visible-activation-glyphs/glyph-data.zip) contains all six full line fields, six corresponding centroid fields, the reference skin, labeled region surfaces for visibility analysis, two visibility NPZ files, source snapshots, and a README. The ZIP is approximately 485 MB. The [rate-0.3 endpoint VTP](../data/74-visible-activation-glyphs/glyphs/rate03-on.vtp) is also available separately.

The full line VTPs preserve all 288,235 active GlobalCellIds, including cells hidden in the images. Each also records MuscleId and name, ActivationControlId, MuscleFraction, qNormSquared, exact commanded shortening, LearnedAxisRest, and the sign-invariant LearnedAxisDyadRest. The files under `samples/` contain every active centroid despite the directory name; they are not spatial samples. Open a prebuilt `glyphs/<state>.vtp` in ParaView and color it by CommandedShorteningPercent with a fixed 0–100 range. The view-specific masks are stored separately under `visibility/`; the full VTP remains unfiltered when rotated.

The [renderer receipt](../data/74-visible-activation-glyphs/summary.json) records 34 input and 30 output hashes. The [independent verification](../data/75-visible-glyph-report-verification/activation-glyphs.json) passed for all six fields: active-cell coverage, saved controls, muscle metadata, centers, axes, dyads, exact shortening, the common linear length scale, camera projection, raster-label mask replay, retained interior counts, and all 19 ZIP members. Maximum length error was below 5 × 10⁻¹⁶ m. The final overview and mouth images were visually checked for glyph visibility and caption/legend bounds.

## Reproduction

Run from `exp/2026/09/09/activation-space-smoothness/` with a new empty destination:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LIBGL_ALWAYS_SOFTWARE=1 \
CHERRIES_NAME='visible-activation-glyphs-no-outlines' \
CHERRIES_TAGS='activation,glyph,full-field,visibility,report' \
uv run python src/74-render-visible-activation-glyphs.py \
  --output-dir data/74-visible-activation-glyphs
```

The [Comet run](https://www.comet.com/liblaf/apple/7c5be9d87a0946f8b2d310649c295101) completed successfully. Recorded metadata:

```text
Name: visible-activation-glyphs-no-outlines
Interpreter: .venv/bin/python3
Start: 2026-09-09 13:48:38.099411+08:00
End: 2026-09-09 13:49:10.101702+08:00
Git SHA: d56fa1b553b287b22b2cf7bb82d46117e34ed6bb
```

The experiment source directory is untracked; the exact renderer and visibility-helper snapshots are included in the output and archive. The run used the normal Comet/local evidence profile with Git commits disabled. Terminal output is retained in `logs/74-visible-activation-glyphs-terminal.log`. Earlier visualization revisions are preserved in their original output directories.

The independent checker is [75-verify-visible-activation-glyphs.py](../src/75-verify-visible-activation-glyphs.py). It ran with `CHERRIES_NAME='verify-visible-activation-glyphs'` and `CHERRIES_TAGS='activation,glyph,verification,report'` under the same one-thread numerical environment. Its terminal output is retained in `logs/75-visible-activation-glyphs-verification-terminal.log`.

Renderer receipt SHA-256: `6c669a15d67b32ab9a92e3341406021fd3392a2088eba26eaa16bf321253a25b`.

ParaView ZIP SHA-256: `ded14eb13ad5b61388a3d1d4256128e1ac33d376bc43cb0d5a8d6c3ff2540000`.
