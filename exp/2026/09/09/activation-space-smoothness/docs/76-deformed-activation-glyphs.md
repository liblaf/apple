# Activation glyphs on saved deformed shapes

The gallery contains an overview and a mouth-corner view for each of six saved learned-axis states. Every complete 3D field contains one line per active tetrahedron: **288,235 lines across 103 activation regions**. The centers, directions, skin context, and visibility now follow the saved deformed state. No inverse fitting or mechanical solve was run for this visualization. The previous reference-frame gallery remains preserved under `data/74-visible-activation-glyphs/`.

## Geometry and control meaning

For tetrahedron vertices Xᵢ and their saved nodal displacements uᵢ, use xᵢ = Xᵢ + uᵢ and center c = mean(xᵢ). Let Dₘ = [X₁−X₀, X₂−X₀, X₃−X₀] and Dₛ = [x₁−x₀, x₂−x₀, x₃−x₀], with edges as columns. The deformation gradient is F = Dₛ Dₘ⁻¹. The reference learned axis nᵣ = v/‖v‖ is displayed in the spatial direction nₛ = F nᵣ / ‖F nᵣ‖.

Checkpoint q rows map to tetrahedron indices through `active_ids`; `ActivationControlId` labels the 103 regions and does not index q. The active set is exactly `MuscleFraction > 0`. Checkpoint `rest_points` equals fixture volume points, and fixture skin points equal volume points indexed by skin `GlobalPointId`. Thus the skin uses `(rest_points + u)[GlobalPointId]` with its original connectivity.

The learned control satisfies q = v, C = vvᵀ, and B = I + C = A⁻¹. With s = ‖v‖², the commanded axial shortening is a = s/(1+s). Line endpoints are c ± 0.00225 a nₛ, in metres:

**Line length = 4.5 mm × commanded shortening fraction.**

This is the same common scale as the preceding 74 gallery: a 50% command gives a 2.25 mm line, and a 25% command gives a 1.125 mm line. Transport by F changes direction only; ‖F nᵣ‖ does not multiply the display length. There is no minimum line length, cell-size scaling, or per-state normalization. Color uses one linear 0–100% viridis range. `MuscleFraction` is exported but does not multiply the shortening command.

The two ends have equal meaning because v and −v define the same tensor. These are learned control directions, not anatomical fibers or observed tissue strain. Line lengths are display lengths, not physical fiber lengths or displacement magnitudes. The saved deformation is shown at scale 1 without smoothing, decimation, amplification, or reconstruction of failed states. Inverted cells remain present and each panel retains its saved fit, motion, and global inversion annotations.

## Visibility

Muscle shape emerges from the tetrahedron glyphs. No muscle surfaces, outline curves, or silhouettes are drawn. The deformed skin is faint grey at opacity 0.06. Lines use a 1 px stroke in standalone 1,800 × 1,800 PNGs.

For each state, the renderer extracts every activation region separately from that state's deformed volume. For each frozen orthographic camera it renders an opaque region-ID image, with lighting and antialiasing disabled. A tetrahedron is retained if the frontmost region at its projected deformed centroid matches its own region. This keeps internal tetrahedra throughout the visible region's depth and hides different regions behind it. The masks are recomputed separately for all 12 state/view pairs; none reuse the reference masks. Full VTP exports remain unfiltered.

| State | View | Retained cells | Retained interior cells | Retained boundary-source cells | Behind another region |
| --- | --- | ---: | ---: | ---: | ---: |
| `original-off` | Overview | 125,124 | 77,511 | 47,613 | 162,929 |
| `original-off` | Mouth corner | 12,785 | 7,391 | 5,394 | 29,212 |
| `original-on` | Overview | 125,608 | 77,851 | 47,757 | 162,446 |
| `original-on` | Mouth corner | 11,713 | 6,904 | 4,809 | 26,608 |
| `quarter-off` | Overview | 125,206 | 77,719 | 47,487 | 162,847 |
| `quarter-off` | Mouth corner | 12,265 | 7,299 | 4,966 | 28,778 |
| `quarter-on` | Overview | 127,968 | 79,731 | 48,237 | 160,085 |
| `quarter-on` | Mouth corner | 15,034 | 9,778 | 5,256 | 24,596 |
| `rate03-on-128` | Overview | 124,074 | 76,929 | 47,145 | 163,979 |
| `rate03-on-128` | Mouth corner | 11,429 | 6,371 | 5,058 | 29,949 |
| `rate03-on` | Overview | 124,476 | 77,167 | 47,309 | 163,577 |
| `rate03-on` | Mouth corner | 11,954 | 7,002 | 4,952 | 29,209 |

Interior cells contribute no triangle to their region's extracted surface. The masks include projected pixel coordinates, the full front-label image, source cell/region IDs, and counts. This is a centroid visibility rule: it can leave line endpoints crossing region boundaries, and it retains overlapping lines within the same visible region. Dense, strongly activated areas therefore remain crowded. Weak commands can be smaller than a pixel; the rate-0.3 endpoint median is 0.1032% shortening. The exported numeric fields retain these values exactly.

## Saved states

These are unmatched descriptive states with different fits and update budgets. They do not add a matched smoothness test or change the study's learning-rate findings. All checkpoint paths below are relative to the original experiment's `data/` directory.

| State | Full checkpoint | Learning rate | Fit RMS (mm) | Motion RMS (mm) | Inverted cells, full volume |
| --- | --- | ---: | ---: | ---: | ---: |
| `original-off` | `24-learned-axis/step-0016.npz` | 7.85798579455474 | 3.6007 | 4.4723 | 1,035 |
| `original-on` | `25-learned-axis-smooth/step-0128.npz` | 7.85798579455474 | 0.6811 | 5.1796 | 94 |
| `quarter-off` | `48-axis-off-lr-quarter-64/step-0064.npz` | 1.96449644863869 | 2.5878 | 4.3589 | 409 |
| `quarter-on` | `49-axis-on-lr-quarter-64/step-0064.npz` | 1.96449644863869 | 5.1521 | 7.3081 | 2,189 |
| `rate03-on-128` | `55-axis-on-lr03-128/step-0128.npz` | 0.3 | 4.3827 | 1.7457 | 1 |
| `rate03-on` | `56-axis-on-lr03-256/step-0256.npz` | 0.3 | 2.8768 | 4.0475 | 73 |

`original-off` uses update 16, its latest complete checkpoint. Update 28 has only a saved surface/optimizer state; failed update-29 controls are not used. The rate-0.3 update-128 and update-256 states are separate saved checkpoints of the continuation.

![Rate-0.3 update-256 activation overview on saved deformation](../data/76-deformed-activation-glyphs/geometry/rate03-on/side-context.png)

![Rate-0.3 update-256 activation mouth-corner view on saved deformation](../data/76-deformed-activation-glyphs/geometry/rate03-on/region1-mouth-corner.png)

## Outputs and reproduction

The [render receipt](../data/76-deformed-activation-glyphs/summary.json) records all inputs, outputs, source snapshots, states, and semantics. The [ParaView bundle](../data/76-deformed-activation-glyphs/glyph-data.zip) contains six full line VTPs, six centroid VTPs, six deformed skins, six labeled deformed region surfaces, 12 masks, source snapshots, and README. Its size is 1,183,318,752 bytes. Each PNG remains separately available under `data/76-deformed-activation-glyphs/geometry/<state>/` for later slide layout.

Open matching `context/<state>-skin.vtp` and `glyphs/<state>.vtp` in ParaView, and color lines by `CommandedShorteningPercent` with a fixed 0–100 range. The `samples/` fields also contain `LearnedAxisSpatial` for custom centered glyphs. `LearnedAxisRest`, both axis dyads, `RestCentroid`, `DeformationGradient`, `DetF`, `AxisTransportStretch`, and `GlobalCellId` retain the reference-to-spatial construction. Deformation gradients are nine components in row-major order.

The execution root was `${APPLE_HISTORICAL_WORKTREE}/exp/2026/09/09/activation-space-smoothness`. Inputs were read from the original experiment at `exp/2026/09/09/activation-space-smoothness`; fixture and frozen cameras were read from that original checkout's September 7 and September 8 experiment groups. Generated receipts keep those execution paths. Verified files are copied into the original experiment with separate integration receipts, without rewriting their generation provenance.

```bash
cd ${APPLE_HISTORICAL_WORKTREE}/exp/2026/09/09/activation-space-smoothness
set -o pipefail
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LIBGL_ALWAYS_SOFTWARE=1 \
CHERRIES_NAME='Deformed muscle activation glyphs for six saved states' \
CHERRIES_TAGS='activation-space,deformed-geometry,full-tetrahedron-glyphs,visible-muscles,no-fit' \
.venv/bin/python src/76-render-deformed-activation-glyphs.py \
  2>&1 | tee logs/76-render-deformed-activation-glyphs.terminal.log
```

The renderer refuses to overwrite a nonempty output directory. Use `--output-dir data/<new-run-name>` for a new reproduction. The [executed renderer](../data/76-deformed-activation-glyphs/sources/76-render-deformed-activation-glyphs.py), visibility helper, and `ProfileCometNoCommit` are frozen in `sources/`. Normal Cherries/Comet recording was enabled and Git commits were disabled. The successful process and shutdown hooks exited with code 0. Its Comet summary was:

```text
name: Deformed muscle activation glyphs for six saved states
url: https://www.comet.com/liblaf/apple/048f7e69867a49049bc5856a1374e557
render/states: 6
render/lines_per_state: 288235
cherries/start_time: 2026-09-09 13:59:56.980079+08:00
cherries/end_time: 2026-09-09 14:01:18.068668+08:00
cherries/git/sha: d56fa1b553b287b22b2cf7bb82d46117e34ed6bb
```

See the [complete terminal log](../logs/76-render-deformed-activation-glyphs.terminal.log) for the full summary. The Git revision identifies the base checkout; untracked executed sources are identified by their frozen file hashes. No commit or push was performed.

## Verification

The independent checker is [77-verify-deformed-activation-glyphs.py](../src/77-verify-deformed-activation-glyphs.py). Its [receipt](../data/77-deformed-glyph-verification/activation-glyphs.json) passed for all six states and 12 views. It independently solves DₘᵀFᵀ = Dₛᵀ and checks F Dₘ = Dₛ, all line endpoints and scalar fields, exact skin and region-surface vertices, analytic camera projection, direct label-image lookup, retained interior cells, and archive bytes. Each view has six distinct masks, all different from its reference mask. The largest F discrepancy was 1.50 × 10⁻¹⁴; the largest line-vector and length errors were 1.99 × 10⁻¹⁵ m and 4.91 × 10⁻¹⁶ m. The [verification Comet run](https://www.comet.com/liblaf/apple/41741f621d6d43ff887948040fdf7b45) and [verification log](../logs/77-verify-deformed-activation-glyphs.log) retain the execution record. Renderer and verifier used Python 3.14.6, NumPy 2.4.6, PyVista 0.48.4, and VTK 9.6.2. All 12 PNGs were visually inspected for line clarity, caption/legend bounds, and absence of muscle surface/outline overlays. Dense fields retain same-region overlap by design. Browser viewport layout is outside this image inspection.
