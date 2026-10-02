# Public anatomy import and forehead reference trial

Date: 2026-09-12. Repository: `liblaf/apple`, HEAD `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`.

## Result

Two publicly downloadable anatomy sources have been converted into reusable local assets. A separate simulation fixture uses Z-Anatomy's frontal muscle geometry to replace the existing forehead's transverse PCA field with two longitudinal directions. The existing forward solver consumes this field, and a controlled trial produces upward motion in a bilateral lower-forehead skin band where the baseline produces slight downward motion.

This is an improvement in anatomical plausibility supported by a modeling experiment. It does **not** establish subject-specific anatomical accuracy. The replacement directions are geometric estimates with an explicit anatomical prior; no measured fascicles, dermal insertion maps, or new attachment laws were obtained.

![Baseline fibers, registered public reference, and candidate](../data/20-reference-transfer/forehead-reference-comparison.png)

## Public assets actually obtained

| Source | Pinned version and local result | Useful contribution | Boundary |
| --- | --- | --- | --- |
| [Z-Anatomy](https://github.com/LluisV/Z-Anatomy) | Commit `6c7f9016bd5899ac8edafd31b9900c151df42ed6`; two FBXs, 61 extracted PLYs and a GLB | 42 named facial muscle meshes, 8 fascia/aponeurosis meshes, 11 reference bones; separate frontalis bellies and epicranial aponeuroses | Modeled surface anatomy; no measured fibers, dermal attachments, or skin mesh in this pinned FBX package |
| [ArtiSynth models](https://github.com/artisynth/artisynth_models) | Commit `e37be3f1b4831f347c055cbf7124b6bd57414f98`; face FEM, muscle polylines and attachment sets exported to VTK | 8,720 points; 6,342 elements; explicit modeled paths for 11 muscle groups; source index semantics preserved | No frontalis path, regional SMAS geometry, or measured fascicle/dermal attachment map in the imported Badin package |

The earlier [model survey](../../../../../../docs/research/2026-09-12-human-head-anatomy-models.md) covers the wider alternatives. This experiment concentrates on sources downloadable without a purchase or individual access request.

Z-Anatomy's source and extracted manifests record SHA-256 and Git blob hashes, source URLs, object transforms, coordinate conventions, and attribution. The exact root and model-specific license/readme files are retained. They identify Z-Anatomy as CC BY-SA 4.0 and preserve BodyParts3D attribution and its stated CC BY-SA 2.1 Japan terms. The pinned model readme also warns that the FBX release may be outdated. See [source manifest](../data/12-public-models/zanatomy/source/source-manifest.json), [extraction manifest](../data/12-public-models/zanatomy/extracted/zanatomy-manifest.json), and [model license](../data/12-public-models/zanatomy/source/Models-License.txt).

ArtiSynth's own license, README, Java consumers, ANSYS files, and attachment files are retained in its source cache. The [audit](../data/12-public-models/artisynth/audit.json) distinguishes source data from runtime derivatives:

- The FEM contains 6,024 hex8 and 318 wedge6 elements. The importer resolves the repeated ANSYS slots used for wedges and verifies connectivity and element/node identifiers after reloading VTK.
- The macro declares 22 left-side paths. Eighteen resolve against the selected marker file. Four MAS paths have absent markers; the Java consumer also removes MAS. The export preserves that absence. Runtime mirroring produces 36 paths.
- The 887 attachment entries comprise 432 domain-edge node numbers, 120 nose node numbers, six zygomatic-named fixed node numbers, and 329 jaw point-list indices. Jaw indices are zero-based list positions; the other sets use ANSYS node identifiers.
- Zygomatic-named fixed nodes are boundary conditions, not segmented ligament surfaces or dermal endpoints. The paths are modeled embedded polylines, not specimen-measured fascicles.

### Geometry quality

All 42 extracted Z-Anatomy facial muscle meshes are single-component, watertight, and winding-consistent. Six of the eight fascia/aponeurosis meshes are open sheets; only the two epicranial aponeuroses are watertight. Three reference bones have topology defects. The bones and fascia therefore remain inspection/alignment assets.

The 21 bilateral muscle pairs and four bilateral fascia pairs are mirror copies to within 0.24 micrometre at the vertices. Separate object names provide useful semantic distinctions, but do not supply independent anatomical asymmetry. No selected object references an image texture. Reproducible details are in [geometry QA](../data/12-public-models/zanatomy/geometry-qa.json).

## The current forehead problem

The baseline fixture has 1,146,517 tetrahedra, 120,020 active muscle cells, and 35 active semantic regions. MuscleId 28, named `Occipitofrontalis epicranius001_Head_muscles_0`, is a broad merged forehead region containing 35,171 active cells. Its current global PCA direction is approximately:

```text
(0.9999971, 0.0018722, 0.0015281)
```

The current coordinates are X lateral, Y superior, and Z anterior. The field is consequently almost entirely transverse. The [baseline audit](../data/10-baseline/regions.json) records the original fixture hashes, region volumes, extents, and fiber statistics.

The anatomical basis for preferring a longitudinal forehead direction is external to the PCA calculation. Histology describes parallel frontalis bundles extending between the supraorbital region and galea. Detailed cadaveric and dynamic studies also show that fascial adherence and sliding vary by forehead region. Thus correcting direction alone cannot reproduce the complete mechanism. See [Human Frontalis Muscle Innervation and Morphology, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8932476/) and [Angrigiani et al., 2024](https://doi.org/10.1093/asj/sjad320).

## Registration and candidate construction

The atlas is registered with a proper three-dimensional similarity transform using 22 corresponding muscle centroids, representing 11 bilateral pairs. Atlas centroids use triangle-area weighting; current centroids use muscle-fraction-weighted cell volumes. Frontalis is excluded from fitting. Platysma and buccinator are also excluded because their coverage and geometry differ substantially across the cropped models.

The scale is 1.01397. The training-centroid RMS is **4.91 mm**; leaving an entire bilateral muscle pair out gives **5.55 mm RMS**. This validation avoids retaining the opposite member of the same mirrored pair during its test. These numbers measure cross-atlas correspondence, not anatomical accuracy. Nearest registered frontalis-vertex distances from current forehead cell centers have medians of about 4 mm and 95th percentiles of about 7.3 mm; maximum distances approach 19 mm.

These errors are too large to justify silently transferring thin fascia, direct insertions, or fine material labels. The registered [atlas](../data/20-reference-transfer/registered-atlas.vtm) is provided for inspection, while the solver change is confined to the forehead direction field.

For each frontalis belly, the candidate regresses surface position against the atlas superior coordinate. With surface covariance C and atlas superior Z, the unnormalized direction is `C[:, Z] / C[Z, Z]`. This is a stated brow-to-galea longitudinal prior, not a fiber measurement. It also avoids selecting the widest extent of a broad belly as its contraction axis. After transformation and normalization, the two directions are approximately:

```text
source .r: (-0.0901, 0.9255, -0.3678)
source .l: (+0.0886, 0.9258, -0.3675)
```

The candidate keeps one direction per belly and does not represent local fascicle curvature. It retains the entire original active forehead region, including any segmentation errors. Source-side labels are used as atlas identifiers; the correspondence is based on the observed lateral coordinates.

The [candidate fixture](../data/20-reference-transfer/fixture/volume.vtu) preserves the original points, tetrahedra, material fractions, skin, fixed constraints, activation mask, control IDs, and all non-forehead fibers. Only `ActivationFiber` on the 35,171 active forehead cells affects the solver differently. Four additional arrays record reference identity, distance, names and method; the solver does not use them. All original arrays are checked after saving and reloading. [Transfer receipt](../data/20-reference-transfer/transfer.json).

## Controlled forward comparison

The experiment uses the existing `FacePhysics` and activation adapter, with exact source snapshots and hashes retained. Each solve starts from zero displacement. Activation mode F has `q = 0.1` in the active forehead and zero elsewhere, with `gamma = 0.5`. All passive materials and constraints are identical. Both solves use relative tolerance `1e-5`, absolute tolerance `1e-12`, and the existing strict line search.

| Measurement | Baseline | Candidate |
| --- | ---: | ---: |
| Forehead mean squared superior fiber alignment | 0.00000351 | 0.856845 |
| Lower-forehead mean superior displacement, area weighted | −0.0383 mm | +0.1137 mm |
| Lower-forehead mean absolute lateral displacement, area weighted | 0.1000 mm | 0.0380 mm |
| Converged optimizer steps | 672 | 696 |
| Final gradient norm | 2.164e−12 | 1.853e−12 |
| Minimum det(F) | 0.91326 | 0.93868 |

The primary skin ROI is a bilateral band in the lower third of the baseline active forehead's superior-coordinate range. It lies inside the forehead's lateral extent, within 12 mm of active forehead cell centers, anterior to the nearest center, and excludes fixed vertices. It contains **851 vertices**, with 412 on the lower-X side and 439 on the higher-X side. Area-weighted upward displacement increases by 0.1531 mm and 0.1509 mm on the two sides, respectively.

An initial 6 mm proximity threshold selected only 32 asymmetric vertices. That selection is preserved in the evidence as a limited diagnostic. The broader geometry-based band was inspected for bilateral coverage and recomputed from the saved displacements; it required no new solve and did not fit an expression target.

![Displacements and the bilateral lower-forehead region](../data/30-forward/paired-skin-displacement-front.png)

The F activation matrices differ by up to 0.1527 in their packed components, proving that the consumer receives a different active tensor. A separate Raw6 check gives exactly zero matrix difference when only the fibers change. Consequently, this candidate changes F/Shared-style use of the fibers; it does not change Raw6 behavior merely by being loaded.

The result supports the intended change from transverse contraction toward upward lower-forehead motion in this fixture. No measured expression, eyebrow landmark trajectory, wrinkle depth, or anatomical segmentation was used as ground truth. The absolute upward displacement is small, and the existing passive anatomy and attachment assumptions still govern the response. Solver convergence and positive determinants are numerical checks, not evidence of anatomical fidelity. Full data: [evidence.json](../data/30-forward/evidence.json), [saved arrays](../data/30-forward/forward-arrays.npz), [baseline deformation](../data/30-forward/baseline-deformed.vtu), and [candidate deformation](../data/30-forward/candidate-deformed.vtu).

## Reproduction and run evidence

Run from `exp/2026/09/12/public-head-anatomy`. The active project interpreter is used directly to preserve the existing environment instead of updating dependencies. CPU registration uses NumPy/PyVista; forward solves use the existing CUDA/Warp/Torch implementation. Blender 5.2.1 LTS imports the FBXs. The random visualization sample uses seed 0.

```bash
ANATOMY_PYTHON=.venv/bin/python
export VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow
export COMET_AUTO_LOG_GIT_PATCH=false

"$ANATOMY_PYTHON" src/fetch_zanatomy.py --output-dir data/12-public-models/zanatomy/source
blender --background --factory-startup --python src/zanatomy_source.py -- \
  --muscular-fbx data/12-public-models/zanatomy/source/MuscularSystem100.fbx \
  --skeletal-fbx data/12-public-models/zanatomy/source/SkeletalSystem100.fbx \
  --output-dir data/12-public-models/zanatomy/extracted

CHERRIES_NAME='Public anatomy baseline audit' \
CHERRIES_TAGS='face,anatomy,public-models,baseline' \
  "$ANATOMY_PYTHON" src/10-audit-baseline.py

CHERRIES_NAME='Import pinned ArtiSynth public face anatomy (verified)' \
CHERRIES_TAGS='public-anatomy,artisynth,import,pinned-source,verified' \
  "$ANATOMY_PYTHON" src/12-import-artisynth.py

CHERRIES_NAME='Z-Anatomy geometry QA' \
CHERRIES_TAGS='public-anatomy,zanatomy,geometry-qa' \
  "$ANATOMY_PYTHON" src/14-check-zanatomy.py

CHERRIES_NAME='Public atlas forehead reference transfer' \
CHERRIES_TAGS='face,anatomy,public-models,reference-transfer' \
  "$ANATOMY_PYTHON" src/20-transfer-reference.py

CHERRIES_NAME='Forehead forward fiber comparison' \
CHERRIES_TAGS='public-anatomy,forehead,fiber-transfer,forward' \
  "$ANATOMY_PYTHON" src/30-compare-forward.py
```

The ArtiSynth importer accepts `--offline true` after acquisition. `30-compare-forward.py --analysis-only true` recomputes measurements and the plot from saved displacements, with fixture checks. `DEBUG=1` was used for initial smoke runs only.

| Run | Recorded summary |
| --- | --- |
| [Baseline](https://www.comet.com/liblaf/apple/8cf6caa549d34112a30e3d06081dda1e) | 120,020 active cells; 35,171 forehead cells |
| [ArtiSynth import](https://www.comet.com/liblaf/apple/98ba7b66d2954bb09b2db76822f5d20f) | Source identifiers, mesh topology, paths and attachment sets round-trip successfully |
| [Z-Anatomy geometry QA](https://www.comet.com/liblaf/apple/cb9912c947f24b6281576eec0fd356c3) | 61 object topology records and 29 mirrored-pair comparisons |
| [Reference transfer](https://www.comet.com/liblaf/apple/3e54023af4a4480989b5706a4749ae9f) | RMS 4.90637 mm; held-out pair RMS 5.54977 mm; 35,171 changed fibers |
| [Forward comparison](https://www.comet.com/liblaf/apple/60cb876bb0074f48a79bdb28292671f8) | Both full-mesh solves converge at the declared tolerances |
| [Bilateral ROI and artifact verification](https://www.comet.com/liblaf/apple/549ab0a36c214d4882c5d72206a69958) | Saved-displacement analysis with 851 bilateral skin vertices |
| [Final figure and verification](https://www.comet.com/liblaf/apple/ef55b3032115456cbaf3be1dcb4de1ac) | Successful final analysis-only run; figure layout inspected |

Local outputs and Cherries snapshots are the primary evidence. Initial baseline/forward runs warned that global Git patch/environment capture or final remote logging did not complete; the transfer run also logged a notification timeout. These warnings do not invalidate the completed local computations, but remote summaries should not be treated as complete artifact archives. Later commands disable global Git patch capture while retaining Git SHA and explicit source/data hashes. Source checks passed with Ruff and Python compilation; download hashes, VTK round trips, fixture isolation, and generated-file links were verified. Original experiments and their inputs were not edited, staged, or committed.

PyVista can serialize its hidden boolean-array-name metadata in a different order across processes. Such a byte-hash change must be distinguished from a changed mesh or field. The [fixture receipt](../data/30-forward/fixture-receipt.json) preserves the actual consumed artifact hash and records logical equality separately for the later delivered fixture; it does not rewrite the historical solve receipt.

## What to use next

Use the new fixture for controlled F-mode forehead trials and the registered atlas for visual inspection. The ArtiSynth exports provide a reusable starting point for investigating modeled paths in its supported lower-face muscle groups. They have not been registered or transferred into the current face here.

The remaining high-value anatomy work requires additional evidence: regional fascia segmentation with reliable correspondence, distinct receiving surfaces for dermal attachments, and measured or explicitly validated fascicle trajectories. The imported public files do not supply these together. Automatically relabeling the current heuristic aponeurosis field from a several-millimetre atlas alignment would overstate what the data support.
