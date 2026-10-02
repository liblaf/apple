# Browser report and tailnet serving

The browser report is built from the completed [study report](40-results.md), its saved-state figures, and compact evidence records. The published directory is `../site/`; all linked report assets live inside it.

The builder is [60-build-report-site.py](../src/60-build-report-site.py). Run it from the experiment directory with the repository's Python environment:

```bash
.venv/bin/python src/60-build-report-site.py
```

The private address is PRIVATE_HOST:8769 (private preview omitted), with PRIVATE_HOST:8769 (private preview omitted) as the direct-address alternative. The listener binds only to PC07's verified Tailscale IPv4 address. Readers must have tailnet access, and PC07 must be online.

## Runtime lifecycle

The report uses the transient user unit `activation-smoothness-report.service`. Its definition lives under `/run/user/1000/systemd/transient/` for the current boot. No persistent service is installed or enabled, and no Tailscale Serve or Funnel configuration is changed.

After building `site/index.html`, start the server with:

```bash
systemd-run --user --unit=activation-smoothness-report \
  --description='Activation smoothness study report on tailnet' \
  --property=Restart=on-failure --property=RestartSec=3 \
  --setenv=PYTHONUNBUFFERED=1 \
  /usr/bin/python -m http.server 8769 --bind PRIVATE_HOST \
  --directory exp/2026/09/09/activation-space-smoothness/site
```

Inspect or stop it with:

```bash
systemctl --user status activation-smoothness-report.service
systemctl --user stop activation-smoothness-report.service
```

After a reboot, verify PC07's Tailscale address before running the start command again.

## Initial publication check

The report was published and checked on September 9, 2026, at 10:29 Asia/Shanghai. The [local HTTP receipt](../data/61-tailnet-verification-v2/local-http.json) confirms HTTP 200 and exact file bytes for the index and all 30 relative linked assets. All 17 images have alternative text, and the serving directory contains no symlinks.

The [PC05 receipt](../data/61-tailnet-verification-v2/pc05-http.json) confirms that a separate tailnet device fetched the final index, manifest, and build-source download with matching checksums. The endpoint figure was also fetched successfully from PC05 before the final text-only clarification. Tailscale Serve configuration remained empty.

The initial index SHA-256 was `49369e79ba12abba8b6390917a6112b5888590dd3f4492080048532c614f1f5a`. Its manifest SHA-256 was `0a2589d78218439e2062d5ec8a1c3e19bc9713e014b561bb008d1a865649498e`.

## Plot and implementation revision

The report was revised on September 9, 2026, at 11:03 Asia/Shanghai. Nine figures now use distinct colors, line styles, and markers. The page includes the [learning-rate diagnosis](45-learning-rate-diagnosis.md), a three-model implementation matrix, a seven-run protocol table, and shared solver/material settings from the [implementation audit](46-implementation-comparison.md). The diagnosis extracts existing pilot and primary traces; it does not run new inverse fits. The previous published directory is preserved at `../data/60-report-site-before-colors/`.

The [revision HTTP receipt](../data/62-report-revision-verification/local-http.json) confirms HTTP 200 and exact bytes for all 57 published files, validates local index links and anchor targets, and records 18 images with alternative text and eight tables. The revised trajectory, learning-rate, exact-section, and historical-context figures were visually inspected. Browser automation timed out, so this revision has no browser screenshot or viewport-layout verification.

The [revision PC05 receipt](../data/62-report-revision-verification/pc05-http.json) confirms matching checksums for the final page, manifest, revised trajectory and learning-rate plots, implementation-table download, and site build source from another tailnet device. The existing runtime-only server remains active; no service restart or persistent configuration change was needed.

The plot-revision index SHA-256 was `88bdd4034e21170440105769cefcc37b3ad10e77b0c52d02e8876a416a57764e`. Its manifest SHA-256 was `b590c6f18087c45f803d0b63697cc0bb153f54ace53e4a81dd18424cd187a021`.

## Completed learning-rate follow-up

The report was updated and checked on September 9, 2026, at 12:37 Asia/Shanghai. The rate-0.3 section (private preview omitted) contains the completed 256-update smoothed-arm run, the original and quarter-rate comparisons, inversion-onset brackets, physical update sizes, and exact saved geometry. The [quarter-rate results](49-quarter-rate-results.md) and [rate-0.3 results](56-rate-03-results.md) retain the full interpretation and evidence. The published directory before the follow-up is preserved at `../data/63-report-site-before-rate-results/`.

The [final local HTTP receipt](../data/64-rate-report-verification/local-http.json) confirms HTTP 200 and exact bytes for all 92 published files, validates local links and anchors, and records 30 images with alternative text and 13 tables. The new trajectory, global-fit/motion, section, and endpoint figures were visually inspected. This check does not include browser viewport-layout verification. A separate read-only audit found no material discrepancy between the execution/results documents and the verification/comparison records, and independently confirmed all 43 comparison-input and 15 output hashes.

The [final PC05 receipt](../data/64-rate-report-verification/pc05-http.json) confirms matching checksums from another tailnet device for the index, manifest, rate-0.3 trajectories and endpoint plate, results download, numerical-verification receipt, and site build source. The [service receipt](../data/64-rate-report-verification/service.json) confirms that the existing transient server remains active with PID 262868. No restart or persistent service change was needed.

The learning-rate revision index SHA-256 was `881e249d86adaec9b9320dee63a33823fa4b3b7be124f72f8199da4df03b08a8`. Its manifest SHA-256 was `2d52b5efe06d067f425837e8d4e185a6478441abdc8128e455b14524245bd983`.

## Activation glyph gallery

The glyph gallery (private preview omitted) was published and checked on September 9, 2026, at 13:05 Asia/Shanghai. It adds 12 standalone activation views for six learned-axis states, an exact VTP/ParaView bundle, and the [glyph methods and evidence](65-activation-glyphs.md). The previous 92-file report is preserved at `../data/66-report-site-before-glyphs/`.

The [local HTTP receipt](../data/67-glyph-report-verification/local-http.json) confirms HTTP 200 and exact bytes for all 109 published files, valid page links and anchors, 42 images with alternative text, and 13 tables. The [PC05 receipt](../data/67-glyph-report-verification/pc05-http.json) confirms matching bytes from another tailnet device for the index, manifest, both rate-0.3 glyph views, the ParaView ZIP, methods document, independent field-verification receipt, and both renderer/build sources. The [service receipt](../data/67-glyph-report-verification/service.json) confirms that the existing transient server remains active; no service configuration changed.

Glyph image QA checked axis visibility and caption/legend bounds. The [independent field verification](../data/67-glyph-report-verification/activation-glyphs.json) checked saved controls, sampled IDs and muscle fields, reference centers, axes and dyads, exact shortening values, display lengths, and archive bytes. The 12 VTPs and all renderer input/output hashes also passed a separate read-only audit. This publication does not claim browser viewport-layout verification.

The initial glyph-gallery index SHA-256 was `4f2b7aa3c03613604f252eeaca65970547dc4ce7aaace9f4c80897ce9c23fe14`. The initial glyph-gallery manifest SHA-256 was `876b779ceafbb6ccc82d960e5133e2a186a460db9baa78857d2b8c6dce0d4f76`.

## Glyphs on saved deformed shapes

The deformed glyph gallery (private preview omitted) was published and checked on September 9, 2026, at 14:10 Asia/Shanghai. All six learned-axis states now use their saved deformed tetrahedron centers, spatially transformed directions `normalize(F n_rest)`, and deformed skin. The common line scale remains 4.5 mm × commanded shortening fraction. Visibility is recomputed for each state and camera, retaining interior tetrahedra of the visible muscle. No muscle surfaces or outlines are drawn. The full fields retain all 288,235 active tetrahedra across 103 activation regions. The methods and exact checkpoints are in [76-deformed-activation-glyphs.md](76-deformed-activation-glyphs.md).

The preceding 112-file reference-frame site is preserved at `../data/78-report-site-before-deformed/`, and its original `data/74-visible-activation-glyphs/` assets remain intact. The main study and learning-rate findings are unchanged: 89 copied assets are byte-identical, and the HTML outside the glyph section differs only in the glyph evidence sentence.

The [local HTTP receipt](../data/79-deformed-glyph-report-verification/local-http.json) confirms HTTP 200 and exact bytes for all 112 published files, valid local links and anchors, 42 images with alternative text, and 13 tables. The [PC05 receipt](../data/79-deformed-glyph-report-verification/pc05-http.json) confirms matching checksums for 13 files fetched from another tailnet device: the page, manifest, both rate-0.3 endpoint views, the complete endpoint VTP, the 1,183,318,752-byte ZIP, methods, field receipts, renderer, geometry verifier, visibility helper, and site builder. SSH used BatchMode and strict host-key checking; HTTP requests bypassed proxies.

The [independent geometry receipt](../data/77-deformed-glyph-verification/activation-glyphs.json) verifies all six full fields, an independently solved affine transform, the exact skin point mapping and deformed region triangles, analytic camera projection, direct label-image lookup, and all archive bytes. Each view has six distinct state masks, all different from the reference mask. The [image inspection receipt](../data/79-deformed-glyph-report-verification/image-qa.json) records readability and annotation/legend bounds for all 12 PNGs. This does not claim browser viewport-layout verification. No inverse fit, surface smoothing, decimation, or deformation amplification was performed.

The [integration receipt](../data/79-deformed-glyph-report-verification/integration.json) maps worktree-generated files to their verified original-experiment copies; generation receipts preserve their original execution paths. The [service receipt](../data/79-deformed-glyph-report-verification/service.json) confirms the existing transient server remained active with PID 262868. No restart, persistent service installation, or service configuration change was needed.

The [publication check source](../src/79-verify-deformed-glyph-report.py), [complete terminal log](../logs/79-verify-deformed-glyph-report.terminal.log), and [Comet run](https://www.comet.com/liblaf/apple/880b9bbf62574c83bf677791d30afc54) retain the verification command and execution record. It used `ProfileCometNoCommit` and exited successfully after Cherries shutdown. No commit or push was made.

Deformed-glyph revision index SHA-256: `b369ecd1b33915ae0bba13a4ca4147d7737fcf2db321607fc757c583a195c747`.

Deformed-glyph revision manifest SHA-256: `759294489eba24ec9d1eb10642425ba3a7dac7a5c7265c7c1b96f9f976b820ae`.

## Concise investigation report

The report was recomposed and verified on September 9, 2026, at 14:58 Asia/Shanghai, before the requested 15:00 deadline. The main page (private preview omitted) now follows the four-step investigation: the old-baseline/active-stress observation, the mathematical mapping and determinant bug, the corrected baseline with muscle deformation evidence, and the later uniaxial/smoothness tests. All reported experiments exclude skin energy. The prior full study remains at study-details.html (private preview omitted). The prior serving directory is preserved in `../data/86-report-site-before-ideas/`.

The interactive comparison (private preview omitted) includes three pairs, two cameras, 38 separate figure assets, and an omitted-mode tensor map. The [local HTTP receipt](../data/87-idea-report-verification/local-http.json) confirms exact bytes for all 188 files. The [PC05 receipt](../data/87-idea-report-verification/pc05-http.json) confirms 13 representative files across the tailnet. The existing transient service remains active without a restart or persistent configuration change.

Browser checks passed all 24 comparison/view/display combinations at desktop size and checked mobile overflow, image decoding, and JavaScript errors. The [interactive browser receipt](../data/88-idea-browser-verification/summary.json) and [main report browser receipt](../data/91-narrative-browser-verification/summary.json) retain the results. Main-page desktop and mobile screenshots were visually inspected; all 12 main-page images decoded successfully.

Investigation revision index SHA-256: `f446308dfaa70d343f13b8c74612df2ca34591cc4ba541ef12c77d2b95539b5f`.

Investigation revision manifest SHA-256: `ee54835af3f453a7af93387671f64e017c97e455c22fa0ccec189456b1a30c08`.

## Overview figures and focused muscle neighborhood

The updated main report (private preview omitted) was published and verified on September 9, 2026, at 15:13 Asia/Shanghai. Skin figures now use whole-face overview cameras, and activation figures use the oblique face overview. The interactive gallery opens on Overview. Every displayed figure links to its original image for zooming. The previous serving directory is preserved in `../data/94-report-site-before-overviews/`; the original `data/84-idea-comparison/` package is also intact.

Section 3 now shows the exact 694-cell neighborhood from the earlier tetrahedron analysis, with the corrected baseline added: Rest / Old baseline 194 / Corrected baseline 200 / PSD 1024. Cell 27306 remains gold. The [4000 × 1120 composite](../data/93-focused-muscle-patch/cell-patch.png), four separate 1000 × 1000 state panels, selected-cell IDs, and [provenance receipt](../data/93-focused-muscle-patch/summary.json) are independently reusable. The renderer uses exact `x = X + u`, no centroid normalization, and the original REST/OLD/PSD focal point and scale; the corrected state does not change the framing. The isolated tetrahedron figure remains in an expandable detail.

The [geometry check](../data/95-overview-report-verification/focused-patch.json) confirms the selected IDs are identical to the historical 694-cell selection, historical REST/OLD/PSD patch metrics agree, all source and output hashes match, and the selected-cell determinants agree with the canonical comparison. The [local HTTP check](../data/95-overview-report-verification/local-http.json) verified exact bytes for all 191 files; the [PC05 check](../data/95-overview-report-verification/pc05-http.json) verified 11 representative files through another tailnet device. The [browser check](../data/96-overview-browser-verification/summary.json) passed all 24 gallery combinations, Overview as the initial view, all 12 main-page images, original-image navigation, and desktop/mobile overflow and JavaScript checks. Main-page desktop/mobile and gallery activation screenshots were visually inspected. The existing transient service remained active with PID 262868.

Commands run from `exp/2026/09/09/activation-space-smoothness/`:

```bash
CHERRIES_NAME='Overview comparison gallery' CHERRIES_TAGS='report,overview,comparison,saved-states' uv run python src/84-build-idea-comparison.py --output-dir data/92-overview-comparison
CHERRIES_NAME='Four-state focused mouth muscle patch' CHERRIES_TAGS='saved-state,visualization,baseline-story,muscle-patch' uv run python src/93-render-focused-muscle-patch.py
uv run python src/60-build-report-site.py
CHERRIES_NAME='Overview report publication verification' CHERRIES_TAGS='report,overview,verification,tailnet' uv run python src/95-verify-overview-report.py
node src/96-verify-overview-browser.cjs
```

Completed Comet summaries: [Overview comparison gallery](https://www.comet.com/liblaf/apple/15078cc723d041658a85b8579ec762c2), 3 comparisons and 38 figures; [Four-state focused mouth muscle patch](https://www.comet.com/liblaf/apple/d9714aae194046c7880dd7d665ea7677), 694 cells and 4 states; [Overview report publication verification](https://www.comet.com/liblaf/apple/ed2111806e62432490fc2d76ff513c94), 191 local files and 11 peer files. All completed with `ProfileCometNoCommit` and successful shutdown. No new inverse fit, commit, or push was performed.

Overview revision index SHA-256: `1edaa36c8de4a9c4fcd56b31a7d66b60364ae19fef180e2321eca520023339a7`.

Overview revision manifest SHA-256: `f8bb4410424b7a149beed330b4ac71c9becda76beceafff505430dfcbd878a0c`.

## Consistent oblique baseline shape views

The report (private preview omitted) was updated and verified on September 9, 2026, at 15:25 Asia/Shanghai. Its two earlier shape comparisons now use the same `side-context` camera, gray flat shading, lighting, and light background as the supplied reference and the later shape panels. The two comparisons remain Old / PSD and Old / Corrected / PSD. Their checkpoints and numerical interpretation are unchanged. The previous serving directory is preserved in `../data/98-report-site-before-oblique/`.

[Renderer 97](../src/97-render-baseline-oblique.py) imports the exact study-81 rendering function. It generated three standalone 1800 × 1800 state PNGs and unscaled 3600 × 1800 and 5400 × 1800 comparison images in [data/97-baseline-oblique](../data/97-baseline-oblique/). The [receipt](../data/97-baseline-oblique/summary.json) records exact checkpoint, fixture, camera, renderer, and image hashes. A separate mapping audit confirmed that each NPZ's `rest_points + u` equals its archived `final.vtu.points` bit-for-bit, and the corrected and PSD final checkpoints are byte-identical to the step checkpoints used by study 81.

The [figure check](../data/99-oblique-report-verification/oblique-figures.json) verified 13 source/output hashes, the exact study-81 camera and style, the state order, image dimensions, and pixel-exact unscaled panel juxtaposition. The [local HTTP check](../data/99-oblique-report-verification/local-http.json) verified all 198 served files, and the [PC05 check](../data/99-oblique-report-verification/pc05-http.json) verified eight changed or representative files from another tailnet device. The [browser check](../data/100-oblique-browser-verification/summary.json) passed both revised figures, original-image navigation, desktop/mobile overflow, image decoding, and JavaScript/HTTP error checks. Desktop screenshots of both affected sections were visually inspected. The existing transient service remained active; no persistent service setting changed.

Commands run from `exp/2026/09/09/activation-space-smoothness/`:

```bash
CHERRIES_NAME='Oblique baseline shape comparisons' CHERRIES_TAGS='saved-state,visualization,baseline-story,oblique' uv run python src/97-render-baseline-oblique.py
uv run python src/60-build-report-site.py
CHERRIES_NAME='Oblique baseline report verification' CHERRIES_TAGS='report,oblique,verification,tailnet' uv run python src/99-verify-oblique-report.py
node src/100-verify-oblique-browser.cjs
```

The completed Comet summaries are [Oblique baseline shape comparisons](https://www.comet.com/liblaf/apple/0eb49a49ba904bc085564e6df501f2bb), three states and five images, and [Oblique baseline report verification](https://www.comet.com/liblaf/apple/73dfba6b15ea46169f2dbcfd0bd7a9a4), 198 local files and eight peer files. Both used `ProfileCometNoCommit`, completed shutdown successfully, and created no commit. No fitting, geometry smoothing, or deformation amplification was performed.

Oblique revision index SHA-256: `b256b8de4540c0a5b2bffc2d8d10d71f3d9f3a3563d2a9c7ef13f951e293aed3`.

Oblique revision manifest SHA-256: `7f0198aa024a9cafffa439c7fc82a9fdc87c865723d5df2608b8c21566bdd113`.

## Target shape in both baseline comparisons

The report (private preview omitted) was updated and verified on September 9, 2026, at 15:50 Asia/Shanghai. Both early shape comparisons now show the supplied target first, using the same oblique camera, gray flat shading, lighting, background, and scale as the result panels. The comparison orders are Target / Old / PSD and Target / Old / Corrected / PSD. The original result pixels are unchanged. The prior report is preserved in `../data/102-report-site-before-target/`.

[Renderer 101](../src/101-add-oblique-target.py) created a standalone [1800 × 1800 target image](../data/101-oblique-target/target.png) and two unscaled comparison PNGs at 5400 × 1800 and 7200 × 1800. It imports the exact study-81 rendering function and reuses the verified target VTP. The [receipt](../data/101-oblique-target/summary.json) records source and output hashes, camera, style, and panel order. The first invocation exposed a missing output-directory creation before rendering; the script was corrected, and the successful run completed with no artifact from that failed invocation used.

The [independent verification](../data/103-target-report-verification/summary.json) confirms that all 15,299 target points equal the frozen skin plus the Smile field indexed by GlobalPointId, with identical point IDs and 29,899-triangle connectivity. It also checks the camera/style, exact panel juxtaposition, all ten changed publication files over local HTTP, and seven representative files from PC05. Unchanged published files retain their hashes from the earlier full verification. The [browser check](../data/104-target-browser-verification/summary.json) passed both revised figures, full-resolution navigation, desktop/mobile overflow, and JavaScript/HTTP error checks. The updated corrected-baseline section was visually inspected in the browser screenshot, and a separate visual audit passed the target and both composites.

Commands run from `exp/2026/09/09/activation-space-smoothness/`:

```bash
CHERRIES_NAME='Target in oblique baseline comparisons' CHERRIES_TAGS='target,visualization,baseline-story,oblique' uv run python src/101-add-oblique-target.py
uv run python src/60-build-report-site.py
CHERRIES_NAME='Target shape report verification' CHERRIES_TAGS='target,report,verification,tailnet' uv run python src/103-verify-target-report.py
node src/104-verify-target-browser.cjs
```

Successful Comet summaries: [Target in oblique baseline comparisons](https://www.comet.com/liblaf/apple/3c4bf52c0ea04da6872756cd62fa077d), one target view and two comparisons; [Target shape report verification](https://www.comet.com/liblaf/apple/d9fd80025bec4cd69a586efc2a634b59), 15,299 target points, ten changed local files, and seven peer files. Both used `ProfileCometNoCommit` and completed shutdown successfully. No fitting, geometry smoothing, displacement amplification, commit, or push was performed.

Target-composite revision index SHA-256: `9e4bdfecb3c8ae7bb265fb03f0b21b7a9e67c89146d05589ee7f35a34ed19d3f`.

Target-composite revision manifest SHA-256: `3a1410e612c9fe39ca26cb70aa41f747b32cba520ca2fe128ba920e056ce1b86`.

## Standalone target and muscle-neighborhood locator

The target was separated from the comparison composites on September 9, 2026, at 15:54 Asia/Shanghai. It now appears exactly once as a standalone 1800 × 1800 image beneath the report introduction, with a direct download link. Both baseline comparisons again contain results only. The [browser receipt](../data/106-single-target-browser-verification/summary.json) and [PC05 receipt](../data/106-single-target-browser-verification/peer.json) verified this presentation and the target image bytes. The preceding report is preserved in `../data/105-report-site-before-single-target/`.

At 16:07, section 3 (private preview omitted) gained a boxed rest-face overview immediately above the muscle close-ups. [Renderer 107](../src/107-render-muscle-location.py) uses the same oblique camera and surface material/light settings as the other shape views. Its box encloses the projected extent of all 244 unique vertices in the exact 694-cell reference-selected neighborhood, with an 18-pixel margin. The marker identifies the upper-lip neighborhood shown in the close-ups. The standalone target and result-only comparison figures remain intact.

The [locator PNG](../data/107-muscle-location/overview.png), [projection arrays](../data/107-muscle-location/projection.npz), and [source/output receipt](../data/107-muscle-location/summary.json) retain reproducibility evidence. The [independent geometry and publication check](../data/109-muscle-location-verification/summary.json) confirms exact cell and vertex IDs, agreement between analytic orthographic projection and VTK display coordinates to 5.8e-12 pixels, complete box containment, eight changed local HTTP files, six matching PC05 files, and an active transient server. The [browser check](../data/110-muscle-location-browser-verification/summary.json) verifies the locator placement, both overview images, exactly one standalone target, restored result-only panels, original-image navigation, and desktop/mobile error and overflow checks. Locator screenshots at desktop and mobile sizes were visually inspected. The preceding serving directory is preserved in `../data/108-report-site-before-locator/`.

Commands run from `exp/2026/09/09/activation-space-smoothness/`:

```bash
CHERRIES_NAME='Muscle neighborhood overview locator' CHERRIES_TAGS='report,muscle-patch,overview,locator' uv run python src/107-render-muscle-location.py
uv run python src/60-build-report-site.py
CHERRIES_NAME='Muscle locator report verification' CHERRIES_TAGS='report,muscle-patch,verification,tailnet' uv run python src/109-verify-muscle-location.py
node src/110-verify-muscle-location-browser.cjs
```

Successful Comet summaries: [Muscle neighborhood overview locator](https://www.comet.com/liblaf/apple/9de29c734a2e4e2c831ca2ec71d9f1c8), 694 cells and 244 unique vertices; [Muscle locator report verification](https://www.comet.com/liblaf/apple/e9e3d749bcbc4734bafd2466723edf45), the same patch, eight changed files, and six peer files. Both completed with `ProfileCometNoCommit` and successful shutdown. The renderer emitted a nonfatal VTK `AddActor2D` deprecation warning; output and projection verification passed. No fitting, geometry modification, commit, or push was performed.

Current index SHA-256: `dbb9d95e1eb4a4eabff4cba11e6ba564e7e1b82648a517a0f51ec9f98298d25f`.

Current manifest SHA-256: `28988458223f0fe53601f154c03ae3eb20caef8246d6944dec60376f0891f4d9`.
