# Learning-rate repeatability and continuation

The selected larger-rate LR 0.6 continuation reached step **1024** with
fit RMS **1.610246 mm**, down from
**2.313335 mm** at the common start.
Motion RMS changed from **3.306700 mm**
to **4.125968 mm**. The hard cap was reached while neither settling rule passed. Smoothness and rank remain deferred; no conditional branch is claimed.

The two step-512 probes compare the fixed Adam learning rates **0.3** and
**0.6** with `epsilon = 0.01` in both arms. There was no magnitude calibration
or learning-rate search. Three fresh no-update samples passed the frozen
full-tensor repeatability gate at the common cached sample-0 fit RMS
**2.3133354948563185 mm**. The selected probe was **larger-rate LR 0.6**.
That selection establishes only a local 32-update advantage under this frozen
model and state; it does not identify a universally optimal learning rate. This
is a fixed-budget optimizer result, not an inverse-convergence, capacity,
anatomical, or physiological claim.

Fit and motion RMS values below are area-weighted unless explicitly labeled
otherwise. The optimization retains the original unweighted data objective and
all 1,729,410 tensor controls; the area-weighted metric governs branch selection.

## Probe selection

The source100 sample-0 physical update RMS values were
`1.41127796515e-05 MPa` at LR 0.3 and
`2.82228475186e-05 MPa` at LR 0.6: a measured
full-update ratio of **1.999808**, not a matching target.

The first actual source102 updates replayed their corresponding source100 shadow
updates: LR 0.3 had a full-field relative Q error of
`0`
and normalized-q maximum error `0`;
LR 0.6 had `0`
and `0`. Their actual
first-update RMS ratio was **1.999808**.

| Arm | Maximum N/S | Maximum N/D | Maximum same-gradient projected q error |
| --- | ---: | ---: | ---: |
| baseline | 0.00027449809% | 0.00027455073% | 5.5511151e-15 |
| selected | 0.0002730166% | 0.00054608542% | 4.4408921e-15 |

The limits were N/S ≤ 1% and N/D ≤ 10%. The full tensor-field separation D was
`1.41100740306e-05 MPa`.
Maximum pairwise N was `3.87393111253e-11 MPa`
for LR 0.3 and `7.70530577324e-11 MPa` for LR 0.6.
The aggregate receipt retains every pairwise comparison.

The common step-512 fit RMS was 2.313335 mm. The
baseline final fit RMS was 2.270270 mm; the
larger-rate advantage was 0.040895 mm against the required 0.010000 mm. The [probe decision](../data/103-probe/summary.json) binds the selected checkpoint and both branch endpoints.

Over those 32 updates, LR 0.6 accumulated
**1.973164 times**
the physical update path and ended with motion RMS
**0.048413 mm higher**.
This compares fit at equal update counts; it does not establish equal-path efficiency or improved motion.
The [independent probe audit](../tmp/100-preflight/probe-audit.json) verifies the cached state, all 32 solves, optimizer lineage, and selection rule.

![Step-512 full-tensor update repeatability](../data/104-learning-rate-repeatability-plots/learning-rate-repeatability.png)

## Saved endpoints

| State | Global step | Fit RMS (mm) | Motion RMS (mm) | Minimum detF | Inverted tets |
| --- | ---: | ---: | ---: | ---: | ---: |
| Common step-512 replay | 512 | 2.313335 | 3.306700 | 0.339640 | 0 |
| Baseline probe | 544 | 2.270270 | 3.357114 | 0.362795 | 0 |
| Larger-rate probe | 544 | 2.229375 | 3.405526 | 0.374289 | 0 |
| Selected continuation | 608 | 2.090059 | 3.569348 | 0.348727 | 0 |
| Selected continuation | 672 | 1.978678 | 3.700033 | 0.304941 | 0 |
| Selected continuation | 736 | 1.886851 | 3.807486 | 0.266634 | 0 |
| Selected continuation | 800 | 1.809382 | 3.897730 | 0.232507 | 0 |
| Selected continuation | 864 | 1.742835 | 3.974751 | 0.201759 | 0 |
| Selected continuation | 928 | 1.684835 | 4.041330 | 0.173708 | 0 |
| Selected continuation | 992 | 1.633671 | 4.099520 | 0.147906 | 0 |
| Selected continuation | 1024 | 1.610246 | 4.125968 | 0.135720 | 0 |

![Common root and completed probes](../data/107-learning-rate-render/probe-front.png)

![Common root, final selected endpoint, and exact target](../data/107-learning-rate-render/selected-front.png)

![Mouth detail at the common root, final endpoint, and exact target](../data/107-learning-rate-render/selected-mouth.png)

## Endpoint diagnostics

| Endpoint | Tensor variation / common step-512 value | Normal 5-mm high-pass RMS (mm) | Normal high-pass ratio | Rank mixing fraction | Stress-cap fraction | detF min / max | Inverted tets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline probe | 1.057418 | 0.205128 | 0.178713 | 0.266218 | 0.000000 | 0.362795 / 2.586995 | 0 |
| Larger-rate probe | 1.115044 | 0.207779 | 0.178315 | 0.267407 | 0.000000 | 0.374289 / 2.599179 | 0 |
| Selected step 608 | 1.334942 | 0.216594 | 0.176900 | 0.271729 | 0.000000 | 0.348727 / 2.637391 | 0 |
| Selected step 672 | 1.541637 | 0.223419 | 0.175776 | 0.275445 | 0.000000 | 0.304941 / 2.662500 | 0 |
| Selected step 736 | 1.736935 | 0.228840 | 0.174861 | 0.278691 | 0.000000 | 0.266634 / 2.678315 | 0 |
| Selected step 800 | 1.922336 | 0.233217 | 0.174092 | 0.281559 | 0.000000 | 0.232507 / 2.687514 | 0 |
| Selected step 864 | 2.099027 | 0.236802 | 0.173432 | 0.284116 | 0.000000 | 0.201759 / 2.692110 | 0 |
| Selected step 928 | 2.268030 | 0.239780 | 0.172862 | 0.286413 | 0.000000 | 0.173708 / 2.693466 | 0 |
| Selected step 992 | 2.430210 | 0.242288 | 0.172371 | 0.288487 | 0.000000 | 0.147906 / 2.692697 | 0 |
| Selected step 1024 | 2.508963 | 0.243397 | 0.172150 | 0.289449 | 0.000000 | 0.135720 / 2.691734 | 0 |

Tensor variation is the source105 `smoothness` diagnostic normalized here to the common accepted step-512 value. The 5-mm ratio is the source105 full-face rest-normal displacement high-pass RMS divided by its total normal-field RMS. These, rank mixing, stress-cap fraction, determinants, and inversions are reported diagnostics only; they did not select the learning rate or stop continuation.

From the selected step-544 probe to the final endpoint, the normalized high-pass
ratio decreased from 0.178315 to 0.172150, while
the absolute high-pass RMS increased from 0.207779
to 0.243397 mm. Final tensor variation was
2.508963 times its
common step-512 value. The lower normalized surface ratio does not establish
reduced absolute roughness or smoother controls.

## Stopping evidence

The hard cap was reached while neither settling rule passed. Smoothness and rank remain deferred; no conditional branch is claimed.

| Block | Fit reduction (mm) | Required progress (mm) | Gradient RMS / global 256 | Gradient max / global 256 |
| --- | ---: | ---: | ---: | ---: |
| 544 to 608 | 0.139316 | 0.011147 | 47.88% | 57.67% |
| 608 to 672 | 0.111381 | 0.010450 | 41.87% | 52.29% |
| 672 to 736 | 0.091827 | 0.010000 | 37.27% | 47.76% |
| 736 to 800 | 0.077469 | 0.010000 | 33.61% | 43.89% |
| 800 to 864 | 0.066546 | 0.010000 | 30.65% | 40.54% |
| 864 to 928 | 0.058000 | 0.010000 | 28.18% | 37.63% |
| 928 to 992 | 0.051164 | 0.010000 | 26.11% | 35.07% |
| 992 to 1024 | 0.023425 | 0.010000 | 25.19% | 33.91% |

The final [decision](../data/103-block1024/summary.json) records `stop = true`, `regularization_eligible = false`, and the original global-step-256 projected-gradient anchor.

Every later block loads the exact parent controls, Adam moments and counter, and
saved displacement as the solver seed. It then performs a no-update forward and
adjoint check at the boundary; the refreshed displacement can differ within the
original replay tolerance. This consumes no Adam update. The two step-512 probes
instead reuse the common cached state and gradient exactly, as required.
Independent audit receipts cover [blocks through step 800](../tmp/100-preflight/continuation-audit-through800.json)
and [the remaining blocks through step 1024](../tmp/100-preflight/continuation-audit-864-through1024.json).

![Recorded continuation traces](../data/105-learning-rate-comparison/continuation-optimization.png)

## Viewer and figures

The [saved-state viewer](viewer.html) and its [viewer manifest](../data/107-learning-rate-render/viewer-manifest.json) contain actual saved geometry only. [Probe front PDF](../data/107-learning-rate-render/probe-front.pdf), [probe mouth PDF](../data/107-learning-rate-render/probe-mouth.pdf), [selected front PDF](../data/107-learning-rate-render/selected-front.pdf), [selected mouth PDF](../data/107-learning-rate-render/selected-mouth.pdf), [repeatability PDF](../data/104-learning-rate-repeatability-plots/learning-rate-repeatability.pdf), and [trace PDF](../data/105-learning-rate-comparison/continuation-optimization.pdf) are separate reusable assets.

All six figure pairs passed [PNG and PDF visual checks](../tmp/108-learning-rate-report/visual-qa.json).
Browser interaction could not be tested because this session had no enabled
browser surface. The publication procedure verifies static links, JavaScript
modules, and every served file against its byte count and SHA-256.

## Reproducibility

The [frozen protocol](../docs/101-learning-rate-protocol.md), [source100 aggregate](../data/100-repeatability/summary.json), [source105 comparison receipt](../data/105-learning-rate-comparison/summary.json), and [source107 render receipt](../data/107-learning-rate-render/summary.json) preserve the local hash lineage. The staged subsite provides a text-only reproduction bundle and figure bundle; large states and checkpoints remain local. Conditional receipts: none supplied.

| Stage | Local summary | Terminal capture and wall time | Comet |
| --- | --- | --- | --- |
| No-update sample 0 | [summary](../data/100-step512-sample0/summary.json) | [log](../logs/100-step512-sample0-terminal.log) (0:00:09.151883) | [Comet](https://www.comet.com/liblaf/apple/944a34bac2414e138bffa59b4494add8) |
| No-update sample 1 | [summary](../data/100-step512-sample1/summary.json) | [log](../logs/100-step512-sample1-terminal.log) (0:00:12.269874) | [Comet](https://www.comet.com/liblaf/apple/a3900f77a70e4feeb25c902f23838ca0) |
| No-update sample 2 | [summary](../data/100-step512-sample2/summary.json) | [log](../logs/100-step512-sample2-terminal.log) (0:00:12.277028) | [Comet](https://www.comet.com/liblaf/apple/38d19ec7b63342e9842359f8fbc795fc) |
| Repeatability aggregate | [summary](../data/100-repeatability/summary.json) | [log](../logs/100-repeatability-terminal.log) (0:00:12.537182) | [Comet](https://www.comet.com/liblaf/apple/7b502231bb5c40c3b828a4f2a132526c) |
| Baseline 32-update probe | [summary](../data/102-baseline32/summary.json) | [log](../logs/102-baseline32-terminal.log) (0:04:47.831134) | [Comet](https://www.comet.com/liblaf/apple/02da0a7e0da14709b4e8f5295b588a34) |
| Larger-rate 32-update probe | [summary](../data/102-larger-rate32/summary.json) | [log](../logs/102-larger-rate32-terminal.log) (0:05:14.863258) | [Comet](https://www.comet.com/liblaf/apple/f0a3292f7fbb4c5b8d082a9d7add4eb7) |
| Probe decision | [summary](../data/103-probe/summary.json) | [log](../logs/103-probe-terminal.log) (0:00:00.305498) | [Comet](https://www.comet.com/liblaf/apple/05971f3e31ef47ac822495d2ae9fa034) |
| Repeatability plot | [summary](../data/104-learning-rate-repeatability-plots/summary.json) | [log](../logs/104-learning-rate-repeatability-plots-terminal.log) (0:00:00.593100) | [Comet](https://www.comet.com/liblaf/apple/e5ea32cc842b467299bbd8f10d55fcc6) |
| Endpoint comparison | [summary](../data/105-learning-rate-comparison/summary.json) | [log](../logs/105-learning-rate-comparison-terminal.log) (0:01:26.498186) | [Comet](https://www.comet.com/liblaf/apple/4153f53d11ab49899c8362348fd6f2af) |
| Saved-state rendering | [summary](../data/107-learning-rate-render/summary.json) | [log](../logs/107-learning-rate-render-terminal.log) (0:01:37.575391) | [Comet](https://www.comet.com/liblaf/apple/b1b2a4c66b934ee0abc0a4d3e355d181) |
| Continuation to 608 | [summary](../data/102-fit608/summary.json) | [log](../logs/102-fit608-terminal.log) (0:06:46.784982) | [Comet](https://www.comet.com/liblaf/apple/7c97e2a1131e40dcb9f6dbaee23a8125) |
| Decision at 608 | [summary](../data/103-block608/summary.json) | [log](../logs/103-block608-terminal.log) (0:00:00.384135) | [Comet](https://www.comet.com/liblaf/apple/8b2fa9225a0e4da5949b02f11b8bc07b) |
| Continuation to 672 | [summary](../data/102-fit672/summary.json) | [log](../logs/102-fit672-terminal.log) (0:06:26.340727) | [Comet](https://www.comet.com/liblaf/apple/30b5285dd1f0450d898743e3b0638935) |
| Decision at 672 | [summary](../data/103-block672/summary.json) | [log](../logs/103-block672-terminal.log) (0:00:00.367311) | [Comet](https://www.comet.com/liblaf/apple/e35e66e702a74c92975a5f7d9b498dfc) |
| Continuation to 736 | [summary](../data/102-fit736/summary.json) | [log](../logs/102-fit736-terminal.log) (0:06:09.762051) | [Comet](https://www.comet.com/liblaf/apple/1a098904e267489a968bb3f0005cbfd6) |
| Decision at 736 | [summary](../data/103-block736/summary.json) | [log](../logs/103-block736-terminal.log) (0:00:00.382265) | [Comet](https://www.comet.com/liblaf/apple/3b87c9a80fab462e9be3f041dce9f183) |
| Continuation to 800 | [summary](../data/102-fit800/summary.json) | [log](../logs/102-fit800-terminal.log) (0:05:53.738649) | [Comet](https://www.comet.com/liblaf/apple/5b13bf6c6b7b4c5bb152f57db3c7d7e3) |
| Decision at 800 | [summary](../data/103-block800/summary.json) | [log](../logs/103-block800-terminal.log) (0:00:00.372706) | [Comet](https://www.comet.com/liblaf/apple/4c52d24c55284cf2862c628d46c76eb9) |
| Continuation to 864 | [summary](../data/102-fit864/summary.json) | [log](../logs/102-fit864-terminal.log) (0:05:40.862821) | [Comet](https://www.comet.com/liblaf/apple/c40900eb39164938a9544303db8a0f59) |
| Decision at 864 | [summary](../data/103-block864/summary.json) | [log](../logs/103-block864-terminal.log) (0:00:00.369351) | [Comet](https://www.comet.com/liblaf/apple/c8959096c85b490aa5e09a97536a3be1) |
| Continuation to 928 | [summary](../data/102-fit928/summary.json) | [log](../logs/102-fit928-terminal.log) (0:05:39.603865) | [Comet](https://www.comet.com/liblaf/apple/eebe1e6cb2054bbe80d3d36783cb0e95) |
| Decision at 928 | [summary](../data/103-block928/summary.json) | [log](../logs/103-block928-terminal.log) (0:00:00.409608) | [Comet](https://www.comet.com/liblaf/apple/fe7275d948c64e308d50b8988abec4f5) |
| Continuation to 992 | [summary](../data/102-fit992/summary.json) | [log](../logs/102-fit992-terminal.log) (0:05:43.471603) | [Comet](https://www.comet.com/liblaf/apple/a702080023714874a4911fd227ed29c3) |
| Decision at 992 | [summary](../data/103-block992/summary.json) | [log](../logs/103-block992-terminal.log) (0:00:00.392728) | [Comet](https://www.comet.com/liblaf/apple/131f00cdc3ab4e429c89d6f3f6de274a) |
| Continuation to 1024 | [summary](../data/102-fit1024/summary.json) | [log](../logs/102-fit1024-terminal.log) (0:02:53.474988) | [Comet](https://www.comet.com/liblaf/apple/32cd3d1abcf84b078d134e692d3a5999) |
| Decision at 1024 | [summary](../data/103-block1024/summary.json) | [log](../logs/103-block1024-terminal.log) (0:00:00.451535) | [Comet](https://www.comet.com/liblaf/apple/4139ee44981948e78b1558e30fc9c357) |
