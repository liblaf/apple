# Why the 20-update Smile fit moves little

The saved endpoints move only about **0.49 mm area-weighted skin RMS** from their actual initial equilibrium, against **5.1373 mm** target motion. The main demonstrated optimization issue is overwhelming spatial-smoothness gradients after the first Adam update. This diagnosis uses saved checkpoints and CPU algebra only; it performs no new forward solve or adjoint and changes no fitting settings.

## Evidence

The inherited objective is normalized skin-position error plus strong graph stress smoothness. Magnitude and jaw weights are zero. The retained smoothness coefficient is `5.066584049455902`.

| Hybrid stage | Target RMS, mm | Weighted smoothness | Data loss |
| --- | ---: | ---: | ---: |
| Initial | 5.137331 | 0 | 1.000003 |
| Update 1 | 4.888141 | 18.834801 | 0.905344 |
| Update 2 | 4.890703 | 182.791431 | 0.906294 |
| Update 20 | 4.920538 | 12.087828 | 0.917384 |

At zero stress the smoothness gradient is exactly zero. The first full Adam-0.3 step improves positional fit while introducing large spatial stress differences. Later updates mostly reduce that roughness and give back part of the fit improvement. Both arms' best positional RMS occurs at update 1.

Scalar loss values alone do not establish gradient dominance. The CPU audit recomputed the exact weighted quadratic smoothness gradient and subtracted it from the saved total gradient; magnitude and jaw penalties are zero. At the final hybrid checkpoint, smoothness-gradient coordinate RMS is `4.38823e-4`, versus `1.02445e-6` for the data term: about **428 times larger**. The original arm gives about **429 times**. Their gradient cosine is about `0.0015`, so these are almost orthogonal directions. Volume-dual gradient norms also show dominance (hybrid `290.21` versus `0.5746`). This is a raw-gradient comparison, not a direct decomposition of Adam's history-dependent preconditioned step.

The hybrid's motion from initial equilibrium is `0.488481 mm` weighted RMS, maximum `1.405947 mm`; its projection along the full target displacement is only `4.583%` of target amplitude. The motion-target cosine is `0.48199`. Every postinitial forward evaluation performed iterations, so this is not an unchanged warm start repeatedly returned by the force threshold. These checks do not certify tight-equilibrium shape or gradient accuracy.

The jaw remains at the `0°` lower bound with a positive objective derivative (about `0.1439` per normalized angle at the hybrid endpoint). Its local descent direction lies outside the permitted `0–40°` interval, so projection gives no jaw movement. Removing the jaw prior does not remove this physical bound. Only 12–13 of 864,705 stress eigenvalues reach the upper cap, so that cap is not broadly saturated.

## Interpretation and next diagnostic

The combination of strong inherited smoothness and full Adam-0.3 updates makes the short trajectory largely a stress-smoothing problem after its initial fit improvement. The two forward solvers return similar shapes because they solve the same physical model driven by very similar stress trajectories. Neither inverse fit is stationary.

The next controlled diagnostic would use zero smoothness for a short run, keeping Adam 0.3, full steps, contact, stress bounds, inputs and tolerances unchanged. That would isolate the regularizer's influence on target fitting. It has **not** been run, and this audit does not establish whether the anatomical model can reproduce the target smile fully.

## Reproduction and receipts

From this experiment directory:

```bash
DEBUG=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 \
CHERRIES_NAME='Smile saved endpoint motion diagnosis' \
CHERRIES_TAGS=smile,diagnosis,postprocess \
.venv/bin/python src/27-diagnose-smile-motion.py
```

The local Cherries run completed successfully; DEBUG disabled remote Comet recording. See [motion receipt](../data/smile-shape-diagnosis-001/motion-audit.json), [gradient receipt](../data/smile-shape-diagnosis-001/gradient-audit.json), [terminal log](../data/smile-shape-motion-diagnosis-003.stdout.log), and [completed comparison](22-smile-fit-results.md). Input and checkpoint hashes bind the measurements to the completed run. The checkout has uncommitted research changes.
