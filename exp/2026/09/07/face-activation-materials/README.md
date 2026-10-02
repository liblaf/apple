# Face activation and material experiments

Read [the result report](docs/30-face-results.md) or open the private 3D report (private preview omitted). The completed block study is a separate debugging case at port 8766.

## Environment and inputs

All commands below run from this directory, in the existing Apple checkout. The recorded environment is [data/environment.json](data/environment.json); copies and hashes of the actual Apple and installed Peach runtime sources are under `data/runtime-sources/`. The runs used the existing `.venv` with Python 3.14, Torch 2.12, Warp 1.14, and an RTX 4090. Do not infer package sources from another repository with the same name.

```bash
cd exp/2026/09/07/face-activation-materials
FACE_PYTHON=.venv/bin/python
export COMET_AUTO_LOG_GIT_METADATA=false
export COMET_AUTO_LOG_GIT_PATCH=false
export COMET_AUTO_LOG_ENV_DETAILS=false
```

Normal numbered stages use `ProfileCometNoCommit`: metrics go to Comet, local evidence is retained, and Git auto-commit is disabled. `DEBUG=1` selects local debugging only. The installed Cherries asset-upload method does not upload these mesh files; local artifacts are the evidence. Scripts reject nonempty output directories. Use new output names for reruns and preserve the published inputs.

The fixture is `data/10-fixture/{volume.vtu,skin.vtp,summary.json}`. It depends on three pinned upstream files, whose SHA-256 checks are enforced in `src/face_fixture.py`:

- `exp/2026/06/17/human-face-smile-prestrain-v2/data/10-human-face-prepared.vtu`
- `exp/2026/08/17/human-face-smile-material-heuristic-sweep/data/10-material-candidates/skin-e100-p000.vtp`
- `exp/2026/08/18/human-face-smile-plane-stress-skin/data/10-corrected-baseline/skin-isface-e0200-p000.vtp`

These paths are relative to the Apple repository. `10-prepare-face.py` builds the fixture; `12-prepare-controls.py` builds the target-independent four-control Gaussian partition. They can be rerun with new explicit output paths. The fixture reconstruction receipt distinguishes byte-level serialization from verified semantic equality. Use the existing archived fixture when reproducing the recorded runs.

The downloadable `records/reproducibility.zip` contains current scripts, frozen run sources, configuration and solver records, audit records, environment receipts, and runtime source copies. It deliberately omits large original meshes and NPZ/VTU states. It is a reproduction recipe and evidence archive, not a self-contained input dataset. The full fields remain in this local experiment directory. The web viewer separately includes the saved boundary geometry needed for visual inspection.

## Repeat the primary inverse screen

For example, reproduce the FiberModes settings with a new output directory:

```bash
CHERRIES_NAME='Face FiberModes reproduction' \
CHERRIES_TAGS='face,inverse,reproduction,fat-nu049' \
"$FACE_PYTHON" src/20-run-face-inverse.py \
  --method FiberModes --steps 40 --fat-nu 0.49 \
  --gradient-audit true --forward-rtol 1e-5 --forward-atol 1e-12 \
  --adjoint-rtol 1e-7 --output-dir data/reproduction-fiber-modes
```

Change the method to `Region5Modes` or `Raw6` and choose a different output directory for the other zero-initialization cases. Exact effective settings, including priors, bounds, target mask, and material coefficients, are in each run's `config.json`, `provenance.json`, and `summary.json`. Raw6's recorded run was deliberately stopped after accepted step 8; rerunning the 40-step command is not a reproduction of that stopping decision. Its stop receipt, immutable checkpoint, and CPU export are retained separately.

The learned-axis experiment uses the same strict forward solver through its isolated runner. It has a nonzero initialization, 99 free parameters and 105 stored scalar slots. Before running it, ensure `data/26-latent-region-initial-v2.npz` exists. If absent, generate it with `"$FACE_PYTHON" src/26-run-latent-region.py --prepare-initial`; preparation refuses to overwrite an existing initializer. Then check the initializer and parameterization with `"$FACE_PYTHON" src/26-run-latent-region.py --cpu-validate`.

```bash
CHERRIES_NAME='Face learned regional axes reproduction' \
CHERRIES_TAGS='face,inverse,reproduction,latent-axes' \
"$FACE_PYTHON" src/26-run-latent-region.py \
  --steps 40 --gradient-audit true \
  --forward-rtol 1e-5 --forward-atol 1e-12 --adjoint-rtol 1e-7 \
  --output-dir data/reproduction-latent-region
```

`13-prepare-latent-controls.py` and `learned_fiber_models.py` are a rejected CPU preparation, not the executed learned-axis inverse model.

## Fixed-control materials and validation

`22-replay-fat-and-depressor.py` consumes the saved 35-scalar endpoint and changes one factor at a time. `29-fat0499-activation-ramp.py` consumes the saved FiberModes controls and increases them from zero in ten equal increments at fat ν = 0.499. Neither script refits activation. The ramp is quasistatic continuation, not physical time. Its exact zero state is analytical; every nonzero stage must pass the strict solver. Failed runs and the failed direct rest reset at ν = 0.499 remain explicit records.

`40-audit-face-results.py` recomputes volume ratios, principal stretches, static intersections, and intrinsic surface high-pass diagnostics from saved meshes. `47-audit-raw6-activation.py` checks the exact raw activation tensors. `45-summarize-comparisons.py` collects explicitly listed endpoints and audits, verifies source and field hashes, and distinguishes inverse termination from fixed-control replay and diagnostic export. The report links its final JSON and CSV outputs.

Validation limits are substantive: the determinant rejection floor is not a physiological threshold; the projected-gradient diagnostic covers activation bounds only; and static intersection audits do not establish collision-free trajectories. Preserve the actual solver tolerance and initialization when comparing results. Some GPU jobs overlapped, so recorded wall times are not a fair performance comparison.

## Rebuild and serve the report

The final viewer manifest is `docs/55-main-activation-comparisons.json`. `55-render-main-activation-comparisons.py` renders saved states and creates lazy local geometry bundles. The final bundle is `data/75-final-face-comparisons-caption-legend`. The publisher copies existing artifacts and does not invoke physics:

```bash
"$FACE_PYTHON" src/60-publish-face-report.py \
  --viewer-bundle data/75-final-face-comparisons-caption-legend \
  --report docs/30-face-results.md --site site
```

The report is served by the current-boot user unit `face-activation-report.service`, bound only to the Tailscale address `PRIVATE_HOST:8767`. It has no public Funnel or Tailscale Serve exposure. To create that service after a reboot, when the named unit is absent:

```bash
systemd-run --user --unit=face-activation-report \
  --property=Restart=on-failure \
  /usr/bin/python -m http.server 8767 --bind PRIVATE_HOST \
  --directory exp/2026/09/07/face-activation-materials/site
```

To inspect the existing service, use `systemctl --user status face-activation-report.service`. Port 8766 belongs to the previous block report and is independent.

## Provenance note

One plotting stage initially used Cherries' default Git plugin and created an unintended local commit. The repository was restored to its original HEAD, with all experiment files preserved and unstaged; no push occurred. `data/git-hook-recovery.json` records the exact recovery. Receipts captured while that temporary commit existed retain the original observed hash rather than being rewritten. The plotting stage was then rerun with the no-commit profile. Current production code was not changed by these experiments.
