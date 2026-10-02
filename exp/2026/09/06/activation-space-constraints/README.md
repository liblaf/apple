# Activation constraints and surface bumpiness

Controlled nonlinear tetrahedral muscle–fat experiments using the repository's
`StableNeoHookeanActive` energy and implicit adjoint. Work started on 2026-09-06.

- Design draft: [10-experiment-plan.md](docs/10-experiment-plan.md).
- Frozen initial execution scope: [12-execution-protocol.md](docs/12-execution-protocol.md).
- Forward mechanism results: [15-frequency-findings.md](docs/15-frequency-findings.md).
- Measured report: [20-results.md](docs/20-results.md), published as `site/index.html`.

The Tailnet report is served at
PRIVATE_HOST:8766 (private preview omitted), with
PRIVATE_HOST:8766 (private preview omitted) as the direct-address alternative.
The HTTP listener is bound only to the machine's Tailscale address. Tailscale
Serve HTTPS was not enabled on this tailnet; no tailnet administration or public
Funnel configuration was changed.

The user service `activation-constraints-report.service` runs the static server.
It restarts on failure and is a transient service for the current boot:

```bash
systemctl --user status activation-constraints-report.service
```

To start the same report after a reboot, from any directory:

```bash
systemd-run --user --unit=activation-constraints-report \
  --description='Activation constraints research report on tailnet' \
  --property=Restart=on-failure --property=RestartSec=3 \
  --setenv=PYTHONUNBUFFERED=1 \
  /usr/bin/python -m http.server 8766 --bind PRIVATE_HOST \
  --directory exp/2026/09/06/activation-space-constraints/site
```

All experiments run from this directory. The explicit Cherries profile records
local snapshots and Comet metrics, and disables automatic Git staging/commits.
For subsequent runs, set these Comet variables to avoid unrelated, slow
environment and whole-repository Git-patch collection:

```bash
export COMET_AUTO_LOG_GIT_METADATA=false
export COMET_AUTO_LOG_GIT_PATCH=false
export COMET_AUTO_LOG_ENV_DETAILS=false
```

The local JSON/NPZ/VTU/CSV files are the numerical evidence. Cherries 3.0.2 does
not currently upload assets through its Comet asset hook. Figures, tabular
records, and source snapshots are in the report downloads; full NPZ fields and
VTU mesh checkpoints remain in the local data directories.

Use the repository interpreter without changing the environment:

```bash
APPLE_PYTHON=.venv/bin/python
CHERRIES_NAME='Activation forward frequency probe' \
  CHERRIES_TAGS='activation-priors,3d-fem,forward,frequency' \
  "$APPLE_PYTHON" src/10-forward-frequency-probe.py --output-dir data/10-frequency
CHERRIES_NAME='Activation priors 33-fit nonlinear tetrahedral screen' \
  CHERRIES_TAGS='activation-priors,3d-fem,inverse,screen,noise-002' \
  "$APPLE_PYTHON" src/20-inverse-constraint-matrix.py \
  --nx 24 --ny 10 --steps 160 --noise-rms 0.02 --output-dir data/20-matrix
CHERRIES_NAME='Activation prior strength and held-out robustness checks' \
  CHERRIES_TAGS='activation-priors,3d-fem,frontier,heldout,fiber-sensitivity' \
  "$APPLE_PYTHON" src/30-run-followups.py --steps 240 --output-dir data/30-followups
CHERRIES_NAME='Smoothing-only fiber held-out checks' \
  CHERRIES_TAGS='activation-priors,3d-fem,heldout,smoothing-only' \
  "$APPLE_PYTHON" src/32-smoothing-only-holdout.py --steps 240 \
  --output-dir data/32-smoothing-holdout
CHERRIES_NAME='Activation endpoint numerical polish' \
  CHERRIES_TAGS='activation-priors,3d-fem,stationarity,gradient-audit' \
  "$APPLE_PYTHON" src/35-polish-endpoints.py --steps 100 --output-dir data/35-polish
CHERRIES_NAME='Activation gradient finite-difference scale audit' \
  CHERRIES_TAGS='activation-priors,3d-fem,gradient-audit,finite-difference' \
  "$APPLE_PYTHON" src/37-gradient-scale-audit.py --output-dir data/37-gradient-scale
CHERRIES_NAME='Fiber-direction endpoint gradient scale audit' \
  CHERRIES_TAGS='activation-priors,3d-fem,gradient-audit,finite-difference' \
  "$APPLE_PYTHON" src/37-gradient-scale-audit.py --endpoints fiber-10deg/F-MS \
  --output-dir data/37-gradient-scale-fiber10
CHERRIES_NAME='Activation refined-forward discretization check' \
  CHERRIES_TAGS='activation-priors,3d-fem,refinement,heldout' \
  "$APPLE_PYTHON" src/40-refinement-check.py --steps 240 --output-dir data/40-refinement
"$APPLE_PYTHON" src/45-audit-surfaces.py --include-checkpoints
CHERRIES_NAME='Activation constraint measured analysis' \
  CHERRIES_TAGS='activation-priors,analysis,report' \
  "$APPLE_PYTHON" src/50-analyze-results.py --output-dir data/50-analysis
"$APPLE_PYTHON" src/60-publish-report.py
```

These commands document execution order and parameters. Preserve existing run
folders before repeating the numerical work; the smoothing-only runner refuses
to overwrite its endpoints. It reuses the exact held-out fixtures from step 30.
Step 35 starts from saved step-30 controls. Step 40 verifies the archived
step-20 source hashes and transfers coarse tetrahedral controls to nested fine
tetrahedra. The publisher requires the completed measured Markdown report and
validates all local figure and download links before replacing the site.
