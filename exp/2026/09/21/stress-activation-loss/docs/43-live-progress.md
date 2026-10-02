# Live Smile fit progress through the tailnet

The read-only live viewer is available at **PRIVATE_PREVIEW_URL while the transient service is running. It serves the active `41-l2-unrestricted-active-strain-lr0p05-nosmooth-001` fit, with an interactive fitted surface, target comparison, target overlay, a fixed 0–10 mm position-error display, loss components, and position RMS. Status refreshes every five seconds; surface geometry reloads only after a new checkpoint.

The page reports the saved shape step separately from the current inner Newton progress. The current curves begin at step zero of the fresh active-strain learning-rate-0.05, zero-smoothness run. They do not include the previous trajectory. The viewer also supports separate ancestry segments when a run is explicitly resumed. Approximate-solve markers and inversion counts are diagnostic only. Update/skip/approximate counts refer to the current process. The no-skin fit and its Adam state are unchanged by the viewer.

The remaining-time card estimates time to the attempt budget and displays the expected finish in the browser's local time. Its pace uses the latest ten timing intervals from the current process, excluding initialization and parent runs. Step gaps count skipped attempts, including trailing skips recorded in the summary. The estimate subtracts time already spent in the current attempt; when that attempt exceeds the recent pace, the card reports the overrun and the expected finish moves later. This is a rolling estimate, not a convergence prediction, and excludes final post-fit exports or diagnostics. It refreshes every five seconds with the status. Reload a page opened before this addition to load the new card.

## Runtime

The server is bound only to the Tailscale address `PRIVATE_HOST`, on the previously unused port 8774. It exposes explicit page, JavaScript and JSON API routes, without a directory listing or arbitrary filesystem access. Existing preview ports and Tailscale Serve configuration are unchanged. Geometry extraction uses NumPy on the CPU; all interactive rendering runs in the visiting browser.

This is a transient user service under `/run/user/1000/systemd/transient/`, with a 24-hour runtime limit. No persistent unit was installed or enabled; it does not survive reboot. Stop and clean it up with:

```sh
systemctl --user stop apple-smile-live-20260923.service
```

Launch command from the experiment directory:

```sh
systemd-run --user --unit=apple-smile-live-20260923 --collect \
  --property=RuntimeMaxSec=24h \
  --property=WorkingDirectory=exp/2026/09/21/stress-activation-loss \
  .venv/bin/python src/43-serve-live.py \
  --source data/41-l2-unrestricted-active-strain-lr0p05-nosmooth-001 \
  --console tmp/41-fit-l2-unrestricted-active-strain-lr0p05-nosmooth-001.console.log \
  --fit-pid 3235943 --bind PRIVATE_HOST --port 8774
```

The PID is for this specific active fit; update it when serving a different process. The viewer checks both PID and command identity before reporting that the fit is alive. Its logs are available through `journalctl --user -u apple-smile-live-20260923.service`.

## Verification and provenance

- HTTP requests through the tailnet address return the page, local JavaScript, status, topology, target and current fit data.
- During the preceding resumed run, API fitted vertices exactly matched the saved checkpoint displacement plus rest position at step 13. Target errors and triangle connectivity matched the source arrays. The surface has 15,299 vertices and 29,899 triangles.
- Live status was previously verified to advance from step 12 to step 13 without restarting the viewer. The preceding fresh stress run was verified through step 5: its history starts at zero, contains one segment, reports LR 0.05 and zero smoothness, and excludes the earlier trajectory. Checkpoint/summary consistency is checked before publishing a new shape.
- The extracted JavaScript module passes `node --check`; both SVG chart functions were evaluated with actual live data and produced finite coordinates and separate parent/resume segments. Ruff passes for the server.
- ETA verification: six CPU tests cover recent timing, resumed clocks, skipped attempts, current-attempt age, an overdue final attempt, completion, stopped processes, and insufficient history. JavaScript rendering checks cover live API data and every ETA display state. During the preceding stress run, at step 47/200, the API reported about 105 minutes remaining at 41.2 seconds per attempt; timing will change as solve cost changes. Only the viewer service was restarted; the fit retained PID 3160064 and its original LR 0.05 / zero-smoothness settings.
- Requests for repository files and unexposed data paths return 404.
- The browser tab opened with the correct page title, but the browser inspection tool repeatedly timed out while binding the tab. A visual screenshot check could not be completed in this session.

The UI uses the existing local Three.js files and camera presets from `exp/2026/09/07/tensor-active-stress/data/107-learning-rate-render/`. It needs no external CDN. The viewer is a serving helper, not a new solver experiment; the underlying run's [Comet record](https://www.comet.com/liblaf/apple/2bbacc689f894f6a9c8acb9b78e2f4d2), checkpoints and [fit report](41-l2-unrestricted-active-strain-lr0p05-nosmooth.md) retain the experiment provenance.

The current viewer follows the transient fit service `apple-smile-fit-20260923.service`. At the user's request, it now shows a fresh active-strain run from identity activation with Adam learning rate 0.05 and smoothness weight 0. Both determinant terms use physical det(F); activation enters only the norm term. The previous stress trajectory remains preserved separately. The current fit PID is 3235943. Live API verification confirmed the strain model, LR 0.05, zero smoothness, a new history starting at step zero, and an advancing first Adam update.
