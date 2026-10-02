# Low-memory active-strain bulk BSR plus IPC contact check

## Method

[`active_bulk_contact.py`](../src/active_bulk_contact.py) provides
`ActiveBulkContactHessian`. It stores the full-coordinate active-strain bulk
Hessian in a reusable BSR matrix, applies it to a full displacement direction,
adds the existing `GpuContactHessian` product from an IPC Hessian precomputed
with `PSDProjectionMethod.CLAMP`, and finally restricts to the current free
DOFs. It does not create the free-coordinate FEM-plus-contact CSR union.

The final test used standard IPC and a vertex-patch filter that assigns all
fixed vertices to patch zero and every free vertex a distinct patch. Thus it
excludes fixed/fixed pairs while retaining free/free and fixed/free pairs. It
compared three deterministic free directions against `gpu_contact` at rest
with zero `S` and at a saved nonzero-`S`, deformed MouthOpen state.

## Command

```bash
CHERRIES_NAME='Low memory active bulk contact HVP validation' \
CHERRIES_TAGS='mouthopen,contact,ipc,gpu,active-strain,hessian,validation,low-memory' \
uv run python src/13-check-active-bulk-contact.py
```

The process required at least 2.5 GB free VRAM and was capped at 8% of total
GPU memory. Cherries used `Git(commit=False)` and the corrected filter run
completed at [a0974c9ff2704c448c42411a031ed837](https://www.comet.com/liblaf/apple/a0974c9ff2704c448c42411a031ed837).

## Results

The BSR topology has 3,098,894 blocks. Persistent bulk storage was
249,734,728 bytes, while the uploaded IPC Hessian used 2,380,684 bytes at rest
and 2,555,968 bytes in the deformed state. Peak allocated process memory was
about 854 MB.

The corrected run had 157,381 active IPC pairs at rest and 141,276 in the
deformed state. Maximum relative HVP discrepancy versus `gpu_contact` was
`3.520e-16` at rest and `3.541e-16` in the deformed state. The initial topology and BSR assembly
took 19.32 s; material/displacement refresh for the deformed state took 0.751
s. After preparation, per-product times were 3.37–3.72 ms, compared with
3.30–3.78 ms for `gpu_contact` in this contended run.

The BSR topology remains reusable while mesh topology and the DOF map stay
fixed. Its numeric values refresh in place after a displacement or material
change. The IPC upload is cached for a collision state and refreshed when its
Hessian changes. This makes the backend mechanically exact for the tested
operator, while its first-use and state-refresh costs must be amortized across
multiple linear iterations.

The full receipt is
[`data/13-active-bulk-contact-check-002/summary.json`](../data/13-active-bulk-contact-check-002/summary.json).
