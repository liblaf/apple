# Fixed-reference GPU wiring check

The extended fixed-reference model runs with the core GPU contact Hessian.
This bounded check does not run a continuation.

Command:

```bash
CHERRIES_NAME='Fixed reference extended GPU wiring' \
CHERRIES_TAGS='mouthopen,contact,fixed-reference,gpu,hvp' \
uv run python src/53-check-fixed-reference-gpu.py \
  --output 53-fixed-reference-gpu-check
```

The receipt is [summary.json](../data/53-fixed-reference-gpu-check/summary.json).
Comet recorded the successful run as
[90c7c914af6a4ba9b16bc371eb5052b7](https://www.comet.com/liblaf/apple/90c7c914af6a4ba9b16bc371eb5052b7).

The model has 256,249 nodes after appending the rigid sources. Its 604,872
physical free DOFs are unchanged, and its 85,047 appended rigid coordinates
are fixed. At the repaired rest reference, total free force is
`2.51e-21 MPa m²`, which is floating-point roundoff; contact is empty and has
no scoped intersections.

At 0.002 of the saved MouthOpen pose, CCD accepted the complete chord. The
state has 105 active IPC stencils, minimum active separation `77.04 µm`, and
no scoped intersections. The core `gpu_contact` HVP agreed with the core
matrix-free HVP at relative error `7.93e-17`.

Peak allocation was 850,975,232 bytes while another GPU job occupied most of
the card. This verifies extended Warp material indexing and the GPU contact
operator for this state; it does not validate later continuation states.
