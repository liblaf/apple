# MouthOpen to Smile activation animation

The [completed animation](../data/96-mouthopen-to-smile-animation/mouthopen-to-smile-transition.mp4) plays the 121 previously equilibrated transition states from MouthOpen to Smile. It retains the full tetmesh boundary rendering, prescribed jaw motion, common camera, principal activation glyphs, and one-second endpoint holds. The title and counter now show MouthOpen to Smile and Smile blend from zero to one.

This is reverse playback of the existing continuation, with no new forward solves or independent reverse continuation. Its tensor path is `S(beta) = (1-beta) S_MouthOpen + beta S_Smile`. The original numerical results and mechanical limitations, including inverted cells and surface intersections, apply unchanged; see the [numerical report](94-activation-transition-results.md).

The H.264 MP4 is 1920 by 1080 pixels, 30 fps, 179 frames, and 5.966667 seconds long. It uses yuv420p and faststart. Source hashes, frame mapping, and output hashes are recorded in the [manifest](../data/96-mouthopen-to-smile-animation/manifest.json).

FFprobe metadata checks and full-stream FFmpeg decoding passed. The decoded first and last frames were visually inspected and show the correct expressions and labels. First, midpoint, and last decoded frames match the corresponding original frames in reverse order, excluding the rewritten header, with PSNR values of 45.40, 43.55, and 44.57 dB. Differences arise from video re-encoding. The [verification receipt](../tmp/96-video-qa/verification.json) records these checks.

The [Cherries run](https://www.comet.com/liblaf/apple/4b3a1197d8f144c4ac9247e7e9e8e045) completed successfully, including shutdown. See the [terminal log](../logs/96-reverse-animation-terminal.log). No Git commits or pushes were made.

Run from this experiment directory:

```bash
CHERRIES_NAME='MouthOpen to Smile activation animation' \
CHERRIES_TAGS='mouthopen,smile,activation-transition,reverse,render' \
.venv/bin/python -u \
  src/96-reverse-activation-animation.py
```
