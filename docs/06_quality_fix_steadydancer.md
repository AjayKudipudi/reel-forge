# 06 — SteadyDancer Output Quality Fix

This document records the diagnosis and remediation of two visible defects
in early end-to-end outputs from the pipeline. It supersedes the
implementation-time plan and checklist; everything below describes
**code as it ships today** and is supported by the references in
`docs/01`–`05`.

| Defect | Status |
|---|---|
| Raw model output played in slow-motion / "stuck" | **Fixed** |
| Generated hands looked like white blobs / melted geometry | **Fixed** |
| Folded fingers (fists, gripping) lack micro-detail | **Bounded, documented** — at L40S 48 GB resolution ceiling |

---

## 1. Defect: slow-motion / "stuck" raw output

### Symptom

The dance in `animated.mp4` (raw 16 fps generator output, before any
post-processing) played at roughly half the speed of the source reel
choreography. Consecutive frames looked nearly identical — the
"stuck-y" feel.

### Root cause

SteadyDancer outputs at a fixed 16 fps (`wan/configs/shared_config.py`
`sample_fps=16`) and consumes exactly 81 pose JPGs per inference. The
pipeline was passing the driving reel to `pose_align.py` at its native
30 fps (typical Instagram Reels capture rate). The resulting
`pose/0000.jpg` … `pose/0080.jpg` covered only `81 / 30 ≈ 2.7` real
seconds of source motion — which the model then played back over
`81 / 16 = 5.06` seconds, a ~53% playback speed.

Upstream demos hid this assumption: their training data (X-Dance) and
demo clips are 16 fps natively, so 81 source frames ≈ 5 real seconds
by construction. The 16-fps expectation is implicit, not documented.

### Fix

`DwPoseExtractor.extract_aligned` (`reel_forge/ec2/models/dwpose.py`)
now resamples the driving video to 16 fps before invoking either
pose-extraction script:

```python
normalized = work_dir / "driving_16fps.mp4"
run_tool(FFMPEG, [
    "-hide_banner", "-loglevel", "error",
    "-y", "-i", str(driving_video),
    "-r", "16",
    "-vsync", "cfr",   # not -fps_mode cfr; AMI ships ffmpeg 4.x
    "-an",             # audio_attach reads from the ORIGINAL reference.mp4
    str(normalized),
])
driving_video = normalized
```

`-vsync cfr` forces constant frame rate via frame drop/duplicate; this
is the ffmpeg 4.x-compatible spelling of `-fps_mode cfr`. The AMI ships
ffmpeg 4.x — `-fps_mode` is a 5.0+ flag and was rejected mid-pipeline
during initial integration.

### Result

| Source fps | Frames in 81-JPG window | Output duration | Playback speed |
|---|---|---|---|
| Before — 30 fps native | 2.7 s of motion | 5.06 s | ~53% slow-mo |
| After — resampled to 16 fps | 5.06 s of motion | 5.06 s | 100% real-time |

Audio sync is unaffected: the `audio_attach` phase reads from the
original `reference.mp4` (`reel_forge/ec2/phases/audio_attach.py:32`),
not from the audio-stripped `driving_16fps.mp4`.

### RIFE target now derives from source fps

Companion change in `reel_forge/ec2/phases/interp.py`: the post-model
RIFE interpolator used to target a hardcoded 60 fps regardless of
source. That meant 4× interpolation distance for any 30-fps source,
which is exactly where RIFE hallucinates on fast hand motion.

`_detect_target_fps` now returns `min(max(src_fps, 16), 60)`:

| Source fps | RIFE target | RIFE multi | Notes |
|---|---|---|---|
| < 16 | 16 | passthrough | model output already 16 fps; just copy |
| 24 | 24 | 2 | RIFE produces 32, --fps stamps 24 |
| 30 | 30 | 2 | typical Instagram |
| 60 | 60 | 4 | |
| > 60 | 60 | 4 | clamp ceiling |

Lower multipliers reduce RIFE artifact rate on fast motion.

---

## 2. Defect: hand artifacts on fast motion

### Symptom

Hands in the generated video looked melted, fingers were missing, or
the entire hand rendered as a white blob during the first ~2 seconds
and last ~1 second of output. Mid-video (~2-4 s) hands rendered
cleanly.

### Diagnostic process

The artifact zones correlated with intro/outro fast hand gestures, but
that left two possible causes:

| Hypothesis | Where to fix |
|---|---|
| DWPose mis-tracks hand keypoints during fast motion | upstream pose extraction (smoothing, MediaPipe Hands refinement) |
| DWPose is fine, model under-uses the clean pose signal in late denoising | model conditioning (CFG / DC-CFG window) |

To decide between them, `reel_forge/ec2/phases/pose_extract.py` now
uploads `pose_overlay.mp4` (the skeleton drawn on top of the source
frames) to `s3://<bucket>/<prefix>/<job>/_runtime-logs/pose_overlay.mp4`
after pose extraction. The aligned-pose video that conditions the
model was already produced; only the upload was missing.

Inspecting the overlay at the artifact timestamps confirmed:
**DWPose tracks hands cleanly across all zones, including the
fast-motion frames.** The skeleton's hand region shows distinct finger
keypoints at 1.0 s, 2.5 s, and 4.5 s with comparable quality.

Conclusion: the model wasn't using the clean pose signal effectively
during the detail-emergence phase of denoising. The fix lives in
conditioning, not extraction.

### Fix

Three configuration adjustments applied in
`reel_forge/ec2/models/steadydancer.py` (both the single-chunk
`animate()` CLI args and the chunked daemon spec):

| Parameter | Was | Now | Upstream default | Rationale |
|---|---|---|---|---|
| `condition_guide_scale` | 1.0 | 1.5 | argparse 1.5 (README example 1.0) | Pose track verified clean — safe to amplify pose signal. Matches upstream argparse default. |
| `end_cond_cfg` | 0.4 | 0.6 | argparse 0.4 / `__init__` docstring 0.5 | Keeps pose-aware suppression active through 60% of the timeline (steps 4–24 of 40), covering the detail-emergence phase. Within upstream design intent. |
| `sample_steps` | 40 | 50 | i2v default 40 | 25% more denoising compute for fine details. Costs ~25% more wall-clock per chunk. |

### Result

Progression observed across config iterations on the same source pair:

| Config (cond_scale / end_cond_cfg / steps) | Open-hand quality | Folded-finger quality |
|---|---|---|
| 1.0 / 0.4 / 40 (paper recipe) | white blobs ❌ | n/a |
| 1.0 / 0.6 / 40 | recognizable ✓ | fuzzy |
| 1.3 / 0.6 / 40 | clearer ✓ | partially melted |
| **1.5 / 0.6 / 50 (ship)** | **sharp ✓** | melted on close-up |

Open-hand renderings (extended fingers, pointing, waving) now look
correct. Folded fingers (fists, gripping motions) still lose
micro-detail — see §4.

---

## 3. Operational fix: timeout alignment

Discovered while debugging the OOM in §4: `GENERATE_DANCER_TOOL.timeout_s`
was 5400 s (90 min) while its parent `AnimatePhase.timeout_s` was 12600 s
(3.5 h). On a cold spot without FSR, weight reads from EBS are
throttled to 7–11 MB/s, which pushed total `generate_dancer.py`
runtime to 89 min — over the inner-tool timeout but well under the
phase budget. The inner timeout fired ~47 s after the model logged
`Finished.` and successfully wrote `animated.mp4` to local disk —
SIGKILLing the process before the orchestrator could upload to S3.
The mp4 was lost with the spot's EBS.

Fix: bumped `GENERATE_DANCER_TOOL.timeout_s` to 12600 to match the
phase wrapper. The phase wrapper is now authoritative.

The general invariant: **the inner subprocess timeout must be at
least as large as the outer phase timeout.** Otherwise the inner
timeout pre-empts the phase wrapper and the orchestrator never gets
the chance to capture the output file. See comments in
`reel_forge/ec2/models/steadydancer.py` for the full context.

---

## 4. Known limitation: folded-finger micro-detail

### What we tried

The defect from §2 has a residual: when fingers are folded into a fist
or gripping pose, the hand silhouette is correct but individual
fingers blur into a smooth knob. To address this we tried generating
at the next available supported resolution: `--size 720*1280` (+56 %
pixel area vs `576*1024`).

Result: **CUDA OOM at sampling step 0 on L40S 48 GB.**

```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 3.16 GiB. GPU has 44.39 GiB total, 1.30 GiB free.
Process is using 43.09 GiB.
```

L40S's nominal 48 GB has only ~44.4 GiB usable after CUDA driver
reservation. At `720*1280`, peak DiT activations exceed the budget.
`offload_model=True` already pushes T5 + CLIP to CPU after the
encoding step; the DiT itself cannot be further offloaded without
upstream changes.

### Bound

**`576*1024` is the practical resolution ceiling on L40S 48 GB** at our
current config (steps=50, cond_scale=1.5, end_cond_cfg=0.6,
frame_num=81). Hand regions render at ~30–50 px wide; folded fingers
at ~5 px each — at the model's resolution floor.

### Escape hatches (not implemented)

1. **Larger GPU (H100 80 GB / H200 141 GB)** — would allow `720*1280`
   and probably `1024*800` portrait, giving fingers ~40 % more pixels
   each.
2. **Hand-specific post-process step** — e.g., MediaPipe Hands +
   per-frame hand-region inpainting / refinement. Adds a new pipeline
   stage and a new ML model dependency.
3. **Source-clip selection** — choreographies without prolonged
   folded-finger close-ups avoid the artifact entirely.

Spot reclamations are a separate concern (AWS capacity pressure, not
quality) — the orchestrator's marker-based resume handles them
gracefully; no fix required.

---

## 5. Final config (as shipped)

| Parameter | Value | Where |
|---|---|---|
| Driving video fps | **resampled to 16** | `ec2/models/dwpose.py:extract_aligned` |
| `--size` | `576*1024` portrait | `ec2/models/steadydancer.py` |
| `--frame_num` | 81 (= 5.06 s @ 16 fps) | `core/manifest.py` |
| `--sample_steps` | **50** | `ec2/models/steadydancer.py` |
| `--sample_shift` | 5.0 | upstream i2v default |
| `--sample_solver` | unipc | upstream i2v default |
| `--sample_guide_scale` | 5.0 | paper recipe |
| `--condition_guide_scale` | **1.5** | upstream argparse default |
| `--st_cond_cfg` | 0.1 | paper recipe |
| `--end_cond_cfg` | **0.6** | extended from paper's 0.4 |
| `--offload_model` | True | T5 + CLIP → CPU after encode |
| RIFE target fps | **`min(max(source_fps, 16), 60)`** | `ec2/phases/interp.py` |
| `GENERATE_DANCER_TOOL.timeout_s` | **12600** | `ec2/models/steadydancer.py` |
| `pose_overlay.mp4` uploaded to S3 | yes | `ec2/phases/pose_extract.py` |

Items in **bold** were changed in this fix; the rest are unchanged
from the paper recipe / upstream defaults.

---

## 6. Files changed

| File | What |
|---|---|
| `reel_forge/ec2/models/dwpose.py` | Driving-video 16 fps resample at start of `extract_aligned()`; ffmpeg-4.x-compatible `-vsync cfr` |
| `reel_forge/ec2/models/steadydancer.py` | `--sample_steps`, `--condition_guide_scale`, `--end_cond_cfg` CLI flags (single-chunk path) and matching spec values (chunked daemon path). `GENERATE_DANCER_TOOL.timeout_s` raised to 12600 |
| `reel_forge/ec2/phases/interp.py` | `_detect_target_fps` clamps to [16, 60]; `_interp_one` passthrough when target ≤ 16 |
| `reel_forge/ec2/phases/pose_extract.py` | Upload `pose_overlay.mp4` to S3 `_runtime-logs/` for diagnostic visibility |
| `docs/01_architecture.md` | New §4a "Driving video FPS normalization"; §4 RIFE notes updated for source-fps tracking |
| `docs/02_settings_audit.md` | Driving-video fps row; RIFE target row updated |
| `docs/03_findings_and_limitations.md` | §1.7 slow-motion finding; §1.8 RIFE target finding; OOM result on `720*1280` |
| `docs/04_bugs_encountered.md` | Bug 48 (driving-video fps mismatch → slow motion) |

---

## 7. Verification

Smoke tests (local, no GPU):

```bash
pytest tests/                              # 133 pass
mypy reel_forge/                           # clean
ruff check reel_forge/                     # clean
```

End-to-end on a 30 fps Instagram-style source reel:

| Check | Pass criterion |
|---|---|
| `animated.mp4` from a fresh GPU run | 576×1024, 16 fps CFR, 81 frames, **5.06 s duration** (real-time, not slow-motion) |
| `final.mp4` after the full pipeline | 1080×1920, fps matches source reel fps clamped to [16, 60], dance plays at source choreography speed |
| `s3://.../<job>/_runtime-logs/pose_overlay.mp4` | exists; visual inspection shows clean DWPose tracking on hands |
| Open-hand frames in `final.mp4` | hands recognizable with finger structure visible |
| Folded-finger close-ups | hands recognizable; finger micro-detail limited (documented in §4) |

---

## 8. References

- Paper: arXiv 2511.19320 — *SteadyDancer: Harmonized and Coherent Human
  Image Animation with First-Frame Preservation*
- Upstream code: [MCG-NJU/SteadyDancer](https://github.com/MCG-NJU/SteadyDancer)
- Training dataset card: [MCG-NJU/X-Dance on Hugging Face](https://huggingface.co/datasets/MCG-NJU/X-Dance)
- Base model: [Wan-AI/Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)
- DWPose: [yzd-v/DWPose](https://github.com/IDEA-Research/DWPose) — 133-keypoint whole-body pose estimator
- Practical-RIFE: [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE) — RIFE v4.26 frame interpolation
