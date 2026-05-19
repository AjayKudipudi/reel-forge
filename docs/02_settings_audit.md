# 02 — Settings Audit

Every generation, post-process, and pipeline setting documented with
its source authority. Settings without a primary source citation are
flagged.

---

## 1. Generation (SteadyDancer-14B)

### 1.1 Classifier-free guidance scales

| Setting | Value | Authority |
|---|---|---|
| `sample_guide_scale` (text CFG, w^txt) | **5.0** | Paper arxiv 2511.19320 §Implementation Details: `w^txt = 5.0` |
| `condition_guide_scale` (pose CFG, w^pose) | **1.0** | Paper §Implementation Details: `w^pose = 1.0`. README example matches |
| DC-CFG start (`st_cond_cfg`) | **0.1** | Paper DC-CFG window `[0.1, 0.4]` |
| DC-CFG end (`end_cond_cfg`) | **0.4** | Paper DC-CFG window `[0.1, 0.4]` |

A short experiment raised `sample_guide_scale` from 5.0 to 6.0 to
amplify identity / hand / no-flower negative tokens. The result was
classical CFG over-satisfaction: extra hands when "complete hands" was
enumerated in the prompt, deformed fingers on fast motion. Wan2.1's
README documents that `guide_scale = 6.0` applies only to T2V-1.3B,
not to I2V-14B; SteadyDancer derives from I2V-14B. The setting was
reverted to 5.0.

`condition_guide_scale` was previously set to 1.5 (argparse default
from upstream); the README inference example explicitly overrides it
to 1.0 and raising it amplified pose-detector noise as extra-people
artefacts.

### 1.2 Sampling

| Setting | Value | Authority |
|---|---|---|
| `sample_steps` | **40** | Wan I2V-14B `_validate_args` default. Reducing to 25-30 would trade quality for ~25-37% wall savings; deferred until visual A/B available |
| `sample_shift` | **5.0** | Upstream default for non-480p sizes. `sample_shift=3.0` applies only to 832x480 / 480x832 |
| `frame_num` (per chunk) | **81** | Wan2.1 native; paper author confirmation in MCG-NJU/SteadyDancer issue #17 ("our primary goal is currently to generate 81 frames") |
| `size` | **576x1024** (portrait) | Wan2.1-I2V-14B-480P trained at ~590K pixel area. 576x1024 portrait = 589,824 px (same area as the README's `1024x576` landscape, just rotated). Reels-native aspect |

The model is area-optimised, not resolution-optimised. Generating at
720x1280 (921,600 px, +56%) exceeds the training distribution and
empirically degrades quality. Sharpness in the final 1080x1920 Reel
comes from Lanczos upscale in `reels_format` plus RIFE interpolation
in `interp`, not from generating at higher resolution.

### 1.3 Memory and offloading

| Setting | Value | Authority |
|---|---|---|
| `offload_model` | **False** | On g6e.2xlarge (L40S 48 GB) the entire 28 GB DiT bf16 fits on-device. `True` streams layers via PCIe per step and is ~30-40% slower for no benefit. On g6e.xlarge (30 GB RAM) `True` was actively dangerous due to swap thrashing |
| `t5_cpu` | **False** | T5-XXL on CPU with single-threaded bf16 is unusably slow (multi-hour). T5 fits comfortably on the L40S alongside the DiT |

### 1.4 Prompts

| Setting | Authority |
|---|---|
| Positive prompt | Single natural-language sentence per Wan2.1 issue #496 (long attribute-list prompts produce worse output than short natural-language prompts on Wan-14B) |
| Negative prompt | Upstream Wan I2V-14B `sample_neg_prompt` (Chinese: `多余的手指` / `手指融合` / `形态畸形的肢体` / `三条腿` / `画得不好的手部` / `背景人很多`, plus 22 other quality negatives in `wan/configs/shared_config.py:18`) **augmented** with project-specific additions (identity drift, eye state, hair accessories, floating particles). The negative is `upstream + ", " + additions`, never a replacement |

UMT5-XXL is multilingual; Chinese and English negatives are equivalent
at the text encoder. Overriding the upstream default would lose its
model-trained suppression of finger / limb artefacts.

Prior attempts enumerated anatomy in the positive ("both hands
visible, all fingers visible, complete hands and arms"). At CFG 5.0+
this caused CFG over-satisfaction (extra hands). Anatomical
enumeration belongs in the negative or is delegated to upstream
defaults.

---

## 2. Pose extraction (DWPose)

| Setting | Value | Authority |
|---|---|---|
| Dual-pass extraction | `pose_align.py` + `pose_align_withdiffaug.py` | Paper DC-CFG requires both a positive pose folder (`cond_pos_folder`) and a "negative" augmented-pose folder (`cond_neg_folder`) for the conditional/null branches |
| `--max_frame` | **500** | Sized for up to 6 chunks (6 x 81 = 486 frames, + headroom). Pose extraction caps at `min(max_frame, source_frames)` |
| Driving video fps (before pose_align) | **16** | Model output is fixed 16 fps (`wan/configs/shared_config.py` `sample_fps=16`). Feeding source at native 30/60 fps produced slow-motion output (Bug 48). `dwpose.py:extract_aligned` resamples via `ffmpeg -r 16 -fps_mode cfr -an` to `driving_16fps.mp4` before either pose_align script reads the video |
| Yolox checkpoint | `yolox_l_8x8_300e_coco.pth` | mmdetection release URL; baked into the AMI |
| DWPose checkpoint | `dw-ll_ucoco_384.pth` | HuggingFace `yzd-v/DWPose` |

---

## 3. Frame interpolation (Practical-RIFE)

| Setting | Value | Authority |
|---|---|---|
| RIFE model version | **v4.26** | Most recent stable release at integration time. Apache-2.0-compatible MIT license |
| `--multi` | **`ceil(target_fps / 16)`** | Derived per-job from source reel fps (30 fps → 2, 60 fps → 4). **Mandatory** — without explicit `--multi`, RIFE defaults to 2 regardless of `--fps`, producing half-duration output (1.88x fast-forward observable in playback, Bug 42) |
| `--fps` (target) | **source reel fps, clamped to [16, 60]** | Tracks the user's source reel (Instagram-native output). 16 fps lower bound avoids playback judder; 60 fps upper bound caps RIFE multiplier and file size. Was previously hardcoded 60 — caused over-interpolation on 30 fps source, amplifying RIFE hallucinations on fast hand motion. When target ≤ 16, RIFE is skipped entirely (passthrough copy) |
| Per-chunk interp | **enabled** | Interpolating each chunk separately before concatenation avoids flow-based artefacts across the F6 re-anchor boundary |

RIFE replaced `ffmpeg minterpolate` (both `mci:aobmc:vsbmc=1` and
`blend` modes). `mci` hallucinated yellow/green blobs at boundaries
between adjacent source frames with large optical-flow delta. `blend`
opacity-blends adjacent frames and never hallucinates content but
produces motion blur on fast movement. RIFE synthesises clean ML
frames without either failure mode.

---

## 4. Face restoration (GFPGAN)

| Setting | Value | Authority |
|---|---|---|
| Model | **GFPGANv1.4** (`clean` arch) | Apache 2.0; StyleGAN2 prior; sharpens the ~50 px-wide generated face area |
| `alpha` (blend strength) | **0.30** | Implemented as `cv2.addWeighted(restored, 0.30, original, 0.70, 0)` on the OUTPUT. Conservative; higher values amplify the "plasticky" feel |
| `randomize_noise` | **False** (monkey-patched) | Default `True` injects fresh Gaussian noise on every call, producing per-frame variation in skin micro-details that reads as flicker. No public API exists; the StyleGAN2 forward is replaced with a wrapper that always passes `randomize_noise=False`. Cited: GFPGAN issue #533; arxiv 2410.11828v1 (video-restoration benchmark) |
| Laplacian sharpness gate | **threshold = 120** | Skip restoration entirely when the source face crop's variance > 120. For already-clear faces the lossy 95 px -> 512 px upsample -> restore -> 95 px downsample round-trip only adds artefacts. Threshold from GFPGAN discussion #639 |

### 4.1 Inert `weight=` kwarg

`GFPGANer.enhance(weight=N)` is a no-op when the underlying
architecture is `clean`. Source-read
`gfpgan/archs/gfpganv1_clean_arch.py:277`:

```python
def forward(self, x, return_latents=False, return_rgb=True,
            randomize_noise=True, **kwargs):
    ...
```

The `weight` kwarg falls into `**kwargs` and is never referenced
elsewhere in the file (verified by `grep -i weight` returning zero
matches). The same is true for `arch='original'`. The Replicate hosted
GFPGAN API does not expose `weight` for this reason.

The only restoration-strength control in the public GFPGAN API is a
manual alpha-blend on the output. This is what AUTOMATIC1111 webui's
"GFPGAN visibility" slider implements (confirmed via webui discussion
#5257); the pipeline mirrors that approach.

---

## 5. Reels format and audio

| Setting | Value | Authority |
|---|---|---|
| Output resolution | **1080x1920** | Instagram Reels standard 9:16 portrait |
| Upscale filter | **Lanczos** | ffmpeg recommended for 16:9 -> high-fidelity upsample |
| Audio mux | `ffmpeg -shortest` | Truncates to `min(video_duration, audio_duration)`. Verified compatible with N-chunk output where video is `N x 5.06 s` and source reel audio is the full reel length |

---

## 6. Pipeline-internal

| Setting | Value | Notes |
|---|---|---|
| `randomize_noise=False` monkey-patch | Replaces `restorer.gfpgan.forward` (see §4) | Required because GFPGAN exposes no public switch |
| Phase output checks | `_PHASE_OUTPUT_CHECKS` dict (`ec2/orchestrator.py`) | Marker is only honoured when local output exists. See [`04_bugs_encountered.md`](./04_bugs_encountered.md) Bug 46 |
| FSR credit wait | **>= 360 s** after `state=enabled` | AWS issues credits at ~`60 / snapshot_gib` per minute (~1 credit per 5 min for a 300 GB snapshot). Volume creation consumes one credit. If the bucket is empty at `create_volume`, AWS silently falls back to lazy-load (`FastRestored: False` on the volume, no error) |

---

## 7. Paper-recipe checklist

| Setting | Paper authority | Implementation |
|---|---|---|
| `w^txt = 5.0` | §Implementation Details | `sample_guide_scale = 5.0` |
| `w^pose = 1.0` | §Implementation Details | `condition_guide_scale = 1.0` |
| DC-CFG `[0.1, 0.4]` | §Implementation Details | `st_cond_cfg = 0.1`, `end_cond_cfg = 0.4` |
| Augmented negative-pose folder | DC-CFG section | `pose_align_withdiffaug.py` writes `pose_neg/`; passed as `--cond_neg_folder` |
| First-frame preservation | §Method | Reference photo opened fresh at each chunk via F6 (`ec2/inference/generate_dancer_chunked.py`) |

---

## 8. Where each value lives in code

| Setting | File |
|---|---|
| Generation CFG, steps, shift, size, prompts | `reel_forge/ec2/models/steadydancer.py` (single-chunk CLI args) and `ec2/inference/generate_dancer_chunked.py` (daemon spec) |
| Pose dual-pass | `reel_forge/ec2/models/dwpose.py` |
| RIFE flags | `reel_forge/ec2/phases/interp.py` |
| GFPGAN alpha, sharpness gate, noise patch | `reel_forge/ec2/phases/face_restore.py` |
| Reels format | `reel_forge/ec2/phases/reels_format.py` |
| Positive / negative prompts | `reel_forge/data/prompts/animate.py` |
| Marker + output guards | `reel_forge/ec2/orchestrator.py` (`_PHASE_OUTPUT_CHECKS`) |
| FSR enable / wait / disable | `reel_forge/ec2/launch.py` (`enable_fsr`, `disable_fsr`) |
