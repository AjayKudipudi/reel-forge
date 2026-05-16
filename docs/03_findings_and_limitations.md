# 03 — Findings and Limitations

What production runs revealed about the model stack, the upstream
tools, and the pipeline design. Items below are organised by where the
finding sits — model, post-process, infrastructure — and each carries
a citation back to the originating evidence.

---

## 1. Model-level findings (SteadyDancer / Wan2.1 stack)

### 1.1 Identity drift is bounded by chunk re-anchoring (F6)

A single SteadyDancer pass generates 81 frames (~5 s at 16 fps).
Multi-chunk output requires a chaining strategy.

Initial implementation passed the previous chunk's last frame as the
next chunk's first-frame condition (in-memory tensor handoff). Frame-
by-frame visual review (12 keyframes at n = 0, 10, 20, 40, 60, 80, 81,
82, 100, 120, 140, 160) showed:

- Frame 0 matches the reference photo exactly (first-frame preservation works).
- Drift begins within ~0.6 s (frame 10).
- By frame 80 (chunk 0 last) the face is clearly different from the reference.
- Frame 81 (chunk 1 first) inherits frame 80's drifted identity, not
  the reference's identity.
- Drift then compounds through chunk 1.

Fix (`F6`, implemented in `ec2/inference/generate_dancer_chunked.py`):
open the original reference photo fresh at every chunk. The daemon
still amortises model load + DiT bf16 conversion across chunks
(~12 min saved per run for N >= 2). Identity drift is now bounded to
one chunk's worth (~5 s).

**Trade-off accepted**: chunk N+1's frame 0 is the rest-pose reference
while its pose-track frame 0 is mid-dance (continuous from chunk N's
pose frame 80). First-frame preservation is literal, so the
concatenated video has a brief pose discontinuity at the boundary.
Per-chunk RIFE interpolation before concat (see
[`01_architecture.md`](./01_architecture.md) §4) keeps optical-flow
artefacts from spanning the boundary; the visible result is a 17 ms
hard cut, below the threshold of perception.

### 1.2 Hand artefacts on fast motion are an upstream limitation

DWPose's 133-keypoint estimator has lower confidence on the wrist and
finger keypoints during high-velocity hand motion. When confidence
drops, the conditioning signal to SteadyDancer is noisy in exactly the
locations where the generator must render detailed structure.
Empirically, the artefact appears as:

- Melted or fused fingers during fast side-to-side hand motion;
- Brief hand disappearance when the arm crosses the body fast;
- Motion-blurred limbs even in the 16 fps source generation (i.e., the
  blur originates pre-RIFE).

Cross-checked: the same hand blur appears in native 16 fps `final.mp4`
and in 60 fps RIFE output at the corresponding timestamp. The
generator is the source.

The SteadyDancer paper §Limitations confirms this is a fundamental
constraint of pose-driven I2V: "highly dynamic poses with rapid
self-occlusion remain challenging." Mitigations tried:

| Attempted fix | Result |
|---|---|
| Enumerate "both hands visible, all fingers visible" in positive | Caused CFG over-satisfaction at `sample_guide_scale=6.0`: model added extra hands trying to please prompt. Reverted |
| Negative tokens (`missing hands, fingerless, deformed hands`) | Mild improvement; the upstream Chinese `多余的手指 / 手指融合` already covered most of this and was being overridden |
| Bump `condition_guide_scale` (pose CFG) to 1.5 | Amplified pose-detector noise as extra-people artefacts in the background |

The current stance is that hand artefacts on fast motion are accepted
as a known defect; the only durable fix is upstream improvement to
DWPose or replacement with a higher-confidence hand-keypoint detector.

### 1.3 GFPGAN `weight=` is inert in `arch='clean'`

Source-read `gfpgan/archs/gfpganv1_clean_arch.py:277`:

```python
def forward(self, x, return_latents=False, return_rgb=True,
            randomize_noise=True, **kwargs):
    ...
```

`grep -i weight` on the file returns zero matches. The `weight` kwarg
passed by `GFPGANer.enhance()` falls into `**kwargs` and is never
referenced. The same is true for `arch='original'`. The hosted
Replicate GFPGAN API does not expose `weight` because the maintainers
know it is inert. AUTOMATIC1111 webui's "GFPGAN visibility" slider
implements restoration strength as a manual alpha-blend on the output
(webui discussion #5257).

Implication for any pipeline using GFPGAN: passing `weight=0.5` to
`enhance()` does nothing; the pipeline is silently applying GFPGAN at
100% strength on every frame. Manual `cv2.addWeighted` on the output
is the only working restoration-strength control.

### 1.4 Long attribute-list prompts hurt Wan-14B output

Wan2.1 issue #496 documents that long attribute-list positive prompts
produce worse output than short natural-language sentences on Wan-14B.
A ~50-token positive enumerating anatomy ("both hands visible, all
fingers visible, complete hands and arms") combined with CFG 5.0+
produces classical CFG over-satisfaction artefacts (extra hands,
deformed fingers).

The fix is to keep the positive as a single natural-language sentence
and rely on the negative for anatomical suppression.

### 1.5 The upstream negative prompt must be augmented, not replaced

The upstream Wan I2V-14B `sample_neg_prompt` (Chinese, in
`wan/configs/shared_config.py:18`) already covers:

> `多余的手指` (extra fingers), `手指融合` (fused fingers),
> `形态畸形的肢体` (misshapen limbs), `三条腿` (three legs),
> `画得不好的手部` (poorly drawn hands), `背景人很多` (many background people)

UMT5-XXL is multilingual; Chinese and English negatives are equivalent
at the text encoder. Overriding the upstream default with an English
negative that does not enumerate these terms loses the model-trained
suppression of finger / limb artefacts.

The fix is to define `NEGATIVE = upstream_english_translation + ", " +
our_additions`. Additions cover identity drift, eye state, hair
accessories, and floating particles — categories the upstream does
not handle.

### 1.6 Generation resolution should match the trained pixel area

Wan2.1-I2V-14B-480P was trained at ~480p (the suffix is literal):
~590K pixels per frame. The model is **area-optimised**, not
resolution-optimised. Concrete derivation:

| Size | Pixel area | vs training |
|---|---|---|
| `1024 x 576` landscape | 589,824 | baseline (README example) |
| `576 x 1024` portrait | 589,824 | same area, rotated -> Reels-native |
| `720 x 1280` portrait | 921,600 | +56% -> beyond training distribution |

Generating at 720x1280 empirically degraded coherence. Sharpness in
the final 1080x1920 Reel comes from Lanczos upscale in `reels_format`
plus RIFE interpolation in `interp`, not from generating at higher
resolution.

---

## 2. Post-process findings

### 2.1 `ffmpeg minterpolate=mci` hallucinates at chunk boundaries

The first non-stub `interp` phase used `ffmpeg
minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:vsbmc=1`. With F6
chunk re-anchoring, the boundary frame pair is (chunk 0's last source
frame = mid-dance, chunk 1's first source frame = rest pose). The
optical-flow estimator could not resolve a flow field across that
delta and synthesised yellow/green hallucinated content where the
hands "should be."

Smaller versions of the same defect appeared mid-chunk on fast hand
motion (frames 150, 225) where intra-chunk hand motion was too fast
for `mci` to track cleanly.

`mi_mode=blend` (opacity-blend adjacent source frames) never invents
content. Trade-off: produces natural motion blur on fast motion.
Practical-RIFE v4.26 produces clean ML-interpolated frames without
either failure mode and replaces `minterpolate` entirely on the AMI.

### 2.2 GFPGAN `randomize_noise=True` produces per-frame skin flicker

The StyleGAN2 prior injects fresh Gaussian noise on every call by
default. On still images this is invisible; on a video, per-frame
noise samples vary the skin micro-details and read as flicker. No
public API exists to disable this; the pipeline replaces
`restorer.gfpgan.forward` with a wrapper that always passes
`randomize_noise=False`.

Cited: GFPGAN issue #533 (open, no maintainer fix), arxiv 2410.11828v1
(video-restoration benchmark documents the same defect).

---

## 3. Infrastructure findings

### 3.1 EBS FSR lazy-load throttles unrestored snapshots to ~7-11 MB/s

A gp3 volume created from an AMI snapshot reads at ~7-11 MB/s
effective on first-touch blocks, regardless of configured IOPS or
throughput. This is the AWS-documented lazy-load behaviour for
non-FSR-enabled snapshots; the volume is "initialised" only after
every block has been read at least once.

Confirmed empirically:

```
volume vol-...
  type: gp3, 300 GB, 3000 IOPS, 125 MB/s throughput
  FastRestored: False                    <-- root cause
snapshot snap-...
  FSR records: []                        <-- FSR not enabled in any AZ
```

Sequential `cat` and mmap random faults hit the same bottleneck.

The fix is to enable Fast Snapshot Restore (FSR) in the launch AZ
before creating the volume, wait for credits to accumulate, then
launch the spot. FSR can be disabled immediately after — its effect
is consumed at `create_volume`, and the volume keeps its initialised
state.

### 3.2 FSR credits require a 360-second wait after `state=enabled`

AWS issues FSR credits at approximately `60 / snapshot_gib` credits
per minute. For a 300 GB snapshot that is ~1 credit per 5 minutes.
Volume creation consumes one credit. If the bucket is empty at
`create_volume`, AWS silently falls back to lazy-load: `FastRestored:
False` on the volume, no error, no warning.

The launch driver reads `EnabledTime` from the API after FSR reaches
`state=enabled` and sleeps until at least 360 seconds have elapsed.
This is idempotent; if FSR was already enabled long enough, no further
sleep is performed.

### 3.3 Spot interruption + marker-vs-files inconsistency

The orchestrator's marker-skip logic was initially unconditional: if
the S3 marker existed, the phase was skipped. After a spot
interruption mid-animate, the re-launched spot found
`markers/pose_extract.done` in S3 but the per-frame pose JPGs lived
on the reclaimed spot's local EBS. Animate then immediately failed
because `pose/0000.jpg` was missing.

Fix: `_PHASE_OUTPUT_CHECKS` maps each phase to its primary on-disk
output; the marker is only honoured when the local output exists. See
[`04_bugs_encountered.md`](./04_bugs_encountered.md) Bug 46.

The orthogonal long-term fix is uploading pose-extract outputs to S3
so a fresh spot can preseed them. The current trade-off (re-run
pose_extract for ~2.5 min vs ship several hundred MB of JPGs over the
network) leaves them local.

### 3.4 Cloud-init dependency race conditions

Three categories of dependency issues had to be handled in cloud-init
because the AMI was not re-baked after their root causes were found:

| Race | Resolution |
|---|---|
| `numpy<2` pin gets bumped by transitive resolution when `gfpgan` is installed | Pin `numpy<2` in the `gfpgan` pip command **and** defensively `pip install --force-reinstall --no-deps "numpy<2"` after. Verify with `python -c "from xtcocotools.coco import COCO"` |
| `basicsr/data/degradations.py` imports `torchvision.transforms.functional_tensor` (removed in torchvision 0.17+) | `sed`-patch to use `torchvision.transforms.functional`. Locate the file via `find /opt/.../.venv -path '*basicsr/data/degradations.py'`, not via `python -c "import basicsr.data.degradations"` (the latter is exactly what fails) |
| `skvideo` uses `np.float` / `np.int` / `np.bool` (removed in numpy 1.20+) | `sed`-patch across `skvideo/**/*.py` to use bare `float` / `int` / `bool` |

Diagnostic block at the end of cloud-init prints `[ok]` or `[FAIL]`
per runtime import so dep issues surface immediately on next boot
instead of after a 90-minute animate phase.

All three races are eliminated by an AMI re-bake that includes these
fixes pre-applied.

### 3.5 The wrong checkpoint silently inits 45+ tensors to noise

`WanModel.from_pretrained()` silently random-initialises any tensor
not present in the checkpoint and emits only a stderr warning. With
the base `Wan-AI/Wan2.1-I2V-14B-480P` checkpoint specified instead of
the fine-tuned `MCG-NJU/SteadyDancer-14B`, 45+ tensors of
SteadyDancer's pose-conditioning adapter
(`condition_embedding_align.cross_attn.in_proj_weight`,
`condition_embedding_align.ffn_pose.0.weight`,
`patch_embedding_fuse.weight`, `patch_embedding_ref_c.weight`, ...)
were random noise. The sampler never converged; runs timed out at the
60-minute animate budget.

The warning was caught only after a `stream_to_log` diagnostic merged
upstream stdout into the captured stderr stream. See
[`04_bugs_encountered.md`](./04_bugs_encountered.md) Bug 27.

---

## 4. Validation and process findings

### 4.1 Smoke tests must exercise the full inference path

The original smoke harness ran `import ...` + `--help` on the upstream
CLIs. This caught dependency issues at import time but missed every
runtime bug that surfaced only when actual data flowed through:
missing pose negative folder, wrong checkpoint loaded, video-vs-frames
confusion at `--cond_pos_folder`, removed `--convert_model_dtype`
flag, and the inert `weight=` kwarg.

A 1-frame, 1-step inference smoke catches all of these at AMI bake
time, adding ~3 minutes to the bake.

### 4.2 Postprocess-only iteration loop saves cost on post-animate fixes

Iterating on prompts, RIFE settings, or GFPGAN parameters does not
require re-running the 90-minute animate phase. The
`seed_postprocess.py` script:

1. Downsamples a local 60 fps mp4 to 16 fps;
2. Uploads it to S3 as `animated.mp4` under a given job_id;
3. Uploads `pose_extract.done` and `animate.done` markers;
4. Deletes any leftover `interp.done` / `face_restore.done` /
   `audio_attach.done` / `reels_format.done` markers;
5. Resets `status.json` to `prepared`.

A subsequent `generate` run skips pose_extract and animate, performing
only interp + face_restore + audio_attach + reels_format (~25 min wall
vs ~2h 40m for a full run).

### 4.3 KEEP_ALIVE_ON_FAILURE for in-place debugging

A failed run that terminates immediately re-imposes the full
FSR-enable + boot + cloud-init cost on the next attempt. With
`KEEP_ALIVE_ON_FAILURE=1` and `KEEP_ALIVE_SECONDS=3600`, a non-zero
exit causes the spot to sleep before termination. The operator can
SSH in, fix the issue interactively, and re-invoke the orchestrator
on the same spot — no re-paying boot overhead.

This is the right default for any infrastructure-change run.
