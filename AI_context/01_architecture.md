# 01 — Architecture

How the pipeline is structured, what each phase does, and how the
system survives spot interruption.

---

## 1. High-level flow

```
Local (driver):                       EC2 spot (g6e.2xlarge, eu-south-2):
----------------                      ----------------------------------
prepare  --reel  | --video             cloud-init -> orchestrator
   |  --photo  --clips N                 |
   |                                     +-- pose_extract  (DWPose dual pass)
   |  upload inputs to S3                +-- animate       (SteadyDancer-14B)
   |  enqueue job_id                     +-- interp        (RIFE v4.26)
   |  enable FSR on AMI snapshot         +-- face_restore  (GFPGAN)
   |  wait FSR credits accumulate        +-- audio_attach  (ffmpeg)
   |  launch spot in matching AZ         +-- reels_format  (ffmpeg)
   |  disable FSR                        |
   |                                     +-- upload artefacts to S3
   |                                     +-- self-terminate
   |
   poll status.json from S3
   download final.mp4 when state=completed
```

Job ids are content-addressed
(`sha256(canonical_inputs)[:12]`); preparing the same inputs twice
yields the same job id.

---

## 2. Phase responsibilities

| Phase | Input | Output | Wall (typical) | Tool |
|---|---|---|---|---|
| `pose_extract` | reference reel mp4, target photo | `pose/0000.jpg`..`pose/XXXX.jpg`, `pose_neg/...` | ~2 min (dual pass) | upstream `preprocess/pose_align.py` + `pose_align_withdiffaug.py` |
| `animate` | photo + per-chunk pose dirs | `animated_chunk_<i>.mp4`, then concatenated `animated.mp4` | ~30 min per chunk (40 steps, 576x1024) | upstream `generate_dancer.py` via daemon wrapper |
| `interp` | `animated.mp4` (16 fps) | `animated_60fps.mp4` | ~30 s | Practical-RIFE `inference_video.py --multi=4` |
| `face_restore` | `animated_60fps.mp4` | `animated_60fps_face.mp4` | ~4 min | GFPGAN per-frame + alpha-blend |
| `audio_attach` | restored video + source reel audio | `animated_w_audio.mp4` | <5 s | `ffmpeg -shortest` |
| `reels_format` | audio-attached video | `final.mp4` (1080x1920) | <5 s | `ffmpeg` Lanczos upscale + pad |

Each phase writes a marker file `markers/<phase>.done` to S3 on success
and an artifact mp4 either to local disk only (interp/face_restore) or
to S3 (animate, final.mp4, status.json).

---

## 3. F6 — chained chunking removal

A single SteadyDancer pass generates exactly **81 frames** at 16 fps
(5.06 s). Longer output requires chunking. Two chaining strategies
were considered:

| Strategy | Description | Outcome |
|---|---|---|
| Chained (rejected) | Each chunk's first-frame condition is the previous chunk's last frame, fed in-memory between chunks | Identity drift compounded across chunks: face at frame 160 was clearly a different person from the reference |
| F6 re-anchor (shipped) | Every chunk's first-frame condition is the original reference photo | Identity drift bounded to one chunk's worth (~5 s) |

F6 was validated against the same job id on the same source pair.
Frame 0 of every chunk after F6 matches the reference photo exactly
(first-frame preservation is literal). The trade-off is a brief pose
discontinuity at chunk boundaries: frame 80 is mid-dance, frame 81
re-anchors to the rest pose because preservation is literal. RIFE
interpolation runs **per-chunk** before concatenation, so flow-based
artefacts (`mci` blob hallucinations, `blend` motion-blur smears) at
the boundary are avoided entirely (see Bug 17/18-class issues in
[`04_bugs_encountered.md`](./04_bugs_encountered.md)).

Implementation: `ec2/inference/generate_dancer_chunked.py` — a
daemon-style batch script that loads SteadyDancer-14B once and runs N
inferences sequentially. Each chunk gets a fresh PIL.Image opened from
the reference photo path, never from the previous chunk's GPU tensor.

The daemon design also amortises the ~10 min one-time DiT bf16
conversion across all chunks in a run; previously each chunk paid that
cost.

---

## 4. Per-chunk RIFE interpolation + concat

```
animated_chunk_0.mp4  (16 fps, 81 frames)
animated_chunk_1.mp4  (16 fps, 81 frames)
            |
            v   (RIFE v4.26, --multi=4, per chunk)
animated_chunk_0_60fps.mp4
animated_chunk_1_60fps.mp4
            |
            v   (ffmpeg -f concat -c copy)
animated_60fps.mp4  (60 fps, 2*81*4 frames)
```

RIFE runs on each chunk independently. The interpolator never sees a
chunk boundary, so it cannot hallucinate content across the
re-anchor discontinuity. The boundary in the concatenated 60 fps output
is a 1/60 s hard cut (~17 ms), below the threshold of perception.

RIFE requires explicit `--multi=ceil(target_fps / source_fps)`. The
`--fps` flag alone does not change the multiplier; without
`--multi=4`, RIFE produces only 2x the source frames stamped at the
target fps, halving the output duration.

---

## 5. Cloud-init flow on spot launch

Steps the user-data script performs before invoking the orchestrator:

1. Mount root volume, expand filesystem to 300 GiB.
2. Activate the pre-baked Python virtualenv at
   `/opt/insta-influencer/.venv`.
3. Pull the project source tarball from S3 (~85 KB) and extract over
   the baked snapshot so newly-shipped code paths take effect without
   re-baking.
4. Page-cache prewarm: `cat` the SteadyDancer-14B and Wan2.1 weight
   files in the order the loader will read them so first-byte reads hit
   page cache.
5. Install/pin dependencies missing from the AMI:
   - `pip install "numpy<2" basicsr==1.4.2 facexlib==0.3.0 gfpgan==1.3.8 scikit-video`
   - `pip install --force-reinstall --no-deps "numpy<2"` (defensive)
6. Patch `basicsr/data/degradations.py`: replace `from
   torchvision.transforms.functional_tensor import rgb_to_grayscale`
   with `from torchvision.transforms.functional import rgb_to_grayscale`
   (removed in torchvision 0.17+). The file is located via `find ...`,
   not `python -c "import basicsr.data.degradations"` (the latter is
   exactly what fails).
7. Patch `skvideo/**/*.py`: rewrite `np.float(` -> `float(` and same
   for `np.int`, `np.bool` (removed in numpy 1.20+).
8. Download Practical-RIFE clone (~3 MB shallow) and the
   `RIFEv4.26_0921.zip` weights bundle (~22 MB).
9. Download `GFPGANv1.4.pth` (~332 MB).
10. Prewarm facexlib detection weights into `~/.cache`.
11. Diagnostic check: each runtime import prints `[ok]` or `[FAIL]` to
    the cloud-init log so dep issues surface immediately rather than
    after a 90-minute animate phase.
12. Exec `python -m insta_influencer.ec2.orchestrator process-job
    <job_id>`.

Re-baking the AMI to include these would eliminate ~7-10 minutes per
cold launch.

---

## 6. Marker-based resume after spot interruption

Each phase writes `s3://<bucket>/<prefix>/<job>/markers/<phase>.done`
on success. On a fresh spot the orchestrator iterates phases and, for
each:

```python
if storage.exists(marker):
    if _phase_outputs_present_locally(phase.name, work):
        continue            # skip; both marker and on-disk output exist
    log("phase.marker_present_but_outputs_missing.rerun")
    # fall through and re-run
```

`_PHASE_OUTPUT_CHECKS` maps each phase to its primary on-disk artefact:

| Phase | On-disk output |
|---|---|
| `pose_extract` | `pose/0000.jpg` |
| `animate` | `animated.mp4` |
| `interp` | `animated_60fps.mp4` |
| `face_restore` | `animated_60fps_face.mp4` |
| `audio_attach` | `animated_w_audio.mp4` |
| `reels_format` | `final.mp4` |

`animate` outputs (`animated.mp4` plus each chunk) are uploaded to S3
on completion, so a fresh spot can pull them back via the orchestrator
artefact-preseed step at job startup. Pose extraction outputs (per-frame
JPGs, several hundred MB) are not uploaded; on spot interruption, the
orchestrator correctly re-runs pose_extract on the fresh spot because
the marker is present but the JPGs are not.

---

## 7. Design contracts

The codebase enforces seven structural contracts:

| Contract | File | Purpose |
|---|---|---|
| Phase protocol | `core/phase.py` | Every pipeline step honours `phase.run(ctx) -> PhaseResult` |
| ObjectStore | `core/storage.py` | S3 / local / in-memory implementations; no bare boto3 elsewhere |
| ExternalTool runner | `core/external_tool.py` | Unified subprocess wrapper for ffmpeg / yt-dlp / upstream CLIs |
| Status state machine | `core/status.py`, `core/status_models.py` | Declared `TRANSITIONS`; illegal transitions raise |
| Content-addressed job id | `prepare/job_id.py` | `sha256(canonical_inputs)[:12]` -> idempotent `prepare` |
| Pydantic schemas | `core/manifest.py`, `core/status_models.py` | Discriminated unions, validation, JSON round-trip |
| AnimationModel / PoseExtractor protocols | `ec2/models/_base.py` | `Fake*` impls enable GPU-free local dev and tests |

Subprocess isolation per phase keeps VRAM usage clean; each phase runs
in a fresh Python interpreter.

---

## 8. Upstream integration model

The upstream SteadyDancer repository exposes `generate_dancer.py` and
`preprocess/pose_align.py` as CLI tools, not as importable Python APIs.
Wrappers in `ec2/models/steadydancer.py` and `ec2/models/dwpose.py`
invoke these as subprocesses via the unified `run_tool` runner, with
output paths fed through a shared work directory.

For chunked output, `ec2/inference/generate_dancer_chunked.py` imports
the upstream `wan` package directly (PYTHONPATH includes both project
root and the upstream repo), loading `wan.WanI2VDancer` once and
iterating chunks. This is the only direct-import dependency on
upstream internals.
