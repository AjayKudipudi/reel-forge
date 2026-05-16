# insta-influencer — AI Context

Documentation index and executive summary for the insta-influencer
pipeline. This directory contains the engineering record behind the
project: design decisions, settings audit, findings, and bug catalogue.

---

## Project summary

An image-to-video dance generator targeting Instagram Reels. Given a
single reference photo of a target person and a reference dance reel,
the pipeline produces a vertical (1080x1920) Reel of that person
performing the choreography from the reference.

The system is **technically functional**: it produces watchable output
end-to-end. A persistent defect remains on fast-motion hand frames; the
root cause is an upstream limitation in pose-estimation confidence on
high-velocity hand keypoints, not in the pipeline itself (see
[`03_findings_and_limitations.md`](./03_findings_and_limitations.md)).

---

## Stack

| Stage | Component | License | Role |
|---|---|---|---|
| Pose extraction | DWPose (mmpose-based) | Apache 2.0 | Extract 133-keypoint pose sequence from driving reel |
| Generation | SteadyDancer-14B (MCG-NJU + Tencent PCG) | Apache 2.0 | Image-to-video model. Reference photo literally becomes frame 1 of output ("first-frame preservation") |
| Frame interpolation | Practical-RIFE v4.26 | MIT | 16 fps native -> 60 fps target |
| Face restoration | GFPGAN v1.4 (StyleGAN2 prior) | Apache 2.0 | Sharpen ~50 px-wide generated faces |
| Mux + format | ffmpeg | LGPL | Audio reattach, 1080x1920 reels format |

Base model is `Wan-AI/Wan2.1-I2V-14B-480P` (Apache 2.0), to which
SteadyDancer adds a pose-conditioning adapter.

References:
- Paper: arxiv 2511.19320 (`SteadyDancer: Harmonized and Coherent Human
  Image Animation with First-Frame Preservation`, Jiaming Zhang et al.)
- Code: https://github.com/MCG-NJU/SteadyDancer
- Weights: https://huggingface.co/MCG-NJU/SteadyDancer-14B
- Practical-RIFE: https://github.com/hzwer/Practical-RIFE
- GFPGAN: https://github.com/TencentARC/GFPGAN

---

## Pipeline phases

```
pose_extract  ->  animate  ->  interp  ->  face_restore  ->  audio_attach  ->  reels_format
   ~2 min      ~30 min/chunk   ~30 sec      ~4 min          <5 sec          <5 sec
   DWPose       SteadyDancer    RIFE x4    GFPGAN +        ffmpeg          ffmpeg
   x2 passes    + DC-CFG       multi=4    alpha=0.30      -shortest       1080x1920
```

Phase outputs are persisted to per-job S3 prefixes; per-phase marker
files allow resume after spot interruption.

Long videos use a chunked approach: each chunk renders 81 native frames
(~5 seconds at 16 fps) and re-anchors to the original reference photo
(see `F6` in [`03_findings_and_limitations.md`](./03_findings_and_limitations.md)).
RIFE interpolation runs per-chunk before concatenation, eliminating
flow-based artefacts across boundaries.

---

## Infrastructure

- **Compute**: AWS g6e.2xlarge spot instances (L40S 48 GB GPU, 64 GB RAM).
- **Region**: eu-south-2.
- **Storage**: per-job S3 prefix for inputs, outputs, logs, status, and
  phase markers.
- **Resume**: phase markers + per-phase output checks (see Bug 46 in
  [`04_bugs_encountered.md`](./04_bugs_encountered.md)).
- **First-boot cost**: an EBS volume created from an AMI snapshot
  lazy-loads blocks from S3 at ~7-11 MB/s unless Fast Snapshot Restore
  (FSR) is enabled in the launch AZ. The launch driver enables FSR
  just-in-time, waits for credits to accumulate, then disables FSR
  after the spot is running.

---

## Cost envelope

| Item | Cost |
|---|---|
| g6e.2xlarge spot, effective | ~$0.04/hr (with AWS credits applied to an on-demand floor of ~$2.24/hr) |
| Per-run total (2-clip / 10 sec output) | ~$3 net of credits |
| Per-run wall time | ~2h 20m end-to-end |
| Animate phase share | ~1h 33m of the 2h 20m |

Wall-time is dominated by spot boot (~10 min), one-time DiT bf16
conversion (~10 min), and 33 min per chunk of GPU sampling.

---

## Documentation index

| File | Contents |
|---|---|
| [`01_architecture.md`](./01_architecture.md) | Phase-by-phase architecture; F6 chained-chunking removal; cloud-init flow; marker-based resume |
| [`02_settings_audit.md`](./02_settings_audit.md) | Each generation and post-process setting with citation and rationale |
| [`03_findings_and_limitations.md`](./03_findings_and_limitations.md) | What was learned from production runs; known defects with upstream causes |
| [`04_bugs_encountered.md`](./04_bugs_encountered.md) | Bug catalogue (~47 entries) with symptom, root cause, fix location |
| [`05_aws_setup.md`](./05_aws_setup.md) | IAM, AMI bake, FSR lifecycle, spot interruption recovery |

Older drafts (`00_overview.md`, `01_model_research.md`,
`02_implementation_plan.md`, `03_configuration.md`, `04_observability.md`,
`05_alternative_models_discussions.md`, `06_upstream_integration_notes.md`,
`07_session_log_handoff.md`, `08`/`09_run_timing_*.md`) remain in the
parent directory for historical reference.

---

## Status (as of the most recent validation runs)

| Aspect | State |
|---|---|
| End-to-end pipeline | Functional |
| Identity stability across chunks | Bounded to one chunk (~5 s) via F6 re-anchoring |
| Hand artefacts on fast motion | Persistent; upstream pose-estimator limitation |
| Face naturalness | Stable at GFPGAN alpha=0.30 with `randomize_noise=False` |
| Spot interruption recovery | Functional after Bug 46 fix (marker-vs-files guard) |
| Reproducible AMI | Not yet baked from the post-v8.10 stack; cloud-init handles dep race conditions for now |
