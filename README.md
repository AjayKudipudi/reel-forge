# insta-influencer

**AI-driven Instagram Reels generator.** Take a photo of a person and a reference Reel — get back a Reel where that person performs the same dance, in your chosen output format, ready to post.

Built around [SteadyDancer-14B](https://huggingface.co/MCG-NJU/SteadyDancer-14B) for pose-conditioned video generation, with RIFE for frame interpolation, GFPGAN for face restoration, and an AWS EC2 spot orchestration layer for cost-effective GPU inference.

---

## What it does

1. **prepare** — ingest a reference Reel + a portrait photo, extract pose/keypoints, build a job manifest.
2. **generate** — launch an EC2 spot GPU, run SteadyDancer-14B inference, post-process (interpolation + face restore), download the result.
3. **post** — stage the finished Reel for manual or API-based upload to Instagram.

Designed for a single operator running dance-content Reels at scale without renting a GPU 24/7.

---

## Tech stack

- **Generative model**: SteadyDancer-14B (Wan2.1 base, GGUF quantized for 24 GB VRAM)
- **Pose extraction**: DWPose
- **Post-processing**: RIFE (24 fps → 60 fps interp), GFPGAN (face restoration)
- **Orchestration**: AWS EC2 spot (g6.xlarge), S3-backed job state
- **Language / runtime**: Python 3.11+, pydantic-settings, boto3, Click
- **Quality**: ruff, mypy strict, pytest

---

## Architecture

A thin local CLI on your laptop builds a job manifest and uploads inputs to S3. It then launches a pre-baked EC2 spot AMI that contains the model weights and inference code. The instance pulls the manifest, runs the pipeline (pose → animate → interp → restore → mux audio), writes the output back to S3, and self-terminates. The CLI downloads the result. All state lives in S3 so jobs survive spot interruptions and laptop reboots.

See `AI_context/00_overview.md` and `AI_context/02_implementation_plan.md` for the full design.

---

## Quickstart

### Prerequisites

- Python 3.11+
- AWS account with EC2 G-family quota approved (see `.env.example`)
- A HuggingFace token with access to SteadyDancer-14B and Wan2.1
- ffmpeg on PATH

### Install

```bash
git clone https://github.com/AjayKudipudi/reel-forge.git
cd insta-influencer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in HF_TOKEN, AWS creds, EC2_AMI_ID, etc.
pytest                  # run unit + contract tests (smoke tests excluded)
```

### Local dry-run (no AWS, no GPU)

```bash
export STORAGE_BACKEND=local ANIMATE_FAKE=1
python -m insta_influencer prepare \
    --video tests/fixtures/sample.mp4 \
    --photo tests/fixtures/jane.png
python -m insta_influencer generate
```

### Full pipeline on AWS

```bash
# One-time: bake the AMI (downloads weights, installs deps, ~45 min)
python scripts/setup_ami.py

# Per-job:
insta prepare --video <reel.mp4> --photo <portrait.jpg>
insta generate
insta status
insta post --job-id <id>
```

---

## CLI

```
insta prepare    build job manifest from a reference Reel + photo
insta generate   launch spot GPU, animate, download
insta status     show job table with state + heartbeat
insta logs       tail S3 logs for a job
insta retry      re-queue a recoverable failure
insta cancel     mark a job cancelled
insta post       copy finished Reel to ready/ for upload
insta cleanup    enforce retention policy
insta stats      aggregate cost + runtime per job
```

---

## Configuration

All knobs are env vars loaded by pydantic-settings (see `insta_influencer/config.py`). The full schema with comments is in `.env.example`. Key categories:

- AWS infra (region, S3 bucket, EC2 AMI/instance type, spot pricing)
- Storage backend (`s3` for prod, `local` for dev)
- Model quantization (`fp16` / `gguf-q4-s` / `gguf-q5-m` / ...)
- Output shape (frames, fps, Reels 1080×1920, letterbox vs pillarbox)
- Behaviour toggles (audio mux, background replace, frame interp, content moderation)
- Retention / cost reporting

---

## Roadmap & known limitations

See `AI_context/` for the working design docs and open items. Highlights:

- Currently single-GPU, single-job at a time. Batching across one spot launch is on the roadmap.
- Auto-posting to Instagram via Graph API is gated behind phase F; manual upload is the default.
- Content moderation is opt-in and requires you to plug in your own binary.
- SteadyDancer-14B is research code — quality varies by reference Reel composition, occlusion, and lighting.

---

## License

Source code: MIT (see [LICENSE](LICENSE)).
SteadyDancer-14B model weights: Apache-2.0 (governed by the upstream HuggingFace repo terms).
