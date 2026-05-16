# reel-forge

[![CI](https://github.com/AjayKudipudi/reel-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/AjayKudipudi/reel-forge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-blueviolet)](https://github.com/astral-sh/ruff)
[![Types: mypy](https://img.shields.io/badge/types-mypy-blue.svg)](http://mypy-lang.org/)

> AI dance-video generator for Instagram Reels — SteadyDancer-14B + Practical-RIFE + GFPGAN on AWS spot GPUs.

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

See `docs/00_overview.md` and `docs/02_implementation_plan.md` for the full design.

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
cd reel-forge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in HF_TOKEN, AWS creds, EC2_AMI_ID, etc.
pytest                  # run unit + contract tests (smoke tests excluded)
```

### Local dry-run (no AWS, no GPU)

```bash
export STORAGE_BACKEND=local ANIMATE_FAKE=1
python -m reel_forge prepare \
    --video tests/fixtures/sample.mp4 \
    --photo tests/fixtures/jane.png
python -m reel_forge generate
```

### Full pipeline on AWS

```bash
# One-time: bake the AMI (downloads weights, installs deps, ~45 min)
python scripts/setup_ami.py

# Per-job:
forge prepare --video <reel.mp4> --photo <portrait.jpg>
forge generate
forge status
forge post --job-id <id>
```

---

## CLI

```
forge prepare    build job manifest from a reference Reel + photo
forge generate   launch spot GPU, animate, download
forge status     show job table with state + heartbeat
forge logs       tail S3 logs for a job
forge retry      re-queue a recoverable failure
forge cancel     mark a job cancelled
forge post       copy finished Reel to ready/ for upload
forge cleanup    enforce retention policy
forge stats      aggregate cost + runtime per job
```

---

## Configuration

All knobs are env vars loaded by pydantic-settings (see `reel_forge/config.py`). The full schema with comments is in `.env.example`. Key categories:

- AWS infra (region, S3 bucket, EC2 AMI/instance type, spot pricing)
- Storage backend (`s3` for prod, `local` for dev)
- Model quantization (`fp16` / `gguf-q4-s` / `gguf-q5-m` / ...)
- Output shape (frames, fps, Reels 1080×1920, letterbox vs pillarbox)
- Behaviour toggles (audio mux, background replace, frame interp, content moderation)
- Retention / cost reporting

---

## Roadmap & known limitations

See `docs/` for the working design docs and open items. Highlights:

- Currently single-GPU, single-job at a time. Batching across one spot launch is on the roadmap.
- Auto-posting to Instagram via Graph API is gated behind phase F; manual upload is the default.
- Content moderation is opt-in and requires you to plug in your own binary.
- SteadyDancer-14B is research code — quality varies by reference Reel composition, occlusion, and lighting.

---



## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the
local development setup, branch flow, and code-style guide. Good places to
start:

- Browse [`good-first-issue`](https://github.com/AjayKudipudi/reel-forge/labels/good-first-issue) labelled issues
- Browse [`help-wanted`](https://github.com/AjayKudipudi/reel-forge/labels/help-wanted) for larger tasks
- Ask in [GitHub Discussions](https://github.com/AjayKudipudi/reel-forge/discussions) for design questions

Security issues: see [SECURITY.md](SECURITY.md). Community standards:
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

Source code: MIT (see [LICENSE](LICENSE)).
SteadyDancer-14B model weights: Apache-2.0 (governed by the upstream HuggingFace repo terms).
