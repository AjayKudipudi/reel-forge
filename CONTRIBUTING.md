# Contributing to reel-forge

Thanks for your interest in contributing. This project is an AI dance-video
generation pipeline (SteadyDancer-14B → RIFE → GFPGAN → ffmpeg) that runs
on AWS spot GPUs. Contributions are welcome at every level — bug reports,
documentation, prompt engineering, new pipeline phases, or full model
swaps.

## Quick links

- Architecture: [`docs/01_architecture.md`](docs/01_architecture.md)
- Settings rationale (every CFG / sampling value with paper citation): [`docs/02_settings_audit.md`](docs/02_settings_audit.md)
- Known limitations: [`docs/03_findings_and_limitations.md`](docs/03_findings_and_limitations.md)
- Bug catalogue (47 documented bugs + fixes): [`docs/04_bugs_encountered.md`](docs/04_bugs_encountered.md)
- AWS setup (IAM, AMI bake, FSR): [`docs/05_aws_setup.md`](docs/05_aws_setup.md)

## Ways to contribute

- **Open an issue** for bugs, feature requests, or questions. Use the
  templates — they ask for the right context up front.
- **Pick a `good-first-issue`** if you're new to the codebase.
- **Pick a `help-wanted`** if you want to tackle something larger.
- **Improve the docs** — typos, clarifications, examples, demos.
- **Propose a new pipeline phase** — e.g., a different frame interpolator,
  upscaler, audio sync model.

## Local development setup

Prerequisites:
- Python 3.11
- `ffmpeg` and `ffprobe` on `$PATH`
- AWS account with EC2/S3 access if you want to run the spot pipeline end-to-end
- A Hugging Face account + token if you want to download SteadyDancer-14B weights

```bash
git clone https://github.com/AjayKudipudi/reel-forge.git
cd reel-forge
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# edit .env with your AWS keys, HF token, etc.
```

Run the local test suite (no GPU required — tests use a Fake model):

```bash
pytest tests -q
```

Run linters before pushing:

```bash
ruff check .
mypy reel_forge
```

## Branch + PR flow

1. Fork the repo on GitHub.
2. Create a feature branch: `git checkout -b feat/short-description` or `fix/short-description`.
3. Make focused changes. Keep PRs small — one concern per PR.
4. Run `pytest`, `ruff check`, `mypy reel_forge` and make sure all pass.
5. Push your branch and open a Pull Request against `main`.
6. The PR template prompts for context — fill it in.
7. CI (GitHub Actions) runs ruff + mypy + pytest on every PR.

## Code style

- Type annotations on all public functions
- `ruff` settings live in `pyproject.toml` — match them
- One natural-language sentence per docstring; no marketing fluff
- Bugs / non-obvious behaviour: include the WHY in a comment (we lean
  on the bug catalogue style — see existing comments in
  `reel_forge/ec2/phases/face_restore.py` and `interp.py` for tone)

## Commit messages

Plain English, present tense, what + why:

> `interp: switch RIFE multi from 2 to ceil(target_fps/16) to preserve duration`

Not:

> ~~"fix bug"~~

## Reporting security issues

See [`SECURITY.md`](SECURITY.md). Don't open public issues for security
problems — use the private channel described there.

## Code of Conduct

Participation is governed by the [Contributor Covenant](CODE_OF_CONDUCT.md).
Be kind, be specific, assume good intent.
