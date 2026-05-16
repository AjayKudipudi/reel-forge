# 04 — Bugs Encountered

Catalogue of issues found during AMI bakes and production runs. Each
entry: symptom, root cause, fix location. Numbering preserved from
the original session log for cross-reference. Bugs 1-9 (AMI bake
era) and 28+ (production runs) are summarised; the bake-era entries
have been collapsed where they were specific to one transient
mis-configuration.

---

## Bake-era bugs (1-9): AMI baking and smoke validation

### Bug 1 — `NoCredentialsError` on local boto3 calls

**Symptom**: `botocore.exceptions.NoCredentialsError`. `pydantic-settings`
loads `.env` into a `Config` object but does not push to `os.environ`,
so boto3's credential chain finds nothing.

**Fix**: `load_config()` now performs
`os.environ.setdefault("AWS_ACCESS_KEY_ID", cfg.AWS_ACCESS_KEY_ID)`
and equivalents at startup. Covers all boto3 callers.

### Bug 2 — `.env` inline comment parsed as value

**Symptom**: `git fetch origin '# comment'` -> invalid refspec.
`STEADYDANCER_GIT_SHA=                     # comment` -> python-dotenv
parsed the trailing comment as the value, the bash check `if [ -n
"$VAR" ]` was too permissive.

**Fix**: comments moved off empty-value lines in `.env`; defensive
bash regex `[[ "$SDANCER_SHA" =~ ^[0-9a-f]{7,40}$ ]]` in cloud-init.

### Bug 3 — `VcpuLimitExceeded` from instance not yet terminated

**Symptom**: rapid re-bake hits the 4-vCPU G-family quota because the
prior bake instance is still in `shutting-down`.

**Fix**: poll for `terminated` before retry in `launch.py`.

### Bug 4 — 200 GB root volume filled mid-download

**Symptom**: HF download fails at 100% disk usage. DLAMI base + pip
installs occupy ~140 GB before the HF weights begin.

**Fix**: `ROOT_VOLUME_GB = 300`; mid-script cache cleanup
(`apt clean`, `pip cache purge`, drop conda pkgs) recovers ~50 GB;
`df -h` checkpoints at every download boundary.

### Bug 5 — Upstream API guessed from paper, not from repo

**Symptom**: wrappers imported
`from preprocess.pose_align import extract_and_align` and
`from steadydancer.pipeline import SteadyDancerPipeline`. Neither
symbol exists at upstream main. Upstream exposes CLI entry points
(`preprocess/pose_align.py`, `generate_dancer.py`), not Python APIs.

**Fix**: wrappers in `ec2/models/dwpose.py` and
`ec2/models/steadydancer.py` invoke the upstream CLIs via the
unified `run_tool` subprocess runner.

### Bug 6 — mmcv compile failed: missing Python.h

**Symptom**: `fatal error: Python.h: No such file or directory` during
mmcv build.

**Fix**: add `python3.11-dev build-essential` to the apt install line.

### Bug 7 — mmpose install: `chumpy` does `import pip` under PEP 517

**Symptom**: `chumpy` (transitive mmpose dep) `setup.py` does
`import pip` from a subprocess. Under PEP 517 build isolation, the
build venv has only setuptools+wheel — no pip.

**Fix**: `pip install --no-build-isolation chumpy==0.70` **before**
`pip install mmpose==1.3.2`.

### Bug 8 — HF cache path mismatch

**Symptom**: cache layout written by `snapshot_download` did not match
the path the upstream loader read from.

**Fix**: align the cache layout with `HF_HOME` and the loader's
expected `models--<owner>--<repo>/snapshots/<sha>` layout.

### Bug 9 — Smoke test ran on synthetic gray photo

**Symptom**: smoke harness used a synthetic gray image, which passed
import-level checks but exercised none of the real-data code paths.

**Fix**: smoke now exercises a 1-frame 1-step real inference. Caught
later bugs at bake time rather than first-production-run time.

---

## Production-run bugs (10-22): cloud-init, AWS, and first inference

### Bug 10 — yt-dlp transient rate-limit

**Symptom**: `prepare --reel <url>` intermittently fails with HTTP 429.

**Fix**: exponential-backoff retry in `prepare/reel_fetcher.py`.

### Bug 11 — Cloud-init template was Python `.format()` over `${VAR}`

**Symptom**: `${VAR}` placeholders in the cloud-init template were
fed to Python `str.format()`, raising `KeyError`. Bash `${VAR}` and
Python `{var}` overlap badly.

**Fix**: switched to `string.Template.safe_substitute` (uses `${}`
syntax natively); Python f-string interpolation deferred to a
preprocessing step before substitution.

### Bug 12 — `request_ondemand` passed `SubnetId=None`

**Symptom**: boto3 rejected `SubnetId=None`. The optional field was
forwarded unconditionally.

**Fix**: drop `SubnetId` from the request kwargs when unset.

### Bug 13 — Cloud-init did not export `HF_TOKEN`

**Symptom**: `snapshot_download` of gated repos failed; HF returned
401 because the token environment variable was not set when the
orchestrator's Python subprocess ran.

**Fix**: cloud-init now exports `HF_TOKEN` before invoking the
orchestrator.

### Bug 14 — `VcpuLimitExceeded` on rapid relaunch (recurring)

**Symptom**: same root cause as Bug 3; recurred after a different
launch path was added.

**Fix**: poll for `terminated` before launch in both code paths.

### Bug 15 — Cluster: DWPose ckpts missing + numpy 2.x ABI + JIT SEGV

**Symptom**: pose_extract SEGV. Three independent issues, each
masking the next:

1. DWPose `yolox_l_8x8_300e_coco.pth` at the default path; HF cache
   had `dw-ll_ucoco_384.pth` and `yolox_l.onnx` but not the `.pth`.
   The default-path lookup failed inside a C extension -> NULL deref
   -> SEGV.
2. numpy 2.4.3 vs mmcv compiled against numpy 1.x -> C ABI break at
   runtime.
3. `torch.distributed.optim.functional_*` SEGV at import on the
   DLAMI's libstdc++/libtorch combo (upstream-deprecated, marked by
   torch's own deprecation warning).

**Fix**:
1. `setup_ami.py` `wget`s `yolox_l_8x8_300e_coco.pth` from the
   mmdetection release URL and symlinks `dw-ll_ucoco_384.pth` from HF
   cache into the upstream-expected `pretrained_weights/dwpose/`.
2. Pin `numpy>=1.26,<2` and `opencv-python-headless>=4.9,<4.13` in
   `pyproject.toml [ec2]`; defensive final pip step pins them again
   and uninstalls `opencv-python` (which can transitively bump numpy).
3. `export PYTORCH_JIT=0` in `/etc/profile.d/insta-influencer.sh` and
   inline in cloud-init.

### Bug 16 — `--convert_model_dtype` unrecognised

**Symptom**: `generate_dancer.py: unrecognized argument:
--convert_model_dtype`. Argument from a stale README example.

**Fix**: removed the flag from `ec2/models/steadydancer.py`. Surviving
args verified against `generate_dancer.py --help`.

### Bug 17 — `--cond_pos_folder` expects frames, not a video

**Symptom**: `FileNotFoundError: .../pose/0000.jpg`. The DWPose
wrapper produced `aligned_pose.mp4`; upstream expected numbered JPGs.

**Fix**: after `pose_align.py` succeeds, run
`ffmpeg -i aligned_pose.mp4 -start_number 0 -q:v 2 %04d.jpg` to
extract per-frame JPGs into the same `pose_dir`.

### Bug 18 — `--cond_neg_folder` missing -> `os.path.join(None, ...)`

**Symptom**: `TypeError: expected str, bytes or os.PathLike object,
not NoneType`. Upstream builds
`os.path.join(args.cond_neg_folder, "", f"{i:04d}.jpg")` for every
frame, used for the classifier-free guidance null branch.

**Fix**: `ec2/models/steadydancer.py` builds a sibling `pose_neg/`
directory and passes `--cond_neg_folder`. Initial implementation used
blank-black JPGs (Bug 33 cleaned this up to use real augmented poses).

### Bug 19 — GGUF quantised weights do not satisfy diffusers' loader

**Symptom**: `FileNotFoundError: ... does not appear to have a file
named diffusion_pytorch_model-00001-of-00007.safetensors`. The plan
was to use GGUF Q5_K_M weights; upstream `generate_dancer.py` calls
`WanModel.from_pretrained(checkpoint_dir)` via diffusers, which only
recognises `*.safetensors`.

**Fix**: `setup_ami.py` `allow_patterns` for
`Wan-AI/Wan2.1-I2V-14B-480P` now includes
`diffusion_pytorch_model*`, pulling the 7 bf16 shards (~28 GB). Total
HF cache size on the AMI is ~38 GB.

### Bug 20 — Insufficient host RAM causes swap thrashing

**Symptom**: on g6e.xlarge (30 GB RAM) with `--offload_model True`,
the system swapped during DiT layer streaming and animate stalled.

**Fix**: switched the production target to g6e.2xlarge (64 GB RAM).
Also flipped `--offload_model False` since the L40S has 48 GB VRAM
and the DiT fits on-device.

### Bug 21 — AMI baked stale `status.json`

**Symptom**: a baked AMI contained a previous job's `status.json` in
the work directory. Fresh jobs found pre-existing status and skipped
phases.

**Fix**: bake script clears `/opt/.../work/*` before `create-image`.

### Bug 22 — Cloud-init `set -e` aborted before `terminate-instances`

**Symptom**: orchestrator failure left the spot running because the
trailing `aws ec2 terminate-instances` never ran under `set -e`.

**Fix**: `trap` an explicit terminate on EXIT; remove `set -e` for
the terminate line; idempotent.

### Bug 23 — `AnimatePhase.timeout_s = 1800` too short for cold EBS

**Symptom**: first run after a fresh spot saw the DiT load take >30
min due to cold EBS lazy-load (pre-FSR fix). Phase timed out before
the model finished loading.

**Fix**: bumped to 5400 (later 12600 for chunked output).

### Bug 24 — `pip uninstall opencv-python` wiped `opencv-python-headless`

**Symptom**: both packages share the `cv2/` directory; uninstalling
one removed it for the other. `import cv2` failed.

**Fix**: explicit `pip install --force-reinstall opencv-python-headless`
after any uninstall step.

### Bug 25 — `create_ami` called `stop_instances` on a spot

**Symptom**: spot instances cannot be stopped (only terminated or
hibernated). `UnsupportedOperation`.

**Fix**: detect spot instances; skip the stop step (create-image
handles a running instance directly).

### Bug 26 — Buffered stderr lost on `TimeoutExpired`

**Symptom**: timeouts in animate left the real upstream error
invisible because `proc.communicate(timeout=...)` discarded the
buffered output on `TimeoutExpired`.

**Fix**: stream stdout+stderr through a thread that writes to the
phase log file as they arrive; on timeout, the log already has the
upstream banner.

### Bug 27 — Loading the wrong checkpoint silently random-inits 45+ tensors

**Symptom**: two consecutive 60-minute animate timeouts after all
other fixes landed. Caught via the stream-stderr diagnostic (Bug 26):

```
Some weights of WanModel were not initialized from the model checkpoint
at /opt/.../models--Wan-AI--Wan2.1-I2V-14B-480P/snapshots/... and are
newly initialized: ['condition_embedding_align.cross_attn.in_proj_weight',
'condition_embedding_align.ffn_pose.0.weight', 'patch_embedding_fuse.weight',
'patch_embedding_ref_c.weight', ... (45+ tensors)]
You should probably TRAIN this model on a down-stream task to be able to
use it for predictions and inference.
```

These tensors are SteadyDancer's pose-conditioning adapter; they
exist only in `MCG-NJU/SteadyDancer-14B`, not in the base
`Wan-AI/Wan2.1-I2V-14B-480P`. `from_pretrained` does not raise on
missing tensors — only prints a warning. With 45+ tensors of random
noise, the sampler never converged.

**Fix**: cloud-init downloads `MCG-NJU/SteadyDancer-14B` (~28 GB) on
fresh spot boot and passes that path as `--ckpt_dir`.

---

## Production-run bugs (28-47): post-first-run, quality, infrastructure

### Bug 28 — `--t5_cpu` with bf16 T5-XXL is unusably slow

**Symptom**: animate stalled for an hour. T5-XXL on CPU at bf16 with
single-threaded matmul is multi-hour-class slow.

**Fix**: `--t5_cpu False`. T5 fits on the L40S 48 GB alongside the DiT.

### Bug 29 — Tool timeout pre-empted phase timeout

**Symptom**: `GENERATE_DANCER_TOOL.timeout_s = 3600` raised
`TimeoutExpired` before the phase's 5400s budget elapsed.

**Fix**: bumped tool timeout to 5400, then to 12600 for chunked
output. Phase timeout always >= tool timeout.

### Bug 30 — Stream-to-log captured only stderr

**Symptom**: upstream `generate_dancer.py` logs warnings (including
the Bug 27 banner) to stdout. The phase log captured only stderr,
hiding diagnostic output.

**Fix**: `run_tool` `stream_to_log` mode now merges stdout into the
captured stream via `stderr=STDOUT`.

### Bug 31 — `--size 832x480` is sub-native

**Symptom**: README documents `--size 1024*576` for I2V-14B; the
pipeline was passing `832*480`. Output was visibly less coherent.

**Fix**: `--size 1024*576` (later `576*1024` portrait for Reels).

### Bug 32 — `--condition_guide_scale` default 1.5 not from README

**Symptom**: argparse default in upstream is 1.5; the README example
explicitly overrides to 1.0. Higher values amplified pose-detector
noise as extra-people artefacts.

**Fix**: `--condition_guide_scale 1.0` (paper §Implementation Details
`w^pose = 1.0`).

### Bug 33 — `pose_neg/` was 81 blank black JPGs

**Symptom**: the Bug 18 fix used blank-black JPGs as the negative
pose folder. This broke the CFG signal: the model could not learn the
difference between "real pose" and "no pose."

**Fix**: run `pose_align_withdiffaug.py` to produce real augmented
poses for the negative branch. Dual-pass extraction is now standard.

### Bug 34 — Generic prompt "a person dancing" had no identity anchor

**Symptom**: output identity drifted toward generic-dancer averages
because the prompt provided no anchoring information.

**Fix**: positive prompt is a single natural-language sentence with
subject description matching the target photo. See
[`02_settings_audit.md`](./02_settings_audit.md) §1.4.

### Bug 35 — DiT bf16 conversion takes ~10 min CPU per process

**Symptom**: every `generate_dancer.py` subprocess paid a ~10-minute
one-time CPU conversion of the DiT weights from fp32-stored to bf16.

**Fix**: daemonised inference
(`ec2/inference/generate_dancer_chunked.py`) loads the model once and
serves N chunks. Re-baking the AMI with pre-converted bf16 shards
would eliminate the per-launch cost as well.

### Bug 36 — `proc.returncode` referenced before assignment

**Symptom**: `NameError: name 'proc' is not defined` in stream-to-log
mode. The `else` branch bound `proc = subprocess.run(...)`; the `if
stream_to_log:` branch bound only `proc_p = Popen(...)`.

**Fix**: read `result.returncode` (populated in both branches).

### Bug 37 — `pose_align.py --max_frame 300` insufficient for chunked output

**Symptom**: chunked animate needed 81 pose frames per chunk; with up
to 10 chunks that is 810 pose frames. The 300 cap was a relic of
single-chunk era.

**Fix**: `--max_frame 500` (sized for up to 6 chunks comfortably).

### Bug 38 — `import skvideo.io` failed with ModuleNotFoundError

**Symptom**: Practical-RIFE `inference_video.py` line 10
`import skvideo.io` failed. The package was not installed.

**Fix**: add `scikit-video` to the cloud-init pip install line.

### Bug 39 — `skvideo` uses `np.float` (removed in numpy 1.20+)

**Symptom**: `AttributeError: module 'numpy' has no attribute
'float'` at `skvideo/io/abstract.py`.

**Fix**: `sed`-patch `np.float(`/`np.int(`/`np.bool(` -> `float(`/`int(`/`bool(`
across `skvideo/**/*.py`.

### Bug 40 — `basicsr` patch lookup imported the broken module

**Symptom**: the cloud-init patch step located
`basicsr/data/degradations.py` via `python -c "import
basicsr.data.degradations; print(...)"`. The import was exactly what
needed patching (it failed on
`torchvision.transforms.functional_tensor`), so the lookup returned
nothing and the patch silently skipped.

**Fix**: `find /opt/.../.venv -path '*basicsr/data/degradations.py'`
to locate the file without importing.

### Bug 41 — `pip install gfpgan` upgraded numpy past xtcocotools's wheel ABI

**Symptom**: `numpy.dtype size changed` ABI error in `xtcocotools`
(mmpose dep) after `gfpgan` install.

**Fix**: pin `numpy<2` in the gfpgan pip command and add a defensive
`pip install --force-reinstall --no-deps "numpy<2"` after. Verify
with `python -c "from xtcocotools.coco import COCO"`.

### Bug 42 — RIFE produced 5.18 s instead of 9.75 s output

**Symptom**: RIFE output was 1.88x fast-forward. Cause: RIFE's
`--multi` defaults to 2 regardless of `--fps`. `--fps=60` alone does
not change the multiplier.

**Fix**: pass `--multi=ceil(60/16)=4` explicitly in
`ec2/phases/interp.py`.

### Bug 43 — `KeyError: 'what'` from bare `{}` in template

**Symptom**: cloud-init template rendering crashed with `KeyError`
because bare `{}` in `find -exec` syntax and Python heredoc f-strings
collided with `str.format` placeholder syntax.

**Fix**: escape `{}` to `{{}}` for `find -exec` and `{e}` to `{{e}}`
inside Python heredoc f-strings within the template.

### Bug 44 — Postprocess seed left stale phase markers

**Symptom**: `seed_postprocess.py` set up `animate.done` but left
`interp.done` / `face_restore.done` / etc from a prior completion.
The orchestrator skipped every phase and reported "no final.mp4
produced."

**Fix**: `seed_postprocess.py` step 3b now explicitly deletes
`interp.done`, `face_restore.done`, `audio_attach.done`, and
`reels_format.done` markers.

### Bug 45 — `KEEP_ALIVE` template render: `Replacement index 0 out of range`

**Symptom**: the KEEP_ALIVE branch of the cloud-init template
contained a bare `{}` in `find -exec` that `str.format` interpreted
as a positional placeholder.

**Fix**: same as Bug 43 — escape to `{{}}`.

### Bug 46 — Marker honoured when phase outputs missing locally

**Symptom**: spot interruption mid-animate left
`markers/pose_extract.done` in S3 but the pose JPGs lived on the
reclaimed spot's EBS. The fresh spot skipped pose_extract because the
marker was present, then animate failed immediately on missing
`pose/0000.jpg`.

**Root cause**: `process_job`'s marker-skip check was unconditional:
`if storage.exists(marker): skip`. It never checked whether the
phase's outputs were present on this spot's disk.

**Fix**: `ec2/orchestrator.py` declares `_PHASE_OUTPUT_CHECKS` mapping
each phase to its primary on-disk output, and the skip check now
calls `_phase_outputs_present_locally(phase.name, work)`. When the
marker is present but outputs are missing, the phase re-runs and
logs `phase.marker_present_but_outputs_missing.rerun`.

### Bug 47 — `Config.HF_TOKEN` required on manual orchestrator launch

**Symptom**: pydantic-settings rejected `Config()` construction
because `HF_TOKEN` was missing from env. Cloud-init exports it before
invoking the orchestrator; a manual SSH `nohup` launch does not.

**Workaround**: scp a launcher script with the token inlined.

**Proper fix (pending)**: default `HF_TOKEN=""` in Config. `HF_TOKEN`
is only needed for the cloud-init SteadyDancer-14B download; once
the AMI is re-baked to include the model, it is not needed at all.
