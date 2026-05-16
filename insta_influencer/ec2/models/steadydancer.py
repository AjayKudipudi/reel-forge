"""Real SteadyDancer loader — subprocess wrapper around upstream `generate_dancer.py`.

Upstream is CLI-driven, not a Python API. We invoke it via run_tool and
read the produced mp4. Args verified against MCG-NJU/SteadyDancer main
README inference example:
  --task i2v-14B
  --ckpt_dir <SteadyDancer-14B weights dir from HF cache>
  --image <reference image, 1024x576>
  --cond_pos_folder <pose JPGs from pose_align.py>
  --cond_neg_folder <pose JPGs from pose_align_withdiffaug.py>
  --prompt "..." / --frame_num 81 / --size 1024*576
  --save_file <output.mp4>  / --base_seed <int>
  --offload_model True (encoders offload back to CPU; DiT stays on GPU)
  --condition_guide_scale 1.0 (README example value, not argparse default 1.5)
"""
from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import structlog

from ...core.errors import ErrorClass
from ...core.external_tool import ToolSpec, run_tool

UPSTREAM_REPO_DIR = Path("/opt/insta-influencer/third_party/SteadyDancer")
log = structlog.get_logger(__name__)


def _classify_generate(_rc: int, _out: str, err: str) -> ErrorClass:
    e = err.lower()
    if "out of memory" in e or "cuda out of memory" in e:
        return ErrorClass.MODEL_OOM
    if "checkpoint" in e and ("not found" in e or "no such" in e):
        return ErrorClass.MODEL_LOAD_FAILED
    return ErrorClass.INFERENCE_ERROR


GENERATE_DANCER_TOOL = ToolSpec(
    name="generate_dancer.py",
    binary=sys.executable,
    # Match AnimatePhase.timeout_s — the phase wrapper protects us. Earlier
    # value of 3600 was tighter than the phase timeout and pre-empted it,
    # SIGKILLing generate_dancer.py at exactly 60 min while it was still
    # mid-T5-prompt-encoding on CPU.
    timeout_s=5400,
    classifier=_classify_generate,
)


# Separate ToolSpec for the chunked batch script (loads model once, runs N
# inferences). Timeout scales with chunk count: one model load + N samplings.
# Set to the same 12600s ceiling as AnimatePhase.timeout_s (3.5 h) — covers
# up to ~6 chunks worst case.
GENERATE_DANCER_CHUNKED_TOOL = ToolSpec(
    name="generate_dancer_chunked",
    binary=sys.executable,
    timeout_s=12600,
    classifier=_classify_generate,
)


def _hf_cache_dir() -> Path:
    # AMI bake (cloud-init root) and runtime (orchestrator under root) both
    # set HF_HOME=/opt/insta-influencer/hf-cache via setup_ami user-data so
    # the cache is predictable across boot states.
    return Path(os.environ.get("HF_HOME", "/opt/insta-influencer/hf-cache")) / "hub"


def _resolve_ckpt_dir(repo_id: str) -> Path:
    """Resolve a snapshot_download-style path under the HF cache."""
    snapshot_root = _hf_cache_dir() / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshot_root.exists():
        raise FileNotFoundError(f"HF cache miss: {snapshot_root}")
    snapshots = sorted(snapshot_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not snapshots:
        raise FileNotFoundError(f"no snapshots under {snapshot_root}")
    return snapshots[0]


class SteadyDancerModel:
    name: str = "steadydancer-14b"

    def __init__(self) -> None:
        self._quant: str | None = None
        self._ckpt_dir: Path | None = None
        self._gguf_dir: Path | None = None

    def load(self, *, quant: str) -> None:
        from ...config import get_config

        cfg = get_config()
        # The CKPT_DIR must point at MCG-NJU/SteadyDancer-14B — the fine-tuned
        # model. Wan-AI/Wan2.1-I2V-14B-480P is just the BASE Wan2.1 weights
        # and is missing SteadyDancer's pose-conditioning adapter layers
        # (45+ tensors: condition_embedding_*, patch_embedding_fuse, etc.).
        # Loading the base silently randomly-initializes those tensors,
        # producing a model that generates garbage and never converges
        # (Bug 27 — see handoff).
        self._ckpt_dir = _resolve_ckpt_dir(cfg.HF_STEADYDANCER)
        self._gguf_dir = _resolve_ckpt_dir(cfg.HF_STEADYDANCER_GGUF)
        self._quant = quant
        log.info(
            "steadydancer.resolved",
            ckpt_dir=str(self._ckpt_dir),
            gguf_dir=str(self._gguf_dir),
            quant=quant,
        )

    def animate(
        self,
        *,
        reference_image_path: Path,
        pose_dir: Path,
        pose_neg_dir: Path,
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        fps: int,
        seed: int,
        output_path: Path,
        progress_cb: Callable[[str], None],
    ) -> Path:
        if self._ckpt_dir is None:
            self.load(quant=self._quant or "gguf-q5-m")
        assert self._ckpt_dir is not None

        env = os.environ.copy()
        env["PYTHONPATH"] = str(UPSTREAM_REPO_DIR)

        # --cond_neg_folder must contain per-frame JPGs from pose_align_withdiffaug.py
        # (the differentially-augmented pose track). Caller passes the path
        # explicitly so chunked generation can supply different per-chunk
        # pose_neg dirs without renaming gymnastics inside this wrapper.
        if not pose_neg_dir.exists() or not (pose_neg_dir / "0000.jpg").exists():
            raise FileNotFoundError(
                f"expected per-frame JPGs in {pose_neg_dir} (produced by "
                f"DwPoseExtractor's pose_align_withdiffaug.py step); got nothing"
            )

        progress_cb("invoking generate_dancer.py")
        t0 = time.time()
        # Stream stderr live to a file so we have a post-mortem even when
        # generate_dancer.py is SIGKILLed by timeout — the `capture_output`
        # buffer is lost on TimeoutExpired but a streaming file isn't.
        stream_log = output_path.parent / "logs" / "generate_dancer.live.log"

        def _upload_log_to_s3() -> None:
            try:
                import boto3

                from ...config import get_config
                cfg = get_config()
                if not stream_log.exists() or not cfg.S3_BUCKET:
                    return
                key = (
                    f"{cfg.S3_PREFIX}/{output_path.parent.name}"
                    "/_runtime-logs/generate_dancer.live.log"
                )
                boto3.client("s3", region_name=cfg.AWS_REGION).upload_file(
                    str(stream_log), cfg.S3_BUCKET, key,
                )
                log.info("generate_dancer.live_log_uploaded", key=key)
            except Exception as up_err:
                log.warning("generate_dancer.live_log_upload_failed", err=str(up_err))

        try:
            run_tool(
                GENERATE_DANCER_TOOL,
                stream_to_log=stream_log,
                args=[
                str(UPSTREAM_REPO_DIR / "generate_dancer.py"),
                "--task", "i2v-14B",
                "--ckpt_dir", str(self._ckpt_dir),
                "--image", str(reference_image_path),
                "--cond_pos_folder", str(pose_dir),
                "--cond_neg_folder", str(pose_neg_dir),
                "--prompt", prompt,
                "--frame_num", str(num_frames),
                # 576*1024 portrait — matches Instagram Reels native 9:16
                # aspect (1080x1920 target after reels_format upscale, NO
                # letterbox bars). Was 1024*576 landscape which produced
                # heavy top/bottom black bars in the final Reel and made
                # the dancer appear small.
                "--size", "576*1024",
                "--save_file", str(output_path),
                "--base_seed", str(seed),
                # Pose-condition CFG strength. README example uses 1.0; the
                # upstream argparse default of 1.5 is too strong and amplifies
                # pose-detector noise (spurious detections leak into output as
                # extra people / artifacts).
                "--condition_guide_scale", "1.0",
                # offload_model=True moves T5 + CLIP back to CPU after their
                # one-time encoding step, freeing VRAM before DiT is pinned to
                # GPU for the sampling loop. DiT itself stays on GPU throughout
                # sampling (upstream image2video_dancer.py:366 — .to(device) is
                # OUTSIDE the loop, contrary to the previous handoff's read).
                # Peak GPU: ~24 GB during encode, ~30 GB during sample → fits
                # comfortably on L40S 48 GB. The per-step CPU↔GPU shuffle for
                # noise predictions adds ~1-3% wall, not the 30-40% earlier
                # claimed.
                #
                # We DROP --t5_cpu: running T5-XXL (11B params, bf16) forward
                # on CPU stalled prompt encoding for 40+ min (single-threaded
                # bf16 oneDNN kernels). With T5 on GPU, encoding is sub-second.
                "--offload_model", "True",
            ],
                cwd=UPSTREAM_REPO_DIR,
                env=env,
            )
        except Exception:
            _upload_log_to_s3()
            raise
        _upload_log_to_s3()
        log.info("animate.done", out=str(output_path), wall_s=round(time.time() - t0, 2))
        return output_path

    def animate_chunks(
        self,
        *,
        first_image_path: Path,
        chunk_specs: list[dict[str, Path | int]],
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        work_dir: Path,
        progress_cb: Callable[[str], None],
    ) -> None:
        """Run N inferences with ONE model load via
        `generate_dancer_chunked.py`. Each chunk_spec is a dict with keys
        `pose_dir`, `pose_neg_dir`, `output_path`, `seed`. Every chunk
        uses `first_image_path` as its first-frame condition — identity
        re-anchors per chunk (post-§5.15 fix). `negative_prompt` is
        forwarded via the spec to `wan_i2v.generate(n_prompt=...)`; if
        empty, the upstream model falls back to its default Chinese
        sample_neg_prompt.

        Saves ~10 min x (N-1) chunks vs N independent run_tool calls (each
        of which would otherwise pay the DiT bf16 conversion cost separately).
        """
        if self._ckpt_dir is None:
            self.load(quant=self._quant or "gguf-q5-m")
        assert self._ckpt_dir is not None

        env = os.environ.copy()
        # Daemon imports `wan` from the upstream repo; PYTHONPATH must include
        # both our project root (for the daemon module itself) and the upstream.
        existing = env.get("PYTHONPATH", "")
        proj_root = Path(__file__).resolve().parent.parent.parent.parent
        env["PYTHONPATH"] = (
            f"{proj_root}:{UPSTREAM_REPO_DIR}" + (f":{existing}" if existing else "")
        )

        # Build spec JSON.
        chunks_payload: list[dict[str, str | int]] = []
        for s in chunk_specs:
            chunks_payload.append(
                {
                    "cond_pos_folder": str(s["pose_dir"]),
                    "cond_neg_folder": str(s["pose_neg_dir"]),
                    "save_file": str(s["output_path"]),
                    "base_seed": int(s["seed"]),  # type: ignore[arg-type]
                },
            )
        spec = {
            "shared": {
                "task": "i2v-14B",
                "ckpt_dir": str(self._ckpt_dir),
                "size": "576*1024",
                "frame_num": int(num_frames),
                "offload_model": True,
                "t5_cpu": False,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "sample_solver": "unipc",
                "sample_steps": 40,
                "sample_shift": 5.0,
                # Paper recipe (arxiv 2511.19320 §Implementation Details):
                # w^txt=5.0, w^pose=1.0, DC-CFG window [0.1, 0.4]. Reverted
                # from 2026-05-14's 6.0 after researching upstream defaults:
                # the higher CFG was causing classical over-satisfaction
                # artifacts (extra hands when "complete hands" enumerated in
                # positive prompt; deformed fingers on fast motion). Wan2.1
                # README also documents 6.0 only for T2V-1.3B, not I2V-14B.
                "sample_guide_scale": 5.0,
                "condition_guide_scale": 1.0,
                "st_cond_cfg": 0.1,
                "end_cond_cfg": 0.4,
            },
            "first_image": str(first_image_path),
            "chunks": chunks_payload,
        }
        spec_path = work_dir / "_chunked_spec.json"
        spec_path.write_text(__import__("json").dumps(spec, indent=2))

        progress_cb(f"invoking generate_dancer_chunked.py ({len(chunk_specs)} chunks)")
        t0 = time.time()
        stream_log = work_dir / "logs" / "generate_dancer_chunked.live.log"

        def _upload_log_to_s3() -> None:
            try:
                import boto3

                from ...config import get_config
                cfg = get_config()
                if not stream_log.exists() or not cfg.S3_BUCKET:
                    return
                key = (
                    f"{cfg.S3_PREFIX}/{work_dir.name}"
                    "/_runtime-logs/generate_dancer_chunked.live.log"
                )
                boto3.client("s3", region_name=cfg.AWS_REGION).upload_file(
                    str(stream_log), cfg.S3_BUCKET, key,
                )
                log.info("generate_dancer_chunked.live_log_uploaded", key=key)
            except Exception as up_err:
                log.warning("generate_dancer_chunked.live_log_upload_failed", err=str(up_err))

        try:
            run_tool(
                GENERATE_DANCER_CHUNKED_TOOL,
                stream_to_log=stream_log,
                args=[
                    "-m",
                    "insta_influencer.ec2.inference.generate_dancer_chunked",
                    str(spec_path),
                ],
                cwd=UPSTREAM_REPO_DIR,
                env=env,
            )
        except Exception:
            _upload_log_to_s3()
            raise
        _upload_log_to_s3()

        # Verify all chunk outputs exist.
        for s in chunk_specs:
            out_raw = s["output_path"]
            assert isinstance(out_raw, Path), (
                f"chunk_specs['output_path'] must be Path, got {type(out_raw)}"
            )
            if not out_raw.exists():
                raise FileNotFoundError(
                    f"chunked daemon exited 0 but did not produce {out_raw}",
                )

        log.info(
            "animate_chunks.done",
            num_chunks=len(chunk_specs),
            wall_s=round(time.time() - t0, 2),
        )
