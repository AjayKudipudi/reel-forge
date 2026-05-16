"""Batched generate_dancer — loads the model ONCE and runs N inferences.

The upstream `generate_dancer.py` CLI loads SteadyDancer-14B + does a CPU
fp32→bf16 conversion of the 28GB DiT weights on every invocation
(~10-13 min). For chunked generation (manifest.output.num_clips > 1) we
amortize that cost: load once, run N inferences, exit.

This script mirrors the upstream's i2v branch of `generate_dancer.py:358-440`
and `wan/image2video_dancer.py:generate()` but is driven by a JSON spec
file instead of argparse.

Spec JSON format:
{
  "shared": {
    "task": "i2v-14B",
    "ckpt_dir": "/opt/.../SteadyDancer-14B/snapshots/...",
    "size": "1024*576",
    "frame_num": 81,
    "offload_model": true,
    "t5_cpu": false,
    "prompt": "...",
    "negative_prompt": "...",
    "sample_solver": "unipc",
    "sample_steps": 40,
    "sample_shift": 5.0,
    "sample_guide_scale": 5.0,
    "condition_guide_scale": 1.0,
    "st_cond_cfg": 0.1,
    "end_cond_cfg": 0.4
  },
  "first_image": "/opt/.../photo.png",
  "chunks": [
    {"cond_pos_folder": "...", "cond_neg_folder": "...",
     "save_file": "...animated_chunk_0.mp4", "base_seed": 42},
    {"cond_pos_folder": "...", "cond_neg_folder": "...",
     "save_file": "...animated_chunk_1.mp4", "base_seed": 43}
  ]
}

Every chunk uses `first_image` (the original reference photo) for its
first-frame condition. Prior versions chained chunks (chunk N+1's first
image = chunk N's GPU-tensor last frame) to make pose transitions smooth,
but per the 2026-05-14 frame review (handoff §5.15) that compounded
identity drift across chunks — each chunk inherited the prior chunk's
already-drifted face, never re-anchoring to the real photo. Re-anchoring
every chunk caps drift to one chunk's worth (81 frames @ 16fps = 5s).
The resulting pose discontinuity at chunk boundaries is mitigated
downstream by the interp phase's `minterpolate` motion-bridging.

Usage:
  python -m reel_forge.ec2.inference.generate_dancer_chunked <spec.json>

Runs under PYTHONPATH=/opt/insta-influencer/third_party/SteadyDancer so
`import wan` resolves to the upstream repo. Caller sets that env var.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
import wan  # type: ignore[import-not-found]
from PIL import Image
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # type: ignore[import-not-found]
from wan.utils.utils import cache_video  # type: ignore[import-not-found]


def _build_condition_frames(
    folder: Path,
    frame_num: int,
    target_size: tuple[int, int],
) -> list[Image.Image]:
    """Load `frame_num` per-frame JPGs (0000.jpg..NNNN.jpg) from `folder`
    and resize each to `target_size`. Mirrors upstream's loop at
    generate_dancer.py:371-378.
    """
    out = []
    for i in range(frame_num):
        path = folder / f"{i:04d}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"pose frame missing: {path}")
        img = Image.open(path).convert("RGB").resize(target_size, Image.Resampling.BICUBIC)
        out.append(img)
    return out


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: generate_dancer_chunked <spec.json>", file=sys.stderr)
        sys.exit(2)
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text())
    shared = spec["shared"]
    chunks = spec["chunks"]
    first_image_path = Path(spec["first_image"])

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    cfg = WAN_CONFIGS[shared["task"]]
    logging.info(
        f"Loading WanI2VDancer (one-time setup; will serve {len(chunks)} chunks)",
    )
    wan_i2v = wan.WanI2VDancer(
        config=cfg,
        checkpoint_dir=shared["ckpt_dir"],
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=shared["t5_cpu"],
        st_cond_cfg=shared["st_cond_cfg"],
        end_cond_cfg=shared["end_cond_cfg"],
    )
    logging.info("Model loaded; starting batch inference.")

    # Open the reference photo ONCE — every chunk uses it as its first-frame
    # condition. See module docstring + handoff §5.15: chaining compounded
    # identity drift across chunks; re-anchoring caps drift per chunk.
    reference_img = Image.open(first_image_path).convert("RGB")

    n_prompt = shared.get("negative_prompt", "") or ""

    for idx, chunk in enumerate(chunks):
        logging.info(f"=== Chunk {idx + 1}/{len(chunks)} → {chunk['save_file']} ===")
        img = reference_img.copy()
        img_x = img.copy()

        cond_pos_folder = Path(chunk["cond_pos_folder"])
        cond_neg_folder = Path(chunk["cond_neg_folder"])
        frame_num = shared["frame_num"]

        condition_pos = _build_condition_frames(cond_pos_folder, frame_num, img.size)
        condition_neg = _build_condition_frames(cond_neg_folder, frame_num, img.size)
        img_c = condition_pos[0]  # first pose frame, resized to img.size

        logging.info(f"Generating chunk {idx + 1}/{len(chunks)} ...")
        video = wan_i2v.generate(
            shared["prompt"],
            img,
            img_x=img_x,
            img_c=img_c,
            condition=condition_pos,
            condition_null=condition_neg,
            max_area=MAX_AREA_CONFIGS[shared["size"]],
            frame_num=frame_num,
            shift=shared["sample_shift"],
            sample_solver=shared["sample_solver"],
            sampling_steps=shared["sample_steps"],
            guide_scale=shared["sample_guide_scale"],
            condition_guide_scale=shared["condition_guide_scale"],
            n_prompt=n_prompt,
            seed=chunk["base_seed"],
            offload_model=shared["offload_model"],
        )

        save_file = chunk["save_file"]
        logging.info(f"Saving chunk {idx + 1} to {save_file}")
        cache_video(
            tensor=video[None],
            save_file=save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        # Free the video tensor before next iteration to keep GPU mem flat.
        del video
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logging.info(f"All {len(chunks)} chunks complete.")


if __name__ == "__main__":
    main()
