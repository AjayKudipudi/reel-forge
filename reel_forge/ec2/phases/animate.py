"""Phase 2: SteadyDancer-14B inference via upstream generate_dancer.py.

For num_clips=1 (default), runs a single 81-frame animate against the
prepared photo + pose dirs and writes animated.mp4.

For num_clips>1, runs animate N times within the same orchestrator
process (model stays loaded on GPU between chunks — only the per-chunk
sampling work repeats). **Every chunk uses the original reference photo
as its first-frame condition** — re-anchoring identity per chunk. This
replaced the chained approach (chunk N's last frame → chunk N+1's first
frame, paper author's recommendation in MCG-NJU/SteadyDancer issue #17)
after the 2026-05-14 frame review (handoff §5.15) showed chaining
compounded identity drift across chunks. Per-chunk pose slices are
materialized as symlink dirs `pose_chunk_<i>/` and `pose_neg_chunk_<i>/`
under work_dir. The N produced clips are ffmpeg-concat'd losslessly
into animated.mp4 for downstream phases; minterpolate in the interp
phase motion-bridges any pose discontinuity at chunk boundaries.
"""
from __future__ import annotations

import time
from pathlib import Path

from ...core import keys as K
from ...core.errors import classify
from ...core.external_tool import run_tool
from ...core.result import PhaseContext, PhaseResult
from ...core.seed import seed_everything
from ...core.tools.ffmpeg import FFMPEG
from ..models.factory import get_animation_model


class AnimatePhase:
    name: str = "animate"
    # 12600s = 3.5 hours. Covers up to ~6 chunks of 81 frames at 1024x576 on
    # g6e.2xlarge L40S (each chunk samples 40 steps x 3 forwards ~= 30 min,
    # plus one-time 13 min model load + bf16 conversion). For num_clips=1
    # the actual wall is ~45 min so the timeout is a generous safety margin.
    timeout_s: int = 12600

    def run(self, ctx: PhaseContext) -> PhaseResult:
        seed_everything(ctx.seed)
        t0 = time.time()
        try:
            model = get_animation_model()
            model.load(quant=ctx.manifest.model.quant)

            num_clips = ctx.manifest.output.num_clips
            num_frames = ctx.manifest.output.num_frames

            full_pose_dir = ctx.work_dir / "pose"
            full_pose_neg_dir = ctx.work_dir / "pose_neg"
            original_photo = ctx.work_dir / K.PHOTO
            final_out = ctx.work_dir / K.ANIMATED

            if num_clips == 1:
                # Fast path — no slicing, no concat. Single inference straight
                # into animated.mp4.
                model.animate(
                    reference_image_path=original_photo,
                    pose_dir=full_pose_dir,
                    pose_neg_dir=full_pose_neg_dir,
                    prompt=ctx.manifest.prompt,
                    negative_prompt=ctx.manifest.negative_prompt or "",
                    num_frames=num_frames,
                    fps=ctx.manifest.output.fps,
                    seed=ctx.seed,
                    output_path=final_out,
                    progress_cb=ctx.on_progress,
                )
            else:
                # Chunked path. Pre-build all per-chunk pose dirs (cheap —
                # they're symlinks), then call the daemon ONCE to run all
                # inferences with a single model load (~10 min DiT bf16
                # conversion paid once instead of N times). The daemon
                # passes the original reference photo as the first-frame
                # condition for EVERY chunk — re-anchoring identity each
                # chunk (post-§5.15 fix; chaining compounded face drift).
                chunk_outputs: list[Path] = []
                chunk_specs: list[dict[str, Path | int]] = []
                for i in range(num_clips):
                    pose_chunk = _build_chunk_pose_dir(
                        full_pose_dir, ctx.work_dir / f"pose_chunk_{i}", i, num_frames,
                    )
                    pose_neg_chunk = _build_chunk_pose_dir(
                        full_pose_neg_dir, ctx.work_dir / f"pose_neg_chunk_{i}", i, num_frames,
                    )
                    chunk_out = ctx.work_dir / f"animated_chunk_{i}.mp4"
                    chunk_outputs.append(chunk_out)
                    chunk_specs.append({
                        "pose_dir": pose_chunk,
                        "pose_neg_dir": pose_neg_chunk,
                        "output_path": chunk_out,
                        # Vary seed across chunks; same seed would freeze
                        # diffusion sampling identically (waste of compute).
                        "seed": ctx.seed + i,
                    })

                # Real SteadyDancer model exposes animate_chunks; the fake
                # doesn't (and num_clips>1 doesn't make sense in tests).
                # Fall back to the per-chunk loop if the model lacks it.
                animate_chunks = getattr(model, "animate_chunks", None)
                if callable(animate_chunks):
                    animate_chunks(
                        first_image_path=original_photo,
                        chunk_specs=chunk_specs,
                        prompt=ctx.manifest.prompt,
                        negative_prompt=ctx.manifest.negative_prompt or "",
                        num_frames=num_frames,
                        work_dir=ctx.work_dir,
                        progress_cb=ctx.on_progress,
                    )
                else:
                    # Fallback for models that don't expose animate_chunks
                    # (currently only the Fake test model). Every chunk
                    # uses the original photo — matches the daemon path's
                    # post-§5.15 behavior.
                    for i, spec in enumerate(chunk_specs):
                        ctx.on_progress(f"animate chunk {i + 1}/{num_clips}")
                        model.animate(
                            reference_image_path=original_photo,
                            pose_dir=spec["pose_dir"],  # type: ignore[arg-type]
                            pose_neg_dir=spec["pose_neg_dir"],  # type: ignore[arg-type]
                            prompt=ctx.manifest.prompt,
                            negative_prompt=ctx.manifest.negative_prompt or "",
                            num_frames=num_frames,
                            fps=ctx.manifest.output.fps,
                            seed=int(spec["seed"]),  # type: ignore[arg-type]
                            output_path=spec["output_path"],  # type: ignore[arg-type]
                            progress_cb=ctx.on_progress,
                        )

                _concat_chunks(chunk_outputs, final_out, ctx.work_dir)

            # Upload animate's outputs to S3 so a later "postprocess" launch
            # can skip pose_extract + animate by pre-seeding work_dir from
            # S3. Cheap upload (a 5-10MB animated.mp4 + small chunk files)
            # vs the ~1h 34m animate phase saves ~$1.50 / 1.5h per iteration
            # when only post-process tuning (RIFE, GFPGAN, prompts that don't
            # affect generation, audio mux changes) is being validated.
            try:
                ctx.storage.upload(final_out, f"{ctx.s3_prefix}/{K.ANIMATED}")
                if num_clips > 1:
                    for ch in chunk_outputs:
                        if ch.exists():
                            ctx.storage.upload(ch, f"{ctx.s3_prefix}/{ch.name}")
            except Exception as upload_err:
                # Non-fatal — animate succeeded, just don't have post-process
                # short-cut for this job.
                ctx.logger.warning("animate.upload_artifacts_failed", err=str(upload_err))

            return PhaseResult.ok(
                stats={
                    "wall_s": round(time.time() - t0, 2),
                    "quant": ctx.manifest.model.quant,
                    "frames": num_frames * num_clips,
                    "clips": num_clips,
                },
                artifacts={"animated": final_out},
            )
        except Exception as exc:
            info = classify(exc)
            return PhaseResult.fail(
                error_class=info.error_class,
                message=info.message,
                retryable=info.retryable,
                stats={"wall_s": round(time.time() - t0, 2)},
                stderr_tail=info.stderr_tail,
            )


def _build_chunk_pose_dir(
    full_dir: Path,
    chunk_dir: Path,
    clip_index: int,
    frames_per_clip: int,
) -> Path:
    """Materialize a per-chunk pose directory of `frames_per_clip` frames
    named 0000.jpg..NNNN.jpg, symlinking from the full pose track.

    generate_dancer.py reads `cond_pos_folder` via `range(args.frame_num)`
    so frames MUST start at 0000.jpg; we use symlinks to avoid copies.
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    start = clip_index * frames_per_clip
    for j in range(frames_per_clip):
        src = full_dir / f"{start + j:04d}.jpg"
        if not src.exists():
            raise FileNotFoundError(
                f"chunk {clip_index} frame {j} ({src}) missing — pose extraction "
                f"produced fewer than {start + frames_per_clip} frames. Either the "
                f"source video is too short for num_clips={clip_index + 1}+ chunks, "
                f"or --max_frame in dwpose.py needs raising.",
            )
        dst = chunk_dir / f"{j:04d}.jpg"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    return chunk_dir


def _concat_chunks(chunks: list[Path], out: Path, work_dir: Path) -> Path:
    """ffmpeg-concat the N animated chunk mp4s LOSSLESSLY with `-c copy`.

    Was briefly xfade'd in v8.7 for smoother boundaries but visually it
    produced an "attached" double-image (opacity blend shows both clips
    simultaneously during the fade) — worse than the hard cut it replaced.

    The smoothing of chunk boundaries is now done DOWNSTREAM by the interp
    phase: minterpolate's motion-compensated frame interpolation sees the
    concat'd 16fps stream as continuous video and auto-bridges the
    chunk-N → chunk-N+1 transition via optical-flow synthesis, producing
    motion-coherent in-between frames rather than a visible cross-dissolve.

    Output length: N * (frame_num / sample_fps) = 2 * 5.0625 = 10.125s
    (no overlap loss).
    """
    if len(chunks) == 1:
        run_tool(
            FFMPEG,
            ["-hide_banner", "-loglevel", "error", "-y",
             "-i", str(chunks[0]), "-c", "copy", str(out)],
        )
        return out
    listfile = work_dir / "_chunk_concat.txt"
    listfile.write_text("".join(f"file '{p.resolve()}'\n" for p in chunks))
    run_tool(
        FFMPEG,
        [
            "-hide_banner", "-loglevel", "error",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(listfile),
            "-c", "copy",
            str(out),
        ],
    )
    if not out.exists():
        raise FileNotFoundError(
            f"ffmpeg concat exited 0 but did not produce {out}",
        )
    return out
